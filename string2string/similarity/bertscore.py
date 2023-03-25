"""
    This class contains the original implementation of the BERTScore algorithm by Zhang et al. (2020).

    BERTScore: Evaluating Text Generation with BERT

    @inproceedings{bertscore2020,
        title={BERTScore: Evaluating Text Generation with BERT},
        author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
        booktitle={International Conference on Learning Representations},
        year={2020},
        url={https://openreview.net/forum?id=SkeHuCVFDr}
    }

    Disclaimer:
        This code is adapted from https://github.com/Tiiiger/bert_score
"""

from typing import List, Union, Optional, Tuple

import os
import sys
import time
import pandas as pd
from collections import defaultdict
import torch
from bert_score.utils import (bert_cos_score_idf, get_hash, 
                              get_idf_dict, get_model, get_tokenizer,
                              lang2model, model2layers)


class BERTScore:
    """
    This class implements the BERTScore algorithm.
    """
    
    def __init__(self, 
        model_name_or_path: str = None,
        lang: str = None,
        num_layers: int = None,
        all_layers: bool = False,
        use_fast_tokenizer: bool = False,
        device: str = 'cpu',
        baseline_path: str = None,
    ) -> None:
        r"""
        This function initializes the BERTScore class, which computes the BERTScore between two texts.

        Arguments:
            model_name_or_path (str): BERT model type to use (e.g., bert-base-uncased).
            lang (str): Language of the texts (e.g., en).
            num_layers (int): Number of layers to use.
            all_layers (bool): Whether to use all layers
            use_fast_tokenizer (bool): Whether to use the fast tokenizer.
            device (str): Device to use (e.g., cpu or cuda).
            baseline_path (str): Path to the baseline file.

        Returns:
            None

        Raises:
            ValueError: If model_name_or_path and lang are both None.

        .. attention::

            If you use this class, please make sure to cite the following paper:
        
            .. code-block:: latex

                @inproceedings{bertscore2020,
                    title={BERTScore: Evaluating Text Generation with BERT},
                    author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
                    booktitle={International Conference on Learning Representations},
                    year={2020},
                    url={https://openreview.net/forum?id=SkeHuCVFDr}
                }

        
        .. note::
            * If model_name_or_path is not specified, use the default model for the language.
            * If num_layers is not specified, use the default number of layers.
            * If device is not specified, use the GPU if available, otherwise use the CPU.
            * If baseline_path is not specified, use the default baseline file.
        """

        # Check the arguments
        if model_name_or_path is None and lang is None:
            raise ValueError("You must specify either model_name_or_path or lang")
        
        # Set the attributes
        self.model_name_or_path = model_name_or_path
        self.lang = lang
        self.num_layers = num_layers
        self.all_layers = all_layers
        self.use_fast_tokenizer = use_fast_tokenizer
        self.baseline_path = baseline_path

        # If model_name_or_path is not specified, use the default model for the language
        if self.model_name_or_path is None:
            self.lang = lang.lower()
            self.model_name_or_path = lang2model[self.lang]

        # If num_layers is not specified, use the default number of layers
        if num_layers is None:
            self.num_layers = model2layers[self.model_name_or_path]
        
        # Set the device
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        self.tokenizer = get_tokenizer(self.model_name_or_path, self.use_fast_tokenizer)
        self.model = get_model(self.model_name_or_path, self.num_layers, self.all_layers)
        self.model.eval()
        self.model.to(device)


    # Compute the BERTScore between source sentences and target sentences
    def compute(self,
        source_sentences: List[str],
        target_sentences: Union[List[str], List[List[str]]],
        batch_size: int = 4,
        idf: bool = False,
        nthreads: int = 4,
        return_hash: bool = False,
        rescale_with_baseline: bool = False,
        verbose: bool = False,
    ) -> Union[dict, Optional[str]]:
        """
        This function scores the source sentences based on their similarity to the target sentences using BERTScore.

        Arguments:
            source_sentences (list of str): candidate sentences
            target_sentences (list of str or list of list of str): reference sentences
            batch_size (int): bert score processing batch size
            idf (bool or dict): use idf weighting, can also be a precomputed idf_dict
            nthreads (int): number of threads
            return_hash (bool): return hashcode of the setting
            rescale_with_baseline (bool): rescale bertscore with pre-computed baseline
            verbose (bool): turn on intermediate status update

        Returns:
            (Dict[str, Tensor], Optional[str]): A dictionary containing the precision, recall, and F1 score, and the hashcode (if return_hash is True).
                where the precision, recall, and F1 score are tensors of shape (len(source_sentences),

        Raises:
            ValueError: If the number of source sentences and target sentences do not match.
        """

        # Check the arguments
        if len(source_sentences) != len(target_sentences):
            raise ValueError("The number of candidates and references do not match")
        
        # If the target sentences are grouped, flatten them
        ref_group_boundaries = None
        if not isinstance(target_sentences[0], str):
            ref_group_boundaries = []
            ori_source_sentences, ori_target_sentences = source_sentences, target_sentences
            source_sentences, target_sentences = [], []
            count = 0
            for cand, ref_group in zip(ori_source_sentences, ori_target_sentences):
                source_sentences += [cand] * len(ref_group)
                target_sentences += ref_group
                ref_group_boundaries.append((count, count + len(ref_group)))
                count += len(ref_group)

        if rescale_with_baseline and self.baseline_path is None:
            raise ValueError("Need to specify baseline_path when rescaling with baseline")

        # Get the IDF dict
        if not idf:
            idf_dict = defaultdict(lambda: 1.0)
            # set idf for [SEP] and [CLS] to 0
            idf_dict[self.tokenizer.sep_token_id] = 0
            idf_dict[self.tokenizer.cls_token_id] = 0
        elif isinstance(idf, dict):
            if verbose:
                print("using predefined IDF dict...")
            idf_dict = idf
        else:
            if verbose:
                print("preparing IDF dict...")
            start = time.perf_counter()
            idf_dict = get_idf_dict(target_sentences, self.tokenizer, nthreads=nthreads)
            if verbose:
                print("done in {:.2f} seconds".format(time.perf_counter() - start))

        if verbose:
            print("calculating scores...")
        
        start = time.perf_counter()

        # Get all the predictions
        all_preds = bert_cos_score_idf(
            model = self.model,
            refs = target_sentences,
            hyps = source_sentences,
            tokenizer= self.tokenizer,
            idf_dict = idf_dict,
            verbose = verbose,
            device = self.device,
            batch_size=batch_size,
            all_layers=self.all_layers,
        ).cpu()

        # If the target sentences are grouped, take the max score
        if ref_group_boundaries is not None:
            max_preds = []
            for beg, end in ref_group_boundaries:
                max_preds.append(all_preds[beg:end].max(dim=0)[0])
            all_preds = torch.stack(max_preds, dim=0)

        # Rescale with baseline
        use_custom_baseline = self.baseline_path is not None
        if rescale_with_baseline:
            if self.baseline_path is None:
                self.baseline_path = os.path.join(
                    os.path.dirname(__file__), f"rescale_baseline/{self.lang}/{self.model_name_or_path}.tsv"
                )
            if os.path.isfile(self.baseline_path):
                if not self.all_layers:
                    baselines = torch.from_numpy(
                        pd.read_csv(self.baseline_path).iloc[self.num_layers].to_numpy()
                    )[1:].float()
                else:
                    baselines = (
                        torch.from_numpy(pd.read_csv(self.baseline_path).to_numpy())[:, 1:]
                        .unsqueeze(1)
                        .float()
                    )

                all_preds = (all_preds - baselines) / (1 - baselines)
            else:
                print(
                    f"Warning: Baseline not Found for {self.model_name_or_path} on {self.lang} at {self.baseline_path}",
                    file=sys.stderr,
                )

        # Get the final output
        out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F
        scores = {
            "precision": out[0].numpy(),
            "recall": out[1].numpy(),
            "f1": out[2].numpy(),
        }
        
        # Print the time
        if verbose:
            time_diff = time.perf_counter() - start
            print(
                f"done in {time_diff:.2f} seconds, {len(target_sentences) / time_diff:.2f} sentences/sec"
            )

        # If return hash, return both the output and the hash
        if return_hash:
            return tuple(
                [
                    scores,
                    get_hash(
                        self.model_name_or_path,
                        self.num_layers,
                        idf,
                        rescale_with_baseline,
                        use_custom_baseline=use_custom_baseline,
                        use_fast_tokenizer=self.use_fast_tokenizer,
                    ),
                ]
            )
        # Otherwise, just return the output
        return scores
    

# bert_scorer = BERTScore(lang="en")
# predictions = ["hello there", "general kenobi"]
# references = ["hello there", "general kenobi"]
# score = bert_scorer.compute(predictions, references, return_hash=True)

# print('score:')
# print(score)

# # print the type of each element in the score tuple
# print('score type:')
# print(type(score[0][0]))
# print(type(score[0][1]))
# print(type(score[0][2]))
# print(type(score[1]))

# # score:
# # <class 'torch.Tensor'>
# # <class 'torch.Tensor'>
# # <class 'torch.Tensor'>
# # <class 'str'> / optional

# # score type: 
