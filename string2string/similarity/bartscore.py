"""
    This class contains the original implementation of the BARTScore algorithm by Yuan et al. (2021).
    
    BARTScore: BART-based Evaluation Metric for Text Generation

    @inproceedings{bartscore2021,
        author = {Yuan, Weizhe and Neubig, Graham and Liu, Pengfei},
        booktitle = {Advances in Neural Information Processing Systems},
        editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
        pages = {27263--27277},
        publisher = {Curran Associates, Inc.},
        title = {BARTScore: Evaluating Generated Text as Text Generation},
        url = {https://proceedings.neurips.cc/paper/2021/file/e4d2b6e6fdeca3e60e0f1a62fee3d9dd-Paper.pdf},
        volume = {34},
        year = {2021}
    }

    Disclaimer: 
        This code is adapted from https://github.com/neulab/BARTScore/blob/main/bart_score.py
"""

import numpy as np
from typing import List, Union, Dict
import traceback

import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration


# BARTScore class
class BARTScore:
    """
    This class implements the BARTScore algorithm.
    """
    
    def __init__(self, 
        model_name_or_path='facebook/bart-large-cnn',
        tokenizer_name_or_path: str = None,
        device: str = 'cpu',
        max_length=1024, 
        ) -> None:
        r"""
        This function initializes the BARTScore class, which computes the BARTScore between two pieces of text.

        Arguments:
            model_name_or_path (str): The name or path of the model. Defaults to 'facebook/bart-large-cnn'.
            tokenizer_name_or_path (str): The name or path of the tokenizer. Defaults to None.
            device (str): The device to use. Defaults to 'cpu'.
            max_length (int): The maximum length of the input. Defaults to 1024.

        Returns:
            None

        Raises:
            ValueError: If the device is not 'cpu' or 'cuda'.

         .. attention::

            If you use this class, please make sure to cite the following paper:
        
            .. code-block:: latex

                @inproceedings{bartscore2021,
                    author = {Yuan, Weizhe and Neubig, Graham and Liu, Pengfei},
                    booktitle = {Advances in Neural Information Processing Systems},
                    editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
                    pages = {27263--27277},
                    publisher = {Curran Associates, Inc.},
                    title = {BARTScore: Evaluating Generated Text as Text Generation},
                    url = {https://proceedings.neurips.cc/paper/2021/file/e4d2b6e6fdeca3e60e0f1a62fee3d9dd-Paper.pdf},
                    volume = {34},
                    year = {2021}
                }
        
        .. note::
            * The default model is the BART-large-cnn model.
            * If the tokenizer name or path is not specified, then the model name or path will be used.
            * If the device is 'cuda', then the model will be loaded onto the GPU.
            * If device is not specified, use the GPU if available, otherwise use the CPU.

        """

        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        
        # Set the attributes
        self.device = device
        self.max_length = max_length
        
        # Load model and tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_name_or_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    
    # Loads the model weights from a specified path
    def load(self, 
        weights_path=None,
        ) -> None:
        """
        This function loads the model weights from a specified path.

        Arguments:
            weights_path (str): The path to the weights.

        Returns:
            None
        """
        if weights_path is None:
            weights_path = 'models/bart.pth'

        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    
    # Compute the BARTScore between source sentences and target sentences
    def compute(self, 
        source_sentences: List[str],
        target_sentences: Union[List[str], List[List[str]]],
        batch_size: int = 4,
        agg: str = 'mean',
        ) -> Dict[str, List[float]]:
        """
        This function scores the target sentences against the source sentences using BARTScore.

        Arguments:
            source_sentences (List[str]): The source sentences.
            target_sentences (Union[List[str], List[List[str]]]): The target sentences.
            batch_size (int): The batch size to use (default: 4)
            agg (str): The aggregation method. Defaults to 'mean'; used only when target_sentences is a list of lists.

        Returns:
            Dict[str, List[float]]: The BARTScore for each example.

        Raises:
            ValueError: If the number of source sentences and target sentences do not match.
        """
        # Check the number of source sentences and target sentences
        if len(source_sentences) != len(target_sentences):
            raise ValueError(f'Number of source sentences ({len(source_sentences)}) and number of target sentences ({len(target_sentences)}) do not match.')
        
        # If the target sentences are a list of lists, then call the multi_ref_score function
        if isinstance(target_sentences[0], list):
            return self.compute_multi_ref_score(
                source_sentences=source_sentences,
                target_sentences=target_sentences,
                batch_size=batch_size,
                agg=agg
            )
        
        # Score for each example
        score_list = []

        for i in range(0, len(source_sentences), batch_size):
            # Get the current batch
            src_batch = source_sentences[i: i + batch_size]
            tgt_batch = target_sentences[i: i + batch_size]
            try:
                with torch.no_grad():
                    # Encode the batch
                    encoded_src = self.tokenizer(
                        src_batch,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_batch,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )

                    # Get the input ids and attention masks for the source and target sentences
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)
                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)
                    
                    # Feed the batch to the model and get the loss
                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    # Compute the loss
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    # Get the score
                    curr_score_list = [-x.item() for x in loss]
                    # Append the score to the list
                    score_list += curr_score_list

            except:
                # If there is an error, print the traceback
                raise Exception(f'Error in scoring batch {i // batch_size}:\n{traceback.format_exc()}')
        return {'score': np.array(score_list)}

    
    # Score a batch of examples with multiple references    
    def compute_multi_ref_score(self, 
        source_sentences: List[str],
        target_sentences: List[List[str]], 
        batch_size: int = 4,
        agg: str = "mean",
        ) -> Dict[str, List[float]]:
        """
        Score a batch of examples with multiple references.

        Arguments:
            source_sentences (List[str]): The source sentences.
            target_sentences (List[List[str]]): The target sentences.
            agg (str): The aggregation method. Can be "mean" or "max".
            batch_size (int): The batch size.

        Returns:
            Dict[str, List[float]]: The BARTScore for each example.

        Raises:
            ValueError: If the number of source sentences and target sentences do not match.
        """

        # Assert we have the same number of references
        ref_nums = [len(x) for x in target_sentences]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(target_sentences[0])
        score_matrix = []
        for i in range(ref_num):
            curr_target_sentences = [x[i] for x in target_sentences]
            scores = self.compute(source_sentences, curr_target_sentences, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError(f"Aggregation method {agg} not implemented yet.")
        return {"score": score_list}


# bart_scorer = BARTScore(model_name_or_path='facebook/bart-large-cnn')
# source_sentences = ["I'm super happy today.", "This is a good idea."]
# target_sentences = [["I feel good today.", "I feel sad today."], ["Not bad.", "Sounds like a good idea."]] # List[List of references for each test sample]
# score = bart_scorer.compute(source_sentences, target_sentences, agg="max", batch_size=4)

# print(score)
# [-2.500810384750366, -1.626237154006958]