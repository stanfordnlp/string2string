"""
    This module contains a wrapper class for the sacreBLEU metric from https://github.com/mjpost/sacreBLEU.
"""

from typing import Union, Optional, List, Dict
from string2string.misc.default_tokenizer import Tokenizer
from sacrebleu import corpus_bleu

# Pre-defined tokenizers for sacreBLEU
# This list taken from https://github.com/mjpost/sacrebleu/blob/4f4124642c4eb0b7120e50119c669f0570a326a7/sacrebleu/metrics/bleu.py#L18
ALLOWED_TOKENIZERS = {
    'none': 'tokenizer_none.NoneTokenizer',
    'zh': 'tokenizer_zh.TokenizerZh',
    '13a': 'tokenizer_13a.Tokenizer13a',
    'intl': 'tokenizer_intl.TokenizerV14International',
    'char': 'tokenizer_char.TokenizerChar',
    'ja-mecab': 'tokenizer_ja_mecab.TokenizerJaMecab',
    'ko-mecab': 'tokenizer_ko_mecab.TokenizerKoMecab',
    'spm': 'tokenizer_spm.TokenizerSPM',
    'flores101': 'tokenizer_spm.Flores101Tokenizer',
    'flores200': 'tokenizer_spm.Flores200Tokenizer',
}


class sacreBLEU:
    """
    This class contains the sacreBLEU metric.
    """

    def __init__(self) -> None:
        """
        Initializes the BLEU class.
        """
        pass


    def compute(self,
        predictions: List[str],
        references: List[List[str]],
        smooth_method: str = 'exp',
        smooth_value: Optional[float] = None,
        lowercase: bool = False,
        tokenizer_name: str = '13a',
        use_effective_order: bool = False,
        return_only: List[str] = ['score', 'counts', 'totals', 'precisions', 'bp', 'sys_len', 'ref_len']
    ):
        """
        Returns the BLEU score between a list of predictions and list of list of references.

        Arguments:
            predictions (List[str]): The predictions.
            references (List[List[str]]): The references (or ground truth strings).
            smooth_method (str): The smoothing method. Default is "exp". Other options are "floor", "add-k" and "none".
            smooth_value (Optional[float]): The smoothing value for floor and add-k smoothing. Default is None.
            lowercase (bool): Whether to lowercase the text. Default is False.
            tokenizer_name (str): The tokenizer name. Default is "13a".
            use_effective_order (bool): Whether to use the effective order. Default is False.
            return_only (Optional[List[str]]): The list of BLEU score components to return. Default is None

        Returns:
            Dict[str, float]: The BLEU score (between 0 and 1).

        Raises:
            ValueError: If the number of predictions does not match the number of references.
            ValueError: If the tokenizer name is invalid.
        """

        # Check that the number of predictions matches the number of references
        if len(predictions) != len(references):
            raise ValueError('The number of predictions does not match the number of references.')

        # Check that the tokenizer name is valid
        if tokenizer_name not in ALLOWED_TOKENIZERS:
            raise ValueError('The tokenizer name is invalid.')
        
        # Check that the size of each reference list is the same
        reference_size = len(references[0])
        for reference in references:
            if len(reference) != reference_size:
                raise ValueError('The size of each reference list is not the same.')
            

        # Compute the BLEU score using sacrebleu.corpus_bleu
        # This function returns "BLEUScore(score, correct, total, precisions, bp, sys_len, ref_len)"
        bleu_score = corpus_bleu(
            hypotheses=predictions,
            references=references,
            smooth_method=smooth_method,
            smooth_value=smooth_value,
            lowercase=lowercase,
            tokenize=tokenizer_name,
            use_effective_order=use_effective_order,
        )

        # Get a summary of all the relevant BLEU score components
        final_scores = {k: getattr(bleu_score, k) for k in return_only}

        # Return the BLEU score
        return final_scores
    

# predictions = ["hello there general kenobi", "foo bar foobar"]
# references = [["hello there general kenobi", "hello there !"], ["foo bar foobar", "foo bar foobar"]]

# sbleu = sacreBLEU()
# bleu_score = sbleu.compute(predictions, references)
# print(bleu_score)