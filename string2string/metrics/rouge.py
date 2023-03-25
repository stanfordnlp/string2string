"""
    This module contains a wrapper class for the ROUGE metric.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics for evaluating the quality of summaries in machine translation, text summarization, and other natural language generation tasks.
"""

from typing import Union, List, Dict
from rouge_score import rouge_scorer
from rouge_score.scoring import BootstrapAggregator
from string2string.misc.default_tokenizer import Tokenizer

# ROUGE class
class ROUGE:
    """
    This class is a wrapper for the ROUGE metric from Google Research's rouge_score package.
    """

    def __init__(self,
        tokenizer: Tokenizer = None,
    ) -> None:
        """
        This function initializes the ROUGE class, which is a wrapper for the ROUGE metric from Google Research's rouge_score package.

        Arguments:
            rouge_types (Union[str, List[str]]): The ROUGE types to use. Default is ["rouge1", "rouge2", "rougeL", "rougeLsum"].

        Returns:
            None
        """
        # Set the tokenizer
        if tokenizer is None:
            self.tokenizer = Tokenizer(word_delimiter=' ')
        else:
            self.tokenizer = tokenizer

    # Compute the ROUGE score
    def compute(self,
        predictions: List[str],
        references: List[List[str]],
        rouge_types: Union[str, List[str]] = ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer: bool = False,
        interval_name: str = 'mid',
        score_type: str = 'fmeasure',
    ) -> Dict[str, float]:
        """
        This function returns the ROUGE score between a list of predictions and list of list of references.

        Arguments:
            predictions (List[str]): The predictions.
            references (List[List[str]]): The references (or ground truth strings).
            rouge_types (Union[str, List[str]]): The ROUGE types to use. Default is ["rouge1", "rouge2", "rougeL", "rougeLsum"].
            use_stemmer (bool): Whether to use a stemmer. Default is False.
            interval_name (str): The interval name. Default is "mid".
            score_type (str): The score type. Default is "fmeasure".
            
        Returns:
            Dict[str, float]: The ROUGE score (between 0 and 1).

        Raises:
            ValueError: If the number of predictions does not match the number of references.
            ValueError: If the interval name, score type or ROUGE type is invalid.
            ValueError: If the prediction or reference is invalid.

        
        .. note::
            * The ROUGE score is computed using the ROUGE metric from Google Research's rouge_score package.
            * By default, BootstrapAggregator is used to aggregate the scores.
            * By default, the interval name is "mid" and the score type is "fmeasure".
        """

        # Check if the predictions and references are valid
        if len(predictions) != len(references):
            raise ValueError(f'Number of predictions ({len(predictions)}) does not match number of references ({len(references)})')
        
        # Check if the interval name is valid
        if interval_name not in ['low', 'mid', 'high']:
            raise ValueError(f'Invalid interval name: {interval_name}')
        
        # Check if the score type is valid
        if score_type not in ['precision', 'recall', 'fmeasure']:
            raise ValueError(f'Invalid score type: {score_type}')

        # Check if the ROUGE types are valid
        if not isinstance(rouge_types, list):
            rouge_types = [rouge_types]
        for rouge_type in rouge_types:
            if rouge_type not in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
                raise ValueError(f'Invalid ROUGE type: {rouge_type}')

        # Set the ROUGE scorer
        scorer = rouge_scorer.RougeScorer(
            rouge_types=rouge_types,
            use_stemmer=use_stemmer,
            tokenizer=self.tokenizer
        )

        # Set the aggregator
        aggregator = BootstrapAggregator()

        # Compute the ROUGE score
        for prediction, reference in zip(predictions, references):
            # Check if the prediction and reference are valid
            if not isinstance(prediction, str):
                raise ValueError(f'Invalid prediction: {prediction}')
            if not isinstance(reference, list):
                raise ValueError(f'Invalid reference: {reference}')

            # Compute the ROUGE score
            scores = scorer.score_multi(
                targets=reference,
                prediction=prediction
            )
            aggregator.add_scores(scores)

        # Aggregate the scores
        aggregate_score = aggregator.aggregate()

        # Get a summary of all the relevant BLEU score components
        final_scores = {rouge_type: getattr(aggregate_score[rouge_type], interval_name).__getattribute__(score_type) for rouge_type in rouge_types}

        # Return the final scores
        return final_scores