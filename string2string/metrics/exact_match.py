"""
    This module contains a class for the exact match metric.
"""

from typing import List, Dict

# Exact match class
class ExactMatch:
    def __init__(self) -> None:
        pass

    def compute(self,
        predictions: List[str],
        references: List[List[str]],
        lowercase: bool = True,
    ) -> Dict[str, float]:
        """
        This function returns the exact match score between the predictions and the references.

        Arguments:
            predictions (List[str]): The list of predictions.
            references (List[List[str]]): The list of references.

        Returns:
            float: The exact match score.

        Raises:
            AssertionError: If the number of predictions does not match the number of references.
        """

        # Check that the number of predictions and references are the same length and that the length is not 0
        assert len(predictions) == len(references) and len(predictions) > 0

        # Compute the exact match score
        num_correct = 0.
        for prediction, reference in zip(predictions, references):
            # Lowercase the prediction and reference
            if lowercase:
                prediction = prediction.lower()
                reference = [ref.lower() for ref in reference]

            # Check if the prediction is in the reference
            if prediction in reference:
                num_correct += 1

        # Summary of the final scores
        final_scores = {
            'score': num_correct / len(predictions),
            'num_correct': num_correct,
            'num_total': len(predictions),
        }

        # Return the final scores
        return final_scores