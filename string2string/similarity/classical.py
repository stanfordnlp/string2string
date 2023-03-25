"""
    This module contains the classes for the similarity metrics and functions.

    Here is a list of the similarity metrics:
        (a) Longest Common Subsequence (LCS) based similarity
        (b) Jaro similarity
        (c) Cosine similarity (using word embeddings)
        (d) BERTScore
        (e) BARTScore
        (f) BLEURT
        (g) ROUGE
        (h) METEOR
        (i) TER
        (j) WER
        (k) CER

"""


from typing import List, Union, Tuple, Optional
import numpy as np

# # Import the LongestCommonSubsequence class
from string2string.alignment.classical import LongestCommonSubsequence

# Longest Common Subsequence based similarity class
class LCSimilarity(LongestCommonSubsequence):
    """
    This class contains the Longest Common Subsequence similarity metric.

    This class inherits from the LongestCommonSubsequence class.
    """

    def __init__(self):
        super().__init__()

    def compute(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
        denominator: str = 'max',
    ) -> float:
        """
        Returns the LCS-similarity between two strings.

        Arguments:
            str1 (Union[str, List[str]]): The first string or list of strings.
            str2 (Union[str, List[str]]): The second string or list of strings.
            denominator (str): The denominator to use. Options are 'max' and 'sum'. Default is 'max'.

        Returns:
            float: The similarity between the two strings.

        Raises:
            ValueError: If the denominator is invalid.
        """
        if denominator == 'max':
            return super().compute(str1, str2) / max(len(str1), len(str2))
        elif denominator == 'sum':
            return 2. * super().compute(str1, str2) / (len(str1) + len(str2))
        else:
            raise ValueError('Invalid denominator.')


# Jaro similarity class
class JaroSimilarity:
    """
    This class contains the Jaro similarity metric.
    """

    def __init__(self):
        pass

    def compute(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
    ) -> float:
        """
        This function returns the Jaro similarity between two strings.

        Arguments:
            str1 (Union[str, List[str]]): The first string or list of strings.
            str2 (Union[str, List[str]]): The second string or list of strings.

        Returns:
            float: The Jaro similarity between the two strings.
        """
        # Get the length of the strings
        len1 = len(str1)
        len2 = len(str2)

        # Get the maximum distance, which we denote by k
        k = max(len1, len2) // 2 - 1

        # Initialize the number of matching characters and the number of transpositions
        num_matches = 0
        num_transpositions = 0

        # Initialize the list of matching flags for the strings
        matches1 = [False] * len1
        matches2 = [False] * len2

        # Loop through the characters in the first string and find the matching characters
        for i in range(len1):
            # Get the lower and upper bounds for the search
            lower_bound = max(0, i - k)
            upper_bound = min(len2, i + k + 1)

            # Loop through the characters in the second string
            for j in range(lower_bound, upper_bound):
                # Check if the characters match
                if not matches2[j] and str1[i] == str2[j]:
                    # Increment the number of matches
                    num_matches += 1

                    # Set the matching flags
                    matches1[i] = True
                    matches2[j] = True

                    # Break out of the loop
                    break

        # Check if there are no matches
        if num_matches == 0:
            return 0.
        
        # Loop through again but this time find the number of transpositions
        # That is, the number of times where there are two matching characters but there is another "matched" character in between them
        moving_index = 0
        for i in range(len1):
            # Check if the character is a match
            if matches1[i]:
                # Find the next match
                for j in range(moving_index, len2):
                    # Check if the character is a match
                    if matches2[j]:
                        # Set the moving index
                        moving_index = j + 1

                        # Check if the characters are not in the right order
                        if str1[i] != str2[j]:
                            # Increment the number of transpositions
                            num_transpositions += 1

                        # Break out of the loop
                        break
        
        num_transpositions = num_transpositions // 2

        # Return the Jaro similarity
        return (num_matches / len1 + num_matches / len2 + (num_matches - num_transpositions) / num_matches) / 3.0

       
# jaro = JaroSimilarity()
# print(jaro.compute('FAREMVIEL', 'FARMVILLE'))
