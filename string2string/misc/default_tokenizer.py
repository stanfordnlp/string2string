"""
This file contains the default tokenizer.
"""

from typing import List

# Tokenizer class
class Tokenizer:
    """
    This class contains the tokenizer.

    This class is a wrapper for the ROUGE metric from Google Research's rouge_score package.
    """

    def __init__(self,
        word_delimiter: str = " ",
    ):
        """
        Initializes the Tokenizer class.

        Arguments:
            word_delimiter (str): The word delimiter. Default is " ".
        """
        # Set the word delimiter
        self.word_delimiter = word_delimiter

    # Tokenize
    def tokenize(self,
        text: str,
    ) -> List[str]:
        """
        Returns the tokens from a string.

        Arguments:
            text (str): The text to tokenize.

        Returns:
            List[str]: The tokens.
        """
        return text.split(self.word_delimiter)
    
    # Detokenize
    def detokenize(self,
        tokens: List[str],
    ) -> str:
        """
        Returns the string from a list of tokens.

        Arguments:
            tokens (List[str]): The tokens.

        Returns:
            str: The string.
        """
        return self.word_delimiter.join(tokens)