"""
This module contains the ModelEmbeddings class.
"""

from typing import List, Union
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModel


class ModelEmbeddings:
    """
    This class is an abstract class for neural word embeddings.
    """

    def __init__(self,
        model_name_or_path: str = 'facebook/bart-large',
        tokenizer_name_or_path: str = None,
        device: str = 'cpu',
    ) -> None:
        """
        Constructor.

        Arguments:
            model_name_or_path (str): The name or path of the model to use (default: 'facebook/bart-large').
            tokenizer (Tokenizer): The tokenizer to use (if None, the model name or path is used).
            device (str): The device to use (default: 'cpu').

        Returns:
            None

        Raises:
            ValueError: If the model name or path is invalid.
        """
        # Set the device
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # If the tokenizer is not specified, use the model name or path
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        # Load the model
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)

        # Set the model to evaluation mode (since we do not need the gradients)
        self.model.eval()


    # Auxiliary function to get the last hidden state
    def get_last_hidden_state(self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the last hidden state (e.g., [CLS] token's) of the input embeddings.

        Arguments:
            embeddings (torch.Tensor): The input embeddings.

        Returns:
            torch.Tensor: The last hidden state.
        """

        # Get the last hidden state
        last_hidden_state = embeddings.last_hidden_state

        # Return the last hidden state
        return last_hidden_state[:, 0, :]
    

    # Auxiliary function to get the mean pooling
    def get_mean_pooling(self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the mean pooling of the input embeddings.

        Arguments:
            embeddings (torch.Tensor): The input embeddings.

        Returns:
            torch.Tensor: The mean pooling.
        """

        # Get the mean pooling
        mean_pooling = embeddings.last_hidden_state.mean(dim=1)

        # Return the mean pooling
        return mean_pooling
    
    # Get the embeddings
    def get_embeddings(self,
        text: Union[str, List[str]],
        embedding_type: str = 'last_hidden_state',
    ) -> torch.Tensor:
        """
        Returns the embeddings of the input text.

        Arguments:
            text (Union[str, List[str]]): The input text.
            embedding_type (str, optional): The type of embedding to use. Defaults to 'last_hidden_state'.

        Returns:
            torch.Tensor: The embeddings.

        Raises:
            ValueError: If the embedding type is invalid.
        """

        # Check if the embedding type is valid
        if embedding_type not in ['last_hidden_state', 'mean_pooling']:
            raise ValueError(f'Invalid embedding type: {embedding_type}. Only "last_hidden_state" and "mean_pooling" are supported.')

        # Tokenize the input text
        encoded_text = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )

        # Move the input text to the device
        encoded_text = encoded_text.to(self.device)

        # encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}

        # Get the embeddings
        with torch.no_grad():
            embeddings = self.model(**encoded_text)

        # Get the proper embedding type
        if embedding_type == 'last_hidden_state':
            # Get the last hidden state
            embeddings = self.get_last_hidden_state(embeddings)
        elif embedding_type == 'mean_pooling':
            # Get the mean pooling
            embeddings = self.get_mean_pooling(embeddings)

        # Return the embeddings
        return embeddings