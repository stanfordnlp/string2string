"""
This module contains a wrapper for the Faiss library by Facebook AI Research.
"""

from typing import List, Union, Optional, Dict, Any
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset

import pandas as pd

# FAISS library wrapper class
class FaissSearch:
    def __init__(self, 
        model_name_or_path: str = 'facebook/bart-large',
        tokenizer_name_or_path: str = 'facebook/bart-large',
        device: str = 'cpu',
        ) -> None:
        r"""
        This function initializes the wrapper for the FAISS library, which is used to perform semantic search.


        .. attention::

            * If you use this class, please make sure to cite the following paper:

                .. code-block:: latex

                    @article{johnson2019billion,
                        title={Billion-scale similarity search with {GPUs}},
                        author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
                        journal={IEEE Transactions on Big Data},
                        volume={7},
                        number={3},
                        pages={535--547},
                        year={2019},
                        publisher={IEEE}
                    }

            * The code is based on the following GitHub repository:
                https://github.com/facebookresearch/faiss

        Arguments:
            model_name_or_path (str, optional): The name or path of the model to use. Defaults to 'facebook/bart-large'.
            tokenizer_name_or_path (str, optional): The name or path of the tokenizer to use. Defaults to 'facebook/bart-large'.
            device (str, optional): The device to use. Defaults to 'cpu'.

        Returns:
            None
        """

        # Set the device
        self.device = device

        # If the tokenizer is not specified, use the model name or path
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        # Load the model
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)

        # Set the model to evaluation mode (since we do not need the gradients)
        self.model.eval()

        # Initialize the dataset
        self.dataset = None


    # Auxiliary function to get the last hidden state
    def get_last_hidden_state(self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        This function returns the last hidden state (e.g., [CLS] token's) of the input embeddings.

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
        This function returns the mean pooling of the input embeddings.

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
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> torch.Tensor:
        """
        This function returns the embeddings of the input text.

        Arguments:
            text (Union[str, List[str]]): The input text.
            embedding_type (str, optional): The type of embedding to use. Defaults to 'last_hidden_state'.
            batch_size (int, optional): The batch size to use. Defaults to 8.
            num_workers (int, optional): The number of workers to use. Defaults to 4.

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
    

    # Add FAISS index
    def add_faiss_index(self,
        column_name: str = 'embeddings',
        metric_type: Optional[int] = None,
        batch_size: int = 8,
        **kwargs,
    ) -> None:
        """
        This function adds a FAISS index to the dataset.

        Arguments:
            column_name (str, optional): The name of the column containing the embeddings. Defaults to 'embeddings'.
            index_type (str, optional): The index type to use. Defaults to 'Flat'.
            metric_type (str, optional): The metric type to use. Defaults to 'L2'.

        Returns:
            None

        Raises:
            ValueError: If the dataset is not initialized.
        """

        # Check if the dataset is initialized
        if self.dataset is None:
            raise ValueError('The dataset is not initialized. Please initialize the dataset first.')

        print('Adding FAISS index...')
        self.dataset.add_faiss_index(
            column_name,
            # metric_type=metric_type,
            # device=self.device,
            # batch_size=batch_size,
            faiss_verbose=True,
            # **kwargs,
        )

    def save_faiss_index(self,
        index_name: str,
        file_path: str,
    ) -> None:
        """
        This function saves the FAISS index to the specified file path.
            * This is a wrapper function for the `save_faiss_index` function in the `Dataset` class.

        Arguments:
            index_name (str): The name of the FAISS index  (e.g., "embeddings")
            file_path (str): The file path to save the FAISS index.

        Returns:
            None

        Raises:
            ValueError: If the dataset is not initialized.
        """

        # Check if the dataset is initialized
        if self.dataset is None:
            raise ValueError('The dataset is not initialized. Please initialize the dataset first.')

        print('Saving FAISS index...')
        self.dataset.save_faiss_index(index_name=index_name, file=file_path)

    
    def load_faiss_index(self,
        index_name: str,
        file_path: str,
        device: str = 'cpu',
    ) -> None:
        """
        This function loads the FAISS index from the specified file path.
            * This is a wrapper function for the `load_faiss_index` function in the `Dataset` class.

        Arguments:
            index_name (str): The name of the FAISS index  (e.g., "embeddings")
            file_path (str): The file path to load the FAISS index from.
            device (str, optional): The device to use ("cpu" or "cuda") (default: "cpu").

        Returns:
            None

        Raises:
            ValueError: If the dataset is not initialized.
        """

        # Check if the dataset is initialized
        if self.dataset is None:
            raise ValueError('The dataset is not initialized. Please initialize the dataset first.')

        print('Loading FAISS index...')
        self.dataset.load_faiss_index(index_name=index_name, file=file_path, device=device)

    
    # Initialize the corpus using a dictionary or pandas DataFrame or HuggingFace Datasets object
    def initialize_corpus(self,
        corpus: Union[Dict[str, List[str]], pd.DataFrame, Dataset],
        section: str = 'text',
        index_column_name: str = 'embeddings',
        embedding_type: str = 'last_hidden_state',
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> Dataset:
        """
        This function initializes a dataset using a dictionary or pandas DataFrame or HuggingFace Datasets object.

        Arguments:
            dataset_dict (Dict[str, List[str]]): The dataset dictionary.
            section (str): The section of the dataset to use whose embeddings will be used for semantic search (e.g., 'text', 'title', etc.) (default: 'text').
            index_column_name (str): The name of the column containing the embeddings (default: 'embeddings')
            embedding_type (str): The type of embedding to use (default: 'last_hidden_state').
            batch_size (int, optional): The batch size to use (default: 8).
            max_length (int, optional): The maximum length of the input sequences.
            num_workers (int, optional): The number of workers to use. 
            save_path (Optional[str], optional): The path to save the dataset (default: None).

        Returns:
            Dataset: The dataset object (HuggingFace Datasets).

        Raises:
            ValueError: If the dataset is not a dictionary or pandas DataFrame or HuggingFace Datasets object.
        """

        # Create the dataset
        if isinstance(corpus, dict):
            self.dataset = Dataset.from_dict(corpus)
        elif isinstance(corpus, pd.DataFrame):
            self.dataset = Dataset.from_pandas(corpus)
        elif isinstance(corpus, Dataset):
            self.dataset = corpus
        else:
            raise ValueError('The dataset must be a dictionary or pandas DataFrame.')
        
        # Set the embedding_type
        self.embedding_type = embedding_type
            

        # Tokenize the dataset
        # self.dataset = self.dataset.map(
        #     lambda x: x[section],
        #     batched=True,
        #     batch_size=batch_size,
        #     num_proc=num_workers,
        # )

        # Map the section of the dataset to the embeddings
        self.dataset = self.dataset.map(
            lambda x: {
                index_column_name: self.get_embeddings(x[section], embedding_type=self.embedding_type).detach().cpu().numpy()[0]
                },
            # batched=True,
            batch_size=batch_size,
            num_proc=num_workers,
        )

        # Save the dataset
        if save_path is not None:
            self.dataset.to_json(save_path)

        # Add FAISS index
        self.add_faiss_index(
            column_name=index_column_name,
        )

        # Return the dataset
        return self.dataset
    

    # Initialize the dataset using a JSON file
    def load_dataset_from_json(self,
        json_path: str,
    ) -> Dataset:
        """
        This function loads a dataset from a JSON file.

        Arguments:
            json_path (str): The path to the JSON file.

        Returns:
            Dataset: The dataset.
        """

        # Load the dataset
        self.dataset = Dataset.from_json(json_path)

        # Return the dataset
        return self.dataset
    

    # Search for the most similar elements in the dataset, given a query
    def search(self,
        query: str,
        k: int = 1,
        index_column_name: str = 'embeddings',
    ) -> pd.DataFrame:
        """
        This function searches for the most similar elements in the dataset, given a query.
        
        Arguments:
            query (str): The query.
            k (int, optional): The number of elements to return  (default: 1).
            index_column_name (str, optional): The name of the column containing the embeddings (default: 'embeddings')

        Returns:
            pd.DataFrame: The most similar elements in the dataset (text, score, etc.), sorted by score.

        Remarks:
            The returned elements are dictionaries containing the text and the score.
        """
        
        # Get the embeddings of the query
        query_embeddings = self.get_embeddings([query], embedding_type=self.embedding_type).detach().cpu().numpy()

        # Search for the most similar elements in the dataset
        scores, similar_elts = self.dataset.get_nearest_examples(
            index_name=index_column_name,
            query=query_embeddings, 
            k=k,
        )

        # Convert the results to a pandas DataFrame
        results_df = pd.DataFrame.from_dict(similar_elts)
        
        # Add the scores
        results_df['score'] = scores

        # Sort the results by score
        results_df.sort_values("score", ascending=True, inplace=True)

        # Return the most similar elements
        return results_df