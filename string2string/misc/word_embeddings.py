"""
    This module implements the word embeddings class.
"""
# from tqdm import tqdm
import numpy as np
from typing import List, Union
import torch
import os
from torch import Tensor
from torch.nn import functional as F
import fasttext
import fasttext.util
from string2string.misc.default_tokenizer import Tokenizer


class NeuralEmbeddings:
    """
    This class is an abstract class for neural word embeddings.
    """

    def __init__(self,
        tokenizer: Tokenizer = None,
    ) -> None:
        """
        Constructor.

        Arguments:
            tokenizer (Tokenizer): The tokenizer to use.
        """
        # Set the tokenizer
        if tokenizer is None:
            self.tokenizer = Tokenizer(word_delimiter=" ")


    
    def __call__(self,
        tokens: Union[List[str], str],
        ) -> Tensor:
        """
        This function returns the embeddings of the given tokens.

        Arguments:
            tokens (Union[List[str], str]): The tokens to embed.

        Returns:
            Tensor: The embeddings of the given tokens.
        """
        # Check the tokens
        if isinstance(tokens, str):
            tokens = self.tokenizer.tokenize(tokens)

        # Embed the tokens
        return self.embedding_layer(torch.tensor([self.vocabulary_dict[token] for token in tokens]))
    

    def get_embedding(self,
        tokens: Union[List[str], str]
        ) -> Tensor:
        """
        This function returns the embeddings of the given tokens.

        Arguments:
            tokens (Union[List[str], str]): The tokens to embed.

        Returns:
            Tensor: The embeddings of the given tokens.
        """
        return self.__call__(tokens)


# GloVe embeddings class
class GloVeEmbeddings(NeuralEmbeddings):
    """
    This class implements the GloVe word embeddings.
    """
    # Pre-trained GloVe embeddings
    # Source: https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors
    MODEL_OPTIONS = {
        'glove.6B.200d': {
            'Description': 'Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download)',
            'URL': 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip',
            },
        'glove.twitter.27B': {
            'Description': 'Twitter (27B tokens, 1.2M vocab, uncased, 200d vectors, 1.42 GB download)',
            'URL': 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip',
        },
        'glove.42B.300d': {
            'Description': 'Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download)',
            'URL': 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip',
        },
        'glove.840B.300d': {
            'Description': 'Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)',
            'URL': 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip',
        },
    }

    def __init__(self, 
        model: str = 'glove.6B.200D',
        dim: int = 50,
        force_download: bool = False,
        dir = None,
        tokenizer: Tokenizer = None,
        ) -> None:
        r"""
        This function initializes the GloVe embeddings class.

        Arguments:
            model (str): The model to use. Default is 'glove.6B.200D'. (Options are: 'glove.6B.200D', 'glove.twitter.27B', 'glove.42B.300d', 'glove.840B.300d'.)
            dim (int): The dimension of the embeddings. Default is 300.
            force_download (bool): Whether to force download the model. Default is False.
            dir (str): The directory to save or load the model. Default is None.
            tokenizer (Tokenizer): The tokenizer to use. Default is None.

        Returns:
            None

        Raises:
            ValueError: If the model is not in the MODEL_OPTIONS [glove.6B.200D', 'glove.twitter.27B', 'glove.42B.300d', 'glove.840B.300d'].

        
        .. attention::

            If you use this class, please make sure to cite the following paper:
        
            .. code-block:: latex

                 @inproceedings{pennington2014glove,
                    title={Glove: Global vectors for word representation},
                    author={Pennington, Jeffrey and Socher, Richard and Manning, Christopher D},
                    booktitle={Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)},
                    pages={1532--1543},
                    year={2014}
                }

        
        .. note::
            * If directory is None, the model will be saved in the torch hub directory.
            * If the model is not downloaded, it will be downloaded automatically.
        """
        # Check model
        if model not in self.MODEL_OPTIONS:
            raise ValueError(f'Invalid model: {model}.')
        
        # Set the attributes
        self.model = model
        self.force_download = force_download
        self.dir = dir
        self.token_size = self.model.split('.')[1]
        self.dim = dim

        # Set the path
        if self.dir is None:
            self.dir = f'{torch.hub.get_dir()}/{self.model}'

        # Remove the trailing slash
        if self.dir[-1] == '/':
            self.dir = self.dir[:-1]

        # Download the embeddings if they do not exist or if force_download is True
        if not os.path.exists(self.dir) or self.force_download:

            # Create the directory if it does not exist            
            if not (os.path.exists(self.dir)):
                os.system(f'mkdir {self.dir}')

            # Download the glove .zip file
            print(f'Downloading the {self.model} zip file...')
            torch.hub.download_url_to_file(
                url=self.MODEL_OPTIONS[self.model]['URL'],
                dst=f'{self.dir}/glove.zip',
            )

            # Unzip the glove .txt files
            print(f'Unzipping the {self.model} zip file...')
            os.system(f'unzip {self.dir}/glove.zip -d {self.dir}')

            # Delete the zip file
            os.system(f'rm {self.dir}/glove.zip')

            # Process each glove .txt file and save it as a .pt file
            for file in os.listdir(self.dir):
                # Extract the words and the embeddings from the glove .txt file and save them as a .pt file
                
                # Example of a glove .txt file:
                # the 0.418 0.24968 -0.41242 0.1217 ... 
                # ...
                # and 0.26818 0.14346 -0.27877 0.016257 ...
                # ...

                print(f'Processing {file}...')

                # Load the file
                with open(f'{self.dir}/{file}', 'r') as f:
                    lines = f.readlines()
                
                # Extract the dimension of the embeddings from the file name (e.g. glove.6B.200d.txt -> 200)
                file_embed_dim = file.split('.')[2][:-1]

                # Extract the words and the embeddings
                words = []
                embeddings = np.zeros((len(lines), int(file_embed_dim)))
                # for line in tqdm(lines):
                # for i, line in tqdm(enumerate(lines)):
                for i, line in enumerate(lines):
                    line = line.split(' ')
                    words.append(line[0])
                    embeddings[i] = np.array([float(x) for x in line[1:]])
                
                # Convert the embeddings to a tensor
                embeddings = torch.from_numpy(embeddings)

                # Save the words and the embeddings as a .pt file
                torch.save(words, f'{self.dir}/{file[:-4]}.words.pt')
                torch.save(embeddings, f'{self.dir}/{file[:-4]}.embeddings.pt')

            # Delete the glove .txt files
            os.system(f'rm -r {self.dir}/*.txt')

            # Load the weights and the vocabulary
            weights = torch.load(f'{self.dir}/glove.{self.token_size}.{self.dim}d.embeddings.pt')
            vocabulary = torch.load(f'{self.dir}/glove.{self.token_size}.{self.dim}d.words.pt')
        
        # If the embeddings already exist
        else:
            # Load the weights and the vocabulary
            weights = torch.load(f'{self.dir}/glove.{self.token_size}.{self.dim}d.embeddings.pt')
            vocabulary = torch.load(f'{self.dir}/glove.{self.token_size}.{self.dim}d.words.pt')

        # Create the vocabulary dictionary to be fed to the embedding layer
        self.vocabulary_dict = {word: i for i, word in enumerate(vocabulary)}

        # Create the embedding layer
        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            embeddings=weights,
            freeze=True,
        )

        # Set the tokenizer
        if tokenizer is None:
            self.tokenizer = Tokenizer()
        else:
            self.tokenizer = tokenizer


    def __call__(self,
        tokens: Union[List[str], str],
        ) -> Tensor:
        """
        This function returns the embeddings of the given tokens.

        Arguments:
            tokens (Union[List[str], str]): The tokens to embed.

        Returns:
            Tensor: The embeddings of the given tokens.
        """
        return super().__call__(tokens)
    

    def get_embedding(self,
        tokens: Union[List[str], str]
        ) -> Tensor:
        r"""
        This function returns the embeddings of the given tokens.

        Arguments:
            tokens (Union[List[str], str]): The tokens to embed.

        Returns:
            Tensor: The embeddings of the given tokens.
        """
        return self.__call__(tokens)


# FastTextEmbeddings class
class FastTextEmbeddings(NeuralEmbeddings):
    """
    This class implements the FastText embeddings.        
    """
    def __init__(self,
        model: str = 'cc.en.300.bin',
        force_download: bool = True,
        dir: str = None,
        ) -> None:
        r"""
        This function initializes the FastTextEmbeddings class.

        Arguments:
            model (str): The model to use. Some of the available models are:

                - 'cc.en.300.bin': The English model trained on Common Crawl (300 dimensions)
                - 'cc.hi.300.bin': The Hindi model trained on Common Crawl (300 dimensions)
                - 'cc.fr.300.bin': The French model trained on Common Crawl (300 dimensions)
                - 'cc.yi.300.bin': The Yiddish model trained on Common Crawl (300 dimensions)
                -  ... 
                - 'wiki.en': The English model trained on Wikipedia (300 dimensions)
                - 'wiki.simple': The Simple English model trained on Wikipedia (300 dimensions)
                - 'wiki.ar': The Arabic model trained on Wikipedia (300 dimensions)
                - 'wiki.bg': The Bulgarian model trained on Wikipedia (300 dimensions)
                - 'wiki.ca': The Catalan model trained on Wikipedia (300 dimensions)
                - 'wiki.zh': The Chinese model trained on Wikipedia (300 dimensions)
                - 'wiki.sw': The Swahili model trained on Wikipedia (300 dimensions)
                - 'wiki.fr': The French model trained on Wikipedia (300 dimensions)
                - 'wiki.de': The German model trained on Wikipedia (300 dimensions)
                - 'wiki.es': The Spanish model trained on Wikipedia (300 dimensions)
                - 'wiki.it': The Italian model trained on Wikipedia (300 dimensions)
                - 'wiki.pt': The Portuguese model trained on Wikipedia (300 dimensions)
                - 'wiki.ru': The Russian model trained on Wikipedia (300 dimensions)
                - 'wiki.tr': The Turkish model trained on Wikipedia (300 dimensions)
                - 'wiki.uk': The Ukrainian model trained on Wikipedia (300 dimensions)
                - 'wiki.vi': The Vietnamese model trained on Wikipedia (300 dimensions)
                - 'wiki.id': The Indonesian model trained on Wikipedia (300 dimensions)
                - 'wiki.ja': The Japanese model trained on Wikipedia (300 dimensions)
                - ... 
            
            force_download (bool): Whether to force the download of the model. Default: False.
            dir (str): The directory to save and load the model. 

        Returns:
            None

        Raises:
            ValueError: If the given model is not available.

        .. attention::

            If you make use of this code, please cite the following papers (depending on the model you use):
    
            .. code-block:: latex 

                @inproceedings{mikolov2018advances,
                    title={Advances in Pre-Training Distributed Word Representations},
                    author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
                    booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
                    year={2018}
                }

            .. code-block:: latex 

                @article{bojanowski2017enriching,
                    title={Enriching Word Vectors with Subword Information},
                    author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
                    journal={Transactions of the Association for Computational Linguistics},
                    volume={5},
                    year={2017},
                    issn={2307-387X},
                    pages={135--146}
                }

            .. code-block:: latex 

                @article{joulin2016fasttext,
                    title={FastText.zip: Compressing text classification models},
                    author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
                    journal={arXiv preprint arXiv:1612.03651},
                    year={2016}
                }

        .. note::

            * The models are downloaded from https://fasttext.cc/docs/en/english-vectors.html.
            * The models are saved in the torch hub directory, if no directory is specified.
            * 
        """ 

        # Set the attributes
        self.model = model
        self.dir = dir
        self.force_download = force_download

        # Set the path
        if self.dir is None:
            # For convenience, we save the model in the torch hub directory
            self.dir = f'{torch.hub.get_dir()}/{self.model}'

        # Remove the trailing slash
        if self.dir[-1] == '/':
            self.dir = self.dir[:-1]

        # Download the embeddings if they do not exist or if force_download is True
        if not os.path.exists(self.dir) or self.force_download:
            # Create the directory if it does not exist            
            if not os.path.exists(self.dir):
                os.system(f'mkdir {self.dir}')

            # Download using wget
            if 'wiki' in model:
                # https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
                os.system(f'wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/{model}.zip -P {self.dir}')
                os.system(f'unzip {self.dirl}.zip -d {self.dir}')
                os.system(f'rm {self.dir}.zip')
            else:
                # https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
                os.system(f'wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{model}.gz -P {self.dir}')
                os.system(f'gunzip {self.dir}.gz -d {self.dir}')
                os.system(f'rm {self.dir}.gz')
            
            # Load the model
            ft = fasttext.load_model(f'{self.dir}/{model}')

            # Get the vocabulary
            words = ft.get_words()

            # Convert the embeddings to a tensor
            embeddings =torch.tensor(ft.get_input_matrix())

            # Save the words and the embeddings as a .pt file
            torch.save(words, f'{self.dir}/{model}.words.pt')
            torch.save(embeddings, f'{self.dir}/{model}.embeddings.pt')

            # Delete the model
            del ft

        else:
            try:
                # Load the words and the embeddings
                words = torch.load(f'{self.dir}/{model}.words.pt')
                embeddings = torch.load(f'{self.dir}/{model}.embeddings.pt')
            except:
                raise Exception(f'Please install the {model} model first by setting force_download to True.') 

        # Create the vocabulary dictionary to be fed to the embedding layer
        self.vocabulary_dict = {word: i for i, word in enumerate(words)}

        # Create the embedding layer
        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            embeddings=embeddings,
            freeze=True,
        )

    def __call__(self,
        tokens: Union[List[str], str],
        ) -> Tensor:
        """
        This function returns the embeddings of the given tokens.

        Arguments:
            tokens (Union[List[str], str]): The tokens to embed.

        Returns:
            Tensor: The embeddings of the given tokens.
        """
        return super().__call__(tokens)
    

    def get_embedding(self,
        tokens: Union[List[str], str]
        ) -> Tensor:
        """
        This function returns the embeddings of the given tokens.

        Arguments:
            tokens (Union[List[str], str]): The tokens to embed.

        Returns:
            Tensor: The embeddings of the given tokens.
        """
        return self.__call__(tokens)
        

# # Test
# glove = GloVeEmbeddings(
#     model='glove.6B.200d',
#     dim=50,
#     force_download=False,
#     dir='/Users/machine/.cache/torch/hub/glove.6B.200d'
# )

# # debug
# print(glove('the'))

# # debug
# # print(glove(['the', 'and']))


# ft = FastTextEmbeddings(
#     model='cc.en.300.bin',
#     # model='wiki.en.bin',
#     force_download=False,
#     dir='/Users/machine/.cache/torch/hub/cc.en.300.bin'
# )

# # debug
# print(ft('the'))