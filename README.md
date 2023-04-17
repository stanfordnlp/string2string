<p align="center">
    <img src="https://github.com/suzgunmirac/string2string/blob/main/fables/string2string-overview.png" class="center" />
</p>

# string2string

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/string2string)](https://badge.fury.io/py/string2string)
[![PyPI version](https://badge.fury.io/py/string2string.svg)](https://badge.fury.io/py/string2string)
[![Downloads](https://pepy.tech/badge/string2string)](https://pepy.tech/project/string2string)
[![license](https://img.shields.io/github/license/suzgunmirac/string2string.svg)](https://github.com/suzgunmirac/string2string/blob/main/LICENSE.txt)
[![arXiv](https://img.shields.io/badge/arXiv-2211.07634-b31b1b.svg)](https://arxiv.org/abs/2211.07634)

* **Table of Contents:** [**Getting Started**](#getting-started) | [**Tutorials**](#tutorials) | [**Example Usage**](#example-usage) | [**Documentation**](https://string2string.readthedocs.io/en/latest/) | [**Citation**](#citation) | [**Thanks**](#thanks) 

The **string2string** library is an open-source tool that offers a comprehensive suite of efficient algorithms for a broad range of string-to-string problems. It includes both traditional algorithmic solutions and recent advanced neural approaches to address various problems in pairwise string alignment, distance measurement, lexical and semantic search, and similarity analysis. Additionally, the library provides several helpful visualization tools and metrics to facilitate the interpretation and analysis of these methods. 

The library features notable algorithms such as the [Smith-Waterman algorithm](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm) for pairwise local alignment, the [Hirschberg algorithm](https://en.wikipedia.org/wiki/Hirschberg%27s_algorithm) for global alignment, the [Wagner-Fisher algorithm](https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm) for [edit distance](https://en.wikipedia.org/wiki/Edit_distance), [BARTScore](https://github.com/neulab/BARTScore) and [BERTScore](https://github.com/Tiiiger/bert_score) for similarity analysis, the [Knuth-Morris-Pratt](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm) algorithm for lexical search, and [Faiss](https://github.com/facebookresearch/faiss) for [semantic search](https://en.wikipedia.org/wiki/Semantic_search). Moreover, it wraps existing highly efficient and widely-used implementations of certain frameworks and metrics, such as [sacreBLEU](https://github.com/mjpost/sacrebleu) and [ROUGE](https://github.com/google-research/google-research/tree/master/rouge), whenever it is appropriate and suitable. 

In general, the string2string library seeks to provide extensive coverage and increased flexibility compared to existing libraries for strings. It can be used for many downstream applications, tasks, and problems in natural-language processing, bioinformatics, and computational social sciences. With its comprehensive suite of algorithms, visualization tools, and metrics, the string2string library is a valuable resource for researchers and practitioners in various fields.

## Getting Started

Install the string2string library by running the following command in your terminal:

```python
pip install string2string
```

Once the installation is complete, you can import the library and start using its functionalities.

**Remark**: We recommend using Python 3.7+ for the library.

## Tutorials

* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][def1] [Tutorial: Alignment Tasks and Algorithms][def1]
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][def2] [Tutorial: Distance Tasks and Algorithms][def2]
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][def3] [Tutorial: Search Tasks and Algorithms][def3]
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][def4] [Tutorial: Similarity Tasks and Algorithms][def4]
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][def5] [Hands-on Tutorial: Semantic Search and Visualization of USPTO Patents][def5]
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][def6] [Hands-On Tutorial: Plagiarism Detection of Essays][def6]

## Example Usage

### Alignment

In the following example, we illustrate how to align two sequences of strings globally by using the [Needleman-Wunsch algorithm](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm). This algorithm, along with other alignment techniques, can be found in the alignment module of the library. The code snippet below demonstrates how to apply the Needleman-Wunsch algorithm to perform a global alignment of two given strings.

This example provides a practical illustration of how to use the Needleman-Wunsch algorithm to solve the problem of sequence alignment, which is a fundamental problem in bioinformatics.

```python
>>> # Import the NeedlemanWunsch class from the alignment module
>>> from string2string.alignment import NeedlemanWunsch

>>> # Create an instance of the NeedlemanWunsch class
>>> nw = NeedlemanWunsch()

>>> # Let's create a list of strings (resembling DNA sequences), but they can be any strings (e.g., words), of course.
>>> seq1 = ['X', 'ATT', 'GC', 'GC', 'A', 'A', 'G']
>>> seq2 = ['ATT', 'G', 'GC', 'GC', 'A', 'C', 'G']

>>> # Compute the alignment between two strings
>>> aligned_seq1, aligned_seq2 = nw.get_alignment(seq1, seq2)

>>> # Print the alignment between the sequences, as computed by the Needleman-Wunsch algorithm.
>>> nw.print_alignment(aligned_seq1, aligned_seq2)
X | ATT | - | GC | GC | A | A | G
- | ATT | G | GC | GC | A | C | G

>>> alg_path, alg_seq1_parts, alg_seq2_parts = nw.get_alignment_strings_and_indices(aligned_seq1, aligned_seq2)
>>> plot_pairwise_alignment(seq1_pieces = alg_seq1_parts, seq2_pieces = alg_seq1_parts, alignment = alignment_path, str2colordict = {'-': 'lightgray', 'ATT': 'indianred', 'GC': 'darkseagreen', 'A': 'skyblue', 'G': 'palevioletred', 'C': 'steelblue'}, title = 'Global Alignment Between Two Sequences of Strings')
```

<p align="center">
    <img src="https://github.com/suzgunmirac/string2string/blob/main/fables/alignment-example.png" class="center" />
</p>

### Distance

The following code snippet demonstrates how to use the [Levenshtein edit distance algorithm](https://en.wikipedia.org/wiki/Levenshtein_distance) to compute the edit distance between two strings, at the character level and at the word level.

```python
>>> # Let's create a Levenshtein edit distance class instance, with the default (unit cost) weights, from the distance module
>>> from string2string.distance import LevenshteinEditDistance
>>> edit_dist = LevenshteinEditDistance()

>>> # Let's also create a Tokenizer class instance with the default word delimiter (i.e., space)
>>> from string2string.misc import Tokenizer
>>> tokenizer = Tokenizer(word_delimiter=' ')

>>> # Let's create two strings
>>> text1 = "The quick brown fox jumps over the lazy dog"
>>> text2 = "The kuack brown box jumps over the lazy dog"

>>> # Get the edit distance between them at the character level
>>> edit_dist_score  = edit_dist.compute(text1, text2)

>>> print(f"Edit distance between these two texts at the character level is {edit_dist_score}")
# Edit distance between these two texts at the character level is 3.0

>>> # Tokenize the two texts
>>> text1_tokens = tokenizer.tokenize(text1)
>>> text2_tokens = tokenizer.tokenize(text2)

>>> # Get the distance between them at the word level
>>> edit_dist_score  = edit_dist.compute(text1_tokens, text2_tokens)

>>> print(f"Edit distance between these two texts at the word level is {edit_dist_score}")
# Edit distance between these two texts at the word level is 2.0
```

### Search

The following code snippet demonstrates how to use the [Knuth-Morrs-Pratt (KMP) search algorithm](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm) to find the index of a pattern in a text.

```python
>>> # Let's create a KMPSearch class instance from the search module
>>> from string2string.search import KMPSearch
>>> knuth_morris_pratt = KMPSearch()

>>> # Let's define a pattern and a text
>>> pattern = Jane Austen'
>>> text = 'Sense and Sensibility, Pride and Prejudice, Emma, Mansfield Park, Northanger Abbey, Persuasion, and Lady Susan were written by Jane Austen and are important works of English literature.'

>>> # Now let's find the index of the pattern in the text, if it exists (otherwise, -1 is returned).
>>> idx = knuth_morris_pratt.search(pattern=pattern,text=text)

>>> print(f'The index of the pattern in the text is {idx}.')
# The index of the pattern in the text is 127.
```

### Faiss Semantic Search

The example below demonstrates how to use the [Faiss](https://github.com/facebookresearch/faiss) tool developed by FAIR to perform semantic search. First, we use the [BART-Large model from Hugging Face](https://huggingface.co/facebook/bart-large) to generate embeddings for a small corpus of 25 sentences. To perform the search, we first encode a query sentence using the same BART model and use it to search the corpus. Specifically, we search for sentences that are most similar in meaning to the query sentence. After performing the search, we print the top thre sentences from the corpus that are most similar to the query sentence. 

This approach can be useful in a variety of natural-processing applications, such as question-answering and information retrieval, where it is essential to find relevant information quickly and accurately.

```python
>>> # Let's create a FaissSearch class instance from the search module to perform semantic search
>>> from string2string.search import FaissSearch
>>> faiss_search = FaissSearch(model_name_or_path = 'facebook/bart-large')

>>> # Let's create a corpus of strings (e.g., sentences)
>>> corpus = {
        'text': [
            "Coffee is my go-to drink in the morning.", 
            "I always try to make time for exercise.", 
            "Learning something new every day keeps me motivated.", 
            "The sunsets in my hometown are breathtaking.", 
            "I am grateful for the support of my friends and family.", 
            "The book I'm reading is incredibly captivating.", 
            "I love listening to music while I work.", 
            "I'm excited to try the new restaurant in town.", 
            "Taking a walk in nature always clears my mind.", 
            "I believe that kindness is the most important trait.", 
            "It's important to take breaks throughout the day.", 
            "I'm looking forward to the weekend.", 
            "Reading before bed helps me relax.", 
            "I try to stay positive even in difficult situations.", 
            "Cooking is one of my favorite hobbies.", 
            "I'm grateful for the opportunity to learn and grow every day.", 
            "I love traveling and experiencing new cultures.", 
            "I'm proud of the progress I've made so far.", 
            "A good night's sleep is essential for my well-being.", 
            "Spending time with loved ones always brings me joy.", 
            "I'm grateful for the beauty of nature around me.", 
            "I try to live in the present moment and appreciate what I have.", 
            "I believe that honesty is always the best policy.", 
            "I enjoy challenging myself and pushing my limits.", 
            "I'm excited to see what the future holds."
        ],
    }

>>> # Next we need to initialize and encode the corpus
>>> faiss_search.initialize_corpus(
    corpus=corpus,
    section='text', 
    embedding_type='mean_pooling',
    )

>>> # Let's define a query, and the number of top results we want to retrieve; then, let's perform the semantic search.
>>> query = 'I like going for a run in the morning.'
>>> top_k = 5
>>> top_k_results = faiss_search.search(query=query, k = top_k)

>>> # Let's define a function to print the results of the search.
>>> def print_results(query, results, top_k):
        # Let's first print the query.
        print(f'Query: "{query}"\n')

        # Let's now print the top k results.
        print(f'Top {top_k} most similar sentences in the corpus to the query (smallest score is most similar):')
        for i in range(top_k):
            print(f' - {i+1}: "{results["text"][i]}" with a similarity score of {top_k_results["score"][i]:.2f}')
>>> print_results(query=query, results=top_k_results, top_k=top_k)
# Query: "I like going for a run in the morning."

# Top 3 most similar sentences in the corpus to the query (smallest score is most similar):
#  - 1: "I always try to make time for exercise." with a similarity score of 170.65
#  - 2: "The sunsets in my hometown are breathtaking." with a similarity score of 238.20
#  - 3: "Coffee is my go-to drink in the morning." with a similarity score of 238.85
```

### Cosine Similarity (with Glove and fastText Embeddings)

The following example demonstrates how to use [pre-trained GloVe embeddings](https://nlp.stanford.edu/projects/glove/) to calculate the cosine similarity between different pairs of words. Specifically, we compute the cosine similarity between the embeddings of four words: "cat", "dog", "phone", and "computer". We then create a similarity matrix and use the plot_heatmap function in the visualization module to plot this matrix.

Overall, this example provides a practical demonstration of how to use pre-trained embeddings such as [GloVe](https://nlp.stanford.edu/projects/glove/) and [fastText](https://fasttext.cc/) to quantify the semantic similarity between pairs of words, which can be useful in a variety of natural-language processing tasks.

```python
>>> # Let's create a Cosine Similarity class instance from the similarity module
>>> from string2string.similarity import CosineSimilarity
>>> cosine_similarity = CosineSimilarity()

>>> # Let's also create an instance of the GloVeEmbeddings class from the misc module to compute the embeddings of words
>>> from string2string.misc import GloVeEmbeddings
>>> glove = GloVeEmbeddings(model='glove.6B.200d', dim=50, force_download=True, dir='./models/glove-model/')

>>> # Let's define a list of words
>>> words = ['cat', 'dog', 'phone', 'computer']

>>> # Let's create a list to store the embeddings of the words and compute them
>>> embeds = []
>>> for word in words:
>>>     embedding = glove.get_embedding(word)
>>>     embeds.append(embedding)

>>> # Let's create a similarity matrix to store the cosine similarity between each pair of embeddings
>>> similarity_matrix = np.zeros((len(words), len(words)))
>>> for i in range(len(embeds)):
        similarity_matrix[i, i] = 1
        for j in range(i + 1, len(embeds)):
            result = cosine_similarity.compute(embeds[i], embeds[j], dim=1).item()
            similarity_matrix[i, j] = result
            similarity_matrix[j, i] = result

>>> # Let's visualize the similarity matrix
>>> from string2string.misc.plotting_functions import plot_heatmap
>>> plot_heatmap(
        similarity_matrix, 
        title='Similarity Between GloVe Embeddings',
        x_ticks = words,
        y_ticks = words,
        x_label = 'Words',
        y_label = 'Words',
        valfmt = '{x:.2f}',
        cmap="Blues",
    )
```
<p align="center">
    <img src="https://github.com/suzgunmirac/string2string/blob/main/fables/similarity-example.png" class="center" />
</p>

### Metrics (sacreBLEU and ROUGE)

In the code snippet below, you can see an example of how to employ the sacreBLEU metric for calculating the BLEU score. This metric is implemented as a wrapper around the [sacreBLEU library](https://github.com/mjpost/sacrebleu). The computation is performed by providing a list of candidate sentences and a list of reference sentences as input and calling the compute method of the metric instance.

```python
>>> # Let's import the sacreBLEU metric from the metrics module and create an instance of it
>>> from string2string.metrics import sacreBLEU
>>> sbleu = sacreBLEU()

>>> # Let's define a list of candidate sentences and a list of reference sentences
>>> candidates = ['The sun is shining.', 'The birds are chirping.', 'She is playing the guitar.', 'He is cooking dinner.']
>>> references = [['The sun is shining.', 'The sun is bright.'], ['The birds are singing.', 'The harold is singing.'], ['Julie is playing the flute.', 'She is playing the piano.'], ['Chef is cooking dinner.', 'He is cooking lunch.']]

>>> # Compute the sacreBLEU score
>>> result = sbleu.compute(candidates, references)
>>> print(result)
# {'score': 67.92604743211312, 'counts': [19, 13, 9, 4], 'totals': [21, 17, 13, 9], 'precisions': [90.47619047619048, 76.47058823529412, 69.23076923076923, 44.44444444444444], 'bp': 1.0, 'sys_len': 21, 'ref_len': 21}
```

Similarly, we can use the ROUGE metric to calculate the ROUGE score. This metric is implemented as a wrapper around the [ROUGE library](https://github.com/google-research/google-research/tree/master/rouge). 

```python
>>> # Let's import the ROUGE metric from the metrics module and create an instance of it
>>> from string2string.metrics import ROUGE
>>> rogue = ROUGE()

>>> # Let's define a list of candidate sentences and a list of reference sentences
>>> candidates = ["The cat is sitting on the mat.", "The dog is barking at the mailman.", "The bird is singing in the tree."] 
>>> references = [["The cat is sitting on the mat."], ["The dog is barking at the postman."], ["The bird sings on the tree."]]

>>> # Compute the ROUGE score
>>> result = rogue.compute(candidates, references)
>>> print(result)
# {'rouge1': 0.8241758241758242, 'rouge2': 0.7323232323232324, 'rougeL': 0.8241758241758242, 'rougeLsum': 0.8241758241758242}
```

### BARTScore and BERTScore

This code snippet shows how to utilize the BARTScore and BERTScore metrics to compute their corresponding scores. These similarity metrics serve as wrappers around the BARTScore and BERTScore libraries.

```python
>>> # Importing the BERTScore and BARTScore classes from the similarity module
>>> from string2string.similarity import BERTScore, BARTScore

>>> # Initializing the BARTScore metric
>>> bart_scorer = BARTScore(model_name_or_path='facebook/bart-large-cnn')

>>> # Initializing the BERTScore metric
>>> bert_scorer = BERTScore(lang="en")

>>> # Define the source and target sentences
>>> source_sentences = ["I'm super happy today.", "John von Neumann was a mathematician and physicist."]
>>> target_sentences = ["I feel pretty wonderful today.", "Kurt Vonnegut's Slaughterhouse Five had a profound impact on many people."]

>>> # Compute the BARTScore
>>> score = bart_scorer.compute(source_sentences, target_sentences, agg="mean", batch_size=4)
>>> print(score)
# {'score': array([-2.77911878, -4.60774326])}

>>> # Compute the BERTScore
>>> score = bert_scorer.compute(source_sentences, target_sentences)
>>> print(score)
# {'precision': array([0.943804 , 0.8559195], dtype=float32), 'recall': array([0.943804  , 0.85145634], dtype=float32), 'f1': array([0.943804 , 0.8536821], dtype=float32)}
```


## Potential Future Enhancements and Improvements

- [ ] Use the [Numba library](https://numba.pydata.org/) to accelerate the execution time of the algorithms.
- [ ] Include additional string-based metrics in the miscellaneous module.
- [ ] Implement [BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi) and [FASTA](https://www.ebi.ac.uk/Tools/sss/fasta/) algorithms. [Work-in-Progress]
- [ ] Add more customizable and useful visualization features.

## Citation

```bibtex
@article{suzgun2023string2string,
  title={string2string: A Modern Python Library for String-to-String Algorithms},
  author={Suzgun, Mirac and Shieber, Stuart M and Jurafsky, Dan},
  journal={arXiv preprint arXiv:[TBD].[TBD]},
  year={2023}
}
```

## Thanks

Our project owes a debt of gratitude to the following individuals for their contributions, comments, and feedback: Federico Bianchi, Corinna Coupette, Sebastian Gehrmann, Tayfun Gür, Şule Kahraman, Deniz Keleş, Luke Melas-Kyriazi, Tolúlopé Ògúnrèmí, Alexander "Sasha" Rush, Kyle Swanson, and Garrett Tanzer.

[def1]: https://colab.research.google.com/drive/11dKisbukdDMaZwp_Tnx_64Z7sn0uQD9c?usp=sharing
[def2]: https://colab.research.google.com/drive/1e8iwBkA7Q4XpmHtxst8_XA-APx4Vsb4j?usp=sharing
[def3]: https://colab.research.google.com/drive/1wu-JOyivxn_52SreF2ukYY7xi4uFVuAx?usp=sharing
[def4]: https://colab.research.google.com/drive/1qNDIkVCEMOVW4WySmzQBvrNAzZ4-zORT?usp=sharing
[def5]: https://colab.research.google.com/drive/1lpXNQn2DSuJB-0iQ-x3h_jx-6-laGpNk?usp=sharing
[def6]: https://colab.research.google.com/drive/1TsMT3DESGY4BNkk-ZRDaL70CRTqrQAtB?usp=sharing