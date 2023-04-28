.. string2string documentation master file, created by
   sphinx-quickstart on Thu Mar 16 16:10:42 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to string2string's documentation!
=========================================

The **string2string** library is an open-source tool that offers a comprehensive suite of efficient algorithms for a broad range of string-to-string problems. It includes both traditional algorithmic solutions and recent advanced neural approaches to address various problems in pairwise string alignment, distance measurement, lexical and semantic search, and similarity analysis. Additionally, the library provides several helpful visualization tools and metrics to facilitate the interpretation and analysis of these methods.

The library features notable algorithms such as the `Smith-Waterman algorithm <https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm>`_ for pairwise local alignment, the `Hirschberg algorithm <https://en.wikipedia.org/wiki/Hirschberg%27s_algorithm>`_ for global alignment, the `Wagner-Fisher algorithm <https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm>`_ for `edit distance <https://en.wikipedia.org/wiki/Edit_distance>`_, `BARTScore <https://github.com/neulab/BARTScore>`_ and `BERTScore <https://github.com/Tiiiger/bert_score>`_ for similarity analysis, the `Knuth-Morris-Pratt <https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm>`_ algorithm for lexical search, and `Faiss <https://github.com/facebookresearch/faiss>`_ for `semantic search <https://en.wikipedia.org/wiki/Semantic_search>`_. Moreover, it wraps existing highly efficient and widely-used implementations of certain frameworks and metrics, such as `sacreBLEU <https://github.com/mjpost/sacrebleu>`_ and `ROUGE <https://github.com/google-research/google-research/tree/master/rouge>`_, whenever it is appropriate and suitable.

In general, the **string2string** library seeks to provide extensive coverage and increased flexibility compared to existing libraries for strings. It can be used for many downstream applications, tasks, and problems in natural-language processing, bioinformatics, and computational social sciences. With its comprehensive suite of algorithms, visualization tools, and metrics, the string2string library is a valuable resource for researchers and practitioners in various fields.

Getting Started
---------------

Install the string2string library by running the following command in your terminal:

.. code-block:: bash

   pip install string2string

Once the installation is complete, you can import the library and start using its functionalities.

**Remark**: We recommend using Python 3.7+ for the library.


Tutorials
---------

.. raw:: html

   <ul>
      <li><a href="https://colab.research.google.com/drive/1TsMT3DESGY4BNkk-ZRDaL70CRTqrQAtB?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://colab.research.google.com/drive/1TsMT3DESGY4BNkk-ZRDaL70CRTqrQAtB?usp=sharing">Tutorial: Alignment Tasks and Algorithms</a></li>
      <li><a href="https://colab.research.google.com/drive/1e8iwBkA7Q4XpmHtxst8_XA-APx4Vsb4j?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://colab.research.google.com/drive/1e8iwBkA7Q4XpmHtxst8_XA-APx4Vsb4j?usp=sharing">Tutorial: Distance Tasks and Algorithms</a></li>
      <li><a href="https://colab.research.google.com/drive/1wu-JOyivxn_52SreF2ukYY7xi4uFVuAx?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://colab.research.google.com/drive/1wu-JOyivxn_52SreF2ukYY7xi4uFVuAx?usp=sharing">Tutorial: Search Tasks and Algorithms</a></li>
      <li><a href="https://colab.research.google.com/drive/1qNDIkVCEMOVW4WySmzQBvrNAzZ4-zORT?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://colab.research.google.com/drive/1qNDIkVCEMOVW4WySmzQBvrNAzZ4-zORT?usp=sharing">Tutorial: Similarity Tasks and Algorithms</a></li>
      <li><a href="https://colab.research.google.com/drive/1lpXNQn2DSuJB-0iQ-x3h_jx-6-laGpNk?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://colab.research.google.com/drive/1lpXNQn2DSuJB-0iQ-x3h_jx-6-laGpNk?usp=sharing">Hands-On Tutorial: Semantic Search and Visualization of USPTO Patents</a></li>
      <li><a href="https://colab.research.google.com/drive/1TsMT3DESGY4BNkk-ZRDaL70CRTqrQAtB?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://colab.research.google.com/drive/1TsMT3DESGY4BNkk-ZRDaL70CRTqrQAtB?usp=sharing">Hands-On Tutorial: Plagiarism Detection of Essays</a></li>
   </ul>

.. toctree::
   :maxdepth: 2
   :caption: Main Modules:

   alignment
   distance
   matching
   similarity
   embedding
   metrics
   miscellaneous

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   hupd_example
   plagiarism_detection


Citation
--------

.. code-block:: bibtex

   @article{suzgun2023string2string,
      title={string2string: A Modern Python Library for String-to-String Algorithms},
      author={Suzgun, Mirac and Shieber, Stuart M and Jurafsky, Dan},
      journal={arXiv preprint arXiv:2304.14395},
      year={2023}
   }



Thanks
------

Our project owes a debt of gratitude to the following individuals for their contributions, comments, and feedback: Federico Bianchi, Corinna Coupette, Sebastian Gehrmann, Tayfun Gür, Şule Kahraman, Deniz Keleş, Luke Melas-Kyriazi, Christopher Manning, Tolúlopé Ògúnrèmí, Alexander "Sasha" Rush, Kyle Swanson, and Garrett Tanzer.