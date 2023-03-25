# The following trick allows us to import the classes directly from the search module:
from .classical import (
    SearchAlgorithm,
    NaiveSearch,
    RabinKarpSearch,
    KMPSearch,
    BoyerMooreSearch,
)
from .faiss_search import FaissSearch