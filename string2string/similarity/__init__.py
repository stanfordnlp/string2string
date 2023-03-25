# The following trick allows us to import the classes directly from the similarity module:
from .bertscore import BERTScore
from .bartscore import BARTScore
from .cosine_similarity import CosineSimilarity
from .classical import LCSimilarity, JaroSimilarity
