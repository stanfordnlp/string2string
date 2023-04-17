"""
Unit tests for the search module.
"""
import random
import unittest
from unittest import TestCase

from string2string.misc import PolynomialRollingHash
from string2string.search import RabinKarpSearch, KMPSearch, BoyerMooreSearch, NaiveSearch

class SearcTestCase(TestCase):
    def test_lexical_search_algs(self):
        # Initialize the rolling hash function
        rolling_hash = PolynomialRollingHash(
            base=10,
            modulus=65537,
        )
        # Initialize the Rabin-Karp search algorithm
        rabin_karp = RabinKarpSearch(hash_function=rolling_hash)
        # Initialize the KMP search algorithm
        knuth_morris_pratt = KMPSearch()
        # Initialize the Boyer-Moore search algorithm
        bayer_moore = BoyerMooreSearch()
        # Initialize the naive search algorithm
        naive = NaiveSearch()

        # Example 1
        pattern = 'Jane Austen'
        text = 'Sense and Sensibility, Pride and Prejudice, Emma, Mansfield Park, Northanger Abbey, Persuasion, and Lady Susan were written by Jane Austen and are important works of English literature.'
        # Search for the pattern in the text using all four algorithms        
        idx_rabin_karp = rabin_karp.search(pattern, text)
        idx_knuth_morris_pratt = knuth_morris_pratt.search(pattern, text)
        idx_bayer_moore = bayer_moore.search(pattern, text)
        idx_naive = naive.search(pattern, text)
        # Check that all the four indices are the same 
        self.assertEqual(idx_rabin_karp, idx_knuth_morris_pratt)
        self.assertEqual(idx_rabin_karp, idx_bayer_moore)
        self.assertEqual(idx_rabin_karp, idx_naive)

        # Example 2-11 (randomly generated)
        for _ in range(10):
            # Randomly generate a pattern and a text, using random strings of length 10 and 100, respectively
            pattern = ''.join(random.choices(['a', 'b', 'c'], k=5))
            text = ''.join(random.choices(['a', 'b', 'c'], k=100))
            # Search for the pattern in the text using all four algorithms
            idx_rabin_karp = rabin_karp.search(pattern, text)
            idx_knuth_morris_pratt = knuth_morris_pratt.search(pattern, text)
            idx_bayer_moore = bayer_moore.search(pattern, text)
            idx_naive = naive.search(pattern, text)
            # Check that all the four indices are the same
            self.assertEqual(idx_rabin_karp, idx_knuth_morris_pratt)
            self.assertEqual(idx_rabin_karp, idx_bayer_moore)
            self.assertEqual(idx_rabin_karp, idx_naive)


if __name__ == "__main__":
    unittest.main()