"""
Unit tests for the distance module.
"""
import unittest
from unittest import TestCase

from string2string.distance import LevenshteinEditDistance, HammingDistance, DamerauLevenshteinDistance, JaccardIndex

class DistanceTestCase(TestCase):
    def test_levenshtein_edit_distance_unit_operations(self):
        ## Case 1: Costs of insertion, deletion, and substitution are all 1.
        edit_distance = LevenshteinEditDistance()
        # Example 0
        dist = edit_distance.compute("", "")
        self.assertEqual(dist, 0.0)
        # Example 1
        dist = edit_distance.compute("aa", "bb")
        self.assertEqual(dist, 2.0)
        # Example 2
        dist = edit_distance.compute("monty-python", "monty-python")
        self.assertEqual(dist, 0.0)
        # Example 3
        dist = edit_distance.compute("kitten", "sitting")
        self.assertEqual(dist, 3.0)
        # Example 4
        dist = edit_distance.compute("sitting", "kitten")
        self.assertEqual(dist, 3.0)
        # Example 5
        dist = edit_distance.compute("aaaaa", "a")
        self.assertEqual(dist, 4.0)
        # Example 6
        dist = edit_distance.compute("", "abcdef")
        self.assertEqual(dist, 6.0)
        # Example 7
        dist = edit_distance.compute("abcdef", "")
        self.assertEqual(dist, 6.0)
        # Example 8
        dist = edit_distance.compute("algorithm", "al-Khwarizmi")
        self.assertEqual(dist, 8.0)
        # Example 9
        dist = edit_distance.compute("qrrq", "rqqr")
        self.assertEqual(dist, 3.0)
        # Example 10
        dist = edit_distance.compute(["kurt", "godel"], ["godel", "kurt"])
        self.assertEqual(dist, 2.0)
        # Example 11
        dist = edit_distance.compute(
            ["kurt", "godel", "kurt"], ["godel", "kurt"]
        )
        self.assertEqual(dist, 1.0)

    def test_levenshtein_edit_distance_weighted_operations(self):
        ## Case 2: insertion = 2., deletion = 2., substitution = 1., match = 0.
        weighted_edit_distance = LevenshteinEditDistance(
            match_weight=0.0,
            insert_weight=2.0,
            delete_weight=2.0,
            substitute_weight=1.0,
        )
        # Example 1
        dist = weighted_edit_distance.compute("aa", "bb")
        self.assertEqual(dist, 2.0)
        # Example 2
        dist = weighted_edit_distance.compute("aca", "bcb")
        self.assertEqual(dist, 2.0)
        # Example 3
        dist = weighted_edit_distance.compute("aa", "")
        self.assertEqual(dist, 4.0)
        # Example 4
        dist = weighted_edit_distance.compute("", "aa")
        self.assertEqual(dist, 4.0)
        # Example 5
        dist = weighted_edit_distance.compute("witty", "witty")
        self.assertEqual(dist, 0.0)
        # Example 6
        dist = weighted_edit_distance.compute("ttss", "stst")
        self.assertEqual(dist, 2.0)

    def test_damerau_levenshtein_edit_distance_unit_operations(self):
        ## Case 1: Costs of insertion, deletion, substitution, and transposition are all 1.
        dameraulevenshteindist = DamerauLevenshteinDistance()
        # Example 0
        dist = dameraulevenshteindist.compute("", "")
        self.assertEqual(dist, 0.0)
        # Example 1
        dist = dameraulevenshteindist.compute("aa", "bb")
        self.assertEqual(dist, 2.0)
        # # Example 2
        dist = dameraulevenshteindist.compute(
            "monty-python", "monty-python"
        )
        self.assertEqual(dist, 0.0)
        # Example 3
        dist = dameraulevenshteindist.compute("ab", "ba")
        self.assertEqual(dist, 1.0)
        # Example 4
        dist = dameraulevenshteindist.compute("sitting", "kitten")
        self.assertEqual(dist, 3.0)
        # Example 5
        dist = dameraulevenshteindist.compute("baaaaa", "ab")
        self.assertEqual(dist, 5.0)
        # Example 6
        dist = dameraulevenshteindist.compute("ababab", "bababa")
        self.assertEqual(dist, 2.0)
        # Example 7
        dist = dameraulevenshteindist.compute("abxymn", "bayxnm")
        self.assertEqual(dist, 3.0)
        # Example 8
        dist = dameraulevenshteindist.compute("wikiepdia", "wikipedia")
        self.assertEqual(dist, 1.0)
        # Example 9
        dist = dameraulevenshteindist.compute("microaoft", "microsoft")
        self.assertEqual(dist, 1.0)
        # Example 10
        dist = dameraulevenshteindist.compute(
            ["kurt", "godel"], ["godel", "kurt"]
        )
        self.assertEqual(dist, 1.0)
        # Example 11
        dist = dameraulevenshteindist.compute(
            ["kurt", "godel", "kurt"], ["godel", "kurt"]
        )
        self.assertEqual(dist, 1.0)
        # Example 12
        dist = dameraulevenshteindist.compute("microaoft", "microsoft")
        self.assertEqual(dist, 1.0)

    def test_hamming_edit_distance(self):
        hamming_distance = HammingDistance()
        # Example 1
        dist = hamming_distance.compute("aa", "bb")
        self.assertEqual(dist, 2.0)
        # Example 2
        dist = hamming_distance.compute("aac", "abc")
        self.assertEqual(dist, 1.0)
        # Example 3
        dist = hamming_distance.compute("Turing1912", "during1921")
        self.assertEqual(dist, 3.0)
        # Example 4
        dist = hamming_distance.compute("John von Neumann", "John von Neumann")
        self.assertEqual(dist, 0.0)
        # Example 5
        dist = hamming_distance.compute("Earth", "earth")
        self.assertEqual(dist, 1.0)
        # Example 6
        with self.assertRaises(ValueError):
            dist = hamming_distance.compute(" ", "abc")
        # Example 7
        dist = hamming_distance.compute("", "")
        self.assertEqual(dist, 0.0)
        # Example 8
        dist = hamming_distance.compute(
            ["", "abc", "234", "#"], ["", "abc", "123", "#"]
        )
        self.assertEqual(dist, 1.0)
        # Example 9
        dist = hamming_distance.compute(
            ["a", "ab", "abc", "abcd", "abc", "ab", "a"],
            ["a", "ab", "abc", "abcd", "abc", "ab", "a"],
        )
        self.assertEqual(dist, 0.0)


    def test_jaccard_indexx(self):
        jaccard_index = JaccardIndex()
        # Example 1
        dist = jaccard_index.compute("aa", "bb")
        self.assertEqual(dist, 1.0)
        # Example 2
        dist = jaccard_index.compute("ab", "ba")
        self.assertEqual(dist, 0.0)
        # Example 3
        dist = jaccard_index.compute("ab", "baaaaab")
        self.assertEqual(dist, 0.0)
        # Example 4
        dist = jaccard_index.compute("ab", "bbbbaaaacd")
        self.assertEqual(dist, 0.5)
        # Example 5
        dist = jaccard_index.compute("ab", "cd")
        self.assertEqual(dist, 1.0)
        # Example 6
        dist = jaccard_index.compute(
            "The quick brown fox jumps over the lazy dog", 
            "The quick brown cat jumps over the lazy dog"
            )
        self.assertEqual(dist, 0.0714285714285714)
        # Example 7
        dist = jaccard_index.compute("apple", "banana")
        self.assertEqual(dist, 0.8333333333333334)
        # Example 8
        dist = jaccard_index.compute(
            ['a','p', 'p', 'l', 'e'], 
            ['b', 'a', 'n', 'a', 'n', 'a']
            )
        self.assertEqual(dist, 0.8333333333333334)
        # Example 9
        dist = jaccard_index.compute(
            ['a','p', 'p', 'l', 'e'],
            ['a','p', 'p', 'p', 'l', 'e', 'e'],
            )
        self.assertEqual(dist, 0.0)


if __name__ == "__main__":
    unittest.main()