"""
Unit tests for the distance module.
"""
import unittest
from unittest import TestCase

from string2string.alignment import (
    NeedlemanWunsch,
    Hirschberg,
    SmithWaterman,
    DTW,
    LongestCommonSubsequence,
    LongestCommonSubstring,
)

class AlignmentTestCase(TestCase):
    # Testing LngestCommonSubsequence
    def test_longest_common_subsequence(self):
        lcsubsequence = LongestCommonSubsequence()
        # Example 1
        length, candidates = lcsubsequence.compute(
            "aa", "aa", 
            returnCandidates=True,
        )
        self.assertEqual(length, 2.0)
        self.assertCountEqual(candidates, ["aa"])
        # Example 2
        length, candidates = lcsubsequence.compute(
            "ab", "ba", returnCandidates=True
        )
        self.assertEqual(length, 1.0)
        self.assertCountEqual(candidates, ["a", "b"])
        # Example 3
        length, candidates = lcsubsequence.compute(
            "ab", "cd", returnCandidates=True
        )
        self.assertEqual(length, 0.0)
        self.assertCountEqual(candidates, [])
        # Example 4
        length, candidates = lcsubsequence.compute(
            "ab", "xxaaabyy", returnCandidates=True
        )
        self.assertEqual(length, 2.0)
        self.assertCountEqual(candidates, ["ab"])
        # Example 5
        length, candidates = lcsubsequence.compute(
            "abcd", "xcxaaabydy", returnCandidates=True
        )
        self.assertEqual(length, 3.0)
        self.assertCountEqual(candidates, ["abd"])
        # Example 6
        length, candidates = lcsubsequence.compute(
            "aabbccdd", "dcdcbaba", returnCandidates=True
        )
        self.assertEqual(length, 2.0)
        self.assertCountEqual(candidates, ["dd", "cc", "bb", "aa", "cd", "ab"])
        # Example 7
        length, candidates = lcsubsequence.compute(
            ["abcd"], ["xcxaaabydy"], 
            returnCandidates=True, 
            boolListOfList=True
        )
        self.assertEqual(length, 0.0)
        self.assertCountEqual(candidates,[])
        # Example 8
        length, candidates = lcsubsequence.compute(
            ["a", "bb", "c"],
            ["a", "bb", "c"],
            returnCandidates=True,
            boolListOfList=True,
        )
        self.assertEqual(length, 3.0)
        self.assertCountEqual(candidates, [["a", "bb", "c"]])
        # Example 9
        length, candidates = lcsubsequence.compute(
            ["a", "b", "c", "dd"],
            ["x", "c", "x", "a", "a", "a", "b", "y", "dd", "y"],
            returnCandidates=True,
            boolListOfList=True,
        )
        self.assertEqual(length, 3.0)
        self.assertCountEqual(candidates, [["a", "b", "dd"]])
        # Example 10
        length, candidates = lcsubsequence.compute(
            ["a", "t", "b", "c", "y", "dd", "xyz"],
            ["x", "c", "x", "t", "a", "a", "a", "b", "y", "dd", "y", "xyz"],
            returnCandidates=True,
            boolListOfList=True,
        )
        self.assertEqual(length, 5.0)
        self.assertCountEqual(
            candidates, [["t", "b", "y", "dd", "xyz"], ["a", "b", "y", "dd", "xyz"]]
        )

    # Testing LongestCommonSubstring
    def test_longest_common_subsubtring(self):
        lcsubstring = LongestCommonSubstring()
        # Example 1
        length, candidates = lcsubstring.compute(
            "aa", "aa", 
            returnCandidates=True,
        )
        self.assertEqual(length, 2)
        self.assertCountEqual(candidates, ["aa"])
        # Example 2
        length, candidates = lcsubstring.compute(
            "aabb", "aa", 
            returnCandidates=True,
        )
        self.assertEqual(length, 2)
        self.assertCountEqual(candidates, ["aa"])
        # Example 3
        length, candidates = lcsubstring.compute(
            "aabbaa", "aa", 
            returnCandidates=True,
        )
        self.assertEqual(length, 2)
        self.assertCountEqual(candidates, ["aa"])
        # Example 4
        length, candidates = lcsubstring.compute(
            "xyxy", "yxyx", 
            returnCandidates=True,
        )
        self.assertEqual(length, 3)
        self.assertCountEqual(candidates, ["xyx", "yxy"])
        # Example 4
        length, candidates = lcsubstring.compute(
            "xyxy", "yxyx", 
            returnCandidates=True,
        )
        self.assertEqual(length, 3)
        self.assertCountEqual(candidates, ["xyx", "yxy"])
        # Example 5
        length, candidates = lcsubstring.compute(
            ["x", "y", "x", "y"],
            ["y", "x", "y", "x"],
            returnCandidates=True,
            boolListOfList=True,
        )
        self.assertEqual(length, 3)
        self.assertCountEqual(
            set(map(tuple, candidates)), 
            set(map(tuple, [["x", "y", "x"], ["y", "x", "y"]]))
            )
        # Example 6
        length, candidates = lcsubstring.compute(
            ["a", "a", "a", "a"], ["a"], 
            returnCandidates=True, 
            boolListOfList=True
        )
        self.assertEqual(length, 1)
        self.assertCountEqual(
            set(map(tuple, candidates)), 
            set(map(tuple, [["a"]]))
        )
        # Example 7
        length, candidates = lcsubstring.compute(
            "x", "xxxx", 
            returnCandidates=True,
        )
        self.assertEqual(length, 1)
        self.assertCountEqual(candidates, ["x"])
        # Example 8
        length, candidates = lcsubstring.compute(
            " julia ", "  julie ", 
            returnCandidates=True,
        )
        self.assertEqual(length, 5)
        self.assertCountEqual(candidates, [" juli"])

    # Testing NeedlemanWunsch
    def test_needleman_wunsch(self):
        # First set of examples
        needlemanwunsch = NeedlemanWunsch(
            match_weight=1, 
            mismatch_weight=-1, 
            gap_weight=-1,
        )
        # Example 1
        aligned_str1, aligned_str2 = needlemanwunsch.get_alignment(
            str1 = ["a", "b", "bb"], 
            str2 = ["a", "bb", "b", "bb"],
        )
        self.assertEqual(aligned_str1, "a | -  | b | bb")
        self.assertEqual(aligned_str2, "a | bb | b | bb")
        # Example 2
        aligned_str1, aligned_str2 = needlemanwunsch.get_alignment(
            str1 = "abcbd", 
            str2 = "abcde",
        )
        self.assertEqual(aligned_str1, "a | b | c | b | d | -")
        self.assertEqual(aligned_str2, "a | b | c | - | d | e")
        # Example 3
        aligned_str1, aligned_str2 = needlemanwunsch.get_alignment(
            str1 = "AATGCATGCGTT",
            str2 = "AATGATTACATT",
        )
        self.assertTrue(aligned_str1 == 'A | A | T | G | C | A | T | G | - | C | G | T | T' or aligned_str1 == "A | A | T | G | C | A | - | T | G | C | G | T | T")
        self.assertEqual(aligned_str2, "A | A | T | G | - | A | T | T | A | C | A | T | T")

        # Another set of examples
        needlemanwunsch = NeedlemanWunsch(
            match_weight=2, 
            mismatch_weight=-1, 
            gap_weight=-2,
        )
        # Example 4
        aligned_str1, aligned_str2 = needlemanwunsch.get_alignment(
            str1 = "AGTACGCA",
            str2 = "TATGC",
        )
        self.assertEqual(aligned_str1, "A | G | T | A | C | G | C | A")
        self.assertEqual(aligned_str2, "- | - | T | A | T | G | C | -")
        # Example 5
        aligned_str1, aligned_str2 = needlemanwunsch.get_alignment(
            str1 = ['G', 'A', 'TWW', 'T', 'AWW', 'C', 'A'],
            str2 = ['G', 'CAA', 'A', 'T', 'XXG', 'C', 'U'],
        )
        self.assertEqual(aligned_str1, "G | A   | TWW | T | AWW | C | A")
        self.assertEqual(aligned_str2, "G | CAA | A   | T | XXG | C | U")
        # Example 6
        aligned_str1, aligned_str2 = needlemanwunsch.get_alignment(
            str1 = "GATTACA",
            str2 = "GCATGCU",
        )
        self.assertEqual(aligned_str1, "G | A | T | T | A | C | A")
        self.assertEqual(aligned_str2, "G | C | A | T | G | C | U")


    # Testing Hirschberg
    def test_hiirschberg(self):
        # First set of examples
        hirschberg = Hirschberg(
            match_weight=1, 
            mismatch_weight=-1, 
            gap_weight=-1,
        )
        # Example 1
        aligned_str1, aligned_str2 = hirschberg.get_alignment(
            str1 = ["a", "b", "bb"], 
            str2 = ["a", "bb", "b", "bb"],
        )
        self.assertEqual(aligned_str1, "a | -  | b | bb")
        self.assertEqual(aligned_str2, "a | bb | b | bb")
        # Example 2
        aligned_str1, aligned_str2 = hirschberg.get_alignment(
            str1 = "abcbd", 
            str2 = "abcde",
        )
        self.assertEqual(aligned_str1, "a | b | c | b | d | -")
        self.assertEqual(aligned_str2, "a | b | c | - | d | e")
        # Example 3
        aligned_str1, aligned_str2 = hirschberg.get_alignment(
            str1 = "AATGCATGCGTT",
            str2 = "AATGATTACATT",
        )
        self.assertTrue(aligned_str1 == 'A | A | T | G | C | A | T | G | - | C | G | T | T' or aligned_str1 == "A | A | T | G | C | A | - | T | G | C | G | T | T")
        self.assertEqual(aligned_str2, "A | A | T | G | - | A | T | T | A | C | A | T | T")

        # Another set of examples
        hirschberg = Hirschberg(
            match_weight=2, 
            mismatch_weight=-1, 
            gap_weight=-2,
        )
        # Example 4
        aligned_str1, aligned_str2 = hirschberg.get_alignment(
            str1 = "AGTACGCA",
            str2 = "TATGC",
        )
        self.assertEqual(aligned_str1, "A | G | T | A | C | G | C | A")
        self.assertEqual(aligned_str2, "- | - | T | A | T | G | C | -")
        # Example 5
        aligned_str1, aligned_str2 = hirschberg.get_alignment(
            str1 = ['G', 'A', 'TWW', 'T', 'AWW', 'C', 'A'],
            str2 = ['G', 'CAA', 'A', 'T', 'XXG', 'C', 'U'],
        )
        self.assertEqual(aligned_str1, "G | A   | TWW | T | AWW | C | A")
        self.assertEqual(aligned_str2, "G | CAA | A   | T | XXG | C | U")
        # Example 6
        aligned_str1, aligned_str2 = hirschberg.get_alignment(
            str1 = "GATTACA",
            str2 = "GCATGCU",
        )
        self.assertEqual(aligned_str1, "G | A | T | T | A | C | A")
        self.assertEqual(aligned_str2, "G | C | A | T | G | C | U")

    # Testing SmithWaterman
    def test_smithwaterman(self):
        smithwaterman = SmithWaterman(
            match_weight=1,
            mismatch_weight=-1,
            gap_weight=-1,
            gap_char="-",
        )
        # Example 1
        aligned_str1, aligned_str2 = smithwaterman.get_alignment(
            str1 = "abcbd",
            str2 = "abcde",
        )
        self.assertEqual(aligned_str1, "a | b | c")
        self.assertEqual(aligned_str2, "a | b | c")
        # Example 2
        aligned_str1, aligned_str2 = smithwaterman.get_alignment(
            str1 = "GAATGCATGCGTT",
            str2 = "TAATGCATGCGGT",
        )
        self.assertEqual(aligned_str1, "A | A | T | G | C | A | T | G | C | G")
        self.assertEqual(aligned_str2, "A | A | T | G | C | A | T | G | C | G")
        # Example 3
        aligned_str1, aligned_str2 = smithwaterman.get_alignment(
            str1 = "TACGGGCCCGCTAC",
            str2 = "TAGCCCTATCGGTCA",
        )
        self.assertEqual(aligned_str1, "T | A | - | C | G | G")
        self.assertEqual(aligned_str2, "T | A | T | C | G | G")
        # Example 4
        aligned_str1, aligned_str2 = smithwaterman.get_alignment(
            str1 = "GAGTCGCTACGGGCCCGCTAC",
            str2 = "TAGCCTATGCACCTATCGGTCA",
        )
        self.assertEqual(aligned_str1, "C | T | A | - | C | G | G")
        self.assertEqual(aligned_str2, "C | T | A | T | C | G | G")

    # Testing DTW
    def test_dtw(self):
        dtw = DTW()
        # Example 1
        alignment  = dtw.get_alignment_path(
            sequence1=[1, 2, 3], 
            sequence2=[1, 2, 3, 4],
            distance='absolute_difference',
        )
        self.assertCountEqual(alignment, [(0, 0), (1, 1), (2, 2), (2, 3)])
        # Example 2
        alignment  = dtw.get_alignment_path(
            sequence1=[1, 2, 3],
            sequence2=[1, 2, 3],
            distance='absolute_difference',
        )
        self.assertCountEqual(alignment, [(0, 0), (1, 1), (2, 2)])
        # Example 3
        alignment  = dtw.get_alignment_path(
            sequence1="abc",
            sequence2="abcd",
            distance='absolute_difference',
        )
        self.assertCountEqual(alignment, [(0, 0), (1, 1), (2, 2), (2, 3)])
        # Example 4
        alignment  = dtw.get_alignment_path(
            sequence1=["a", "b", "c"],
            sequence2=["a", "b", "c", "d"],
            distance='absolute_difference',
        )
        self.assertCountEqual(alignment, [(0, 0), (1, 1), (2, 2), (2, 3)])
        # Example 5
        alignment  = dtw.get_alignment_path(
            sequence1=[10, 20, 30],
            sequence2=[20, 50, 60, 30],
            distance='absolute_difference',
        )
        self.assertCountEqual(alignment, [(0, 0), (1, 0), (2, 1), (2, 2), (2, 3)])


if __name__ == "__main__":
    unittest.main()
