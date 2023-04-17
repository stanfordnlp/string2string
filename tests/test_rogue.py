"""
Unit tests for the ROUGE module.
"""
import unittest
from unittest import TestCase

from string2string.metrics import ROUGE

class ROUGE_TestCase(TestCase):
    def test_rogue(self):
        # Initialize the ROUGE metric
        rogue = ROUGE()
        # Example 1
        candidates = ["The cat is sitting on the mat.", "The dog is barking at the mailman.", "The bird is singing in the tree."] 
        references = [["The cat is sitting on the mat."], ["The dog is barking at the postman."], ["The bird sings on the tree."]]
        # Compute the ROUGE score
        result = rogue.compute(candidates, references)
        print(result)
        r1, r2, rl, rlsum = result['rouge1'], result['rouge2'], result['rougeL'], result['rougeLsum']
        # Check that the score is correct
        self.assertAlmostEqual(r1, 0.824, delta=0.01)
        self.assertAlmostEqual(r2, 0.732, delta=0.01)
        self.assertAlmostEqual(rl, 0.824, delta=0.01)
        self.assertAlmostEqual(rlsum, 0.824, delta=0.01)

        # Example 2
        candidates = ['The quick brown fox jumps over the lazy dog.', 'This is a test.']
        references = [['The quick brown fox jumps over the lazy dog.'], ['This is only a test.']]
        # Compute the ROUGE score
        result = rogue.compute(candidates, references)
        r1, r2, rl, rlsum = result['rouge1'], result['rouge2'], result['rougeL'], result['rougeLsum']
        # Check that the score is correct
        self.assertAlmostEqual(r1, 0.944, delta=0.01)
        self.assertAlmostEqual(r2, 0.786, delta=0.01)
        self.assertAlmostEqual(rl, 0.944, delta=0.01)
        self.assertAlmostEqual(rlsum, 0.944, delta=0.01)

        # Example 3
        candidates = ['I am eating lunch.', 'He is studying.']
        references = [['I am having lunch.'], ['He is studying hard.']]
        # Compute the ROUGE score
        result = rogue.compute(candidates, references)
        r1, r2, rl, rlsum = result['rouge1'], result['rouge2'], result['rougeL'], result['rougeLsum']
        # Check that the score is correct
        self.assertAlmostEqual(r1, 0.661, delta=0.01)
        self.assertAlmostEqual(r2, 0.367, delta=0.01)
        self.assertAlmostEqual(rl, 0.661, delta=0.01)
        self.assertAlmostEqual(rlsum, 0.661, delta=0.01)

        # Example 4
        candidates = ['Random sentence.', 'Random sentence.']
        references = [['Sentence.'], ['Sentence.']]
        # Compute the ROUGE score
        result = rogue.compute(candidates, references)
        r1, r2, rl, rlsum = result['rouge1'], result['rouge2'], result['rougeL'], result['rougeLsum']
        # Check that the score is correct
        self.assertAlmostEqual(r1, 0.0, delta=0.01)
        self.assertAlmostEqual(r2, 0.0, delta=0.01)
        self.assertAlmostEqual(rl, 0.0, delta=0.01)
        self.assertAlmostEqual(rlsum, 0.0, delta=0.01)

        # Example 5
        candidates = ['Random sentence 1.', 'Random sentence 2.']
        references = [['Random sentence 1.'], ['Random sentence 2.']]
        # Compute the ROUGE score
        result = rogue.compute(candidates, references)
        r1, r2, rl, rlsum = result['rouge1'], result['rouge2'], result['rougeL'], result['rougeLsum']
        # Check that the score is correct
        self.assertAlmostEqual(r1, 1.0, delta=0.01)
        self.assertAlmostEqual(r2, 1.0, delta=0.01)
        self.assertAlmostEqual(rl, 1.0, delta=0.01)
        self.assertAlmostEqual(rlsum, 1.0, delta=0.01)

if __name__ == "__main__":
    unittest.main()