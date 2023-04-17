"""
Unit tests for the sacreBLEU module.
"""
import unittest
from unittest import TestCase

from string2string.metrics import sacreBLEU


class SacreBLEUTestCase(TestCase):
    def test_sacrebleu(self):
        # Initialize the sacreBLEU metric
        sbleu = sacreBLEU()
        # Example 1
        candidates = ["The cat is sitting on the mat.", "The dog is barking at the mailman.", "The bird is singing in the tree."] 
        references = [["The cat is sitting on the mat."], ["The dog is barking at the postman."], ["The bird sings on the tree."]]
        # Compute the sacreBLEU score
        result = sbleu.compute(candidates, references)
        score = result['score']
        # Check that the score is correct
        self.assertAlmostEqual(score, 66.37, delta=0.01)

        # Example 2
        candidates = ['The quick brown fox jumps over the lazy dog.', 'This is a test.']
        references = [['The quick brown fox jumps over the lazy dog.'], ['This is only a test.']]
        # Compute the sacreBLEU score
        result = sbleu.compute(candidates, references)
        score = result['score']
        # Check that the score is correct
        self.assertAlmostEqual(score, 81.90, delta=0.01)

        # Example 3
        candidates = ['I am eating lunch.', 'He is studying.']
        references = [['I am having lunch.'], ['He is studying hard.']]
        # Compute the sacreBLEU score
        result = sbleu.compute(candidates, references)
        score = result['score']
        # Check that the score is correct
        self.assertAlmostEqual(score, 32.28, delta=0.01)

        # Example 4
        candidates = ['Random sentence.', 'Random sentence.']
        references = [['Sentence.'], ['Sentence.']]
        # Compute the sacreBLEU score
        result = sbleu.compute(candidates, references)
        score = result['score']
        # Check that the score is correct
        self.assertAlmostEqual(score, 0.0, delta=0.01)

        # Example 5
        candidates = ['Random sentence 1.', 'Random sentence 2.']
        references = [['Random sentence 1.'], ['Random sentence 2.']]
        # Compute the sacreBLEU score
        result = sbleu.compute(candidates, references)
        score = result['score']
        # Check that the score is correct
        self.assertAlmostEqual(score, 100., delta=0.01)

        candidates = ['The sun is shining.', 'The birds are chirping.', 'She is playing the guitar.', 'He is cooking dinner.']
        references = [['The sun is shining.', 'The sun is bright.'], ['The birds are singing.', 'The harold is singing.'], ['Julie is playing the flute.', 'She is playing the piano.'], ['Chef is cooking dinner.', 'He is cooking lunch.']]
        # Compute the sacreBLEU score
        result = sbleu.compute(candidates, references)
        print(result)

if __name__ == "__main__":
    unittest.main()