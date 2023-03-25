"""
    This module contains the following algorithms:
        (-) Naive search algorithm ++
        (a) Rabin-Karp algorithm ++ 
        (b) Boyer-Moore algorithm ++ 
        (c) Knuth-Morris-Pratt algorithm
        (d) Suffix Tree algorithm
        (e) Suffix Array algorithm
        (f) Suffix Automaton algorithm
        (g) Aho-Corasick algorithm (basis of fgrep/grep in Unix) ++ (not implemented)
        (h) Ukkonen's algorithm -- (not implemented)
        (i) Wu-Manber algorithm ++ (not implemented)
        (j) Z-Algorithm ++ (not implemented)
"""

from typing import List, Union, Tuple, Optional
from string2string.misc.hash_functions import HashFunction, PolynomialRollingHash


# Parent class for all search algorithms
class SearchAlgorithm:
    """
    This class contains the parent class for all search algorithms.
    """

    def __init__(self) -> None:
        """
        This function initializes the abstract class for all search algorithms.

        Returns:
            None
        """
        pass

    def search(self,
        pattern: str,
        text: str,
    ) -> int:
        """
        Searches for the pattern in a text.

        Arguments:
            pattern (str): The pattern to search for.
            text (str): The text to search in.

        Returns:
            int: The index of the pattern in the text.
        """
        pass


class NaiveSearch(SearchAlgorithm):
    """
    This class contains the naive search algorithm.
    """

    def __init__(self) -> None:
        """
        Initializes the class.

        Returns:
            None
        """
        super().__init__()


    def search(self,
        pattern: str,
        text: str,
    ) -> int:
        """
        Searches for the pattern in the text.

        Arguments:
            text (str): The text to search in.

        Returns:
            int: The index of the pattern in the text (or -1 if the pattern is not found).

        Raises:
            AssertionError: If the inputs are invalid.
        """
        # Check the inputs
        assert isinstance(pattern, str), 'The pattern must be a string.'
        assert isinstance(text, str), 'The text must be a string.'

        # Set the attributes
        self.pattern = pattern
        self.pattern_length = len(self.pattern)

        # Loop over the text
        for i in range(len(text) - self.pattern_length + 1):
            # Check if the strings match
            if text[i:i + self.pattern_length] == self.pattern:
                return i
            
        # Return -1 if the pattern is not found
        return -1


# Rabin-Karp search algorithm class
class RabinKarpSearch(SearchAlgorithm):
    """
    This class contains the Rabin-Karp search algorithm.
    """

    def __init__(self,
        hash_function: HashFunction = PolynomialRollingHash(),
    ) -> None:
        """
        This function initializes the Rabin-Karp search algorithm class, which uses a hash function to search for a pattern in a text. [RK1987]_

        Arguments:
            hash_function (HashFunction): The hash function to use.

        Returns:
            None

        Raises:
            AssertionError: If the inputs are invalid. 

        .. [RK1987] Karp, R.M. and Rabin, M.O., 1987. Efficient randomized pattern-matching algorithms. IBM Journal of Research and Development, 31(2), pp.249-260.
        """
        assert isinstance(hash_function, HashFunction), 'The hash function must be a HashFunction object.'

        # Set the attributes
        # self.pattern = pattern
        self.hash_function = hash_function

        # # Compute the hash value of the pattern
        # self.pattern_hash = self.hash_function.compute(self.pattern)

        # # Length of the pattern
        # self.pattern_length = len(self.pattern)

    def itialize_pattern_hash(self,
        pattern: str,
    ) -> None:
        """
        This function initializes the pattern hash value.

        Arguments:
            pattern (str): The pattern to search for.

        Returns:
            None

        Raises:
            AssertionError: If the inputs are invalid.
        """
        # Check the inputs
        assert isinstance(pattern, str), 'The pattern must be a string.'

        # Reset the hash function
        self.hash_function.reset()

        # Set the attributes
        self.pattern = pattern

        # Compute the hash value of the pattern
        self.pattern_hash = self.hash_function.compute(self.pattern)

        # Length of the pattern
        self.pattern_length = len(self.pattern)


    def search(self,
        pattern: str,
        text: str,
    ) -> int:
        """
        This function searches for the pattern in the text.

        Arguments:
            pattern (str): The pattern to search for.
            text (str): The text to search in.

        Returns:
            int: The index of the pattern in the text (or -1 if the pattern is not found).

        Raises:
            AssertionError: If the inputs are invalid.

        
        """
        # Check the inputs
        assert isinstance(text, str), 'The text must be a string.'

        # Initialize the pattern hash
        self.itialize_pattern_hash(pattern)

        # Reset the hash function (in case it was used before) [Important!]
        self.hash_function.reset()

        # Compute the hash value of the first window
        window_hash = self.hash_function.compute(text[:self.pattern_length])

        # Loop over the text
        for i in range(len(text) - self.pattern_length + 1):
            # print('Window hash: {}'.format(window_hash))

            # Check if the hash values match
            if window_hash == self.pattern_hash:
                # print('Hash values match at index {}.'.format(i))
                j = 0
                # Check if the strings match
                while text[i + j] == self.pattern[j]:
                    j += 1
                    if j == self.pattern_length:
                        return i
            # Update the hash value of the window
            if i < len(text) - self.pattern_length:
                window_hash = self.hash_function.update(text[i], text[i + self.pattern_length], self.pattern_length)

        # Return -1 if the pattern is not found
        return -1
    

# Knuth-Morris-Pratt (KMP) search algorithm class
class KMPSearch(SearchAlgorithm):
    """
    This class contains the KMP search algorithm.
    """
    
    def __init__(self) -> None:
       r"""
        This function initializes the Knuth-Morris-Pratt (KMP) search algorithm class. [KMP1977]_

        Arguments:
            None

        Returns:
            None

        .. note::
            * The current version of the KMP algorithm utilizes an auxiliary list called the lps_array, which stands for "longest proper prefix which is also a suffix". The lps_array is a list of integers where lps_array[i] represents the length of the longest proper prefix of the pattern that is also a suffix of the pattern[:i+1].
            * By precomputing the lps_array, the KMP algorithm avoids unnecessary character comparisons while searching for the pattern in the text. The algorithm scans the text from left to right and compares characters in the pattern with characters in the text. When a mismatch occurs, the algorithm uses the values in the lps_array to determine the next character in the pattern to compare with the text.
            * An alternative implementation of the KMP algorithm exists, which uses a finite state automaton (FSA) instead of the lps_array, but this is not implemented in this version of the package.

        .. [KMP1977] Knuth, D.E., Morris, J.H. and Pratt, V.R., 1977. Fast pattern matching in strings. SIAM journal on computing, 6(2), pp.323-350.
        """
       super().__init__()

    # Initialize_lps function
    def initialize_lps(self) -> None:
        r"""
        This function initializes the pongest proper prefix suffix (lps) array, which contains the length of the longest proper prefix that is also a suffix of the pattern.

        IOW: For each index i in the lps array, lps[i] is the length of the longest proper prefix that is also a suffix of the pattern[:i + 1]. In other words, if k = lps[i], then pattern[:k] is equal to pattern[i - k + 1:i + 1] (with the condition that pattern[:k+1] is not equal to pattern[i - k:i + 1]). The lps array is used in the Knuth-Morris-Pratt (KMP) algorithm to avoid unnecessary comparisons when searching for a pattern in a text.

        Arguments:
            pattern (str): The pattern to search for.

        Returns:
            None
        """
        # Initialize the list of longest proper prefix which is also a suffix
        self.lps = [0] * self.pattern_length

        # Loop over the pattern
        i = 1 # denotes the index of the character in the pattern
        j = 0 # denotes the length of the longest proper prefix which is also a suffix of the pattern[:i]
        while i < self.pattern_length:
            # Check if the characters match
            if self.pattern[i] == self.pattern[j]:
                j += 1
                self.lps[i] = j
                i += 1
            else:
                if j != 0: 
                    j = self.lps[j - 1]
                else:
                    self.lps[i] = 0
                    i += 1

    # Search for the pattern in the text
    def search(self,
        pattern: str,
        text: str,
    ) -> int:
        """
        This function searches for the pattern in the text.

        Arguments:
            pattern (str): The pattern to search for.
            text (str): The text to search in.

        Returns:
            int: The index of the pattern in the text (or -1 if the pattern is not found)

        Raises:
            AssertionError: If the text is not a string.

        .. note::
            * This is the main function of the KMP search algorithm class.
        """
        # Check the inputs
        assert isinstance(text, str), 'The text must be a string.'

        # Set the attributes
        self.pattern = pattern
        self.pattern_length = len(self.pattern)

        # Initialize the lps array
        self.initialize_lps()

        # Loop over the text
        i = 0
        j = 0
        while i < len(text):
            # Check if the characters match
            if self.pattern[j] == text[i]:
                i += 1
                j += 1
            # Check if the pattern is found
            if j == self.pattern_length:
                return i - j
            # Check if the characters do not match
            elif i < len(text) and self.pattern[j] != text[i]:
                if j != 0:
                    j = self.lps[j - 1]
                else:
                    i += 1

        # Return -1 if the pattern is not found
        return -1
    


# Boyer-Moore search algorithm class
class BoyerMooreSearch:
    """
    This class contains the Boyer-Moore search algorithm.
    """

    def __init__(self) -> None:
        """
        This function initializes the Boyer-Moore search algorithm class. [BM1977]_

        The Bayer-Moore search algorithm is a string searching algorithm that uses a heuristic to skip over large sections of the search string, resulting in faster search times than traditional algorithms such as brute-force or Knuth-Morris-Pratt. It is particularly useful for searching for patterns in large amounts of text.

        .. [BM1977] Boyer, RS and Moore, JS. "A fast string searching algorithm." Communications of the ACM 20.10 (1977): 762-772.
            
        A Correct Preprocessing Algorithm for Boyer–Moore String-Searching

        https://www.cs.jhu.edu/~langmea/resources/lecture_notes/strings_matching_boyer_moore.pdf

        """
        super().__init__()

    
    # This is what we call the "prefix - suffix" match case of the good suffix rule
    def aux_get_suffix_prefix_length(self,
        i: int,
    ) -> int:
        """
        This auxiliary function is used to compute the length of the longest suffix of pattern[i:] that matches a "prefix" of the pattern.

        Arguments:
            i (int): The index of the suffix.

        Returns:
            int: The length of the longest suffix of pattern[i:] that matches a "prefix" of the pattern.
        """
        
        # pattern [ ....... i ................j]
        # Initialize j to the end of the pattern
        j = self.pattern_length - 1
        
        # pattern [ ....... i ....... j .......]
        # Move j to the left until we find a mismatch or until j == i
        while j >= i and self.pattern[j] == self.pattern[j - i]:
            # pattern [ ... j-i ..... i ... j .......]
            j -= 1
        
        return self.pattern_length - (j - 1)
    

    # This is what we call the "substring match" case of the good suffix rule
    def aux_get_matching_substring_length(self,
        j: int,
    ) -> int:
        """
        This auxilary function is used to compute the length of the longess suffix of the patterm that matches a substring of the pattern that ends at the index j. 

        It is used in the "substring match" case of the good suffix rule. More specifically, it is used to find when the suffix of the pattern does not match the text at all. Hence, we find the longest suffix of the pattern that matches a substring of the pattern that ends at the index j.

        Arguments:
            j (int): The end index of the substring.

        Returns:
            int: The length of the longess suffix of the patterm that matches a substring of the pattern that ends at the index j.

        """
        # Loop over the suffixes of the pattern
        for i in range(j, -1, -1):
            # Check if the substring matches the suffix
            if self.pattern[i:i+(j+1)] == self.pattern[self.pattern_length-(j+1):]:
                return j - i + 1
        # Otherwise, if we get here, the substring does not match any suffix of the pattern
        return 0
    

    # Creates the "good suffix" skip table
    def create_skip_gs(self) -> None:
        """
        This function creates the "good suffix" skip table. (It is used in the preprocessing step of the Boyer-Moore search algorithm.)

        Arguments:
            None

        Returns:
            None

        """
        # Create the good suffix "skip" table
        # TODO(msuzgun): Has an error!
        self.skip_gs = [0] * self.pattern_length
        # skip_gs[i] denotes the number of cells to the right we need to skip if the current character is the i-th character of the pattern
        
        # First, we compute the length of the longest suffix of pattern [i:] that matches a prefix of the pattern
        for i in range(self.pattern_length - 1):
            self.skip_gs[i] = self.aux_get_suffix_prefix_length(i)

        # Set the default skip value to the pattern length
        self.skip_gs[-1] = 1

        # Second, we compute the length of the longest suffix of the pattern that matches a substring of the pattern that ends at the index j
        for j in range(self.pattern_length - 2):
            k = (self.pattern_length - 1) - self.aux_get_matching_substring_length(j)
            if self.skip_gs[k] == 0:
                self.skip_gs[k] = self.pattern_length - 1 - j


    # Creates the "bad character" skip table
    def create_skip_bc(self) -> None:
        """
        This function creates the "bad character" skip table. (It is used in the preprocessing step of the Boyer-Moore search algorithm.)

        Arguments:
            None

        Returns:
            None
        """
        # Create the bad character "skip" table
        self.last_occurence = {}

        # last_occurence[c] denotes the index of the last occurence of the character c in the pattern
        for j in range(self.pattern_length - 1):
            self.last_occurence[self.pattern[j]] = j

        # Set the default skip value to the pattern length
        self.last_occurence.setdefault(None, self.pattern_length)

    
    # Searches for the pattern in the text using the Boyer-Moore algorithm
    def search(self,
        pattern: str,
        text: str,
    ) -> int:
        """
        This function searches for the pattern in the text using the Boyer-Moore algorithm.

        Arguments:
            pattern (str): The pattern to search for.
            text (str): The text to search in.

        Returns:
            int: The index of the pattern in the text (or -1 if the pattern is not found)

        Raises:
            AssertionError: If the text or the pattern is not a string.
        """
        # Check both the pattern and the text
        assert isinstance(pattern, str), 'The pattern must be a string.'
        assert isinstance(text, str), 'The text must be a string.'

        # Set the attributes
        self.pattern = pattern

        # Length of the pattern
        self.pattern_length = len(self.pattern)

        # Preprocess the pattern by creating the skip tables for the bad character and good suffix rules, respectively.
        self.create_skip_bc()
        self.create_skip_gs()


        # Loop over the text
        i = 0
        while i <= len(text) - self.pattern_length:
            # Loop over the pattern
            j = self.pattern_length - 1
            while j >= 0 and text[i + j] == self.pattern[j]:
                j -= 1
            # Check if the pattern is found
            if j < 0:
                return i
            # Update i
            i += max(j - self.last_occurence.get(text[i + j], self.pattern_length), 1)

        # Return -1 if the pattern is not found
        return -1



# import time
# import random
# import string

# for _ in range(100):

#     # Create a Boyer-Moore object
#     # Generate a random string of length (1, 10)
#     pattern = ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(random.randint(1, 5)))
#     # pattern = 'AA'
#     bm = BoyerMooreSearch(pattern)

#     # Create a Naive object
#     naive = NaiveSearch()

#     # Create a Rabin-Karp object
#     rk = RabinKarpSearch() #pattern)

#     # Create a KMP object
#     kmp = KMPSearch()

#     # Search for the pattern in the text
#     # Generate a random string of length (11, 100)
#     text = ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(random.randint(11, 10000)))
#     # text = 'AAAAAAAAA'

#     # # Measure time
#     # start = time.time()
#     # print(bm.search(text))
#     # end = time.time()
#     # # print(end - start)

#     # start = time.time()
#     # rk_result = rk.search(text)
#     # end = time.time()
#     # # print(end - start)

#     # start = time.time()
#     # naive_result = naive.search(text)
#     # end = time.time()
#     # print(end - start)
#     naive_result = naive.search(pattern=pattern, text=text)
#     rk_result = rk.search(pattern=pattern, text=text)
#     kmp_result = kmp.search(pattern=pattern, text=text)
#     bm_result = bm.search(text)

#     print(naive_result, rk_result, bm_result, kmp_result)

#     # Check if the results are the same
#     assert naive_result == rk_result == bm_result == kmp_result, 'The results are not the same!'
