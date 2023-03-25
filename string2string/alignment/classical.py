"""
    This module contains the classes for the alignment algorithms.

    Classes:
        StringAlignment: Parent class for all alignment algorithms.
        NeedlemanWunsch: Needleman-Wunsch algorithm.
        SmithWaterman: Smith-Waterman algorithm.
        Hirschberg: Hirschberg algorithm.
"""

from typing import List, Union, Tuple, Optional
import numpy as np
from string2string.misc.basic_functions import cartesian_product

# Parent class for all alignment algorithms
class StringAlignment:
    """
        This class is the parent class for all alignment algorithms implemented in this module.
    """
    # Initialize the class.
    def __init__(self,
        match_weight: int = 1.,
        mismatch_weight: int = -1.,
        gap_weight: int = -1,
        gap_char: str = "-",
        match_dict: dict = None,
        ) -> None:
        r"""
        This function initializes the StringAlignment class.

        Arguments:
            match_weight (int): The weight for a match (default: 1).
            mismatch_weight (int): The weight for a mismatch (default: -1).
            gap_weight (int): The weight for a gap (default: -1).
            gap_char (str): The character for a gap (default: "-").
            match_dict (dict): The match dictionary (default: None).

        Returns:
            None

        .. note::

            The match_dict represents a dictionary of the match weights for each pair of characters. For example, if the match_dict is {"A": {"A": 1, "T": -1}, "T": {"A": -1, "T": 1}}, then the match weight for "A" and "A" is 1, the match weight for "A" and "T" is -1, the match weight for "T" and "A" is -1, and the match weight for "T" and "T" is 1.
            The match_dict is particularly useful when we wish to align (or match) non-identical characters. For example, if we wish to align "A" and "T", we can set the match_dict to {"A": {"T": 1}}. This will ensure that the match weight for "A" and "T" is 1, and the match weight for "A" and "A" and "T" and "T" is 0.
        """
        # Set the weights.
        self.match_weight = match_weight
        self.mismatch_weight = mismatch_weight
        self.gap_weight = gap_weight
        self.gap_char = gap_char
        self.match_dict = match_dict


    def bool_match(self,
        c1: Union[str, List[str]],
        c2: Union[str, List[str]],
    ) -> bool:
        """
        The function returns whether two characters match, according to the match dictionary (if it exists).

        Arguments:
            c1 (str or list of str): The first character or string.
            c2 (str or list of str): The second character or string.

        Returns:
            Whether the two characters match (True or False)
        """

        # If there is no match dictionary, return whether the characters are the same.
        if self.match_dict is None:
            return c1 == c2
        # Otherwise, return whether the characters match according to the match dictionary.
        else:
            if c1 in self.match_dict and c2 in self.match_dict[c1]:
                return self.match_dict[c1][c2] >= 0
            else:
                return c1 == c2


    # Get the match weight.
    def get_match_weight(self, 
        c1: Union[str, List[str]],
        c2: Union[str, List[str]],
    ) -> float:
        """
        This function returns the match weight of two characters.

        Arguments:
            c1 (str or list of str): The first character or string.
            c2 (str or list of str): The second character or string.

        Returns:
            The match weight of the two characters or strings.
        """

        # If there is no match dictionary, return the match weight if the characters are the same, and the mismatch weight otherwise.
        if self.match_dict is None:
            if c1 == c2:
                return self.match_weight
            return self.mismatch_weight
        # Otherwise, return the match weight according to the match dictionary.
        else:
            if c1 in self.match_dict and c2 in self.match_dict[c1]:
                return self.match_dict[c1][c2]
            else:
                if c1 == c2:
                    return self.match_weight
                return self.mismatch_weight


    # Get the gap weight.
    def get_gap_weight(self,
        c: Union[str, List[str]],
    ) -> float:
        """
        This function returns the gap weight of a character or string.

        Arguments:
            c (str or list of str): The character or string.

        Returns:
            The gap weight of the character or string.
        """

        # If there is no match dictionary, return the gap weight.
        if self.match_dict is None:
            return self.gap_weight
        # Otherwise, return the gap weight according to the match dictionary.
        else:
            if c in self.match_dict and self.gap_char in self.match_dict[c]:
                return self.match_dict[c][self.gap_char]
            else:
                return self.gap_weight


    # Get the score of a character pair.
    def get_score(self,
        c1: Union[str, List[str]],
        c2: Union[str, List[str]],
    ) -> float:
        """
        This function returns the score of a character or string pair.

        Arguments:
            c1 (str or list of str): The first character or string.
            c2 (str or list of str): The second character or string.

        Returns:
            The score of the character or string pair.
        """
        # If the characters are the same, return the match weight.
        if c1 == c2:
            return self.match_weight
        # If one of the characters is a gap, return the gap weight.
        elif c1 == self.gap_char or c2 == self.gap_char:
            return self.gap_weight
        # Otherwise, return the mismatch weight.
        else:
            return self.mismatch_weight

    
    # Get the alignment score of two strings (or list of strings). 
    # (This is the sum of the scores of the characters.)
    def get_alignment_score(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
    ) -> float:
        """
        This function returns the alignment score of two strings (or list of strings).

        Arguments:
            str1 (str or list of str): The first string (or list of strings).
            str2 (str or list of str): The second string (or list of strings).

        Returns:
            The alignment score of the two strings (or list of strings).
        """
        # Get the alignment score by summing the scores of the characters.
        score = 0.
        for c1, c2 in zip(str1, str2):
            score += self.get_score(c1, c2)
        return score

    
    # Add gaps to the shorter string.
    def add_space_to_shorter(self,
        str1: str,
        str2: str,
    ) -> Tuple[str, str]:
        """
        This function adds gaps to the shorter string to make the two strings the same length.

        Arguments:
            str1 (str): The first string.
            str2 (str): The second string.

        Returns:
            The two strings with the same length.
        """
        # Get the maximum length of the two strings.
        max_len = max(len(str1), len(str2))

        # Pad the shorter string with gaps.
        if len(str1) < max_len:
            str1 = str1 + ' ' * (max_len - len(str1))
        elif len(str2) < max_len:
            str2 = str2 + ' ' * (max_len - len(str2))

        # Return the padded strings.
        return str1, str2

    
    # Print the alignment.
    def print_alignment(self,
        str1: str,
        str2: str,
    ) -> None:
        """
        This function prints the alignment of two strings.

        Arguments:
            str1 (str): The first string.
            str2 (str): The second string.

        Returns:
            None.
        """
        # Print the alignment.
        # print("Score:", self.get_alignment_score(str1, str2))
        print(str1)
        print(str2)


    
    def get_alignment_strings_and_indices(self,
        str1: str,
        str2: str,
        separator: str = ' | ',
    ) -> Tuple[Tuple[List[int], List[int]], List[str], List[str]]:
        """
        This function returns the indices of the aligned characters, and the two strings separated by the separator.

        Arguments:
            str1 (str): The first string, separated by the separator.
            str2 (str): The second string, separated by the separator.
            separator (str): The separator.

        Returns:
            The indices of the aligned characters, and the two strings separated by the separator.

        """
        sym1 = str1.split(separator)
        sym2 = str2.split(separator)

        alignment_indices = []
        # Get the indices of the aligned characters.
        for i in range(len(sym1)):
            if self.bool_match(sym1[i], sym2[i]): #sym1[i] == sym2[i]:
                alignment_indices.append((i, i))
        
        return alignment_indices, sym1, sym2



# Needleman-Wunsch algorithm class
class NeedlemanWunsch(StringAlignment):
    # Initialize the class.
    def __init__(self,
        match_weight: float = 1.,
        mismatch_weight: float = -1.,
        gap_weight: float = -1.,
        gap_char: str = "-",
        match_dict: dict = None,
        ) -> None:
        r"""
        This function initializes the Needleman-Wunsch algorithm, which is used to get the global alignment of sequences (e.g., strings or lists of strings) such as DNA sequences.
        
        The algorithm is described in the following paper: [Needleman1970]_

        Arguments:
            match_weight (float): The weight of a match (default: 1.).
            mismatch_weight (float): The weight of a mismatch (default: -1.).
            gap_weight (float): The weight of a gap (default: -1.).
            match_dict (dict): The match dictionary (default: None).
            gap_char (str): The gap character (default: "-").
            
        .. [Needleman1970] Needleman, S.B. and Wunsch, C.D., 1970. A General Method Applicable to the Search for Similarities in the Amino Acid Sequence of Two Proteins. Journal of Molecular Biology, 48(3), pp.443-453.
        """
        # Initialize using the parent class.
        super().__init__(
            match_weight=match_weight,
            mismatch_weight=mismatch_weight,
            gap_weight=gap_weight,
            match_dict=match_dict,
            gap_char=gap_char,
        )


    # The auxilary backtrack function.
    def backtrack(self,
        score_matrix: np.ndarray,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
    ) -> Tuple[Union[str, List[str]], Union[str, List[str]]]:
        r"""
        This function is an auxilary function, used by the get_alignment() function, that backtracks the score matrix to get the aligned strings.

        Arguments:
            score_matrix (np.ndarray): The score matrix.
            str1: The first string (or list of strings).
            str2: The second string (or list of strings).
        
        Returns:
            The aligned strings (or list of strings). The aligned strings are padded with spaces to make them the same length.

        .. note::
            * The score matrix is assumed to be a 2D numpy array.
            * There might be multiple optimal alignments. This function returns one of the optimal alignments.
            * The backtracking step has a time complexity of :math:`O(m + n)`, where :math:`n` and :math:`m` are the lengths of the strings str1 and str2, respectively.
        """

        # Lengths of strings str1 and str2, respectively.
        len1 = len(str1)
        len2 = len(str2)

        # Initialize the aligned strings.
        aligned_str1 = ""
        aligned_str2 = ""

        # Initialize the current position.
        i = len1
        j = len2

        # Backtrack until the current position is (0, 0).
        while i > 0 and j > 0:
            # If the current position is the result of a match/mismatch, add the characters to the aligned strings and move to the diagonal.
            if score_matrix[i, j] == score_matrix[i - 1, j - 1] + self.get_score(str1[i - 1], str2[j - 1]):
                insert_str1, insert_str2 = self.add_space_to_shorter(str1[i - 1], str2[j - 1])
                i -= 1
                j -= 1
            # If the current position is the result of a gap in str1, add a gap to str1 and the character to str2 and move to the left.
            elif score_matrix[i, j] == score_matrix[i, j - 1] + self.get_gap_weight(str2[j - 1]):
                insert_str1, insert_str2 = self.add_space_to_shorter(self.gap_char, str2[j - 1])
                j -= 1
            # If the current position is the result of a gap in str2, add a gap to str2 and the character to str1 and move up.
            elif score_matrix[i, j] == score_matrix[i - 1, j] + self.get_gap_weight(str1[i - 1]):
                insert_str1, insert_str2 = self.add_space_to_shorter(str1[i - 1], self.gap_char)
                i -= 1
            
            # Add the characters to the aligned strings.
            aligned_str1 = insert_str1 + ' | ' + aligned_str1
            aligned_str2 = insert_str2 + ' | ' + aligned_str2

        # If there are still characters in str1, add them to the aligned strings.
        while i > 0:
            insert_str1, insert_str2 = self.add_space_to_shorter(str1[i - 1], self.gap_char)
            aligned_str1 = insert_str1 + ' | ' + aligned_str1
            aligned_str2 = insert_str2 + ' | ' + aligned_str2
            i -= 1

        # If there are still characters in str2, add them to the aligned strings.
        while j > 0:
            insert_str1, insert_str2 = self.add_space_to_shorter(self.gap_char, str2[j - 1])
            aligned_str1 = insert_str1 + ' | ' + aligned_str1
            aligned_str2 = insert_str2 + ' | ' + aligned_str2
            j -= 1

        # Remove the last ' | ' from the aligned strings.
        aligned_str1 = aligned_str1[:-3]
        aligned_str2 = aligned_str2[:-3]

        # Return the aligned strings.
        return aligned_str1, aligned_str2
    

    # Get the alignment of two strings (or list of strings).
    def get_alignment(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
        return_score_matrix: bool = False,
    ) -> Tuple[Union[str, List[str]], Union[str, List[str]], Optional[np.ndarray]]:
        r"""
        This is the main function in the NeedlemanWunsch class that gets the alignment of two strings (or list of strings) by using the Needleman-Wunsch algorithm.

        Arguments:
            str1: The first string (or list of strings).
            str2: The second string (or list of strings).
            return_score_matrix (bool): Whether to return the score matrix. (Default: False)

        Returns:
            The aligned strings (or list of strings). The aligned strings are padded with spaces to make them the same length. If return_score_matrix is True, the score matrix is also returned.

        .. note::
            * There might be multiple optimal alignments. This function returns one of the optimal alignments.
            * The time complexity of this function is :math:`O(nm)`, where :math:`n` and :math:`m` are the lengths of the strings str1 and str2, respectively.
            * The space complexity of this function is :math:`O(nm)`.
            * The " | " character is used to separate the elements in the aligned strings.
        """

        # Lengths of strings str1 and str2, respectively.
        len1 = len(str1)
        len2 = len(str2)

        # Initialize the score matrix.
        score_matrix = np.zeros((len1 + 1, len2 + 1))

        # Initialize the first row and column of the score matrix.
        for i in range(1, len1 + 1):
            score_matrix[i, 0] = score_matrix[i - 1, 0] + self.get_gap_weight(str1[i - 1])
        for j in range(1, len2 + 1):
            score_matrix[0, j] = score_matrix[0, j - 1] + self.get_gap_weight(str2[j - 1])

        # Dynamic programming solution (Needleman-Wunsch algorithm):
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                # Get the scores of the three possible paths.
                match_score = score_matrix[i - 1, j - 1] + self.get_match_weight(str1[i - 1], str2[j - 1])
                delete_score = score_matrix[i - 1, j] + self.get_gap_weight(str1[i - 1])
                insert_score = score_matrix[i, j - 1] + self.get_gap_weight(str2[j - 1])
                
                # Get the maximum score.
                max_score = max(match_score, delete_score, insert_score)
                
                # Fill the score matrix.
                score_matrix[i, j] = max_score

        # Get the alignment.
        aligned_str1, aligned_str2 = self.backtrack(score_matrix, str1, str2)

        # Return the alignment and the score matrix.
        if return_score_matrix:
            return aligned_str1, aligned_str2, score_matrix
        return aligned_str1, aligned_str2


# Hirschberg algorithm (linear space algorithm).
class Hirschberg(NeedlemanWunsch):
    def __init__(self,
        match_weight: Union[int, float] = 2,
        mismatch_weight: Union[int, float] = -1,
        gap_weight: Union[int, float] = -2,
        gap_char: str = '-',
        match_dict: dict = None,
    ) -> None:
        r"""
        This function initializes the parameters of the Hirschberg algorithm [H1975]_, a space-efficient solution to the global alignment problem. It inherits from the NeedlemanWunsch class.

        Arguments:
            match_weight (int or float): The weight of a match (default: 2).
            mismatch_weight (int or float): The weight of a mismatch (default: -1).
            gap_weight (int or float): The weight of a gap (default: -2).
            gap_char (str): The character used to represent a gap (default: '-').
            match_dict (dict): The dictionary that maps the characters to their match weights (default: None).

        .. note::
            * The default values are the same as the ones used in the Needleman-Wunsch algorithm.
            * The time complexity of Hirschberg's algorithm is :math:`O(nm)`, where :math:`n` and :math:`m` are the lengths of the strings str1 and str2, respectively.
            * The space complexity of Hirschberg's algorithm is, on the other hand, :math:`O(min(n, m))`.
            * We benefited from the following resources to implement this class: [K2015]_, [W2012]_, [M2010]_, [K2002]_.

        .. [H1975] Hirschberg, Daniel S. "A linear space algorithm for computing maximal common subsequences." Communications of the ACM 18.6 (1975): 341-343.
        .. [K2015] Kellis, Manolis. Computational Biology: Genomes, Networks, Evolution (MIT Course 6.047/6.878) — https://ocw.mit.edu/ans7870/6/6.047/f15/MIT6_047F15_Compiled.pdf (Accessed on 02-16-2023) (Section 2.5.8; Linear Space Alignment, pg. 38-39).
        .. [W2012] Wayne, Kevin. Lecture Slides for Algorithm Design - Dynamic Programming II (https://www.cs.princeton.edu/~wayne/kleinberg-tardos/pdf/06DynamicProgrammingII.pdf) (Accessed on 02-16-2023).
        .. [M2010] Moura, Lucia. Algorithms in Bioinformatics: Lectures 3-5 - Sequence Similarity. Fall 2010. University of Ottawa. (https://www.site.uottawa.ca/~lucia/courses/2010/comp5511/lectures/03-05.pdf) (Accessed on 02-16-2023).
        .. [K2002] Kingsford, Carl. Lecture 7: Dynamic Programming. 2002. Carnegie Mellon University. (https://www.cs.cmu.edu/~ckingsf/class/02-714/Lec07-linspace.pdf) (Accessed on 02-16-2023).
        """

        # Initialize the Needleman-Wunsch algorithm using the super() function.
        super().__init__(
            match_weight=match_weight,
            mismatch_weight=mismatch_weight,
            gap_weight=gap_weight,
            match_dict=match_dict,
            gap_char=gap_char,
        )


    # Get the alignment of two strings (or list of strings).
    def get_alignment(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
    ) -> Tuple[Union[str, List[str]], Union[str, List[str]]]:
        r"""
        This function gets the alignment of two strings (or list of strings) by using the Hirschberg algorithm.

        Arguments:
            str1: The first string (or list of strings).
            str2: The second string (or list of strings).

        Returns:
            The aligned strings as a tuple of two strings (or list of strings).

        .. note::
            * As a notable improvement of the Needleman-Wunsch algorithm, Hirschberg's algorithm combines both divide-and-conquer and dynamic programming principles. This algorithm provides a space-efficient solution, with a time complexity of :math:`O(nm)`, where :math:`n` and :math:`m` are the lengths of the strings str1 and str2, respectively. Its space complexity, however, is :math:`O(min(n, m))`.
            * To further improve the algorithm, one may limit the number of insertions and deletions in the alignment. This strategy can notably reduce the time complexity from :math:`O(mn)` to :math:`O((m + n) * k)`, where k is the maximum number of insertions or deletions allowed. By constraining the alignment around the diagonal in the score matrix with (2 * k + 1) cells, the new version of the algorithm can be called the k-banded Hirschberg algorithm. If k is arbitrarily small, this modification can lead to a significant improvement in the time complexity.
            * The k-banded Hirschberg algorithm, with its time complexity of :math:`O((m + n) * k)`, is a powerful strategy that balances space and time requirements in sequence alignment.
        """
            
        # Lengths of strings str1 and str2, respectively.
        len1 = len(str1)
        len2 = len(str2)

        # Check if the length of str1 is less than or equal to the length of str2.
        if len1 >= len2:
            # Get the alignment.
            aligned_str1, aligned_str2 = self.get_alignment_helper(str1, str2)
        else:
            # Get the alignment.
            aligned_str2, aligned_str1 = self.get_alignment_helper(str2, str1)

        # Remove the trailing " | " from the aligned strings (if any).
        aligned_str1 = aligned_str1.strip(" | ")
        aligned_str2 = aligned_str2.strip(" | ")

        # Replace the " |  | " with " | " in the aligned strings (if any).
        aligned_str1 = aligned_str1.replace('|  |', '|')
        aligned_str2 = aligned_str2.replace('|  |', '|')

        # Return the alignment.
        return aligned_str1, aligned_str2

    
    # Get the alignment of two strings (or list of strings).
    def get_alignment_helper(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
    ) -> Tuple[Union[str, List[str]], Union[str, List[str]]]:
        """
        This is a helper function that is called by the get_alignment() function. This function gets the alignment of two strings (or list of strings) by using the Hirschberg algorithm.
        
        Arguments:
            str1: The first string (or list of strings).
            str2: The second string (or list of strings).

        Returns:
            The aligned strings as a tuple of two strings (or list of strings).

        .. note::
            * We assume that the length of str1 is greater than or equal to the length of str2.
        """

        # Lengths of strings str1 and str2, respectively.
        len1 = len(str1)
        len2 = len(str2)

        # Initialize the aligned strings.
        aligned_str1 = ""
        aligned_str2 = ""

        # Check if the length of str1 is 0.
        if len1 == 0:
            # Add gap characters to the shorter string (i.e., str1).
            for j in range(1, len2+1):
                insert_str1, insert_str2 = self.add_space_to_shorter(self.gap_char, str2[j-1])
                aligned_str1 = aligned_str1 + ' | ' + insert_str1
                aligned_str2 = aligned_str2 + ' | ' + insert_str2
        elif len2 == 0:
            # Add gap characters to the shorter string (i.e., str2).
            for i in range(1, len1+1):
                insert_str1, insert_str2 = self.add_space_to_shorter(str1[i-1], self.gap_char)
                aligned_str1 = aligned_str1 + ' | ' + insert_str1
                aligned_str2 = aligned_str2 + ' | ' + insert_str2
        elif len1 == 1 or len2 == 1:
            # Get the alignment of two strings (or list of strings) by using the Needleman-Wunsch algorithm.
            aligned_str1, aligned_str2 = super().get_alignment(str1, str2)
        else:
            # Get the middle index of str1.
            mid1 = len1 // 2

            # Get the scores of the left and right substrings.
            score_row_left = self.nw_score(str1[:mid1], str2)
            # Score-Right = Reverse ( NW-Score( Reverse(Str1-Mid1), Reverse(Str2) ) ) 
            score_row_right = self.nw_score(str1[mid1:][::-1], str2[::-1])[::-1]

            # Get mid2 = arg max score_row_left + score_row_right
            mid2 = self.get_middle_index(score_row_left, score_row_right)

            # Get the alignment of the left and right substrings.
            aligned_str1_left, aligned_str2_left = self.get_alignment_helper(str1[:mid1], str2[:mid2])
            aligned_str1_right, aligned_str2_right = self.get_alignment_helper(str1[mid1:], str2[mid2:])
            
            # Combine the aligned strings.
            # Make sure to add ' | ' between the aligned strings only if the aligned strings are not empty.
            # This is to avoid adding ' | ' at the beginning and end of the aligned strings.
            if aligned_str1_left != "" and aligned_str1_right != "":
                aligned_str1 = aligned_str1_left + ' | ' + aligned_str1_right
            else:
                aligned_str1 = aligned_str1_left + aligned_str1_right
            if aligned_str2_left != "" and aligned_str2_right != "":
                aligned_str2 = aligned_str2_left + ' | ' + aligned_str2_right
            else:
                aligned_str2 = aligned_str2_left + aligned_str2_right

        # Return the aligned strings.
        return aligned_str1, aligned_str2


    # Return the last row of the score matrix.
    def nw_score(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
    ) -> List[float]:
        """
        This function returns the last row of the score matrix.

        Arguments:
            str1: The first string (or list of strings).
            str2: The second string (or list of strings).

        Returns:
            The last row of the score matrix.
        """

        # Lengths of strings str1 and str2, respectively.
        len1 = len(str1)
        len2 = len(str2)

        # Create a 2 x (len2 + 1) matrix.
        score_matrix = np.zeros((2, len2 + 1))

        # Initialize the first row of the score matrix.
        for j in range(1, len2 + 1):
            score_matrix[0, j] = score_matrix[0, j - 1] + self.get_gap_weight(str2[j - 1]) # insertion cost
        
        # Update the score matrix.
        for i in range(1, len1 + 1):
            score_matrix[1, 0] = score_matrix[0, 0] + self.get_gap_weight(str1[i - 1]) # deletion cost

            for j in range(1, len2 + 1):
                score_matrix[1, j] = max(
                    score_matrix[0, j - 1] + self.get_score(str1[i - 1], str2[j - 1]), # match/mismatch cost
                    score_matrix[0, j] + self.get_gap_weight(str1[i - 1]), # deletion cost
                    score_matrix[1, j - 1] + self.get_gap_weight(str2[j - 1]) # insertion cost
                )

            # Update the score matrix.
            score_matrix[0, :] = score_matrix[1, :]

        # Return the last row of the score matrix.
        return score_matrix[1, :]

    
    # Get the middle index of str2.
    def get_middle_index(self,
        score_left: List[float],
        score_right: List[float],
    ) -> int:
        """
        This function gets the middle index of str2.

        Arguments:
            score_left: The score of the left sublist.
            score_right: The score of the right sublist.

        Returns:
            The middle index of str2.
        """

        # Length of score_left.
        len_score_left = len(score_left)

        # Initialize the middle index.
        mid2 = 0

        # Initialize the maximum score with the possible minimum score.
        # Oh dear, initially I used 0 as the maximum score, but that was wrong. 
        # The maximum score can be negative, so we need to use the possible minimum score instead, which is -float('inf').
        max_score = -float('inf')

        # Get the middle index.
        for i in range(len_score_left):
            if score_left[i] + score_right[i] > max_score:
                mid2 = i
                max_score = score_left[i] + score_right[i]

        # Return the middle index.
        return mid2


# Smith-Waterman algorithm (local alignment).
class SmithWaterman(NeedlemanWunsch):
    def __init__(self,
        match_weight: Union[int, float] = 1,
        mismatch_weight: Union[int, float] = -1,
        gap_weight: Union[int, float] = -1,
        gap_char: str = '-',
        match_dict: dict = None,
    ) -> None:
        r"""
        This function initializes the class variables of the Smith-Waterman algorithm, used for local alignment of sequences (e.g., strings or lists of strings) such as DNA sequences.

        Arguments:
            match_weight (int or float): The weight of a match (default: 1).
            mismatch_weight (int or float): The weight of a mismatch (default: -1).
            gap_weight (int or float): The weight of a gap (default: -1).
            gap_char (str): The character used to represent a gap (default: '-').
            match_dict (dict): The dictionary that maps the characters to their match weights (default: None).
            
        .. note::
            * The default values are the same as the Needleman-Wunsch algorithm.
            * The Smith-Waterman algorithm can be thought of a variant of the Needleman-Wunsch algorithm, where the max function is replaced by the max function with 0 as the default value and the first row and column of the score matrix are initialized to 0.

            .. math::
                :nowrap:

                \begin{align}
                    \texttt{max score} &= \texttt{max}(\texttt{match score}, \texttt{delete score}, \texttt{insert score}, 0)
                \end{align}

            * We only need to override the get_alignment and backtrack functions of the NeedlemanWunsch class.
            * The space and time complexities of the Smith-Waterman algorithm are the same as the Needleman-Wunsch algorithm, that is, :math:`O(mn)` and :math:`O(mn)`, respectively, where :math:`n` and :math:`m` are the lengths of the two strings (or lists of strings).
        """

        # Initialize the SmithWaterman class using its parent class, NeedlemanWunsch.
        super().__init__(
            match_weight=match_weight,
            mismatch_weight=mismatch_weight,
            gap_weight=gap_weight,
            match_dict=match_dict,
            gap_char=gap_char,
        )

    
    # Backtrack the score matrix.
    # Override the backtrack function of the NeedlemanWunsch class.
    def backtrack(self,
        score_matrix: np.ndarray,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
    ) -> Tuple[Union[str, List[str]], Union[str, List[str]]]:
        """
        This function overrides the backtrack function of the NeedlemanWunsch class to get an optimal local alignment between two strings (or list of strings).

        Arguments:
            score_matrix (numpy.ndarray): The score matrix.
            str1 (str or list of str): The first string (or list of strings).
            str2 (str or list of str): The second string (or list of strings).

        Returns:
            The aligned substrings as a tuple of two strings (or list of strings).

        .. note::
            * The backtrack function used in this function is different from the backtrack function used in the Needleman-Wunsch algorithm. Here we start from the position with the highest score in the score matrix and trace back to the first position that has a score of zero. This is because the highest-scoring subsequence may not necessarily span the entire length of the sequences being aligned.
            * On the other hand, the backtrack function used in the Needleman-Wunsch algorithm traces back through the entire score matrix, starting from the bottom-right corner, to determine the optimal alignment path. This is because the algorithm seeks to find the global alignment of two sequences, which means aligning them from the beginning to the end.
        """

        # Initialize the aligned substrings.
        aligned_str1 = ""
        aligned_str2 = ""

        # Get the position with the maximum score in the score matrix.
        # TODO(msuzgun): See if there is a faster way to get the position with the maximum score in the score matrix.
        i, j = np.unravel_index(np.argmax(score_matrix, axis=None), score_matrix.shape)

        # Backtrack the score matrix.
        while score_matrix[i, j] != 0:
            # Get the scores of the three possible paths.
            match_score = score_matrix[i - 1, j - 1] + self.get_match_weight(str1[i - 1], str2[j - 1])
            delete_score = score_matrix[i - 1, j] + self.get_gap_weight(str1[i - 1])
            insert_score = score_matrix[i, j - 1] + self.get_gap_weight(str2[j - 1])

            # Get the maximum score.
            max_score = max(match_score, delete_score, insert_score)

            # Backtrack the score matrix.
            if max_score == match_score:
                insert_str1, insert_str2 = self.add_space_to_shorter(str1[i - 1], str2[j - 1])
                i -= 1
                j -= 1
            elif max_score == delete_score:
                insert_str1, insert_str2 = self.add_space_to_shorter(str1[i - 1], self.gap_char)
                i -= 1
            elif max_score == insert_score:
                insert_str1, insert_str2 = self.add_space_to_shorter(self.gap_char, str2[j - 1])
                j -= 1

            # Add the characters to the aligned strings.
            aligned_str1 = insert_str1 + ' | ' + aligned_str1
            aligned_str2 = insert_str2 + ' | ' + aligned_str2

        # Remove the last ' | '.
        aligned_str1 = aligned_str1[:-3]
        aligned_str2 = aligned_str2[:-3]

        # Return the aligned substrings.
        return aligned_str1, aligned_str2
    

    
    # Get the alignment of two strings (or list of strings).
    # Override the get_alignment function of the NeedlemanWunsch class.
    def get_alignment(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
        return_score_matrix: bool = False,
    ) -> Tuple[Union[str, List[str]], Union[str, List[str]]]:
        """
        This function overrides the get_alignment function of the NeedlemanWunsch class to get the alignment of two strings (or list of strings) by using the Smith-Waterman algorithm.

        Arguments:
            str1 (str or list of str): The first string (or list of strings).
            str2 (str or list of str): The second string (or list of strings).
            return_score_matrix (bool): Whether to return the score matrix (default: False)

        Returns:
            The aligned strings as a tuple of two strings (or list of strings). If return_score_matrix is True, the score matrix is also returned.

        .. note::
            * The Smith-Waterman algorithm is a dynamic programming algorithm that finds the optimal local alignment between two strings (or list of strings).
            * This function is similar to the get_alignment function in the NeedlemanWunsch class, with two differences. First, the first row and column of the score matrix are initialized to 0. Second, the max function used in the dynamic programming solution is replaced with a max function that defaults to 0, i.e., max_score = max(match_score, delete_score, insert_score, 0).
            * Despite these differences, the time and space complexity of the Smith-Waterman algorithm remain the same as that of the Needleman-Wunsch algorithm. It should be noted that most of the code in this function is identical to that in the get_alignment function of the NeedlemanWunsch class.
        """

         # Lengths of strings str1 and str2, respectively.
        len1 = len(str1)
        len2 = len(str2)

        # Initialize the score matrix.
        score_matrix = np.zeros((len1 + 1, len2 + 1))

        # Initialize the first row and column of the score matrix.
        # This time the first row and column are initialized to 0.

        # Dynamic programming solution (Needleman-Wunsch algorithm):
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                # Get the scores of the three possible paths.
                match_score = score_matrix[i - 1, j - 1] + self.get_match_weight(str1[i - 1], str2[j - 1])
                delete_score = score_matrix[i - 1, j] + self.get_gap_weight(str1[i - 1])
                insert_score = score_matrix[i, j - 1] + self.get_gap_weight(str2[j - 1])
                
                # Get the maximum score.
                # Note that this is the only difference between the Smith-Waterman algorithm and the Needleman-Wunsch algorithm.
                # The max function is replaced by the max function with 0 as the default value.
                max_score = max(match_score, delete_score, insert_score, 0.)
                
                # Fill the score matrix.
                score_matrix[i, j] = max_score

        # Get the alignment.
        aligned_str1, aligned_str2 = self.backtrack(score_matrix, str1, str2)

        # Return the alignment and the score matrix.
        if return_score_matrix:
            return aligned_str1, aligned_str2, score_matrix
        return aligned_str1, aligned_str2


# Dynamic time warping (DTW) class.
class DTW:
    def __init__(self) -> None:
        r"""
        This function initializes the Dynamic time warping (DTW) class.
        """
        pass
    
    # Get the alignment indices of two sequences (or list of sequences) by using the DTW algorithm. 
    def get_alignment_path(self,
        sequence1: Union[str, List[str], int, List[int], float, List[float], np.ndarray],
        sequence2: Union[str, List[str], int, List[int], float, List[float], np.ndarray],
        distance = 'absolute_difference',
        p_value: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        This function gets the alignment indices of two sequences (or list of sequences) by using the DTW algorithm.

        Arguments:
            sequence1: The first sequence.
            sequence2: The second sequence.
            distance (str): The distance function to be used (currently only 'absolute_difference' and 'square_difference' are supported).

        Returns:
            The path of the alignment as a list of tuples of two integers.

        Raises:
            TypeError: If the input sequences are not of the same type.
            ValueError: If the distance function is not supported.


        .. note::
            * The DTW algorithm is a dynamic programming algorithm that finds the optimal alignment between two sequences (or list of sequences).
            * The time complexity of the DTW algorithm is :math:`O(nm)`, where :math:`n` and :math:`m` are the lengths of the two sequences, respectively.
        """
    
        # First check if both sequences are of the same type.
        if type(sequence1) != type(sequence2):
            raise TypeError("Both sequences must be of the same type.")

        # Check if the distance function is supported.
        if distance not in ['absolute_difference', 'square_difference']:
            raise ValueError("The distance function must be either 'absolute_difference' or 'square_difference'.")

        # If the sequences are strings or lists of strings, convert them to lists of integers (ASCII codes), in np.ndarray format.
        if type(sequence1) == str or (type(sequence1) == list and type(sequence1[0]) == str):
            sequence1 = np.array([ord(char) for char in sequence1])
            sequence2 = np.array([ord(char) for char in sequence2])

        # Get the lengths of the sequences.
        len1 = len(sequence1)
        len2 = len(sequence2)

        # Initialize the DTW distance matrix with infinity values.
        distance_matrix = np.full((len1 + 1, len2 + 1), np.inf)

        # Initialize the first row and column of the DTW distance matrix with zero.
        distance_matrix[0, 0] = 0.

        # Fill the DTW distance matrix.
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                # Get the distance between the two elements.
                if distance == 'absolute_difference':
                    distance = abs(sequence1[i - 1] - sequence2[j - 1])
                else:
                    # distance == 'square_difference'
                    distance = (sequence1[i - 1] - sequence2[j - 1]) ** 2

                # Fill the DTW distance matrix.
                distance_matrix[i, j] = distance + min(
                        distance_matrix[i - 1, j], 
                        distance_matrix[i, j - 1],
                        distance_matrix[i - 1, j - 1]
                    )

        # Initialize the alignment.
        alignment = []

        # Get the alignment.
        i = len1
        j = len2
        while i > 0 or j > 0:
            alignment.append((i - 1, j - 1))
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                if distance_matrix[i - 1, j] < distance_matrix[i, j - 1] and distance_matrix[i - 1, j] < distance_matrix[i - 1, j - 1]:
                    i -= 1
                elif distance_matrix[i, j - 1] < distance_matrix[i - 1, j - 1]:
                    j -= 1
                else:
                    i -= 1
                    j -= 1

        # Reverse the alignment.
        alignment.reverse()

        # Return the alignment.
        return alignment
    

# Longest common subsequence (LCSubsequenceuence) class
class LongestCommonSubsequence(StringAlignment):
    # Initialize the class
    def __init__(self, 
        list_of_list_separator: str = " ## ",
        ) -> None:
        r"""
        This function initializes the Longest Common Subsequence (LCSubsequenceuence) class, which inherits from the StringAlignment class.

        Longest common subsequence (LCSubsequence) of two strings is a subsequence of maximal length that appears in both of them. 

        The following recurrence relation can be used to solve the LCSubsequence problem:

        .. math::
            :nowrap:
        
            \begin{align}
                L[i,j] =
                \begin{cases}
                    0 &\text{ if } i=0 \text{ or } j=0\\
                    L[i-1,j-1]+1 &\text{ if } i,j>0 \text{ and } str1[i]=str2[j]\\
                    \max(L[i-1,j],L[i,j-1]) &\text{ if } i,j>0 \text{ and } str1[i]\neq str2[j]\\
                \end{cases}
            \end{align}

        where :math:`L[i,j]` denotes the length of the LCSubsequence of the prefixes str1[0:i] and str2[0:j]. The solution to the problem is then given by :math:`L[n,m]`, assuming that str1 and str2 have lengths n and m, respectively.
            
        A dynamic programming solution exists for this problem with a quadratic (i.e., :math:`\mathcal{O}(nm)`) space and time complexity.

        If the vocabulary is fixed, LCSubsequence admits a "Four-Russians speedup," which reduces its overall time complexity to subquadratic :math:`\mathcal{O}(n^2/\log n)`, but this algorithm is not yet implemented in this package.

        Arguments:
            list_of_list_separator (str): Separator to use when the inputs are lists of strings.
        """
        # Initialize the StringAlignment class.
        super().__init__(match_weight=1, mismatch_weight=0, gap_weight=0)

        # Set the list of list separator.
        self.list_of_list_separator = list_of_list_separator


    # Compute the longest common subsequence between two strings
    def compute(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
        returnCandidates: bool = False,
        boolListOfList: bool = False,
    ) -> Tuple[float, Union[List[str], List[List[str]]]]:
        """
        This function computes the longest common subsequence between two strings (or lists of strings).

        Arguments:
            str1 (str or list of str): The first string (or list of strings) to compare.
            str2 (str or list of str): The second string (or list of strings).
            returnCandidates (bool): Whether to return the candidates for the longest common subsequence (default: False).
            boolListOfList (bool): Whether the inputs are lists of strings (default: False).

        Returns:
            * If returnCandidates is False, then the length of the longest common subsequence between the two strings.
            * If returnCandidates is True, then the set of all candidates for the longest common subsequence is also returned.

        .. note::
           * Similar to that of the Levenshtein edit distance problem, the dynamic programming solution for the longest common subsequence problem can be further optimized by using the last row of the two-dimensional array L to compute the length of the LCSubsequence. The key idea behind this optimization is that the last row of the array L only depends on the values of the previous row. Therefore, we can store only two rows of the array L at a time and compute the LCSubsequence using these two rows.
           * This optimization reduces the space complexity of the algorithm to :math:`\mathcal{O}(m)`, where :math:`m` is the length of the shorter input string. This optimization is particularly useful when one of the input strings is much shorter than the other, as it can significantly reduce the amount of memory required to solve the problem.
        """
        # Lengths of strings str1 and str2, respectively.
        n = len(str1)
        m = len(str2)

        # Initialize the distance matrix.
        dist = np.zeros((n + 1, m + 1))

        # Dynamic programming solution to the longest common subsequence.
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # if str1[i-1] == str2[j-1]: # This is the original code. changed: 2023-03-19, 10:05 PM
                if self.bool_match(str1[i-1], str2[j-1]):
                    dist[i, j] = dist[i-1, j-1] + 1
                else:
                    dist[i, j] = max(dist[i-1, j], dist[i, j-1])

        # TODO(msuzgun): At the moment, the backtrack function is not optimized and pretty slow. It should be optimized! 
        def backtrack(i: int, j: int) -> Union[List[str], List[List[str]]]:
            """
            This function, which is called recursively and inside the compute function, is used to backtrack the distance matrix to compute the longest common subsequence.

            Arguments:
                i (int): The row index of the distance matrix.
                j (int): The column index of the distance matrix.

            Returns:
                The set of longest common subsequences between the two strings (or lists of strings).
            """
            # If the row or column index is 0, then the longest common subsequence is empty.
            if i == 0 or j == 0:
                # return [''] if boolListOfList else [] // This is the original code.
                return []

            # If the characters at the current row and column of the distance matrix are equal, then the current character is part of the longest common subsequence.
            # if str1[i-1] == str2[j-1]: # This is the original code. changed: 2023-03-19, 10:05 PM
            if self.bool_match(str1[i-1], str2[j-1]):
                # insert_elt = str1[i-1] if boolListOfList else str1[i-1] // This is the original code.
                insert_elt = [str1[i-1]] if boolListOfList else str1[i-1]
                candidates = list(
                    set(
                        cartesian_product(
                            backtrack(i-1, j-1),
                            insert_elt,
                            boolListOfList=boolListOfList,
                            list_of_list_separator=self.list_of_list_separator,
                        )
                    ) 
                )
                return candidates

            # If the characters at the current row and column of the distance matrix are not equal, then the current character is not part of the longest common subsequence.
            candidates = []
            if dist[i, j-1] >= dist[i-1, j]:
                candidates = backtrack(i, j-1)
            if dist[i-1, j] >= dist[i, j-1]:
                candidates += backtrack(i-1, j)
            return list(set(candidates))

        # Compute the longest common subsequence.
        candidates = None
        if returnCandidates:
            candidates = backtrack(n, m)
            if boolListOfList:
                candidates = [
                    elt.split(self.list_of_list_separator) for elt in candidates
                ]
        return dist[n, m], candidates



# Longest common substring (LCSubstring) class
class LongestCommonSubstring(LongestCommonSubsequence):
    # Initialize the class
    def __init__(self, 
        list_of_list_separator: str = " ## ",
        ) -> None:
        r"""
        This function initializes the LongestCommonSubstring (LCSubstring) class.

        Longest Common Substring (LCSubstring) of two strings is the longest substring that appears in both of them.

        The following recurrence relation can be used to solve the LCSubstring problem:
        
        .. math::
            :nowrap:

            \begin{align}
                L[i,j] =
                \begin{cases}
                    0 &\text{ if } i=0 \text{ or } j=0\\
                    L[i-1,j-1]+1 &\text{ if } i,j>0 \text{ and } str1[i]=str2[j]\\
                    0 &\text{ if } i,j>0 \text{ and } str1[i]\neq str2[j]\\
                \end{cases}
            \end{align}

        where :math:`L[i,j]` denotes the length of the LCSubstring that ends at indices i and j in str1 and str2, respectively. The solution to the problem is then given by the maximum value of :math:`L[i,j]`, assuming that str1 and str2 have lengths n and m, respectively.

        A dynamic programming solution exists for this problem with a quadratic (i.e., :math:`\mathcal{O}(nm)`) space and time complexity.

        Arguments:
            list_of_list_separator (str): Separator to use when the inputs are lists of strings.

        Returns:
            None
        """
        # Separator to use when the inputs are lists of strings.
        super().__init__(list_of_list_separator=list_of_list_separator)


    # Compute the longest common substring between two strings
    def compute(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
        returnCandidates: bool = False,
        boolListOfList: bool = False,
    ) -> Tuple[float, Union[List[str], List[List[str]]]]:
        """
        This function computes the longest common substring between two strings (or lists of strings).

        Arguments:
            str1 (str or list of str): The first string (or list of strings).
            str2 (str or list of str): The second string (or list of strings).
            returnCandidates (bool): A boolean flag indicating whether to return the longest common substring as a list of lists.
            boolListOfList (bool): A boolean flag indicating whether to return the longest common substring as a list of lists.

        Returns:
            If returnCandidates is False, then the length of the longest common substring between the two strings. If returnCandidates is True, then the set of longest common substrings between the two strings (or lists of strings) is also returned.


        .. note::
            * There exists a linear-time solution to LCSubstring problem that uses generalized suffix trees.
            * As with the longest common subsequence problem, the longest common substring is not unique. It is possible to have multiple substrings with the same maximum length that appear in both strings.
            * Similar to the dynamic programming solution for the longest common subsequence problem, the last row of the matrix can also be used to optimize the computation of the longest common substring.
            * It's important to note that the longest common substring is different from the longest common subsequence. The longest common substring is a contiguous sequence of symbols that appears in both strings, while the longest common subsequence is a sequence of characters that may not be contiguous.
            * The longest common substring is a measure of similarity between two strings and is used in various fields, including computational biology, where it is used to compare DNA sequences. Similarly, it is also used in plagirism detection and other applications.
        """
        # Lengths of strings str1 and str2, respectively.
        n = len(str1)
        m = len(str2)

        # Initialize the distance matrix.
        dist = np.zeros((n + 1, m + 1), dtype=int)

        # Initialize the longest common substring length.
        longest_common_substring_length = 0

        # Initialize the longest common substring candidates.
        longest_common_substring_indices = []

        # Dynamic programming solution to the longest common substring.
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # if str1[i-1] == str2[j-1]: # # This is the original code. changed: 2023-03-19, 10:05 PM
                if self.bool_match(str1[i-1], str2[j-1]):
                    dist[i, j] = dist[i-1, j-1] + 1
                    if dist[i, j] > longest_common_substring_length:
                        longest_common_substring_length = dist[i, j]
                        longest_common_substring_indices = [i]
                        # candidates = [str1[i-longest_common_substring_length:i]]
                    elif dist[i, j] == longest_common_substring_length:
                        # candidates.append(str1[i-longest_common_substring_length:i])
                        longest_common_substring_indices.append(i)
                else:
                    dist[i, j] = 0

        # If returnCandidates is True, then additionally return the set of longest common substrings.
        if returnCandidates:
            longest_common_substring_candidates = [str1[i-longest_common_substring_length:i] for i in longest_common_substring_indices]
            if boolListOfList:
                # TODO(msuzgun): Double check this. Correct, but there might be a better way to do this.
                longest_common_substring_candidates = list(set(
                    [
                        f"{self.list_of_list_separator}".join(cand) for cand in longest_common_substring_candidates
                    ]
                ))
                longest_common_substring_candidates = [
                    cand.split(self.list_of_list_separator) for cand in longest_common_substring_candidates
                ]
                longest_common_substring_candidates = set(tuple(elt) for elt in longest_common_substring_candidates)
            else:
                longest_common_substring_candidates = list(set(longest_common_substring_candidates))
            return longest_common_substring_length, longest_common_substring_candidates
        return longest_common_substring_length, None