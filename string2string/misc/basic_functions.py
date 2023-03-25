from typing import List, Union

# Take the Cartesian product of two lists of strings (or lists of lists of strings)
def cartesian_product(
    lst1: Union[List[str], List[List[str]]],
    lst2: Union[List[str], List[List[str]]],
    boolListOfList: bool = False,
    list_of_list_separator: str = " ## ",
) -> Union[List[str], List[List[str]]]:
    """
    This function returns the Cartesian product of two lists of strings (or lists of lists of strings).

    Arguments:
        lst1: The first list of strings (or lists of lists of strings).
        lst2: The second list of strings (or lists of lists of strings).
        boolListOfList: A boolean flag indicating whether the output should be a list of strings (or lists of lists of strings) (default: False).

    Returns:
        The Cartesian product of the two lists of strings (or lists of lists of strings).
    """
    if lst1 == []:
        return lst2
    elif lst2 == []:
        return lst1
    return [
        s1 + ("" if not (boolListOfList) else list_of_list_separator) + s2
        for s1 in lst1
        for s2 in lst2
    ]