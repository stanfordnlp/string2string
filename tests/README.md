# Unit Tests

The purpose of this directory is to provide a set of functional tests that can be used to verify the accuracy of the algorithms and functions utilized within the library.

To run these tests, one can utilize [pytest](https://docs.pytest.org/en/7.3.x/getting-started.html#run-multiple-tests) to execute all files with names of the form `test_*.py` or `*_test.py` located in the current directory and its subdirectories.

To install `pytest`, please run the following command in your terminal:

```bash
pip install -U pytest
```

Executing the `pytest` command in the current directory should generate an output similar to the following:

```python
>>> pytest
============================================================================= test session starts =============================================================================
platform darwin -- Python 3.9.12, pytest-7.2.2, pluggy-1.0.0
rootdir: /Users/machine/string2string
collected 15 items                                                                                                                                                            

test_alignment.py .......                                                                                                                                               [ 46%]
test_distance.py .....                                                                                                                                                  [ 80%]
test_rogue.py .                                                                                                                                                         [ 86%]
test_sacrebleu.py .                                                                                                                                                     [ 93%]
test_search.py .                                                                                                                                                        [100%]

============================================================================= 15 passed in 6.05s ==============================================================================
```
