import numpy as np


def is_increasing(L):
    return all(x <= y for x, y in zip(L, L[1:]))


def is_decreasing(L):
    return all(x >= y for x, y in zip(L, L[1:]))


def is_monotonic(L):
    return is_increasing(L) or is_decreasing(L)


def print_passed_test(testname):
    print(f'Passed test {testname}')

def test_statement(statement, testname):
    if (statement):
        print_passed_test(testname)
    else:
        raise ValueError(f'Test {testname} did not pass.')

def test_function(function, solution, name, *args, **kwargs):
    '''
        Test a generic function on input (args, kwargs) and compare to a given known solution.
    '''
    ans = function(*args, **kwargs)
    if (np.allclose(ans, solution)):
        print(f'Passed test {name}')
    else:
        print(f'Answer: {ans.flatten()}')
        print(f'Solution: {solution.flatten()}')
        raise ValueError(f'Test {name} did not pass.')


def test_in_place_operations(function, mat, solution, name, *args, **kwargs):
    '''
        Test a generic function on input (args, kwargs) and compare to a given known solution.
    '''
    function(*args, **kwargs)  # in place operation
    if (np.allclose(mat, solution)):
        print(f'Passed test {name}')
    else:
        print(f'Answer: {mat.flatten()}')
        print(f'Solution: {solution.flatten()}')
        raise ValueError(f'Test {name} did not pass.')


def test_array(arr1, arr2, name, rtol=1e-5, atol=1e-5):
    arr1 = np.atleast_1d(arr1)
    arr2 = np.atleast_1d(arr2)
    if (np.allclose(arr1.flatten(), arr2.flatten(), rtol=rtol, atol=atol)):
        print(f'Passed test {name}')
    else:
        print(f'Answer = {arr1.flatten()}')
        print(f'Solution = {arr2.flatten()}')
        raise ValueError(f'Test {name} did not pass.')
