import numpy as np

def test(function,input,solution,n):
    print('-------------')
    ans = function(*input)
    if (ans == solution):
        print(f'Passed test {n}')
    else:
        print(f'Answer: {ans}')
        print(f'Solution: {solution}')
    print('-------------')


def test_array(arr1,arr2,n,rtol=1e-5):
    arr1 = np.atleast_1d(arr1)
    arr2 = np.atleast_1d(arr2)
    print('----------------')
    if(np.allclose(arr1.flatten(), arr2.flatten(),rtol=rtol)):
        print(f'Passed test {n}.')
    else:
        print(f'Answer = {arr1.flatten()}')
        print(f'Solution = {arr2.flatten()}')
        raise ValueError(f'Test {n} did not pass.')
    print('----------------')

