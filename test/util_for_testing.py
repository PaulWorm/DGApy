def test(function,input,solution,n):
    print('-------------')
    ans = function(*input)
    if (ans == solution):
        print(f'Passed test {n}')
    else:
        print(f'Answer: {ans}')
        print(f'Solution: {solution}')
    print('-------------')