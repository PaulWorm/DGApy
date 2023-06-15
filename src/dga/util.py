import numpy as np


def mem(mat):
    ''' returns the memory consumption of mat'''
    return np.size(mat) * mat.itemsize * 1e-9