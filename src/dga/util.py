'''
    Provides utility functions for the project
'''

import os
import numpy as np

def mem(mat):
    ''' returns the memory consumption of mat'''
    return np.size(mat) * mat.itemsize * 1e-9

def uniquify(path=None):
    '''

    path: path to be checked for uniqueness
    return: updated unique path
    '''
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + '_' + str(counter) + extension
        counter += 1

    return path + '/'
