import numpy as np
import sys,os
sys.path.append('../src')
sys.path.append('./src')
import BrillouinZone as bz


def example_mat_1():
    nk = (3,3,1)
    mat = np.reshape(np.arange(0,np.prod(nk)),nk)
    return mat


def test_inv_sym_x():
    nk = (3, 3, 1)
    mat = np.reshape(np.arange(0, np.prod(nk)), nk)
    bz.inv_sym(mat,0)
    sol = np.reshape([0,1,2,3,4,5,0,1,2],nk)
    assert np.equal(mat.flatten(),sol.flatten()).all()

def test_inv_sym_y():
    nk = (3, 3, 1)
    mat = np.reshape(np.arange(0, np.prod(nk)), nk)
    bz.inv_sym(mat,1)
    sol = np.reshape([0,1,0,3,4,3,6,7,6],nk)
    assert np.equal(mat.flatten(),sol.flatten()).all()

def test_x_y_sym():
    nk = (3, 3, 1)
    mat = np.reshape(np.arange(0, np.prod(nk)), nk)
    bz.x_y_sym(mat)
    sol = np.reshape([0,1,2,1,4,5,2,5,8],nk)
    assert np.equal(mat.flatten(),sol.flatten()).all()

def test_k_grid():
    nk = (3, 3, 1)
    k_grid = bz.KGrid(nk=nk,symmetries=bz.two_dimensional_square_symmetries())
    sol = np.reshape([0,1,0,1,4,1,0,1,0],nk)
    assert np.equal(k_grid.fbz2irrk.flatten(),sol.flatten()).all()



if __name__ == '__main__':
    test_inv_sym_x()
    test_inv_sym_y()
    test_x_y_sym()
    test_k_grid()
