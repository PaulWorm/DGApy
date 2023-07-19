import numpy as np
import sys

import dga.brillouin_zone as bz


def example_mat_1():
    nk = (3, 3, 1)
    mat = np.reshape(np.arange(0, np.prod(nk)), nk)
    return mat


def test_inv_sym_x():
    nk = (3, 3, 1)
    mat = np.reshape(np.arange(0, np.prod(nk)), nk)
    bz.inv_sym(mat, 0)
    sol = np.reshape([0, 1, 2,
                      3, 4, 5,
                      3, 4, 5], nk)
    assert np.equal(mat.flatten(), sol.flatten()).all()


def test_inv_sym_y():
    nk = (3, 3, 1)
    mat = np.reshape(np.arange(0, np.prod(nk)), nk)
    bz.inv_sym(mat, 1)
    sol = np.reshape([0, 1, 1,
                      3, 4, 4,
                      6, 7, 7], nk)
    assert np.equal(mat.flatten(), sol.flatten()).all()


def test_x_y_sym():
    nk = (3, 3, 1)
    mat = np.reshape(np.arange(0, np.prod(nk)), nk)
    bz.x_y_sym(mat)
    sol = np.reshape([0, 1, 2,
                      1, 4, 5,
                      2, 5, 8], nk)
    assert np.equal(mat.flatten(), sol.flatten()).all()


def test_k_grid():
    nk = (3, 3, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    sol = np.reshape([0, 1, 1,
                      1, 4, 4,
                      1, 4, 4], nk)
    assert np.equal(k_grid.fbz2irrk.flatten(), sol.flatten()).all()


def test_k_grid_even():
    nk = (4, 4, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    sol = np.reshape([0, 1, 2, 1,
                      1, 5, 6, 5,
                      2, 6, 10, 6,
                      1, 5, 6, 5], nk)
    print('----------------')
    if(np.equal(k_grid.fbz2irrk.flatten(), sol.flatten()).all()):
        print('Passed test_k_grid_even.')
    else:
        print(f'Answer = {k_grid.fbz2irrk.flatten()}')
        print(f'Solution = {sol.flatten()}')
    print('----------------')
    # assert np.equal(k_grid.fbz2irrk.flatten(), sol.flatten()).all()


def test_irrk_mapping():
    nk = (3, 3, 1)
    k_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    mat = np.reshape([0, 1, 1,
                      1, 4, 4,
                      1, 4, 4], nk)
    mat_b2fbz = k_grid.map_irrk2fbz(k_grid.map_fbz2irrk(mat))
    assert np.allclose(mat, mat_b2fbz)


def test_irrk_mapping_even():
    nk = (4, 4, 1)
    k_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    mat = np.reshape([0, 1, 2, 1,
                      1, 3, 4, 3,
                      2, 4, 5, 4,
                      1, 3, 4, 3], nk)
    mat_b2fbz = k_grid.map_irrk2fbz(k_grid.map_fbz2irrk(mat))
    assert np.allclose(mat, mat_b2fbz)


if __name__ == '__main__':
    test_inv_sym_x()
    test_inv_sym_y()
    test_x_y_sym()
    test_k_grid()
    test_k_grid_even()
    test_irrk_mapping()
    test_irrk_mapping_even()
