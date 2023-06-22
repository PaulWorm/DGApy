import numpy as np
import sys, os

sys.path.append('../src')
sys.path.append('./src')
import BrillouinZone as bz


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
    test_irrk_mapping()
    test_irrk_mapping_even()

    # # %%

    #
    # nk = (3, 3, 1)
    # _k_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    # hr = hamr.one_band_2d_t_tp_tpp(1, -0.2, 0.1)
    # ek = hamk.ek_3d(_k_grid.grid, hr)
    #
    # ek_irrk = _k_grid.map_fbz2irrk(ek)
    # # print(ek_irrk - ek[_k_grid.fbz2irrk].flatten())
    # ek_b2fbz = _k_grid.map_irrk2fbz(ek_irrk)
    # count_fbz = _k_grid.map_irrk2fbz(_k_grid.irrk_count)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(ek[:, :, 0] - ek_b2fbz[:, :, 0], cmap='RdBu')
    # plt.colorbar()
    # plt.show()
    # print(np.sum(ek[:, :, 0] - ek_b2fbz[:, :, 0]))
    #
    # # plt.figure()
    # # plt.imshow(ek[:,:,0],cmap='RdBu')
    # # plt.colorbar()
    # # plt.show()
    # #
    # # plt.figure()
    # # plt.imshow(ek_b2fbz[:,:,0],cmap='RdBu')
    # # plt.colorbar()
    # # plt.show()
    #
    # plt.figure()
    # plt.pcolormesh(_k_grid.kx, _k_grid.ky, count_fbz[:, :, 0], cmap='RdBu')
    # plt.colorbar()
    # plt.show()
    #
    # plt.figure()
    # plt.pcolormesh(_k_grid.kx, _k_grid.ky, _k_grid.fbz2irrk[:, :, 0], cmap='RdBu')
    # plt.colorbar()
    # plt.show()
    #
    # # # assert np.equal(ek_irrk,ek_b2fbz).all()
    # # #%%
    # # tmp = np.zeros((2,1))
    # # # print()
    #
    # # nk = (3, 3, 1)
    # # mat = np.reshape(np.arange(0, np.prod(nk)), nk)
    # # bz.inv_sym(mat, 0)
    # # bz.inv_sym(mat, 1)
    # # bz.x_y_sym(mat)
    # # plt.figure()
    # # plt.imshow(mat, cmap='RdBu')
    # # plt.show()
