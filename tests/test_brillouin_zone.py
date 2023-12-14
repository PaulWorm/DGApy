import numpy as np

from dga import brillouin_zone as bz
from test_util import util_for_testing as t_util


def example_mat_1():
    nk = (3, 3, 1)
    mat = np.reshape(np.arange(0, np.prod(nk)), nk)
    return mat


def get_example_mat(nk):
    return np.reshape(np.arange(0, np.prod(nk)), nk)


def test_inv_sym():
    '''
        inv_sym changes in-place, hence this testing does not work.
    '''
    nk = (3, 3, 1)
    mat = get_example_mat(nk)
    sol = np.reshape([0, 1, 2,
                      3, 4, 5,
                      3, 4, 5], nk)
    t_util.test_in_place_operations(bz.inv_sym, mat, sol, 'test_inv_sym_x', mat, 0)
    sol = np.reshape([0, 1, 1,
                      3, 4, 4,
                      6, 7, 7], nk)
    mat = get_example_mat(nk)
    t_util.test_in_place_operations(bz.inv_sym, mat, sol, 'test_inv_sym_y', mat, 1)

    nk = (4, 4, 1)
    mat = get_example_mat(nk)
    sol = np.reshape([0, 1, 2, 3,
                      4, 5, 6, 7,
                      8, 9, 10, 11,
                      4, 5, 6, 7], nk)
    t_util.test_in_place_operations(bz.inv_sym, mat, sol, 'test_inv_sym_x_even', mat, 0)
    sol = np.reshape([0, 1, 2, 1,
                      4, 5, 6, 5,
                      8, 9, 10, 9,
                      12, 13, 14, 13], nk)
    mat = get_example_mat(nk)
    t_util.test_in_place_operations(bz.inv_sym, mat, sol, 'test_inv_sym_y_even', mat, 1)


def test_x_y_sym():
    nk = (3, 3, 1)
    sol = np.reshape([0, 1, 2,
                      1, 4, 5,
                      2, 5, 8], nk)
    mat = get_example_mat(nk)
    t_util.test_in_place_operations(bz.x_y_sym, mat, sol, 'test_x_y_sym_odd', mat)

    nk = (4, 4, 1)
    sol = np.reshape([0, 1, 2, 3,
                      1, 5, 6, 7,
                      2, 6, 10, 11,
                      3, 7, 11, 15], nk)
    mat = get_example_mat(nk)
    t_util.test_in_place_operations(bz.x_y_sym, mat, sol, 'test_x_y_sym_even', mat)


def test_x_y_inv():
    nk = (3, 3, 1)
    sol = np.reshape([0, 1, 2,
                      3, 4, 5,
                      6, 5, 4], nk)
    mat = get_example_mat(nk)
    t_util.test_in_place_operations(bz.x_y_inv, mat, sol, 'test_x_y_inv_odd', mat)

    nk = (4, 4, 1)
    sol = np.reshape([0, 1, 2, 3,
                      4, 5, 6, 7,
                      8, 9, 10, 11,
                      12, 7, 6, 5], nk)
    mat = get_example_mat(nk)
    t_util.test_in_place_operations(bz.x_y_inv, mat, sol, 'test_x_y_inv_even', mat)


def test_k_grid():
    nk = (3, 3, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    sol = np.reshape([0, 1, 1,
                      1, 4, 4,
                      1, 4, 4], nk)
    t_util.test_array(k_grid.fbz2irrk, sol, 'test_k_grid_odd')
    sol_irrk = k_grid.map_fbz2irrk(sol)
    mat_b2fbz = k_grid.map_irrk2fbz(k_grid.map_fbz2irrk(sol.flatten(), shape='list'), shape='list')

    t_util.test_array(mat_b2fbz, sol.flatten(), 'test_irrk_mapping_odd_list')
    t_util.test_array(k_grid.map_irrk2fbz(sol_irrk), sol, 'test_irrk_mapping_odd')
    assert k_grid.k_mean(sol_irrk) == k_grid.k_mean(sol,shape='fbz-mesh')

    nk = (4, 4, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    sol = np.reshape([0, 1, 2, 1,
                      1, 5, 6, 5,
                      2, 6, 10, 6,
                      1, 5, 6, 5], nk)
    t_util.test_array(k_grid.fbz2irrk, sol, 'test_k_grid_odd')
    sol_irrk = k_grid.map_fbz2irrk(sol)
    mat_b2fbz = k_grid.map_irrk2fbz(k_grid.map_fbz2irrk(sol.flatten(), shape='list'), shape='list')
    t_util.test_array(mat_b2fbz, sol.flatten(), 'test_irrk_mapping_odd_list')
    t_util.test_array(k_grid.map_irrk2fbz(sol_irrk), sol, 'test_irrk_mapping_odd')
    assert k_grid.k_mean(sol_irrk) == k_grid.k_mean(sol,shape='fbz-mesh')


def test_list_mesh_mapping():
    nk = (9, 9, 1)
    k_grid = bz.KGrid(nk)
    np.random.seed(1)
    mat = np.random.rand(*nk)
    mat_list = k_grid.map_fbz_mesh2list(mat)
    mat_b2mesh = k_grid.map_fbz_list2mesh(mat_list)
    t_util.test_array(mat, mat_b2mesh, 'test_list_mesh_mapping_odd')
    mat = np.random.rand(np.prod(nk))
    mat_b2list = k_grid.map_fbz_mesh2list(k_grid.map_fbz_list2mesh(mat))
    t_util.test_array(mat, mat_b2list, 'test_mesh_list_mapping_odd')


def test_list_mesh_mapping_even():
    nk = (10, 10, 1)
    k_grid = bz.KGrid(nk)
    np.random.seed(1)
    mat = np.random.rand(*nk)
    mat_list = k_grid.map_fbz_mesh2list(mat)
    mat_b2mesh = k_grid.map_fbz_list2mesh(mat_list)
    t_util.test_array(mat, mat_b2mesh, 'test_list_mesh_mapping_even')
    mat = np.random.rand(np.prod(nk))
    mat_b2list = k_grid.map_fbz_mesh2list(k_grid.map_fbz_list2mesh(mat))
    t_util.test_array(mat, mat_b2list, 'test_mesh_list_mapping_even')


def test_list_mesh_mapping_non_uniform():
    nk = (10, 8, 1)
    k_grid = bz.KGrid(nk)
    np.random.seed(1)
    mat = np.random.rand(*nk)
    mat_list = k_grid.map_fbz_mesh2list(mat)
    mat_b2mesh = k_grid.map_fbz_list2mesh(mat_list)
    t_util.test_array(mat, mat_b2mesh, 'test_list_mesh_mapping_non_uniform')
    mat = np.random.rand(np.prod(nk))
    mat_b2list = k_grid.map_fbz_mesh2list(k_grid.map_fbz_list2mesh(mat))
    t_util.test_array(mat, mat_b2list, 'test_mesh_list_mapping_non_uniform')


if __name__ == '__main__':
    test_inv_sym()
    test_x_y_sym()
    test_x_y_inv()
    test_k_grid()
    test_list_mesh_mapping()
    test_list_mesh_mapping_even()
    test_list_mesh_mapping_non_uniform()
