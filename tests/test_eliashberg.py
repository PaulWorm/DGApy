import numpy as np
import matplotlib.pyplot as plt

from dga import eliashberg_equation as eq
from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import wannier
from dga import two_point as twop
from test_util import util_for_testing as t_util
from test_util import test_data as td


def build_matrix_kkp(matrix):
    nkx, nky, nkz, niv = matrix.shape[:-1]
    matrix_full = np.zeros((nkx, nky, nkz, niv, nkx, nky, nkz, niv), dtype=matrix.dtype)

    def get_kmkp(k, kp, nk):
        kmkp = k - kp
        if (kmkp < 0):
            kmkp += nk
        return kmkp

    for ikx in range(nkx):
        for iky in range(nky):
            for ikz in range(nkz):
                for ikpx in range(nkx):
                    for ikpy in range(nky):
                        for ikpz in range(nkz):
                            kmkpx = get_kmkp(ikx, ikpx, nkx)
                            kmkpy = get_kmkp(iky, ikpy, nky)
                            kmkpz = get_kmkp(ikz, ikpz, nkz)
                            matrix_full[ikx, iky, ikz, :, ikpx, ikpy, ikpz, :] = matrix[kmkpx, kmkpy, kmkpz, :, :]
    return matrix_full


def test_on_basic_matrix():
    nkx = 2
    nky = 2
    nkz = 1
    niv = 10
    matrix = np.zeros((nkx, nky, nkz, niv, niv), dtype=complex)
    diag_v = np.arange(0, niv)
    matrix[0, 0, 0, :, :] = np.diag(diag_v)
    gk = np.ones((nkx, nky, nkz, niv), dtype=complex)
    gap0 = np.random.rand(nkx, nky, nkz, niv).astype(complex)
    norm = 1
    n_eig = 3
    eps = 1e-8
    eliashberg_solver = eq.EliashberPowerIteration(gamma=matrix, gk=gk, gap0=gap0, norm=norm, eps=eps, max_count=10000,
                                                   shift_mat=0, n_eig=n_eig)
    t_util.test_array(eliashberg_solver.lam, np.flip(diag_v)[:n_eig], 'basic_matrix', rtol=eps, atol=eps)


def test_on_random_matrix():
    # Create random state to be reproducible:
    np.random.seed(42)
    nkx = 2
    nky = 2
    nkz = 1
    nk_tot = nkx * nky * nkz
    niv = 10
    matrix = np.random.rand(nkx, nky, nkz, niv, niv).astype(complex)
    gk = np.random.rand(nkx, nky, nkz, niv).astype(complex)
    matrix_full = build_matrix_kkp(matrix)
    mat_gg = matrix_full * np.abs(gk[None, None, None, None, :, :, :, :]) ** 2
    mat_gg = mat_gg.reshape(nk_tot * niv, nk_tot * niv)
    eig_val, eig_vec = np.linalg.eig(mat_gg)
    gap0 = np.random.rand(nkx, nky, nkz, niv).astype(complex)
    norm = 1
    n_eig = 2
    eps = 1e-8
    eliashberg_solver = eq.EliashberPowerIteration(gamma=matrix, gk=gk, gap0=gap0, norm=norm, eps=eps, max_count=10000,
                                                   shift_mat=0, n_eig=n_eig)
    eig_sort = eig_val[np.argsort(np.abs(eig_val))]
    t_util.test_array(eliashberg_solver.lam, eig_sort[::-1][:n_eig].real, 'random_matrix', rtol=eps, atol=eps)


def test_eliashberg(input_type='minimal'):
    # load the input:
    ddict, hr = td.load_eliashberg_input(input_type)
    # dmft_input, _ = td.load_testdataset(input_type)
    if(input_type == 'quasi_1d'):
        rtol, atol = 1e-2, 1e-2
        rtol_s, atol_s = 1e-1, 1e-1
    else:
        rtol, atol = 1e-3, 1e-3
        rtol_s, atol_s = 1e-6, 1e-6
    niv_pp = ddict['f_sing_pp'].shape[-1] // 2

    # Create the DGA Green's function:
    siwk_obj = twop.SelfEnergy(ddict['siwk_dga'], ddict['beta'])
    # Create the hr object:
    nk = np.shape(siwk_obj.sigma_core)[:-1]
    sym = ddict['sym']
    k_grid = bz.KGrid(nk, sym)
    ek = hr.get_ek(k_grid)
    # siwk_dmft = twop.SelfEnergy(dmft_input['siw'], dmft_input['beta'])
    # siwk_obj = twop.create_dga_siwk_with_dmft_as_asympt(ddict['siwk_dga'], siwk_dmft, 100)
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'])

    # Cut the Green's function to the right size:
    gk_dga = mf.cut_v(giwk_obj.core, niv_pp, (-1,))
    print(f'Giwk-DGA mu: {giwk_obj.mu}')
    print(f'Giwk-DGA n: {giwk_obj.n}')

    # Define the norm for the power iteration:
    norm = k_grid.nk_tot * ddict['beta']

    # Create starting gap function:
    gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type='random',
                            v_type='random',
                            k_grid=k_grid.grid)

    def test_eigenvalues(channel):
        gamma = - ddict[f'f_{channel}_pp']
        gamma = eq.symmetrize_gamma(gamma, channel)
        n_eig = 1
        powiter = eq.EliashberPowerIteration(gamma=gamma, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
                                             n_eig=n_eig)

        # Obtain the eigenvalues and eigenvectors with numpy:
        matrix_full = build_matrix_kkp(gamma)
        gk_dga_mk = np.flip(gk_dga, axis=(-1))
        bz.apply_symmetries(gk_dga_mk, ['x-inv', 'y-inv'])
        t_util.test_array(np.abs(gk_dga) ** 2, gk_dga * gk_dga_mk, f'{input_type}_|gk|^2_consistent_with_gkgmk')

        mat_gg = matrix_full * np.abs(gk_dga[None, None, None, None, :, :, :, :]) ** 2
        mat_gg = mat_gg.reshape(k_grid.nk_tot * niv_pp * 2, k_grid.nk_tot * niv_pp * 2) / norm
        eig_val, eig_vec = np.linalg.eig(mat_gg)
        sort_ind = np.argsort(eig_val.real)
        eig_val_sort = eig_val[sort_ind].real
        print(f'Eigenvalues: {eig_val_sort[-n_eig:][::-1]}')
        print(f'Eigenvalues pow-iter: {powiter.lam}')
        if powiter.lam_s != 0:  # check only if lam_s is not zero
            t_util.test_array([powiter.lam_s.real, ], eig_val_sort[0], f'{input_type}_eig_val_s_{channel}',
                              rtol=rtol_s, atol=atol_s)
        t_util.test_array(powiter.lam.real, eig_val_sort[-n_eig:][::-1], f'{input_type}_eig_val_{channel}',
                          rtol=rtol, atol=atol)

    test_eigenvalues('sing')
    test_eigenvalues('trip')


def main():
    test_on_basic_matrix()
    test_on_random_matrix()
    input_types = ['minimal', 'high_temperature', 'quasi_1d']
    for input_type in input_types:
        test_eliashberg(input_type)


if __name__ == '__main__':
    main()
