import numpy as np
import matplotlib.pyplot as plt
import dga.eliashberg_equation as eq


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
    assert np.allclose(eliashberg_solver.lam, np.flip(diag_v)[:n_eig], atol=eps)
    print('-------------')
    if (np.allclose(eliashberg_solver.lam, np.flip(diag_v)[:n_eig], atol=eps)):
        print(f'Passed test on basic matrix')
    else:
        print(f'Answer: {eliashberg_solver.lam}')
        print(f'Solution: {np.flip(diag_v)[:n_eig]}')
    print('-------------')

def test_on_random_matrix():
    # Create random state to be reproducible:
    np.random.seed(42)
    nkx = 2
    nky = 2
    nkz = 1
    nk_tot = nkx * nky * nkz
    niv = 10
    matrix = np.random.rand(nkx,nky,nkz,niv,niv).astype(complex)
    gk = np.ones((nkx, nky, nkz, niv), dtype=complex)
    gk = np.random.rand(nkx,nky,nkz,niv).astype(complex)
    matrix_full = build_matrix_kkp(matrix)
    mat_gg = matrix_full * np.abs(gk[None, None, None, None, :, :, :, :]) ** 2
    mat_gg = mat_gg.reshape(nk_tot*niv,nk_tot*niv)
    eig_val, eig_vec = np.linalg.eig(mat_gg)
    gap0 = np.random.rand(nkx, nky, nkz, niv).astype(complex)
    norm = 1
    n_eig = 2
    eps = 1e-8
    eliashberg_solver = eq.EliashberPowerIteration(gamma=matrix, gk=gk, gap0=gap0, norm=norm, eps=eps, max_count=10000,
                                                   shift_mat=0, n_eig=n_eig)
    eig_sort = eig_val[np.argsort(np.abs(eig_val))]

    assert np.allclose(eliashberg_solver.lam, eig_sort[::-1][:n_eig].real, atol=eps)
    print('-------------')
    if (np.allclose(eliashberg_solver.lam, eig_sort[::-1][:n_eig].real, atol=eps)):
        print(f'Passed test on basic matrix')
    else:
        print(f'Answer: {eliashberg_solver.lam}')
        print(f'Solution: {eig_sort[::-1][:n_eig].real}')
    print('-------------')


if __name__ == '__main__':
    test_on_basic_matrix()
    test_on_random_matrix()
    #
    pass
