import numpy as np


def cen2lin(val=None, start=0):
    return val - start


def cut_v_1d(arr=None, niv_cut=0):
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    return arr[niv - niv_cut:niv + niv_cut]


def cut_iv_2d(arr=None, niv_cut=0):
    assert np.size(np.shape(arr)) == 2, 'Array is not 2D.'
    niv = arr.shape[-1] // 2
    return arr[niv - niv_cut:niv + niv_cut, niv - niv_cut:niv + niv_cut]


def cut_v(arr=None, niv_cut=0, axes=(0,)):
    axes = np.flip(np.sort(axes))
    tmp = arr
    for axis in axes:
        tmp = np.apply_along_axis(cut_v_1d, axis=axis, arr=tmp, niv_cut=niv_cut)
    return tmp


def wplus2wfull(mat=None, axis=-1):
    ''' iw dimension must either be first or last.'''
    if (axis == 0):
        mat_full = np.concatenate((np.conj(np.flip(mat[1:, ...], axis=axis)), mat), axis = axis)
    elif (axis == -1):
        mat_full = np.concatenate((np.conj(np.flip(mat[..., 1:], axis=axis)), mat), axis = axis)
    else:
        raise ValueError('axis mus be either first (0) or last (-1)')
    return mat_full


# ----------------------------------------------- FERMIONIC ------------------------------------------------------------
def vn(n=10):
    return np.arange(-n, n)


def v(beta=1.0, n=10):
    return (vn(n=n) * 2 + 1) * np.pi / beta


def iv(beta=1.0, n=10):
    return v(beta=beta, n=n) * 1j


# ------------------------------------------------ BOSONIC -------------------------------------------------------------
def wn(n=10):
    return np.arange(-n, n + 1)


def wn_plus(n=10):
    return np.arange(0, n + 1)


def w(beta=1.0, n=10):
    return (vn(n=n) * 2) * np.pi / beta


def iw(beta=1.0, n=10):
    return v(beta=beta, n=n) * 1j


def wnfind(niw=None, n=None):
    return n - niw


def wn_outer(n_core=None, n_outer=None):
    return np.concatenate((np.arange(-n_outer, -n_core), np.arange(n_core + 1, n_outer + 1)))


def wn_outer_plus(n_core=None, n_outer=None):
    return np.arange(n_core + 1, n_outer + 1)


if __name__ == '__main__':
    niv = 10
    vn = vn(niv)
    niv_cut = 5
    vn_cut_1 = cut_v_1d(vn, niv_cut=niv_cut)
    mat_vn = np.array([vn, vn]).T
    vn_cut = cut_v(vn, niv_cut=niv_cut, axes=(0,))
    mat_vn_cut = cut_v(mat_vn, niv_cut=niv_cut, axes=(0,))

    n = 4
    mat = np.arange(0, n ** 2).reshape(n, n)

    mat_cut = cut_v(mat, niv_cut=1, axes=(0, 1))

    niw = 20
    wn_a = wn(niw)
    n = wnfind(niw, n=20)

    niw_core = 5
    niw_urange = 10
    iw_core = wn(n=niw_core)
    iw_outer = wn_outer(n_core=niw_core, n_outer=niw_urange)
    print(f'{iw_core}')
    print(f'{iw_outer}')
