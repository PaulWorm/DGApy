import numpy as np


def cen2lin(val=None, start=0):
    return val - start

def cut_v_1d(arr=None, niv_cut=0):
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    return arr[niv-niv_cut:niv+niv_cut]

def cut_v(arr=None,niv_cut=0,axes=(0,)):
    axes = np.flip(np.sort(axes))
    tmp = arr
    for axis in axes:
        tmp = np.apply_along_axis(cut_v_1d, axis=axis, arr=tmp, niv_cut=niv_cut)
    return tmp


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


def w(beta=1.0, n=10):
    return (vn(n=n) * 2) * np.pi / beta


def iw(beta=1.0, n=10):
    return v(beta=beta, n=n) * 1j


if __name__=='__main__':
    niv = 10
    vn = vn(niv)
    niv_cut = 5
    vn_cut_1 = cut_v_1d(vn, niv_cut=niv_cut)
    mat_vn = np.array([vn,vn]).T
    vn_cut = cut_v(vn,niv_cut=niv_cut,axes=(0,))
    mat_vn_cut = cut_v(mat_vn, niv_cut=niv_cut, axes=(0,))

    n = 4
    mat = np.arange(0,n**2).reshape(n,n)

    mat_cut = cut_v(mat, niv_cut=1, axes=(0,1))
