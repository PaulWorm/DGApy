import numpy as np


def v_vp_urange(mat,val,niv_urange=0):
    ''' append val to the urange of mat
        assume last two dimensions are v and vp
        niv_urange is the number of additional frequencies
    '''
    niv = mat.shape[-1]//2
    niv_full = niv + niv_urange
    mat_urange = np.ones(mat.shape[:-2]+(niv_full*2,niv_full*2),dtype=mat.dtype)*val
    mat_urange[...,niv_full-niv:niv_full+niv,niv_full-niv:niv_full+niv] = mat
    return mat_urange


def w_to_vmvp(mat):
    '''Transform w -> v-v' '''
    niw = np.shape(mat)[0]//2
    niv = niw//2
    mat_new = np.zeros((2*niv,2*niv)+np.shape(mat)[1:],dtype=mat.dtype)
    vp = vn(niv)
    for i,ivp in enumerate(vp):
        mat_new[:,i,...] = mat[niw-niv-ivp:niw+niv-ivp,...]
    return mat_new

def cen2lin(val=None, start=0):
    return val - start


def wn_cen2lin(wn=0, niw=None):
    return cen2lin(wn, -niw)


def wn_slices(mat=None, n_cut=None, wn=None):
    n = mat.shape[-1] // 2
    mat_grid = np.array([mat[...,n - n_cut - iwn:n + n_cut - iwn] for iwn in wn])
    return mat_grid

def wn_slices_gen(mat=None, n_cut=None, wn=None):
    n = mat.shape[-1] // 2
    mat_grid = np.array([mat[...,n - n_cut - iwn:n + n_cut - iwn] for iwn in wn])
    return mat_grid


def wn_slices_plus(mat=None, n_cut=None, wn=None):
    n = mat.shape[-1] // 2
    mat_grid = np.array([mat[...,n - n_cut + iwn:n + n_cut + iwn] for iwn in wn])
    return mat_grid

def wn_slices_shell(mat,n_shell,n_core=0,wn=None):
    n = mat.shape[-1] // 2
    mat_grid = np.array([mat[...,n+n_core-iwn:n+n_core+n_shell-iwn] for iwn in wn])
    return fermionic_full_nu_range(mat_grid)


def wn_slices_plus_cent(mat=None, n_cut=None, wn=None):
    n = mat.shape[-1] // 2
    # mat_grid = np.array([mat[n - n_cut + iwn:n + n_cut + iwn] for iwn in wn])
    mat_grid = np.array([mat[n - n_cut - (iwn-iwn//2-iwn%2):n + n_cut - (iwn-iwn//2-iwn%2)] for iwn in wn])
    return mat_grid


def cut_iv_fp(mat=None, niv_cut=10):
    ''' Cut fermionic frequencies of four-point object'''
    niv = mat.shape[-1] // 2
    assert (mat.shape[-1] == mat.shape[-2]), 'Last two dimensions of the array are not consistent'
    return mat[..., niv - niv_cut:niv + niv_cut, niv - niv_cut:niv + niv_cut]


def cut_v_1d(arr=None, niv_cut=0):
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    if (niv_cut == -1): niv_cut = niv
    return arr[niv - niv_cut:niv + niv_cut]


def cut_v_1d_pos(arr=None, niv_cut=0, niv_min=0):
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    if (niv_cut == -1): niv_cut = niv
    return arr[niv + niv_min:niv + niv_min + niv_cut]


def cut_v_1d_wn(arr=None, niw_cut=0):
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    return arr[niv - niw_cut:niv + niw_cut + 1]


def cut_iv_2d(arr=None, niv_cut=0):
    assert np.size(np.shape(arr)) == 2, 'Array is not 2D.'
    niv = arr.shape[-1] // 2
    if (niv_cut == -1): niv_cut = niv
    return arr[niv - niv_cut:niv + niv_cut, niv - niv_cut:niv + niv_cut]


def cut_v(arr=None, niv_cut=0, axes=(0,)):
    axes = np.flip(np.sort(axes))
    tmp = arr
    for axis in axes:
        tmp = np.apply_along_axis(cut_v_1d, axis=axis, arr=tmp, niv_cut=niv_cut)
    return tmp

def cut_w(arr=None, niw_cut=0,axes=(0,)):
    axes = np.flip(np.sort(axes))
    tmp = arr
    for axis in axes:
        tmp = np.apply_along_axis(cut_v_1d_wn, axis=axis, arr=tmp, niw_cut=niw_cut)
    return tmp


def cut_v_1d_iwn(arr=None, niv_cut=0, iwn=0):
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    return arr[niv - niv_cut - iwn:niv + niv_cut - iwn]


def wplus2wfull(mat=None, axis=-1):
    ''' iw dimension must either be first or last.'''
    if (axis == 0):
        mat_full = np.concatenate((np.conj(np.flip(mat[1:, ...], axis=axis)), mat), axis=axis)
    elif (axis == -1):
        mat_full = np.concatenate((np.conj(np.flip(mat[..., 1:], axis=axis)), mat), axis=axis)
    else:
        raise ValueError('axis mus be either first (0) or last (-1)')
    return mat_full


def vplus2vfull(mat=None, axis=-1):
    mat_full = np.concatenate((np.conj(np.flip(mat, axis=axis)), mat), axis=axis)
    return mat_full


def fermionic_full_nu_range(mat, axis=-1):
    '''Build full Fermionic object from positive frequencies only along axis.'''
    return np.concatenate((np.conj(np.flip(mat, axis)), mat), axis=axis)


def concatenate_core_asmypt(core, asympt, axis=-1):
    ''' Concatenate core and asympt arrays along axis. v has to be last axis.'''
    niv_asympt = np.shape(asympt)[axis] // 2
    return np.concatenate((asympt[..., :niv_asympt], core, asympt[..., niv_asympt:]), axis=axis)


# ----------------------------------------------- FERMIONIC ------------------------------------------------------------
def vn(n=10, shift=0):
    return np.arange(-n + shift, n + shift)


def vn_plus(n=10, n_min=0):
    return np.arange(n_min, n)


def v(beta=1.0, n=10, shift=0):
    return (vn(n=n, shift=shift) * 2 + 1) * np.pi / beta


def v_plus(beta=1.0, n=10, n_min=0):
    return (vn_plus(n=n, n_min=n_min) * 2 + 1) * np.pi / beta


def iv(beta=1.0, n=10, shift=0):
    return v(beta=beta, n=n, shift=shift) * 1j


def iv_plus(beta=1.0, n=10, n_min=0):
    return v_plus(beta=beta, n=n, n_min=n_min) * 1j


# ------------------------------------------------ BOSONIC -------------------------------------------------------------
def wn(n=10):
    return np.arange(-n, n + 1)

def wn_shell(n=10,n_core=1):
    return np.concatenate((np.arange(-(n+n_core), -n_core),np.arange(n_core+1, n+n_core+1)))


def wn_plus(n=10):
    return np.arange(0, n + 1)


def w(beta=1.0, n=10):
    return (wn(n=n) * 2) * np.pi / beta


def iw(beta=1.0, n=10):
    return w(beta=beta, n=n) * 1j


def wnfind(niw=None, n=None):
    return n - niw


def wn_outer(n_core=None, n_outer=None):
    return np.concatenate((np.arange(-n_outer, -n_core), np.arange(n_core + 1, n_outer + 1)))


def wn_outer_plus(n_core=None, n_outer=None):
    return np.arange(n_core + 1, n_outer + 1)


# ------------------------------------------------ ROUNTINES FOR SUMS -------------------------------------------------------------
def vn_centered_sum_wn(mat, beta, iwn=0, niv_sum=-1):
    '''Assumes [...,v,v'] '''
    niv = mat.shape[-1] // 2
    if (niv_sum == -1):
        return 1 / beta ** 2 * np.sum(mat, axis=(-2, -1))
    else:
        shift = iwn//2
        return 1 / beta ** 2 * np.sum(mat[..., niv - niv_sum + shift:niv + niv_sum + shift, niv - niv_sum + shift:niv + niv_sum + shift], axis=(-2, -1))


def vn_centered_sum(mat, w_n, beta, niv_sum=-1):
    '''Assumes [...,w,v,v'] '''
    result = np.zeros((mat.shape[:-2]), dtype=mat.dtype)
    for i, iwn in enumerate(w_n):
        result[..., i] = vn_centered_sum_wn(mat[..., i, :, :], beta, iwn=iwn, niv_sum=niv_sum)
    return result

def vn_centered_sum_1d(mat, w_n, beta, niv_sum=-1):
    '''Assumes [...,w,v,v'] '''
    niv = mat.shape[-1] // 2
    if(niv_sum == -1): niv_sum = niv
    result = np.zeros((mat.shape[:-1]), dtype=mat.dtype)
    for i, iwn in enumerate(w_n):
        result[..., i,:] = 1 / beta * np.sum(mat[..., i,:,niv - niv_sum + iwn:niv + niv_sum + iwn], axis=(-1))
        # result[..., i,:] = 1 / beta * np.sum(mat[..., i,:,:], axis=(-1))
    return result


def wn_sum(mat, beta, niw_sum=-1):
    ''' Assumes [w,...] '''
    niw = mat.shape[0] // 2
    if (niw_sum == -1):
        return 1 / beta * np.sum(mat, axis=0)
    else:
        return 1 / beta * np.sum(mat[niw - niw_sum:niw + niw_sum + 1], axis=0)


if __name__ == '__main__':
    pass
