# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# This module handles everything around freuquency grids, specifcally Matsubara ones.
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
from multimethod import multimethod
import numbers

KNOWN_FREQ_SHIFTS = ['plus', 'minus', 'center']


def get_freq_shift(iwn, freq_notation):
    if (freq_notation == 'plus'):  # chi_0[w,v] = - beta G(v) * G(v+w)
        iws, iws2 = 0, iwn
    elif (freq_notation == 'minus'):  # chi_0[w,v] = - beta G(v) * G(v-w)
        iws, iws2 = 0, -iwn
    elif (freq_notation == 'center'):  # chi_0[w,v] = - beta G(v+w//2) * G(v-w//2-w%2)
        iws, iws2 = iwn // 2, -(iwn // 2 + iwn % 2)
    else:
        raise NotImplementedError(f'freq_notation {freq_notation} is not known. Known are: {KNOWN_FREQ_SHIFTS}')
    return iws, iws2


# -------------------------------------------- FREQUENCY GRIDS ---------------------------------------------------------


def v_vp_urange(mat, val, niv_urange=0):
    ''' append val to the urange of mat
        assume last two dimensions are v and vp
        niv_urange is the number of additional frequencies
    '''
    niv = mat.shape[-1] // 2
    niv_full = niv + niv_urange
    mat_urange = np.ones(mat.shape[:-2] + (niv_full * 2, niv_full * 2), dtype=mat.dtype) * val
    mat_urange[..., niv_full - niv:niv_full + niv, niv_full - niv:niv_full + niv] = mat
    return mat_urange


def w_to_vmvp(mat):
    '''Transform w -> v-v' '''
    niw = np.shape(mat)[0] // 2
    niv = niw // 2
    mat_new = np.zeros((2 * niv, 2 * niv) + np.shape(mat)[1:], dtype=mat.dtype)
    vp = vn(niv)
    for i, ivp in enumerate(vp):
        mat_new[:, i, ...] = mat[niw - niv - ivp:niw + niv - ivp, ...]
    return mat_new


def cen2lin(val=None, start=0):
    return val - start


def wn_cen2lin(wn=0, niw=None):
    return cen2lin(wn, -niw)


def wn_slices_gen(mat=None, n_cut=None, wn=None, freq_notation='minus'):
    n = mat.shape[-1] // 2
    _,iwn = get_freq_shift(wn,freq_notation)
    mat_grid = np.array([mat[..., n - n_cut + iwn:n + n_cut + iwn] for iwn in wn])
    return mat_grid

def wn_slices_shell(mat, n_shell, n_core=0, wn=None):
    n = mat.shape[-1] // 2
    mat_grid = np.array([mat[..., n + n_core - iwn:n + n_core + n_shell - iwn] for iwn in wn])
    return fermionic_full_nu_range(mat_grid)


def wn_slices_plus_cent(mat=None, n_cut=None, wn=None):
    n = mat.shape[-1] // 2
    mat_grid = np.array([mat[n - n_cut - (iwn - iwn // 2 - iwn % 2):n + n_cut - (iwn - iwn // 2 - iwn % 2)] for iwn in wn])
    return mat_grid

def cut_iv_with_iw_shift_1d(mat, niv_cut=-1, iwn=0):
    ''' Cut the fermionic frequencies centric around iwn '''
    niv = np.size(mat) // 2
    if (niv_cut == -1):
        niv_cut = niv - np.abs(iwn)
    return mat[niv - niv_cut - iwn:niv + niv_cut - iwn]


def cut_iv_with_iw_shift(mat, niv_cut=-1, iwn=0, axes=(-1,)):
    ''' Cut the fermionic frequencies centric around iwn along axes '''
    mat_cut = mat
    for ax in axes:
        mat_cut = np.apply_along_axis(cut_iv_with_iw_shift_1d, axis=ax, arr=mat_cut, niv_cut=niv_cut, iwn=iwn)
    return mat_cut


def cut_v_1d(arr=None, niv_cut=0):
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    if (niv_cut == -1): niv_cut = niv
    return arr[niv - niv_cut:niv + niv_cut]


def inv_cut_v_1d(arr=None, niv_core=0, niv_shell=-1):
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    if (niv_shell == -1): niv_shell = niv
    return np.concatenate((arr[niv - niv_shell - niv_core:niv - niv_core], arr[niv + niv_core:niv + niv_core + niv_shell]))


def cut_v_1d_pos(arr=None, niv_cut=0, niv_min=0):
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    if (niv_cut == -1): niv_cut = niv
    return arr[niv + niv_min:niv + niv_min + niv_cut]


def cut_v_1d_wn(arr=None, niw_cut=0):
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    return arr[niv - niw_cut:niv + niw_cut + 1]


def cut_v(arr=None, niv_cut=0, axes=(0,)):
    axes = np.atleast_1d(axes)
    axes = np.flip(np.sort(axes))
    tmp = arr
    for axis in axes:
        tmp = np.apply_along_axis(cut_v_1d, axis=axis, arr=tmp, niv_cut=niv_cut)
    return tmp


def inv_cut_v(arr=None, niv_core=0, niv_shell=0, axes=(0,)):
    axes = np.atleast_1d(axes)
    axes = np.flip(np.sort(axes))
    tmp = arr
    for axis in axes:
        tmp = np.apply_along_axis(inv_cut_v_1d, axis=axis, arr=tmp, niv_core=niv_core, niv_shell=niv_shell)
    return tmp


def cut_w(arr=None, niw_cut=0, axes=(0,)):
    axes = np.flip(np.sort(axes))
    tmp = arr
    for axis in axes:
        tmp = np.apply_along_axis(cut_v_1d_wn, axis=axis, arr=tmp, niw_cut=niw_cut)
    return tmp


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
'''
    Overloaded functions for fermionic matsubara freuquencies. 
'''


def niv_from_mat(mat, axes=-1, pos=False):
    '''
        Get number of fermionic matsubara frequencies of axes
    '''
    if (pos):
        n = np.shape(mat)[axes]
    else:
        n = np.shape(mat)[axes] // 2
    return n


@multimethod
def vn(n: numbers.Integral, shift: numbers.Integral = 0, pos: bool = False):
    if (pos):
        return np.arange(shift, n + shift)
    else:
        return np.arange(-n + shift, n + shift)


@multimethod
def vn(beta: numbers.Real, n: numbers.Integral, shift: numbers.Integral = 0, pos: bool = False):
    return np.pi / beta * (2 * vn(n, shift=shift, pos=pos) + 1)


@multimethod
def vn(mat: np.ndarray, axes: numbers.Integral = -1, shift: numbers.Integral = 0, pos: bool = False):
    n = niv_from_mat(mat, axes, pos)
    return vn(n, shift=shift, pos=pos)


@multimethod
def vn(beta: numbers.Real, mat: np.ndarray, axes: numbers.Integral = -1, shift: numbers.Integral = 0, pos: bool = False):
    n = niv_from_mat(mat, axes, pos)
    return vn(beta, n, shift=shift, pos=pos)


# ------------------------------------------------ BOSONIC -------------------------------------------------------------
'''
    Overloaded functions for bosonic matsubara freuquencies. 
'''


def niw_from_mat(mat, axes=-1, pos=False):
    '''
        Get number of fermionic matsubara frequencies of axes
    '''
    if (pos):
        n = np.shape(mat)[axes] - 1
    else:
        n = np.shape(mat)[axes] // 2
    return n


@multimethod
def wn(n: numbers.Integral, shift: numbers.Integral = 0, pos: bool = False):
    if (pos):
        return np.arange(shift, n + shift + 1)
    else:
        return np.arange(-n + shift, n + shift + 1)


@multimethod
def wn(beta: numbers.Real, n: numbers.Integral, shift: numbers.Integral = 0, pos: bool = False):
    return np.pi / beta * (2 * wn(n, shift=shift, pos=pos) + 1)


@multimethod
def wn(mat: np.ndarray, axes: numbers.Integral = -1, shift: numbers.Integral = 0, pos: bool = False):
    n = niw_from_mat(mat, axes, pos)
    return wn(n, shift=shift, pos=pos)


@multimethod
def wn(beta: numbers.Real, mat: np.ndarray, axes: int = -1, shift: int = 0, pos: bool = False):
    n = niw_from_mat(mat, axes, pos)
    return wn(beta, n, shift=shift, pos=pos)


# ------------------------------------------------ ROUNTINES FOR SUMS -------------------------------------------------------------
def vn_centered_sum_wn(mat, beta, iwn=0, niv_sum=-1):
    '''Assumes [...,v,v'] '''
    niv = mat.shape[-1] // 2
    if (niv_sum == -1):
        return 1 / beta ** 2 * np.sum(mat, axis=(-2, -1))
    else:
        shift = iwn // 2
        return 1 / beta ** 2 * np.sum(
            mat[..., niv - niv_sum + shift:niv + niv_sum + shift, niv - niv_sum + shift:niv + niv_sum + shift], axis=(-2, -1))


def vn_centered_sum(mat, w_n, beta, niv_sum=-1):
    '''Assumes [...,w,v,v'] '''
    result = np.zeros((mat.shape[:-2]), dtype=mat.dtype)
    for i, iwn in enumerate(w_n):
        result[..., i] = vn_centered_sum_wn(mat[..., i, :, :], beta, iwn=iwn, niv_sum=niv_sum)
    return result


def vn_centered_sum_1d(mat, w_n, beta, niv_sum=-1):
    '''Assumes [...,w,v,v'] '''
    niv = mat.shape[-1] // 2
    if (niv_sum == -1): niv_sum = niv
    result = np.zeros((mat.shape[:-1]), dtype=mat.dtype)
    for i, iwn in enumerate(w_n):
        result[..., i, :] = 1 / beta * np.sum(mat[..., i, :, niv - niv_sum + iwn:niv + niv_sum + iwn], axis=(-1))
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
    niv_core = 30
    niv_shell = 100
    vn_core = vn(niv_core)
    wn_core = wn(niv_core)
    wn_core = wn(np.int64(niv_core))
