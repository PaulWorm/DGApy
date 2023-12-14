"""
    This module handles everything around Matsubara frequencies.
    The main functions are:
        - vn(n) and vn(beta,n) for fermionic frequencies
        - wn(n) and wn(beta,n) for bosonic frequencies
    where n is the number of Matsubara frequencies and beta is the inverse temperature.
"""

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
from multimethod import multimethod
import numbers

KNOWN_FREQ_SHIFTS = ['plus', 'minus', 'center']


def get_freq_shift(iwn, freq_notation):
    if freq_notation == 'plus':  # chi_0[w,v] = - beta G(v) * G(v+w)
        iws, iws2 = 0, iwn
    elif freq_notation == 'minus':  # chi_0[w,v] = - beta G(v) * G(v-w)
        iws, iws2 = 0, -iwn
    elif freq_notation == 'center':  # chi_0[w,v] = - beta G(v+w//2) * G(v-w//2-w%2)
        iws, iws2 = iwn // 2, -(iwn // 2 + iwn % 2)
    else:
        raise NotImplementedError(f'freq_notation {freq_notation} is not known. Known are: {KNOWN_FREQ_SHIFTS}')
    return iws, iws2


# ---------------------------------------- FREQUENCY GRID MANIPULATIONS ----------------------------------------------------------


def append_v_vp_shell(mat, val, niv_shell=0):
    ''' append val to the shell of mat
        assume last two dimensions are v and vp
        niv_urange is the number of additional frequencies
    '''
    niv = mat.shape[-1] // 2
    niv_full = niv + niv_shell
    mat_urange = np.ones(mat.shape[:-2] + (niv_full * 2, niv_full * 2), dtype=mat.dtype) * val
    mat_urange[..., niv_full - niv:niv_full + niv, niv_full - niv:niv_full + niv] = mat
    return mat_urange


# pylint: disable=unexpected-keyword-arg
def w_to_vmvp(mat):
    ''' Transform w -> v-vp '''
    niw = np.shape(mat)[0] // 2
    niv = niw // 2
    mat_new = np.zeros((2 * niv, 2 * niv) + np.shape(mat)[1:], dtype=mat.dtype)
    vp = vn(niv)
    for i, ivp in enumerate(vp):
        mat_new[:, i, ...] = mat[niw - niv - ivp:niw + niv - ivp, ...]
    return mat_new


# pylint: enable=unexpected-keyword-arg

def cen2lin(val, start):
    return val - start


def wn_cen2lin(w, niw):
    return cen2lin(w, -niw)


def wn_slices_gen(mat=None, n_cut=None, w=None):
    n = mat.shape[-1] // 2
    mat_grid = np.array([mat[..., n - n_cut - iwn:n + n_cut - iwn] for iwn in w])
    return mat_grid


def wn_slices_shell(mat, n_shell, n_core=0, w=None):
    ''' v must be the last axis of mat '''
    n = niv_from_mat(mat)  # mat.shape[-1] // 2
    mat_grid = np.array([mat[..., n + n_core - iwn:n + n_core + n_shell - iwn] for iwn in w])
    mat_grid_m = np.array([mat[..., n - n_core - n_shell - iwn:n - n_core - iwn] for iwn in w])
    return np.concatenate((mat_grid_m, mat_grid), axis=-1)
    # return fermionic_full_nu_range(mat_grid, axis=-1)


def cut_iv_with_iw_shift_1d(mat, niv_cut=-1, iwn=0):
    ''' Cut the fermionic frequencies centric around iwn '''
    niv = np.size(mat) // 2
    if niv_cut == -1:
        niv_cut = niv - np.abs(iwn)
    return mat[niv - niv_cut - iwn:niv + niv_cut - iwn]


def cut_iv_with_iw_shift(mat, niv_cut=-1, iwn=0, axes=(-1,)):
    ''' Cut the fermionic frequencies centric around iwn along axes '''
    mat_cut = mat
    for ax in axes:
        mat_cut = np.apply_along_axis(cut_iv_with_iw_shift_1d, axis=ax, arr=mat_cut, niv_cut=niv_cut, iwn=iwn)
    return mat_cut


def cut_v_1d_pos(arr=None, niv_cut=0, niv_min=0):
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    if niv_cut == -1: niv_cut = niv
    return arr[niv + niv_min:niv + niv_min + niv_cut]


def fermionic_full_nu_range(mat, axis=-1):
    '''Build full Fermionic object from positive frequencies only along axis.'''
    return np.concatenate((np.conj(np.flip(mat, axis)), mat), axis=axis)


def bosonic_full_nu_range(mat, axis=-1, freq_axes=None):
    '''Build full Bosonic object from positive frequencies only along axis.'''
    ind = np.arange(1, np.shape(mat)[axis])
    if freq_axes is None:
        freq_axes = axis
    return np.concatenate((np.conj(np.flip(np.take(mat, ind, axis=axis), freq_axes)), mat), axis=axis)


def concatenate_core_asmypt(core, asympt, axis=-1):
    ''' Concatenate core and asympt arrays along axis. v has to be last axis.'''
    niv_asympt = np.shape(asympt)[axis] // 2
    return np.concatenate((asympt[..., :niv_asympt], core, asympt[..., niv_asympt:]), axis=axis)


# ----------------------------------------------- FERMIONIC ------------------------------------------------------------

def cut_v_1d(arr=None, niv_cut=0):
    '''
        Cut fermionic frequencies to range [-niv_cut,niv_cut]
    '''
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    if niv_cut == -1: niv_cut = niv
    return arr[niv - niv_cut:niv + niv_cut]


def cut_v(arr=None, niv_cut=0, axes=(0,)):
    '''
        Cut fermionic frequencies to range [-niv_cut,niv_cut] for all axis in axes
    '''
    axes = np.atleast_1d(axes)
    axes = np.flip(np.sort(axes))
    tmp = arr
    for axis in axes:
        tmp = np.apply_along_axis(cut_v_1d, axis=axis, arr=tmp, niv_cut=niv_cut)
    return tmp


def inv_cut_v_1d(arr=None, niv_core=0, niv_shell=-1):
    '''
        Cut fermionic frequencies to everything OUTSIDE(!) of range [-niv_cut,niv_cut]
    '''
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    if niv_shell == -1: niv_shell = niv
    return np.concatenate((arr[niv - niv_shell - niv_core:niv - niv_core], arr[niv + niv_core:niv + niv_core + niv_shell]))


def inv_cut_v(arr=None, niv_core=0, niv_shell=0, axes=(0,)):
    axes = np.atleast_1d(axes)
    axes = np.flip(np.sort(axes))
    tmp = arr
    for axis in axes:
        tmp = np.apply_along_axis(inv_cut_v_1d, axis=axis, arr=tmp, niv_core=niv_core, niv_shell=niv_shell)
    return tmp


def niv_from_mat(mat, axis=-1, pos=False):
    '''
        Get number of fermionic matsubara frequencies of axes
    '''
    if pos:
        n = np.shape(mat)[axis]
    else:
        n = np.shape(mat)[axis] // 2
    return n


# pylint: disable=function-redefined
# pylint: disable=unexpected-keyword-arg
@multimethod
def vn(n: numbers.Integral, shift: numbers.Integral = 0, pos: bool = False):
    '''
        Overloaded functions for fermionic matsubara freuquencies.
        Warning: beta and n are NOT keyword args!
    '''
    if pos:
        return np.arange(shift, n + shift)
    else:
        return np.arange(-n + shift, n + shift)


@vn.register
def vn_beta(beta: numbers.Real, n: numbers.Integral, shift: numbers.Integral = 0, pos: bool = False):
    return np.pi / beta * (2 * vn(n, shift=shift, pos=pos) + 1)


@vn.register
def vn_from_mat(mat: np.ndarray, axis: numbers.Integral = -1, shift: numbers.Integral = 0, pos: bool = False):
    n = niv_from_mat(mat, axis, pos)
    return vn(n, shift=shift, pos=pos)


@vn.register
def vn_beta_form_mat(beta: numbers.Real, mat: np.ndarray,
                     axis: numbers.Integral = -1, shift: numbers.Integral = 0, pos: bool = False):
    n = niv_from_mat(mat, axis, pos)
    return vn_beta(beta, n, shift=shift, pos=pos)


# pylint: enable=function-redefined
# pylint: enable=unexpected-keyword-arg

# ----------------------------------------------------- BOSONIC ------------------------------------------------------------------

def add_bosonic(mat1, mat2, axis=0):
    '''
        Add two bosonic objects not necessarily of the same size.
    '''
    niw_1 = niw_from_mat(mat1, axis=axis)
    niw_2 = niw_from_mat(mat2, axis=axis)
    if niw_2 <= niw_1:
        result = np.copy(mat1)
        nl, nu = niw_1 - niw_2, niw_1 + niw_2 + 1
        result[nl:nu] += mat2
        return result
    else:
        result = np.copy(mat2)
        nl, nu = niw_2 - niw_1, niw_2 + niw_1 + 1
        result[nl:nu] += mat1
        return result


def cut_v_1d_wn(arr=None, niw_cut=0):
    '''
        Cut the bosonic frequencies centric around 0
    '''
    assert np.size(np.shape(arr)) == 1, 'Array is not 1D.'
    niv = arr.shape[0] // 2
    return arr[niv - niw_cut:niv + niw_cut + 1]


def cut_w(arr=None, niw_cut=0, axes=(0,)):
    '''
        Cut the bosonic frequencies centric around 0 for all axis in axes
    '''
    axes = np.flip(np.sort(axes))
    tmp = arr
    for axis in axes:
        tmp = np.apply_along_axis(cut_v_1d_wn, axis=axis, arr=tmp, niw_cut=niw_cut)
    return tmp


def niw_from_mat(mat, axis=-1, pos=False):
    '''
        Get number of fermionic matsubara frequencies of axes
    '''
    if pos:
        n = np.shape(mat)[axis] - 1
    else:
        n = np.shape(mat)[axis] // 2
    return n


def mat_at_w0(mat, pos=False, axis=0):
    '''
        Returns the matrix at the zeroth bosonic Matsubara frequency.
    '''
    if not pos:
        idx = niw_from_mat(mat, axis=axis, pos=pos)
    else:
        idx = 0
    return np.take(mat, idx, axis=axis)


def get_mat_at_iwn(mat, iwn, pos=False, axis=0):
    '''
        Returns the matrix at the iwn bosonic Matsubara frequency.
    '''
    if not pos:
        idx = niw_from_mat(mat, axis=axis, pos=pos) + iwn
    else:
        idx = iwn
    return np.take(mat, idx, axis=axis)


# pylint: disable=function-redefined
@multimethod
def wn(n: numbers.Integral, shift: numbers.Integral = 0, pos: bool = False):
    '''
        Overloaded function for bosonic matsubara frequencies.
    '''
    if pos:
        return np.arange(shift, n + shift + 1)
    else:
        return np.arange(-n + shift, n + shift + 1)


@wn.register
def wn_beta(beta: numbers.Real, n: numbers.Integral, shift: numbers.Integral = 0, pos: bool = False):
    return np.pi / beta * (2 * wn(n, shift=shift, pos=pos))


@wn.register
def wn_from_mat(mat: np.ndarray, axis: numbers.Integral = 0, shift: numbers.Integral = 0, pos: bool = False):
    n = niw_from_mat(mat, axis, pos)
    return wn(n, shift=shift, pos=pos)


@wn.register
def wn_beta_from_mat(beta: numbers.Real, mat: np.ndarray,
                     axis: numbers.Integral = 0, shift: numbers.Integral = 0, pos: bool = False):
    n = niw_from_mat(mat, axis, pos)
    return wn_beta(beta, n, shift=shift, pos=pos)


# pylint: enable=function-redefined

# ------------------------------------------------ ROUNTINES FOR SUMS ------------------------------------------------------------
def vn_centered_sum_wn(mat, beta, iwn=0, niv_sum=-1):
    '''Assumes [...,v,v'] '''
    niv = mat.shape[-1] // 2
    if niv_sum == -1:
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


def wn_sum(mat, beta, niw_sum=-1):
    ''' Assumes [w,...] '''
    niw = mat.shape[0] // 2
    if niw_sum == -1:
        return 1 / beta * np.sum(mat, axis=0)
    else:
        return 1 / beta * np.sum(mat[niw - niw_sum:niw + niw_sum + 1], axis=0)


# ---------------------------------------- CHANGE FREQUENCY CONVENTIONS ----------------------------------------------------------

def ph2pp_iwn(mat_ph, iwn_pp=0):
    '''
        Transform from (iw,v,vp) to (v+v'-w,v,v') notation
    '''
    assert len(np.shape(mat_ph)) >= 3, 'Input does not have more than or equal three dimensions as required.'
    niw = mat_ph.shape[-3] // 2
    niv = mat_ph.shape[-1] // 2

    niv_pp = np.min((niw // 3, niv // 3))
    iv = np.arange(-niv_pp, niv_pp)
    mat_pp = np.zeros(mat_ph.shape[:-3] + (2 * niv_pp, 2 * niv_pp), dtype=complex)
    for i, vi in enumerate(iv):
        for j, vip in enumerate(iv):
            iwn = niw + vi + vip + 1 - iwn_pp
            ivn = niv + vi
            ivnp = niv + vip
            mat_pp[..., i, j] = mat_ph[..., iwn, ivn, ivnp]
    return mat_pp


def ph2pp(mat_ph):
    '''
        Transform from (iw,v,vp) to (v+v'-w,v,v') notation
    '''
    assert len(np.shape(mat_ph)) >= 3, 'Input does not have more than or equal three dimensions as required.'
    niw_pp = mat_ph.shape[-3] // 6
    w = wn(niw_pp)
    return np.array([ph2pp_iwn(mat_ph=mat_ph, iwn_pp=iwn) for iwn in w])


def ph2pp_iwn_wc(mat_ph, iwn_pp=0):
    '''
        Transform from (iw,v,vp, ss') to (v-v',v,w-v', bar(ss')) notation
    '''
    assert len(np.shape(mat_ph)) >= 3, 'Input does not have more than or equal three dimensions as required.'
    niw = mat_ph.shape[-3] // 2
    niv = mat_ph.shape[-1] // 2

    niv_pp = np.min((niw // 2, niv // 2))
    iv = np.arange(-niv_pp, niv_pp)
    mat_pp = np.zeros(mat_ph.shape[:-3] + (2 * niv_pp, 2 * niv_pp), dtype=complex)
    for i, vi in enumerate(iv):
        for j, vip in enumerate(iv):
            iwn = niw + vi - vip
            ivn = niv + vi
            ivnp = niv + iwn_pp - (vip + 1)
            mat_pp[..., i, j] = -mat_ph[..., iwn, ivn, ivnp]
    return mat_pp


def ph2pp_wc(mat_ph):
    '''
        Transform from (iw,v,vp, ss') to (v-v',v,w-v', bar(ss')) notation
    '''
    assert len(np.shape(mat_ph)) >= 3, 'Input does not have more than or equal three dimensions as required.'
    niw_pp = mat_ph.shape[-3] // 4
    w = wn(niw_pp)
    return np.array([ph2pp_iwn_wc(mat_ph=mat_ph, iwn_pp=iwn) for iwn in w])

def ph2pp_wc_md(mat_dens, mat_magn):
    '''
        Transform from (iw,v,vp) to (v-v',v,w-v') notation
    '''
    mat_dens_ph2pp = ph2pp_wc(mat_dens)
    mat_magn_ph2pp = ph2pp_wc(mat_magn)
    return 0.5*mat_dens_ph2pp+1.5*mat_magn_ph2pp, (mat_dens_ph2pp - mat_magn_ph2pp) * 0.5

if __name__ == '__main__':
    pass
