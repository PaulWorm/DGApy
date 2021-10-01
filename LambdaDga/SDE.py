# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import w2dyn_aux
import numpy as np
import TwoPoint as tp
import FourPoint as fp


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def wn_slices(mat=None, n_cut=None, iw=None):
    n = mat.shape[-1] // 2
    mat_grid = np.array([mat[n - n_cut - wn:n + n_cut - wn] for wn in iw])
    return mat_grid


def local_dmft_sde(vrg: fp.LocalThreePoint = None, chir: fp.LocalSusceptibility = None, u=1.0):
    assert (vrg.channel == chir.channel), 'Channels of physical susceptibility and Fermi-bose vertex not consistent'
    u_r = fp.get_ur(u=u, channel=vrg.channel)
    giw_grid = wn_slices(mat=chir.giw, n_cut=vrg.niv, iw=chir.iw)
    return -u_r / 2. * np.sum((vrg.mat * (1. - u_r * chir.mat[:, None])) * giw_grid,
                              axis=0)  # - np.sign(u_r) * 1./chir.beta


def sde_dga(vrg: fp.FullQ = None, chir: fp.FullQ = None, g_generator: tp.GreensFunctionGenerator = None , mu=0):
    assert (vrg.channel == chir.channel), 'Channels of physical susceptibility and Fermi-bose vertex not consistent'
    niv = vrg.mat.shape[-1] // 2
    sigma = np.zeros((g_generator.nkx(), g_generator.nky(), g_generator.nkz(), 2 * niv), dtype=complex)
    for iqw in range(vrg.qiw.my_size):
        gkpq = g_generator.generate_gk(mu=mu , qiw=vrg.qiw.my_qiw[iqw], niv=niv)
        sigma += - vrg.u_r / (2.0) * (vrg.mat[iqw,:][None,None,None,:] * (1. - vrg.u_r * chir.mat[iqw])  + np.sign(vrg.u_r) * 0.5 / vrg.beta) * gkpq.gk
        # + vrg.u * 0.5 / vrg.beta
    sigma = 1./(vrg.qiw.nq()) * sigma
    return sigma

# ------------------------------------------------ OBJECTS -------------------------------------------------------------


# class local_sde():
#
#     ''' Class to solve the local Schwinger-Dyson equation. Starting Point is the local generalized Susceptibility '''
#
#     def __init__(self, beta=1.0, u=1.0, giw=None, box_sizes=None, iw=None):
#         self._beta = beta
#         self._u = u
#         self._iw = iw
#         self.set_giw(giw=giw)
#         self._box_sizes = box_sizes


# class local_sde():
#     ''' Class to solve the local Schwinger-Dyson equation '''
#
#     def __init__(self, beta=1.0, u=1.0, giw=None, iw=None):
#         self._beta = beta
#         self._u = u
#         self._iw = iw
#         self.set_giw(giw=giw)
#
#
#     def set_giw(self, giw=None):
#         self._giw = giw
#         self._niv_giw = giw.shape[0] // 2
#
#     def get_gloc_grid(self, niv=-1):
#         if(niv == -1):
#             niv = self._niv_giw - np.max(np.abs(self._iw))
#         return np.array([self._giw[self._niv_giw - niv - wn:self._niv_giw + niv - wn] for wn in self._iw])
#
#     def sde(self, vrg=None, chir=None):
#         niv2 = np.shape(vrg)[-1] // 2
#         gloc_grid = self.get_gloc_grid(self, niv=niv2)
#         return  - np.sum(self._u / (2.0) * (vrg * (1. - self._u * chir[:, None]) - 1./self._beta)* gloc_grid, axis=0)
