# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import w2dyn_aux
import copy
import numpy as np
import TwoPoint as twop
import FourPoint as fp


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def wn_slices(mat=None, n_cut=None, iw=None):
    n = mat.shape[-1] // 2
    mat_grid = np.array([mat[n - n_cut - wn:n + n_cut - wn] for wn in iw])
    return mat_grid


def local_dmft_sde(vrg: fp.LocalThreePoint = None, chir: fp.LocalSusceptibility = None, u=None):
    assert (vrg.channel == chir.channel), 'Channels of physical susceptibility and Fermi-bose vertex not consistent'
    u_r = fp.get_ur(u=u, channel=vrg.channel)
    giw_grid = wn_slices(mat=chir.giw, n_cut=vrg.niv, iw=chir.iw)
    return -u_r / 2. * np.sum((vrg.mat * (1. - u_r * chir.mat[:, None]) - 1. / chir.beta) * giw_grid,
                              axis=0)  # The -1./chir.beta is is canceled in the sum. This is only relevant for Fluctuation diagnostics.


def local_rpa_sde(chir: fp.LocalSusceptibility = None, niv_giw=None, u=None):
    u_r = fp.get_ur(u=u, channel=chir.channel)
    giw_grid = wn_slices(mat=chir.giw, n_cut=niv_giw, iw=chir.iw)
    return u_r ** 2 / (2. * chir.beta) * np.sum(chir.mat[:, None] * giw_grid, axis=0)


def sde_dga(vrg: fp.LadderObject = None, chir: fp.LadderSusceptibility = None,
            g_generator: twop.GreensFunctionGenerator = None, mu=0, qiw_grid=None, nq=None, box_sizes=None, q_grid=None):
    assert (vrg.channel == chir.channel), 'Channels of physical susceptibility and Fermi-bose vertex not consistent'
    niv_urange = box_sizes['niv_urange']
    sigma = np.zeros((g_generator.nkx(), g_generator.nky(), g_generator.nkz(), niv_urange), dtype=complex)

    for iqw in range(qiw_grid.shape[0]):
        wn = qiw_grid[iqw][-1]
        q_ind = qiw_grid[iqw][0]
        q = q_grid.irr_kmesh[:, q_ind]
        qiw = np.append(q, wn)
        gkpq = g_generator.generate_gk_plus(mu=mu, qiw=qiw, niv=niv_urange)
        sigma += (vrg.mat[iqw, niv_urange:][None, None, None, :] * (1. - vrg.u_r * chir.mat[iqw]) - 1. / vrg.beta) * gkpq.gk * \
                 q_grid.irrk_count[q_ind]
        if(wn != 0):
            qiw = np.append(q, -wn)
            gkpq = g_generator.generate_gk_plus(mu=mu, qiw=qiw, niv=niv_urange).gk
            sigma += (np.conj(np.flip(vrg.mat[iqw, :],axis=-1)[None, None, None, niv_urange:]) * (1. - vrg.u_r * np.conj(chir.mat[iqw])) - 1. / vrg.beta) * gkpq * \
                     q_grid.irrk_count[q_ind]

    sigma = - vrg.u_r / (2.0) * 1. / (nq) * sigma
    return sigma


def rpa_sde(chir: fp.LocalSusceptibility = None, g_generator: twop.GreensFunctionGenerator = None, niv_giw=None, mu=0,
            nq=None, u=None, qiw_grid=None, q_grid=None):
    u_r = fp.get_ur(u=u, channel=chir.channel)
    sigma = np.zeros((g_generator.nkx(), g_generator.nky(), g_generator.nkz(), niv_giw), dtype=complex)
    for iqw in range(qiw_grid.shape[0]):
        wn = qiw_grid[iqw][-1]
        q_ind = qiw_grid[iqw][0]
        q = q_grid.irr_kmesh[:, q_ind]
        qiw = np.append(q, wn)
        gkpq = g_generator.generate_gk_plus(mu=mu, qiw=qiw, niv=niv_giw)
        sigma += chir.mat[iqw, None] * gkpq.gk * q_grid.irrk_count[q_ind]
        if(wn != 0):
            qiw = np.append(q, -wn)
            gkpq = g_generator.generate_gk_plus(mu=mu, qiw=qiw, niv=niv_giw)
            sigma += np.conj(chir.mat[iqw, None]) * gkpq.gk * q_grid.irrk_count[q_ind]

    sigma = u_r ** 2 / (2. * chir.beta) * 1. / (nq) * sigma
    return sigma


# -------------------------------------------- WRAPPER FUNCTIONS -------------------------------------------------------

# ======================================================================================================================
def local_rpa_sde_correction(dmft_input=None, box_sizes=None, iw=None):
    giw = dmft_input['gloc']
    beta = dmft_input['beta']
    u = dmft_input['u']

    niv_urange = box_sizes['niv_urange']
    niv_asympt = box_sizes['niv_asympt']

    chi0_urange = fp.LocalBubble(giw=giw, beta=beta, niv_sum=niv_urange, iw=iw)
    chi0_asympt = copy.deepcopy(chi0_urange)
    chi0_asympt.add_asymptotic(niv_asympt=niv_asympt)

    chi_rpa_dens = fp.local_rpa_susceptibility(chi0_asympt=chi0_asympt, chi0_urange=chi0_urange, channel='dens', u=u)
    chi_rpa_magn = fp.local_rpa_susceptibility(chi0_asympt=chi0_asympt, chi0_urange=chi0_urange, channel='magn', u=u)

    siw_rpa_dens = local_rpa_sde(chir=chi_rpa_dens, niv_giw=niv_urange, u=u)
    siw_rpa_magn = local_rpa_sde(chir=chi_rpa_magn, niv_giw=niv_urange, u=u)

    rpa_sde = {
        'siw_rpa_dens': siw_rpa_dens,
        'siw_rpa_magn': siw_rpa_magn,
        'chi_rpa_dens': chi_rpa_dens,
        'chi_rpa_magn': chi_rpa_magn
    }

    return rpa_sde


# ======================================================================================================================


# ======================================================================================================================
def local_dmft_sde_from_g2(dmft_input=None, box_sizes=None, g2_dens=None, g2_magn=None):
    giw = dmft_input['gloc']
    beta = dmft_input['beta']
    u = dmft_input['u']
    n = dmft_input['n']

    iw = g2_dens.iw

    niv_core = box_sizes['niv_core']
    niv_urange = box_sizes['niv_urange']

    chi0_core = fp.LocalBubble(giw=giw, beta=beta, niv_sum=niv_core, iw=iw)
    chi0_urange = fp.LocalBubble(giw=giw, beta=beta, niv_sum=niv_urange, iw=iw)

    gchi_dens_loc = fp.LocalFourPoint(matrix=fp.vec_chir_from_g2(g2=g2_dens), giw=g2_dens.giw,
                                      channel=g2_dens.channel,
                                      beta=g2_dens.beta, iw=iw)
    gchi_magn_loc = fp.LocalFourPoint(matrix=fp.vec_chir_from_g2(g2=g2_magn), giw=g2_magn.giw,
                                      channel=g2_magn.channel,
                                      beta=g2_magn.beta, iw=iw)

    gamma_dens = fp.gammar_from_gchir(gchir=gchi_dens_loc, gchi0_urange=chi0_urange, u=u)
    gamma_magn = fp.gammar_from_gchir(gchir=gchi_magn_loc, gchi0_urange=chi0_urange, u=u)

    gamma_dens.cut_iv(niv_cut=niv_core)
    gamma_magn.cut_iv(niv_cut=niv_core)

    gchi_aux_dens_loc = fp.local_gchi_aux_from_gammar(gammar=gamma_dens, gchi0_core=chi0_core, u=u)
    gchi_aux_magn_loc = fp.local_gchi_aux_from_gammar(gammar=gamma_magn, gchi0_core=chi0_core, u=u)

    chi_aux_dens_loc = fp.local_susceptibility_from_four_point(four_point=gchi_aux_dens_loc)
    chi_aux_magn_loc = fp.local_susceptibility_from_four_point(four_point=gchi_aux_magn_loc)

    chi_dens_urange_loc = fp.local_chi_phys_from_chi_aux(chi_aux=chi_aux_dens_loc, chi0_urange=chi0_urange,
                                                         chi0_core=chi0_core,
                                                         u=u)

    chi_magn_urange_loc = fp.local_chi_phys_from_chi_aux(chi_aux=chi_aux_magn_loc, chi0_urange=chi0_urange,
                                                         chi0_core=chi0_core,
                                                         u=u)

    vrg_dens_loc = fp.local_fermi_bose_from_chi_aux_urange(gchi_aux=gchi_aux_dens_loc, gchi0=chi0_core,
                                                           niv_urange=niv_urange,
                                                           u=u)

    vrg_magn_loc = fp.local_fermi_bose_from_chi_aux_urange(gchi_aux=gchi_aux_magn_loc, gchi0=chi0_core,
                                                           niv_urange=niv_urange,
                                                           u=u)

    siw_dens = local_dmft_sde(vrg=vrg_dens_loc, chir=chi_dens_urange_loc, u=u)
    siw_magn = local_dmft_sde(vrg=vrg_magn_loc, chir=chi_magn_urange_loc, u=u)

    siw = siw_dens + siw_magn

    dmft_sde = {
        'chi0_core': chi0_core,
        'chi0_urange': chi0_urange,
        'gamma_dens': gamma_dens,
        'gamma_magn': gamma_magn,
        'vrg_dens': vrg_dens_loc,
        'vrg_magn': vrg_magn_loc,
        'chi_dens': chi_dens_urange_loc,
        'chi_magn': chi_magn_urange_loc,
        'siw_dens': siw_dens,
        'siw_magn': siw_magn,
        'siw': siw,
        'hartree': u / 2. * n
    }

    return dmft_sde

# ======================================================================================================================


# class local_sde():
#
#     ''' Class to solve the local Schwinger-Dyson equation. Starting Point is the local generalized Susceptibility '''
#
#     def __init__(self, beta=1.0, u=1.0, giw=None, box_sizes=None, iw_core=None):
#         self._beta = beta
#         self._u = u
#         self._iw = iw_core
#         self.set_giw(giw=giw)
#         self._box_sizes = box_sizes


# class local_sde():
#     ''' Class to solve the local Schwinger-Dyson equation '''
#
#     def __init__(self, beta=1.0, u=1.0, giw=None, iw_core=None):
#         self._beta = beta
#         self._u = u
#         self._iw = iw_core
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
