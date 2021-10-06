# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import w2dyn_aux
import copy
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

# -------------------------------------------- WRAPPER FUNCTIONS -------------------------------------------------------


# ======================================================================================================================
def local_dmft_sde_from_g2(dmft_input = None, box_sizes = None):

    giw = dmft_input['gloc']
    beta = dmft_input['beta']
    g2_dens = dmft_input['g2_dens']
    g2_magn = dmft_input['g2_magn']
    u = dmft_input['u']
    n = dmft_input['n']

    iw = g2_dens.iw_core

    niv_core = box_sizes['niv_core']
    niv_urange = box_sizes['niv_urange']
    niv_asympt = box_sizes['niv_asympt']

    chi0_core = fp.LocalBubble(giw=giw, beta=beta, niv_sum=niv_core, iw=iw)
    chi0_urange = fp.LocalBubble(giw=giw, beta=beta, niv_sum=niv_urange, iw=iw)
    chi0_asympt = copy.deepcopy(chi0_urange)
    chi0_asympt.add_asymptotic(niv_asympt=niv_asympt)

    gchi_dens_loc = fp.LocalFourPoint(matrix=fp.vec_chir_from_g2(g2=g2_dens), giw=g2_dens.giw,
                                      channel=g2_dens.channel,
                                      beta=g2_dens.beta, iw=g2_dens.iw_core)
    gchi_magn_loc = fp.LocalFourPoint(matrix=fp.vec_chir_from_g2(g2=g2_magn), giw=g2_magn.giw,
                                      channel=g2_magn.channel,
                                      beta=g2_magn.beta, iw=g2_magn.iw_core)

    gamma_dens = fp.gammar_from_gchir(gchir=gchi_dens_loc, gchi0_urange=chi0_urange, u=u)
    gamma_magn = fp.gammar_from_gchir(gchir=gchi_magn_loc, gchi0_urange=chi0_urange, u=u)

    gchi_aux_dens_loc = fp.local_gchi_aux_from_gammar(gammar=gamma_dens, gchi0_core=chi0_core, u=u)
    gchi_aux_magn_loc = fp.local_gchi_aux_from_gammar(gammar=gamma_magn, gchi0_core=chi0_core, u=u)

    chi_aux_dens_loc = fp.local_susceptibility_from_four_point(four_point=gchi_aux_dens_loc)
    chi_aux_magn_loc = fp.local_susceptibility_from_four_point(four_point=gchi_aux_magn_loc)

    chi_dens_urange_loc = fp.local_chi_phys_from_chi_aux(chi_aux=chi_aux_dens_loc, chi0_urange=chi0_urange,
                                                         chi0_core=chi0_core,
                                                         u=u)
    chi_dens_asympt_loc = copy.deepcopy(chi_dens_urange_loc)
    chi_dens_asympt_loc.add_asymptotic(chi0_asympt=chi0_asympt, chi0_urange=chi0_urange)
    chi_magn_urange_loc = fp.local_chi_phys_from_chi_aux(chi_aux=chi_aux_magn_loc, chi0_urange=chi0_urange,
                                                         chi0_core=chi0_core,
                                                         u=u)
    chi_magn_asympt_loc = copy.deepcopy(chi_magn_urange_loc)
    chi_magn_asympt_loc.add_asymptotic(chi0_asympt=chi0_asympt, chi0_urange=chi0_urange)

    vrg_dens_loc = fp.local_fermi_bose_from_chi_aux_asympt(gchi_aux=gchi_aux_dens_loc, gchi0=chi0_core,
                                                           chi_asympt=chi_dens_asympt_loc,
                                                           chi_urange=chi_dens_urange_loc,
                                                           niv_urange=niv_urange,
                                                           u=u)

    vrg_magn_loc = fp.local_fermi_bose_from_chi_aux_asympt(gchi_aux=gchi_aux_magn_loc, gchi0=chi0_core,
                                                           chi_asympt=chi_dens_asympt_loc,
                                                           chi_urange=chi_dens_urange_loc,
                                                           niv_urange=niv_urange,
                                                           u=u)

    siw_dens = local_dmft_sde(vrg=vrg_dens_loc, chir=chi_dens_asympt_loc, u=u)
    siw_magn = local_dmft_sde(vrg=vrg_magn_loc, chir=chi_magn_asympt_loc, u=u)

    siw = siw_dens + siw_magn + u / 2. * n

    dmft_sde = {
        'gamma_dens': gamma_dens,
        'gamma_magn': gamma_magn,
        'chi_dens': chi_dens_asympt_loc,
        'chi_magn': chi_magn_asympt_loc,
        'siw_dens': siw_dens,
        'siw_magn': siw_magn,
        'siw': siw
    }

    return dmft_sde
# ======================================================================================================================

# ======================================================================================================================



# ------------------------------------------------ OBJECTS -------------------------------------------------------------
def dga_susceptibility(dmft_input = None, box_sizes = None, qiw=None):

    chi0q_core_full = fp.FullQ(channel='dens', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)
    chi0q_urange_full = fp.FullQ(channel='dens', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)
    chi0q_asympt_full = fp.FullQ(channel='dens', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)

    chi_dens_asympt = fp.FullQ(channel='dens', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)
    chi_magn_asympt = fp.FullQ(channel='magn', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)

    chi_dens_asympt_lambda = fp.FullQ(channel='dens', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)
    chi_magn_asympt_lambda = fp.FullQ(channel='magn', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)

    vrg_dens = fp.FullQ(channel='dens', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=False)
    vrg_magn = fp.FullQ(channel='magn', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=False)

    g_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=kgrid, hr=t_mat, sigma=dmft1p['sloc'])

    gk_urange = g_generator.generate_gk(mu=dmft1p['mu'], qiw=[0, 0, 0, 0], niv=niv_urange)
    gk_core = copy.deepcopy(gk_urange)
    gk_core.cut_self_iv(niv_cut=niv_core)

    for iqw in range(qiw.my_size):
        gkpq_urange = g_generator.generate_gk(mu=dmft1p['mu'], qiw=qiw.my_qiw[iqw], niv=niv_urange)

        gkpq_core = copy.deepcopy(gkpq_urange)
        gkpq_core.cut_self_iv(niv_cut=niv_core)

        chi0q_core = fp.Bubble(gk=gk_core.gk, gkpq=gkpq_core.gk, beta=gk_core.beta)
        chi0q_urange = fp.Bubble(gk=gk_urange.gk, gkpq=gkpq_urange.gk, beta=gk_urange.beta)
        chi0q_asympt = copy.deepcopy(chi0q_urange)
        chi0q_asympt.add_asymptotic(niv_asympt=box_sizes['niv_asympt'], wn=qiw.my_iw[iqw])

        gchi_aux_dens = fp.construct_gchi_aux(gammar=gamma_dens_loc, gchi0=chi0q_core, u=dmft1p['u'], wn=qiw.wn(iqw))
        gchi_aux_magn = fp.construct_gchi_aux(gammar=gamma_magn_loc, gchi0=chi0q_core, u=dmft1p['u'], wn=qiw.wn(iqw))

        chi_aux_dens = fp.susceptibility_from_four_point(four_point=gchi_aux_dens)
        chi_aux_magn = fp.susceptibility_from_four_point(four_point=gchi_aux_magn)

        chiq_dens_urange = fp.chi_phys_from_chi_aux(chi_aux=chi_aux_dens, chi0_urange=chi0q_urange, chi0_core=chi0q_core)
        chiq_magn_urange = fp.chi_phys_from_chi_aux(chi_aux=chi_aux_magn, chi0_urange=chi0q_urange, chi0_core=chi0q_core)

        chiq_dens_asympt = copy.deepcopy(chiq_dens_urange)
        chiq_dens_asympt.add_asymptotic(chi0_asympt=chi0q_asympt, chi0_urange=chi0q_urange)

        chiq_magn_asympt = copy.deepcopy(chiq_magn_urange)
        chiq_magn_asympt.add_asymptotic(chi0_asympt=chi0q_asympt, chi0_urange=chi0q_urange)

        vrgq_dens = fp.fermi_bose_from_chi_aux_asympt(gchi_aux=gchi_aux_dens, gchi0=chi0q_core, chi_asympt=chiq_dens_asympt
                                                      , chi_urange=chiq_dens_urange, niv_urange=box_sizes['niv_urange'])

        vrgq_magn = fp.fermi_bose_from_chi_aux_asympt(gchi_aux=gchi_aux_magn, gchi0=chi0q_core, chi_asympt=chiq_magn_asympt
                                                      , chi_urange=chiq_magn_urange, niv_urange=box_sizes['niv_urange'])

        chi_dens_asympt.mat[qiw.my_indizes[iqw]] = chiq_dens_asympt.mat
        chi_magn_asympt.mat[qiw.my_indizes[iqw]] = chiq_magn_asympt.mat

        vrg_dens.mat[qiw.my_indizes[iqw]] = vrgq_dens.mat
        vrg_magn.mat[qiw.my_indizes[iqw]] = vrgq_magn.mat

        chi0q_core_full.mat[qiw.my_indizes[iqw]] = chi0q_core.chi0
        chi0q_urange_full.mat[qiw.my_indizes[iqw]] = chi0q_urange.chi0
        chi0q_asympt_full.mat[qiw.my_indizes[iqw]] = chi0q_asympt.chi0

    realt.print_time('Non-local chi ')

    chi_dens_asympt.mat_to_array()
    chi_magn_asympt.mat_to_array()

    vrg_dens.mat_to_array()
    vrg_magn.mat_to_array()

    chi0q_urange_full.mat_to_array()
    chi0q_core_full.mat_to_array()
    chi0q_asympt_full.mat_to_array()


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
