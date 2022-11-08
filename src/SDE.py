# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import w2dyn_aux
import copy
import numpy as np
import TwoPoint as twop
import FourPoint as fp
import Config as conf
import MatsubaraFrequencies as mf


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def wn_slices(mat=None, n_cut=None, wn=None):
    n = mat.shape[-1] // 2
    mat_grid = np.array([mat[n - n_cut - wn:n + n_cut - wn] for wn in wn])
    return mat_grid


def local_dmft_sde(vrg = None, chir: fp.LocalSusceptibility = None,giw=None, u=None, scal_const=1.0):
    #assert (vrg.channel == chir.channel), 'Channels of physical susceptibility and Fermi-bose vertex not consistent'
    u_r = fp.get_ur(u=u, channel=chir.channel)
    niv = vrg.shape[-1] // 2
    giw_grid = wn_slices(mat=giw, n_cut=niv, wn=chir.wn)
    # The u_r in the front stems from Fup = 1/2 * (Fd - Fm)
    return -u_r / 2. * np.sum((vrg * (1. - u_r * chir.mat[:, None]) - scal_const / chir.beta) * giw_grid,
                              axis=0)  # The -1./chir.beta is is canceled in the sum. This is only relevant for Fluctuation diagnostics.

def local_sde_vertex(vertex=None, giw=None, u=None, beta=None):
    niv = vertex.shape[-1] // 2
    niw = vertex.shape[0] // 2
    iw = mf.wn(n=niw)
    giw_grid = wn_slices(mat=giw, n_cut=niv, wn=iw)
    gloc = mf.cut_v_1d(giw,niv_cut=niv)
    # 1/beta^2 is included in the vertex
    return -u/2 * np.sum(vertex*gloc[None,None,:] * giw_grid[:,None,:] * giw_grid[:,:,None], axis=(0,2))


def local_rpa_sde(chir: fp.LocalSusceptibility = None, niv_giw=None, giw=None, u=None):
    u_r = fp.get_ur(u=u, channel=chir.channel)
    giw_grid = wn_slices(mat=giw, n_cut=niv_giw, wn=chir.iw)
    return u_r**2 / (2. * chir.beta) * np.sum(chir.mat_asympt[:, None] * giw_grid, axis=0) # vrg = 1
    #return u_r**2 / (2. * chir.beta) * np.sum(u_r * chir.mat[:, None] * giw_grid, axis=0) # vrg = (1-u chi_u)/(1- u chi_asympt)


def sde_dga(dga_conf: conf.DgaConfig = None, vrg_in=None, chir: fp.LadderSusceptibility = None,
            g_generator: twop.GreensFunctionGenerator = None, mu=0, qiw_grid=None, analyse_spin_fermion=False):
    # assert (vrg.channel == chir.channel), 'Channels of physical susceptibility and Fermi-bose vertex not consistent'
    niv_urange = dga_conf.box.niv_urange
    sigma = np.zeros((g_generator.nkx, g_generator.nky, g_generator.nkz, niv_urange), dtype=complex)
    if (analyse_spin_fermion):
        sigma_spim = np.zeros((g_generator.nkx, g_generator.nky, g_generator.nkz, niv_urange), dtype=complex)

    if(analyse_spin_fermion):
        vrg_im = vrg_in.imag
        vrg = vrg_in.real
    else:
        vrg = vrg_in

    for iqw in range(qiw_grid.shape[0]):
        wn = qiw_grid[iqw][-1]
        q_ind = qiw_grid[iqw][0]
        q = dga_conf.q_grid.irr_kmesh[:, q_ind]
        qiw = np.append(q, wn)
        gkpq = g_generator.generate_gk_plus(mu=mu, qiw=qiw, niv=niv_urange)
        sigma += (vrg[iqw, niv_urange:][None, None, None, :] * (
                1. - chir.u_r * chir.mat[iqw]) - 1.0 / chir.beta) * gkpq.gk * \
                 dga_conf.q_grid.irrk_count[q_ind]
        if(analyse_spin_fermion):
            sigma_spim += (1j * vrg_im[iqw, niv_urange:][None, None, None, :] * (
                        1. - chir.u_r * chir.mat[iqw])) * gkpq.gk * \
                          dga_conf.q_grid.irrk_count[q_ind]
        if (wn != 0):
            qiw = np.append(q, -wn)
            gkpq = g_generator.generate_gk_plus(mu=mu, qiw=qiw, niv=niv_urange).gk
            sigma += (np.conj(np.flip(vrg[iqw, :], axis=-1)[None, None, None, niv_urange:]) * (
                    1. - chir.u_r * np.conj(chir.mat[iqw])) - 1.0 / chir.beta) * gkpq * \
                     dga_conf.q_grid.irrk_count[q_ind]
            if (analyse_spin_fermion):
                sigma_spim += (-1j * np.flip(vrg_im[iqw, :], axis=-1)[None, None, None, niv_urange:] * (
                            1. - chir.u_r * np.conj(chir.mat[iqw]))) * gkpq * \
                              dga_conf.q_grid.irrk_count[q_ind]

    sigma = - chir.u_r / (2.0) * 1. / (dga_conf.q_grid.nk_tot) * sigma
    if (analyse_spin_fermion):
        sigma_spim = - chir.u_r / (2.0) * 1. / (dga_conf.q_grid.nk_tot) * sigma_spim
    if(analyse_spin_fermion):
        return sigma, sigma_spim
    else:
        return sigma

def sde_chi_kernel(dga_conf: conf.DgaConfig = None, kernel=None, g_generator: twop.GreensFunctionGenerator = None, mu=0, qiw_grid=None):
    ''' Sigma = - \sum_q kernel * G(k-q). Everything is within the kernel here. '''
    niv_urange = dga_conf.box.niv_urange
    sigma = np.zeros((g_generator.nkx, g_generator.nky, g_generator.nkz, niv_urange), dtype=complex)
    for iqw in range(qiw_grid.shape[0]):
        wn = qiw_grid[iqw][-1]
        q_ind = qiw_grid[iqw][0]
        q = dga_conf.q_grid.irr_kmesh[:, q_ind]
        qiw = np.append(q, wn)
        gkpq = g_generator.generate_gk_plus(mu=mu, qiw=qiw, niv=niv_urange)
        sigma += kernel[iqw, niv_urange:] * gkpq.gk * dga_conf.q_grid.irrk_count[q_ind]

        if (wn != 0):
            qiw = np.append(q, -wn)
            gkpq = g_generator.generate_gk_plus(mu=mu, qiw=qiw, niv=niv_urange).gk
            sigma += np.conj(np.flip(kernel[iqw,:], axis=-1)[None, None, None, niv_urange:]) * gkpq * dga_conf.q_grid.irrk_count[q_ind]

    sigma = - 1. / (dga_conf.q_grid.nk_tot) * sigma
    return sigma

def get_full_dga_kernel(vrg_d=None,vrg_m=None,chi_d:fp.LadderSusceptibility=None,chi_m=None):
    return chi_d.u/2. * (vrg_d.mat * (1 - chi_d.u_r * chi_d.mat[:,None]) - 3. * vrg_m.mat * (1 - chi_m.u_r * chi_m.mat[:,None]) - 2./chi_d.beta)

def get_dga_kernel_dens(vrg_d=None,chi_d:fp.LadderSusceptibility=None):
    return chi_d.u_r/2. * (vrg_d.mat * (1 - chi_d.u_r * chi_d.mat[:,None]) - 1./chi_d.beta)

def get_dga_kernel_magn(vrg_m=None,chi_m=None):
    return chi_m.u_r/2. * (vrg_m.mat * (1 - chi_m.u_r * chi_m.mat[:,None]) - 1./chi_m.beta)


def sde_dga_wrapper(dga_conf: conf.DgaConfig = None, f_loc_chi0=None, vrg=None, chi=None, qiw_mesh=None, sigma_input=None, mu_input = None,
                    distributor=None):
    g_generator = twop.GreensFunctionGenerator(beta=dga_conf.sys.beta, kgrid=dga_conf.k_grid, hr=dga_conf.sys.hr,
                                               sigma=sigma_input)

    kernel = get_dga_kernel_dens(vrg_d=vrg['dens'], chi_d=chi['dens'])
    sigma_dens = sde_chi_kernel(dga_conf = dga_conf, kernel=kernel, g_generator = g_generator, mu=mu_input, qiw_grid=qiw_mesh)

    kernel = get_dga_kernel_magn(vrg_m=vrg['magn'], chi_m=chi['magn'])
    sigma_magn = sde_chi_kernel(dga_conf = dga_conf, kernel=kernel, g_generator = g_generator, mu=mu_input, qiw_grid=qiw_mesh)

    # kernel = dga_conf.sys.u/dga_conf.sys.beta * 0.5 *(f_loc_chi0['dens'].mat - f_loc_chi0['magn'].mat)
    # sigma_dc_corr = sde_chi_kernel(dga_conf = dga_conf, kernel=kernel, g_generator = g_generator, mu=mu_input, qiw_grid=qiw_mesh)

    #sigma_dc_corr = reduce_and_symmetrize_fbz(dga_conf=dga_conf, mat=sigma_dc_corr, distributor=distributor)
    sigma_dens = reduce_and_symmetrize_fbz(dga_conf=dga_conf, mat=sigma_dens, distributor=distributor)
    sigma_magn = reduce_and_symmetrize_fbz(dga_conf=dga_conf, mat=sigma_magn, distributor=distributor)


    sigma = {
        'dens': sigma_dens,
        'magn': sigma_magn
    }

    return sigma


def reduce_and_symmetrize_fbz(dga_conf: conf.DgaConfig = None, mat=None, distributor=None):
    ''' Reduce the results from the different ranks and create full {k,iv}.'''
    mat = distributor.allreduce(mat)
    mat = mf.vplus2vfull(mat=mat)
    mat = dga_conf.k_grid.symmetrize_irrk(mat=mat)
    return mat


def rpa_sde(dga_conf=None, chir: fp.LocalSusceptibility = None, g_generator: twop.GreensFunctionGenerator = None,
            niv_giw=None, mu=0, qiw_grid=None):
    u_r = fp.get_ur(u=dga_conf.sys.u, channel=chir.channel)
    sigma = np.zeros((g_generator.nkx, g_generator.nky, g_generator.nkz, niv_giw), dtype=complex)
    for iqw in range(qiw_grid.shape[0]):
        wn = qiw_grid[iqw][-1]
        q_ind = qiw_grid[iqw][0]
        q = dga_conf.q_grid.irr_kmesh[:, q_ind]
        qiw = np.append(q, wn)
        gkpq = g_generator.generate_gk_plus(mu=mu, qiw=qiw, niv=niv_giw)
        sigma += chir.mat[iqw, None] * gkpq.gk * dga_conf.q_grid.irrk_count[q_ind]
        if (wn != 0):
            qiw = np.append(q, -wn)
            gkpq = g_generator.generate_gk_plus(mu=mu, qiw=qiw, niv=niv_giw)
            sigma += np.conj(chir.mat[iqw, None]) * gkpq.gk * dga_conf.q_grid.irrk_count[q_ind]

    sigma = u_r ** 2 / (2. * chir.beta) * 1. / (dga_conf.q_grid.nk_tot) * sigma
    return sigma


def rpa_sde_wrapper(dga_conf: conf.DgaConfig = None, sigma_input=None, mu_input=None, chi=None, qiw_grid=None, distributor=None):
    g_generator = twop.GreensFunctionGenerator(beta=dga_conf.sys.beta, kgrid=dga_conf.k_grid, hr=dga_conf.sys.hr,
                                               sigma=sigma_input)
    sigma_dens = rpa_sde(dga_conf=dga_conf, chir=chi['dens'], g_generator=g_generator, niv_giw=dga_conf.box.niv_urange,
                         mu=mu_input, qiw_grid=qiw_grid.my_mesh)
    sigma_magn = rpa_sde(dga_conf=dga_conf, chir=chi['magn'], g_generator=g_generator, niv_giw=dga_conf.box.niv_urange,
                         mu=mu_input, qiw_grid=qiw_grid.my_mesh)

    sigma_dens = reduce_and_symmetrize_fbz(dga_conf=dga_conf, mat=sigma_dens, distributor=distributor)
    sigma_magn = reduce_and_symmetrize_fbz(dga_conf=dga_conf, mat=sigma_magn, distributor=distributor)

    sigma = {
        'dens': sigma_dens,
        'magn': sigma_magn
    }
    return sigma


def build_dga_sigma(dga_conf: conf.DgaConfig = None, sigma_dga=None, sigma_rpa=None, dmft_sde=None, sigma_dmft=None):
    if (dga_conf.box.wn_rpa.size > 0):
        sigma_dga['dens'] = sigma_dga['dens'] + sigma_rpa['dens']
        sigma_dga['magn'] = sigma_dga['magn'] + sigma_rpa['magn']

    niv1 = sigma_dga['dens'].shape[-1] // 2
    niv2 = sigma_dmft.shape[-1] // 2
    sigma_dmft_clip = sigma_dmft[niv2-niv1:niv2+niv1]
    niv_urange = dga_conf.box.niv_urange
    niv_full = dga_conf.box.niv_urange + dga_conf.box.niw_core

    sigma = np.zeros((dga_conf.k_grid.nk) + (2*niv_full,), dtype=complex)
    sigma_nc = np.zeros((dga_conf.k_grid.nk) + (2*niv_full,), dtype=complex)
    sigma[:,:,:,:] = sigma_dmft[None,None,None,niv2-niv_full:niv2+niv_full]
    sigma_nc[:,:,:,:] = sigma_dmft[None,None,None,niv2-niv_full:niv2+niv_full]
    sigma[:,:,:,niv_full-niv_urange:niv_full+niv_urange] = -1*sigma_dga['dens'] + 3 * sigma_dga['magn'] + dmft_sde['hartree'] -  2*dmft_sde[
        'magn'] +  2*dmft_sde['dens']  - dmft_sde['siw'] + sigma_dmft_clip

    sigma_nc[:,:,:,niv_full-niv_urange:niv_full+niv_urange] = sigma_dga['dens'] + 3 * sigma_dga['magn'] - \
                            2*dmft_sde['magn'] + dmft_sde['hartree'] - dmft_sde['siw'] + sigma_dmft_clip

    return sigma, sigma_nc


# ---------------------------------------------- MPI FUNCTIONS ---------------------------------------------------------

# -------------------------------------------- WRAPPER FUNCTIONS -------------------------------------------------------

# ======================================================================================================================
def local_rpa_sde_correction(dga_conf: conf.DgaConfig=None, giw=None, box_sizes=None, iw=None):
    beta = dga_conf.sys.beta
    u = dga_conf.sys.u

    niv_urange = box_sizes.niv_urange

    chi0_urange = fp.LocalBubble(giw=giw, beta=beta, niv_sum=niv_urange, iw=iw)

    chi_rpa_dens = fp.local_rpa_susceptibility(chi0_urange=chi0_urange, channel='dens', u=u)
    chi_rpa_magn = fp.local_rpa_susceptibility(chi0_urange=chi0_urange, channel='magn', u=u)

    siw_rpa_dens = local_rpa_sde(chir=chi_rpa_dens, niv_giw=niv_urange, giw=giw, u=u)
    siw_rpa_magn = local_rpa_sde(chir=chi_rpa_magn, niv_giw=niv_urange, giw=giw, u=u)

    rpa_sde = {
        'dens': siw_rpa_dens,
        'magn': siw_rpa_magn
    }

    chi_rpa = {
        'dens': chi_rpa_dens,
        'magn': chi_rpa_magn
    }

    return rpa_sde, chi_rpa


# ======================================================================================================================


# ======================================================================================================================
def local_dmft_sde_from_g2(dmft_input=None, box_sizes=None, g2_dens=None, g2_magn=None):
    giw = dmft_input['gloc']
    beta = dmft_input['beta']
    u = dmft_input['u']
    n = dmft_input['n']

    iw = g2_dens.wn

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
