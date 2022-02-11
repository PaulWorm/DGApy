# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Contains wrapper functions for local routines


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import Config as conf
import FourPoint as fp
import SDE as sde
import numpy as np

# ---------------------------------------------- FUNCTIONS -------------------------------------------------------------

def add_rpa_correction(dmft_sde=None,rpa_sde_loc=None,wn_rpa = None, sigma_comp=None):
    if (np.size(wn_rpa) > 0):
        dmft_sde['dens'] = dmft_sde['dens'] + rpa_sde_loc['dens']
        dmft_sde['magn'] = dmft_sde['magn'] + rpa_sde_loc['magn']
        dmft_sde['siw'] = dmft_sde['dens'] + dmft_sde['magn'] + dmft_sde['hartree']
        if sigma_comp['dens_re'] is not None: sigma_comp['dens_re'] = sigma_comp['dens_re'] + rpa_sde_loc['dens']
        if sigma_comp['magn_re'] is not None: sigma_comp['magn_re'] = sigma_comp['magn_re'] + rpa_sde_loc['magn']

    else:
        dmft_sde['siw'] = dmft_sde['siw'] + dmft_sde['hartree']
    return dmft_sde

def local_dmft_sde_from_gamma(dga_conf: conf.DgaConfig=None, giw=None, gamma_dmft=None):
    '''

    :param dga_conf: DGA config object
    :param giw: Local dmft green's function
    :param gamma_dens: irreducible vertex in the density channel (local)
    :param gamma_magn: -||- only for magnetic
    :return: Output of the local schwinger dyson equation
    '''
    beta = dga_conf.sys.beta
    u = dga_conf.sys.u
    n = dga_conf.sys.n

    iw = dga_conf.box.wn_core

    niv_core = dga_conf.box.niv_core
    niv_urange = dga_conf.box.niv_urange

    chi0_core = fp.LocalBubble(giw=giw, beta=beta, niv_sum=niv_core, iw=iw)
    chi0_urange = fp.LocalBubble(giw=giw, beta=beta, niv_sum=niv_urange, iw=iw)

    gchi_aux_dens_loc = fp.local_gchi_aux_from_gammar(gammar=gamma_dmft['dens'], gchi0_core=chi0_core, u=u)
    gchi_aux_magn_loc = fp.local_gchi_aux_from_gammar(gammar=gamma_dmft['magn'], gchi0_core=chi0_core, u=u)

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

    if(dga_conf.opt.analyse_spin_fermion_contributions):
        siw_dens_re = sde.local_dmft_sde(vrg=vrg_dens_loc.mat.real, chir=chi_dens_urange_loc, u=u)
        siw_dens_im = sde.local_dmft_sde(vrg=1j*vrg_dens_loc.mat.imag, chir=chi_dens_urange_loc, u=u,scal_const=0.0)
        siw_magn_re = sde.local_dmft_sde(vrg=vrg_magn_loc.mat.real, chir=chi_magn_urange_loc, u=u)
        siw_magn_im = sde.local_dmft_sde(vrg=1j*vrg_magn_loc.mat.imag, chir=chi_magn_urange_loc, u=u,scal_const=0.0)

        siw_dens = siw_dens_re + siw_dens_im
        siw_magn = siw_magn_re + siw_magn_im

    else:
        siw_dens = sde.local_dmft_sde(vrg=vrg_dens_loc.mat, chir=chi_dens_urange_loc, u=u)
        siw_magn = sde.local_dmft_sde(vrg=vrg_magn_loc.mat, chir=chi_magn_urange_loc, u=u)

        siw_dens_re = None
        siw_dens_im = None
        siw_magn_re = None
        siw_magn_im = None

    siw = siw_dens + siw_magn

    dmft_sde = {
        'dens': siw_dens,
        'magn': siw_magn,
        'siw': siw,
        'hartree': u / 2. * n
    }
    vrg_dmft = {
        'dens': vrg_dens_loc,
        'magn': vrg_magn_loc
    }
    chi_dmft = {
        'dens': chi_dens_urange_loc,
        'magn': chi_magn_urange_loc,
    }

    sigma_components = {
        'dens_re': siw_dens_re,
        'dens_im': siw_dens_im,
        'magn_re': siw_magn_re,
        'magn_im': siw_magn_im
    }

    return dmft_sde, chi_dmft, vrg_dmft, sigma_components
