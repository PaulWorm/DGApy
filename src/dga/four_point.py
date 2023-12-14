# ------------------------------------------------ COMMENTS ------------------------------------------------------------
'''
    This module contains the functions to compute the four-point lattice vertex and the susceptibility.
'''

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import re
import gc

import numpy as np
import h5py

from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import config as conf
from dga import bubble as bub
from dga import local_four_point as lfp
from dga import mpi_aux
from dga import plotting


# ----------------------------------------------- SCHWINGER DYSON EQUATION -------------------------------------------------------

def schwinger_dyson_vrg_q(vrg, chir_phys, giwk, beta, u, channel, q_list, q_point_duplicity, wn, nqtot):
    ''' Solve the Schwinger Dyson equation'''
    u_r = lfp.get_ur(u, channel=channel)
    niv_vrg = np.shape(vrg)[-1] // 2
    sigma_f = np.zeros([*np.shape(giwk)[:3], 2 * niv_vrg], dtype=complex)
    for i, q in enumerate(q_list):
        gkpq = bz.shift_mat_by_ind(giwk, ind=[-iq for iq in q])
        for j, iwn in enumerate(wn):
            gkpw_wn_shift = mf.cut_iv_with_iw_shift(gkpq, niv_vrg, iwn)
            sigma_f += (1 - (1 - u_r * chir_phys[i, j, None, None, None, None]) *
                        vrg[i, j, None, None, None, :]) * gkpw_wn_shift * q_point_duplicity[i]
    return 1 / nqtot * u_r / 2 * 1 / beta * sigma_f


def schwinger_dyson_q_shell(chir_phys, giwk, beta, u, n_shell, n_core, wn, q_list, nqtot):
    sigma_f = np.zeros([*np.shape(giwk)[:3], 2 * n_shell], dtype=complex)
    for i, q in enumerate(q_list):
        gkpq = bz.shift_mat_by_ind(giwk, ind=[-iq for iq in q])
        mat_grid = mf.wn_slices_shell(gkpq, n_shell, n_core, w=wn)
        sigma_f += 1 / nqtot * u ** 2 / 2 * 1 / beta * np.sum(chir_phys[i, :, None, None, None, None] * mat_grid, axis=0)
    return sigma_f


def schwinger_dyson_channel_q(vrg, chir_phys, channel, giwk, beta, u, q_list, q_point_duplicity, wn, nqtot, niv_shell=0):
    siwk_core = schwinger_dyson_vrg_q(vrg, chir_phys, giwk, beta, u, channel, q_list, q_point_duplicity, wn, nqtot)
    if niv_shell == 0:
        return siwk_core
    else:
        niv_core = np.shape(vrg)[-1] // 2
        siwk_shell = schwinger_dyson_q_shell(chir_phys, giwk, beta, u, niv_shell, niv_core, wn, q_list, nqtot)
        return mf.concatenate_core_asmypt(siwk_core, siwk_shell)


def schwinger_dyson_full_q(vrg_dens, vrg_magn, chi_dens, chi_magn, kernel_dc, giwk, beta, u, q_list, wn, nqtot,
                           niv_shell=0, logger=None):
    kernel = get_kernel(vrg_dens, chi_dens, u, 'dens')
    kernel += 3 * get_kernel(vrg_magn, chi_magn, u, 'magn')
    kernel -= kernel_dc  # minus because we subtract the double counting part
    siwk_core = schwinger_dyson_kernel_q(kernel, giwk, beta, q_list, wn, nqtot)
    if logger is not None: logger.log_cpu_time(task=' SDE solved in Core. ')

    if niv_shell == 0:
        return siwk_core
    else:
        niv_core = np.shape(siwk_core)[-1] // 2
        siwk_shell = schwinger_dyson_q_shell(chi_dens, giwk, beta, u, niv_shell, niv_core, wn, q_list, nqtot)
        siwk_shell += schwinger_dyson_q_shell(chi_magn, giwk, beta, u, niv_shell, niv_core, wn, q_list, nqtot)
        return mf.concatenate_core_asmypt(siwk_core, siwk_shell)


def get_kernel(vrg, chi_phys, u, channel):
    ''' Kernel for the Schwinger Dyson equation in Hedin form:
        kernel = u_r/2 * (1 - (1 - u_r * chi_phys[q,w]) * vrg[q,w,v])
    '''
    u_r = lfp.get_ur(u, channel)
    return u_r / 2 * (1 - (1 - u_r * chi_phys[:, :, None]) * vrg)


def get_kernel_dc(f: lfp.LocalFourPoint, gchi0_core):
    nq = np.shape(gchi0_core)[0]
    niw = np.shape(gchi0_core)[1]
    niv = np.shape(gchi0_core)[2]
    kernel = np.zeros((nq, niw, niv), dtype=complex)
    for iq in range(nq):
        for iw in range(niw):
            kernel[iq, iw, :] = f.u_r * np.sum(gchi0_core[iq, iw, None, :] * f.mat[None, iw, ...], axis=-1)
    return kernel  # 1/beta is contained in the SDE


def schwinger_dyson_kernel_q(kernel, giwk, beta, q_list, wn, nqtot):
    niv = np.shape(kernel)[-1] // 2
    sigma_f = np.zeros([*np.shape(giwk)[:3], 2 * niv], dtype=complex)
    giwk = mf.cut_v(giwk, niv_cut=niv + np.max(np.abs(wn)), axes=-1)
    for i, q in enumerate(q_list):
        gkpq = bz.shift_mat_by_ind(giwk, ind=[-iq for iq in q])
        for j, iwn in enumerate(wn):
            gkpq_iwn_shift = mf.cut_iv_with_iw_shift(gkpq, niv, iwn)
            sigma_f += kernel[i, j, None, None, None, :] * gkpq_iwn_shift
    return 1 / nqtot * 1 / beta * sigma_f


def schwinger_dyson_dc(gchiq_fupdo, giwk, u, q_list, q_point_duplicity, wn, nqtot):
    '''
        two beta prefactors are contained in F. Additional minus sign from usign gchi0_q
    '''
    niv_core = np.shape(gchiq_fupdo)[-1] // 2
    sigma_dc = np.zeros([*np.shape(giwk)[:3], 2 * niv_core], dtype=complex)
    for iq, q in enumerate(q_list):
        gkpq = bz.shift_mat_by_ind(giwk, ind=[-iq for iq in q])
        mat_grid = mf.wn_slices_gen(gkpq, niv_core, w=wn)
        sigma_dc += + u * 1 / (nqtot) * q_point_duplicity[iq] * np.sum(gchiq_fupdo[iq, :, None, None, None, :] * mat_grid, axis=0)
    return sigma_dc


def schwinger_dyson_shell(chir_phys, giw, beta, u, n_shell, n_core, wn):
    mat_grid = mf.wn_slices_shell(giw, n_shell=n_shell, n_core=n_core, w=wn)
    sigma_f = u ** 2 / 2 * 1 / beta * np.sum((chir_phys[:, None]) * mat_grid, axis=0)
    return sigma_f


# ----------------------------------------------- FOUR POINT FUNCTIONS -----------------------------------------------------------

def get_gchir_from_gamma_loc_q(gammar: lfp.LocalFourPoint = None, gchi0_q=None):
    '''
        Compute the non-local suscptibility using the BSE: chi_r = (chi_0^(-1) + Gamma_r/beta^2)^(-1)
        q_list is distributed among cores.

        WARNING: this assumes that gammar is computed with niv_shell = 0 (i.e. no urange correction)
    '''
    nq = np.shape(gchi0_q)[0]
    nw = np.shape(gchi0_q)[1]
    chir = np.zeros([nq, *gammar.mat.shape], dtype=complex)
    for iq in range(nq):
        for iwn in range(nw):
            chir[iq, iwn, ...] = np.linalg.inv(np.diag(1 / gchi0_q[iq, iwn]) + gammar.mat[iwn])
    return chir


def get_gchir_aux_from_gammar_q(gammar: lfp.LocalFourPoint, gchi0):
    '''
        gchi_aux = (gchi0^(-1) + Gamma/beta^2 - u/beta^2)^(-1)
        [chi_aux]:eV^(-3) = [[gchi0^(-1)]:eV^3 + [Gamma]:eV[1/beta^2]:eV^2 - [u]:eV [1/beta^2]:eV^2]^(-1):eV^(-3)
    '''
    nq = np.shape(gchi0)[0]
    nw = np.shape(gchi0)[1]
    chir_aux = np.zeros([nq, *gammar.mat.shape], dtype=complex)
    for iq in range(nq):
        for iwn in range(nw):
            chir_aux[iq, iwn, ...] = np.linalg.inv(np.diag(1 / gchi0[iq, iwn]) + gammar.mat[iwn] - gammar.u_r / gammar.beta ** 2)
    return chir_aux


def chi_phys_from_chi_aux_q(chi_aux, chi0q_urange, chi0q_core, u, channel):
    ''' chi_phys = ((chi_aux + chi0_urange - chi0_core)^(-1) + u_r)^(-1) '''
    u_r = lfp.get_ur(u, channel)
    chir = 1. / (1. / (chi_aux + chi0q_urange - chi0q_core) + u_r)
    return chir


def chi_phys_asympt_q(chir_urange, chi0_urange, chi0_asympt):
    ''' asymptotic form of the susceptibility '''
    return chir_urange + chi0_asympt - chi0_urange


def vrg_from_gchi_aux_asympt(gchir_aux, gchi0_core, chir_urange, chir_asympt, u, channel):
    '''Note: 1/beta is here included in vrg compared to the old code
        [vrg]:1 = [1/gchi0]:eV^3 [chi_aux]:1/eV^3 *[...]:1
    '''
    u_r = lfp.get_ur(u, channel=channel)
    nq = np.shape(gchir_aux)[0]
    niw = np.shape(gchir_aux)[1]
    niv = np.shape(gchir_aux)[2]
    vrg = np.zeros([nq, niw, niv], dtype=complex)
    for iq in range(nq):
        for iwn in range(niw):
            vrg[iq, iwn, :] = 1 / gchi0_core[iq, iwn] * np.sum(gchir_aux[iq, iwn], axis=-1) * (1 - u_r * chir_urange[iq, iwn]) / (
                    1 - u_r * chir_asympt[iq, iwn])
    return vrg


def vrg_from_gchi_aux(gchir_aux, gchi0_core):
    '''Note: 1/beta is here included in vrg compared to the old code
        [vrg]:1 = [1/gchi0]:eV^3 [chi_aux]:1/eV^3 *[...]:1
    '''
    nq = np.shape(gchir_aux)[0]
    niw = np.shape(gchir_aux)[1]
    niv = np.shape(gchir_aux)[2]
    vrg = np.zeros([nq, niw, niv], dtype=complex)
    for iq in range(nq):
        for iwn in range(niw):
            vrg[iq, iwn, :] = 1 / gchi0_core[iq, iwn] * np.sum(gchir_aux[iq, iwn], axis=-1)
    return vrg


def lam_from_chir_q(gchir, gchi0, channel):
    ''' lambda vertex'''
    sign = lfp.get_sign(channel)
    lam = -sign * (1 - np.sum(gchir * (1 / gchi0)[..., None], axis=-1))
    return lam


def get_lam_tilde(lam_core, chi0q_shell, u, channel):
    u_r = lfp.get_ur(u, channel)
    return (lam_core - u * chi0q_shell[..., None]) / (1 + u_r * chi0q_shell[..., None])


def get_chir_shell(lam_tilde, chi0q_shell, gchi0q_core, beta, u, channel):
    sign = lfp.get_sign(channel)
    chir_shell = -sign * u * chi0q_shell ** 2 \
                 + chi0q_shell * (1 - 2 * u * 1 / beta ** 2 * np.sum((lam_tilde + sign) * gchi0q_core, axis=-1))
    return chir_shell


def chir_tilde(chir_core, lam_tilde, chi0q_shell, gchi0q_core, beta, u, channel):
    chir_shell = get_chir_shell(lam_tilde, chi0q_shell, gchi0q_core, beta, u, channel)
    return (chir_core + chir_shell) / (1 - (u * chi0q_shell) ** 2)


def vrg_q_tilde(lam_tilde, chir_q_tilde, u, channel):
    u_r = lfp.get_ur(u, channel)
    sign = lfp.get_sign(channel)
    return (1 + sign * lam_tilde) / (1 - u_r * chir_q_tilde[..., None])


def get_vrg_and_chir_lad_from_gammar_uasympt_q(gamma_dens: lfp.LocalFourPoint, gamma_magn: lfp.LocalFourPoint,
                                               f_dc, vrg_magn_loc, chi_magn_loc,
                                               bubble_gen: bub.BubbleGenerator, u, my_q_list,
                                               niv_shell=0, logger=None):
    '''
        Compute the fermi-bose vertex and susceptibility using the asymptotics proposed in
        Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005
    '''
    niv_core = gamma_dens.niv
    niv_full = niv_core + niv_shell
    beta = gamma_dens.beta

    # Build the different non-local Bubbles:
    gchi0_q_urange = bubble_gen.get_gchi0_q_list(niv_full, my_q_list)
    chi0_q_urange = 1 / beta ** 2 * np.sum(gchi0_q_urange, axis=-1)
    gchi0_q_core = mf.cut_v(gchi0_q_urange, niv_cut=niv_core, axes=-1)
    chi0_q_core = 1 / beta ** 2 * np.sum(gchi0_q_core, axis=-1)
    chi0q_shell = bubble_gen.get_asymptotic_correction_q(niv_full, my_q_list)
    chi0q_shell_dc = bubble_gen.get_asymptotic_correction_q(niv_full, my_q_list)
    if logger is not None: logger.log_cpu_time(task=' Bubbles constructed. ')

    # double-counting kernel:
    if logger is not None:
        if logger.is_root:
            f_dc.plot(pdir=logger.out_dir + '/', name='F_dc')

    kernel_dc = mf.cut_v(get_kernel_dc(f_dc.mat, gchi0_q_urange), niv_core, axes=(-1,))
    if logger is not None: logger.log_cpu_time(task=' DC kernel constructed. ')

    # Density channel:
    gchiq_aux = get_gchir_aux_from_gammar_q(gamma_dens, gchi0_q_core)
    chiq_aux = 1 / beta ** 2 * np.sum(gchiq_aux, axis=(-1, -2))
    chi_lad_urange = chi_phys_from_chi_aux_q(chiq_aux, chi0_q_urange, chi0_q_core, u, gamma_dens.channel)
    chi_lad_dens = chi_phys_asympt_q(chi_lad_urange, chi0_q_urange, chi0_q_urange + chi0q_shell)

    vrg_q_dens = vrg_from_gchi_aux_asympt(gchiq_aux, gchi0_q_core, chi_lad_urange, chi_lad_dens, u, gamma_dens.channel)

    # Magnetic channel:
    gchiq_aux = get_gchir_aux_from_gammar_q(gamma_magn, gchi0_q_core)
    chiq_aux = 1 / beta ** 2 * np.sum(gchiq_aux, axis=(-1, -2))
    chi_lad_urange = chi_phys_from_chi_aux_q(chiq_aux, chi0_q_urange, chi0_q_core, u, gamma_magn.channel)
    chi_lad_magn = chi_phys_asympt_q(chi_lad_urange, chi0_q_urange, chi0_q_urange + chi0q_shell)

    u_r = lfp.get_ur(u, gamma_magn.channel)
    # 1/beta**2 since we want F/beta**2
    kernel_dc += u_r / gamma_magn.beta * (1 - u_r * chi_magn_loc[None, :, None]) * vrg_magn_loc.mat[None, :, :] * chi0q_shell_dc[
                                                                                                                  :, :, None]
    vrg_q_magn = vrg_from_gchi_aux_asympt(gchiq_aux, gchi0_q_core, chi_lad_urange, chi_lad_magn, u, gamma_magn.channel)

    return vrg_q_dens, vrg_q_magn, chi_lad_dens, chi_lad_magn, kernel_dc


# ----------------------------------------------- BUILD Fq VERTEX ----------------------------------------------------------------
def ladder_vertex_from_chi_aux_components(gchi_aux=None, vrg=None, gchi0=None, beta=None, u_r=None):
    '''
        Look in supplemental of Phys. Rev. B 99, 041115(R) (S.5).

        F(v,vp) = f1 + (1-u_r chi_r(v)) f2
        = beta^2 (gchi0(v))^(-1) [delta(v,vp) - chi_aux(v,vp) gchi0(v)^(-1)] + u_r (1-u_r chi_r(v)) vrg(v) vrg(vp)

        beta**2 originates from units:
        [F]:eV = [beta^2]:1/eV^2 [1/gchi0]:eV^3 [1 - [gchi_aux]:1/eV^3 [1/gchi0]:eV^3 ]
        + [u_r]:eV [1-[u_r]:eV [chi_r]:1/eV]) [vrg]:1 [vrg]:1
     '''
    f1 = beta ** 2 * 1. / gchi0[..., :, None] * (np.eye(gchi0.shape[-1]) - gchi_aux * 1. / gchi0[..., None, :])
    f2 = u_r * vrg[..., :, None] * vrg[..., None, :]
    return f1, f2


def write_vertex_components(d_cfg: conf.DgaConfig, mpi_distributor: mpi_aux.MpiDistributor, channel, gchiq_aux,
                            vrg_q, gchi0_q_core, name='fq_'):
    with mpi_distributor as file:
        for i, iq in enumerate(mpi_distributor.my_tasks):
            for j, iw in enumerate(d_cfg.box.wn):
                f1_slice, f2_slice = ladder_vertex_from_chi_aux_components(gchi_aux=gchiq_aux[i, j],
                                                                           vrg=vrg_q[i, j],
                                                                           gchi0=gchi0_q_core[i, j],
                                                                           beta=d_cfg.sys.beta,
                                                                           u_r=lfp.get_ur(d_cfg.sys.u, channel))
                group = '/' + name + f'irrq{iq:03d}wn{iw:04d}/'

                file[group + f'f1_{channel}/'] = f1_slice
                file[group + f'f2_{channel}/'] = f2_slice


def load_vertex_from_rank_files(output_path, name, mpi_size, nq, niw, niv):
    # Collect data from subfiles (This is quite ugly, as it is hardcoded to my structure.
    # This should be replaced by a general routine):

    # NOTE: they need to be predefined due to the different mpi files
    f1_magn = np.zeros((nq, 2 * niw + 1, 2 * niv, 2 * niv), dtype=complex)
    f2_magn = np.zeros((nq, 2 * niw + 1, 2 * niv, 2 * niv), dtype=complex)
    f1_dens = np.zeros((nq, 2 * niw + 1, 2 * niv, 2 * niv), dtype=complex)
    f2_dens = np.zeros((nq, 2 * niw + 1, 2 * niv, 2 * niv), dtype=complex)

    for ir in range(mpi_size):
        fname = output_path + name + f'Rank{ir:05d}' + '.hdf5'
        with h5py.File(fname, 'r') as file_in:
            keys = (key for key in file_in.keys() if key.startswith('fq_'))  # the group is still hard-coded!
            for key1 in keys:
                # extract the q indizes from the group name!
                irrq = np.array(re.findall(r'\d+', key1), dtype=int)[0]
                iwn = np.array(re.findall(r'\d+', key1), dtype=int)[1] + niw
                f1_magn[irrq, iwn, ...] = file_in.file[key1 + '/f1_magn/'][()]
                f2_dens[irrq, iwn, ...] = file_in.file[key1 + '/f2_dens/'][()]
                f1_dens[irrq, iwn, ...] = file_in.file[key1 + '/f1_dens/'][()]
                f2_magn[irrq, iwn, ...] = file_in.file[key1 + '/f2_magn/'][()]
    return f1_magn, f2_magn, f1_dens, f2_dens


def build_vertex_fq(d_cfg: conf.DgaConfig, comm, chi_lad_magn, chi_lad_dens):
    '''
        Build the ladder vertex from the lambda corrected susceptibility and the precomputed parts.
    '''
    niw = d_cfg.box.niw_core
    niv = d_cfg.box.niv_core
    f1_magn, f2_magn, f1_dens, f2_dens = load_vertex_from_rank_files(output_path=d_cfg.output_path, name='Q',
                                                                     mpi_size=comm.size,
                                                                     nq=d_cfg.lattice.q_grid.nk_irr,
                                                                     niw=niw, niv=niv)

    f1_magn = d_cfg.lattice.q_grid.map_irrk2fbz(f1_magn, shape='mesh')
    f2_magn = d_cfg.lattice.q_grid.map_irrk2fbz(f2_magn, shape='mesh')
    f_magn = f1_magn + (1 + d_cfg.sys.u * chi_lad_magn[..., None, None]) * f2_magn

    f1_dens = d_cfg.lattice.q_grid.map_irrk2fbz(f1_dens, shape='mesh')
    f2_dens = d_cfg.lattice.q_grid.map_irrk2fbz(f2_dens, shape='mesh')
    f_dens = f1_dens + (1 - d_cfg.sys.u * chi_lad_dens[..., None, None]) * f2_dens

    del f1_magn, f2_magn, f1_dens, f2_dens
    gc.collect()
    niw = f_dens.shape[-3] // 2
    plotting.plot_kx_ky(f_dens[:, :, 0, niw, niv, niv],
                        d_cfg.lattice.k_grid.kx,
                        d_cfg.lattice.k_grid.ky, pdir=d_cfg.output_path, name='F_dens_niw0')

    plotting.plot_kx_ky(f_magn[:, :, 0, niw, niv, niv],
                        d_cfg.lattice.k_grid.kx,
                        d_cfg.lattice.k_grid.ky, pdir=d_cfg.output_path, name='F_magn_niw0')
    d_cfg.save_data(f_dens, 'F_dens')
    d_cfg.save_data(f_magn, 'F_magn')
    del f_dens, f_magn
    gc.collect()


if __name__ == '__main__':
    pass
