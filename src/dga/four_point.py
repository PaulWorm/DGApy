# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import copy
import numpy as np

import dga.indizes as ind
import dga.matsubara_frequencies as mf
import dga.config as conf
import dga.local_four_point as lfp
import dga.bubble as bub
import dga.two_point as twop
import dga.brillouin_zone as bz


# ----------------------------------------------- REFACTOR CODE ---------------------------------------------------------

def schwinger_dyson_vrg_q(vrg, chir_phys, giwk, beta, u, channel, q_list, q_point_duplicity, wn, nqtot):
    ''' Solve the Schwinger Dyson equation'''
    u_r = get_ur(u, channel=channel)
    niv_vrg = np.shape(vrg)[-1] // 2
    sigma_F = np.zeros([*np.shape(giwk)[:3], 2 * niv_vrg], dtype=complex)
    for i, q in enumerate(q_list):
        gkpq = bz.shift_mat_by_ind(giwk, ind=[-iq for iq in q])
        for j,iwn in enumerate(wn):
            gkpw_wn_shift = mf.cut_iv_with_iw_shift(gkpq,niv_vrg,iwn)
            sigma_F +=  (1 - (1 - u_r * chir_phys[i, j, None, None, None, None]) *
                       vrg[i, j, None, None, None, :]) * gkpw_wn_shift * q_point_duplicity[i]
    return 1 / nqtot * u_r / 2 * 1 / beta *sigma_F


def schwinger_dyson_q_shell(chir_phys, giwk, beta, u, n_shell, n_core, wn, q_list, nqtot):
    sigma_F = np.zeros([*np.shape(giwk)[:3], 2 * n_shell], dtype=complex)
    for i, q in enumerate(q_list):
        gkpq = bz.shift_mat_by_ind(giwk, ind=[-iq for iq in q])
        mat_grid = mf.wn_slices_shell(gkpq, n_shell, n_core, wn=wn)
        sigma_F += 1 / nqtot * u ** 2 / 2 * 1 / beta * np.sum(chir_phys[i, :, None, None, None, None] * mat_grid, axis=0)
    return sigma_F


def schwinger_dyson_channel_q(vrg, chir_phys, channel, giwk, beta, u, q_list, q_point_duplicity, wn, nqtot, niv_shell=0):
    siwk_core = schwinger_dyson_vrg_q(vrg, chir_phys, giwk, beta, u, channel, q_list, q_point_duplicity, wn, nqtot)
    if (niv_shell == 0):
        return siwk_core
    else:
        niv_core = np.shape(vrg)[-1] // 2
        siwk_shell = schwinger_dyson_q_shell(chir_phys, giwk, beta, u, niv_shell, niv_core, wn, q_list, q_point_duplicity, nqtot)
        return mf.concatenate_core_asmypt(siwk_core, siwk_shell)


def schwinger_dyson_full_q(vrg_dens, vrg_magn, chi_dens, chi_magn, kernel_dc, giwk, beta, u, q_list, wn, nqtot,
                           niv_shell=0, logger=None):
    print(f'vrg-type: {vrg_dens.dtype}; chi-type: {chi_dens.dtype}')
    kernel = get_kernel(vrg_dens, chi_dens, u, 'dens')
    kernel += 3 * get_kernel(vrg_magn, chi_magn, u, 'magn')
    kernel -= kernel_dc  # minus because we subtract the double counting part
    siwk_core = schwinger_dyson_kernel_q(kernel, giwk, beta, q_list, wn, nqtot)
    if (logger is not None): logger.log_cpu_time(task=' SDE solved in Core. ')

    if (niv_shell == 0):
        return siwk_core
    else:
        niv_core = np.shape(siwk_core)[-1] // 2
        siwk_shell = schwinger_dyson_q_shell(chi_dens, giwk, beta, u, niv_shell, niv_core, wn, q_list, nqtot)
        siwk_shell += schwinger_dyson_q_shell(chi_magn, giwk, beta, u, niv_shell, niv_core, wn, q_list, nqtot)
        return mf.concatenate_core_asmypt(siwk_core, siwk_shell)


def get_kernel(vrg, chi_phys, u, channel):
    u_r = get_ur(u, channel)
    return u_r / 2 * (1 - (1 - u_r * chi_phys[:, :, None]) * vrg)


def get_kernel_dc(F, gchi0_core, u, channel):
    u_r = get_ur(u, channel)
    nq = np.shape(gchi0_core)[0]
    niw = np.shape(gchi0_core)[1]
    niv = np.shape(gchi0_core)[-1]
    kernel = np.zeros((nq, niw, niv), dtype=complex)
    for iq in range(nq):
        for iw in range(niw):
            kernel[iq, iw, :] = u_r * np.sum(gchi0_core[iq, iw, None, :] * F[None, iw, ...], axis=-1)
    return kernel # 1/beta is contained in the SDE


def schwinger_dyson_kernel_q(kernel, giwk, beta, q_list, wn, nqtot):
    niv = np.shape(kernel)[-1] // 2
    sigma_F = np.zeros([*np.shape(giwk)[:3], 2 * niv], dtype=complex)
    giwk = mf.cut_v(giwk, niv_cut=niv + np.max(np.abs(wn)), axes=-1)
    for i, q in enumerate(q_list):
        gkpq = bz.shift_mat_by_ind(giwk, ind=[-iq for iq in q])
        for j, iwn in enumerate(wn):
            gkpq_iwn_shift = mf.cut_iv_with_iw_shift(gkpq,niv,iwn)
            sigma_F += kernel[i, j, None, None, None, :] * gkpq_iwn_shift
    return 1 / nqtot * 1 / beta * sigma_F


def schwinger_dyson_dc(gchiq_Fupdo, giwk, u, q_list, q_point_duplicity, wn, nqtot):
    '''
        two beta prefactors are contained in F. Additional minus sign from usign gchi0_q
    '''
    niv_core = np.shape(gchiq_Fupdo)[-1] // 2
    sigma_dc = np.zeros([*np.shape(giwk)[:3], 2 * niv_core], dtype=complex)
    for iq, q in enumerate(q_list):
        gkpq = bz.shift_mat_by_ind(giwk, ind=[-iq for iq in q])
        mat_grid = mf.wn_slices_gen(gkpq, niv_core, wn=wn)
        sigma_dc += + u * 1 / (nqtot) * q_point_duplicity[iq] * np.sum(gchiq_Fupdo[iq, :, None, None, None, :] * mat_grid, axis=0)
    return sigma_dc


def schwinger_dyson_shell(chir_phys, giw, beta, u, n_shell, n_core, wn):
    mat_grid = mf.wn_slices_shell(giw, n_shell=n_shell, n_core=n_core, wn=wn)
    sigma_F = u ** 2 / 2 * 1 / beta * np.sum((chir_phys[:, None]) * mat_grid, axis=0)
    return sigma_F


def schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, giw, u, n, niv_shell=0):
    siw_dens_core = schwinger_dyson_vrg(vrg_dens, chi_dens, giw, u)
    siw_magn_core = schwinger_dyson_vrg(vrg_magn, chi_magn, giw, u)
    hartree = twop.get_smom0(u, n)
    if (niv_shell == 0):
        return siw_dens_core + siw_magn_core + hartree
    else:
        siw_magn_shell = schwinger_dyson_shell(chi_magn, giw, vrg_magn.beta, u, niv_shell, vrg_magn.niv, vrg_magn.wn)
        siw_dens_shell = schwinger_dyson_shell(chi_dens, giw, vrg_dens.beta, u, niv_shell, vrg_dens.niv, vrg_dens.wn)
        return hartree + mf.concatenate_core_asmypt(siw_dens_core + siw_magn_core, siw_magn_shell + siw_dens_shell)


def get_gchir_from_gamma_loc_q(gammar: lfp.LocalFourPoint = None, gchi0=None):
    '''
        Compute the non-local suscptibility using the BSE: chi_r = (chi_0^(-1) + Gamma_r/beta^2)^(-1)
        q_list is distributed among cores.
    '''
    nq = np.shape(gchi0)[0]
    nw = np.shape(gchi0)[1]
    chir = np.zeros([nq, *gammar.mat.shape], dtype=complex)
    for iq in range(nq):
        for iwn in range(nw):
            chir[iq, iwn, ...] = np.linalg.inv(np.diag(1 / gchi0[iq, iwn]) + gammar.mat[iwn])
    return chir

def get_gchir_aux_from_gammar_q(gammar: lfp.LocalFourPoint, gchi0, u):
    ''' chi_aux = (chi0^(-1) + Gamma/beta^2 - u/beta^2)^(-1) '''
    nq = np.shape(gchi0)[0]
    nw = np.shape(gchi0)[1]
    chir_aux = np.zeros([nq, *gammar.mat.shape], dtype=complex)
    u_r = get_ur(u, gammar.channel)
    for iq in range(nq):
        for iwn in range(nw):
            chir_aux[iq, iwn, ...] = np.linalg.inv(np.diag(1 / gchi0[iq, iwn]) + gammar.mat[iwn] - u_r / gammar.beta ** 2)
    return chir_aux


def chi_phys_from_chi_aux_q(chi_aux, chi0q_urange, chi0q_core, u, channel):
    ''' chi_phys = ((chi_aux + chi0_urange - chi0_core)^(-1) + u_r)^(-1) '''
    u_r = get_ur(u, channel)
    chir = 1. / (1. / (chi_aux + chi0q_urange - chi0q_core) + u_r)
    return chir


def chi_phys_asympt_q(chir_urange, chi0_urange, chi0_asympt):
    ''' asymptotic form of the susceptibility '''
    return chir_urange + chi0_asympt - chi0_urange


def vrg_from_gchi_aux(gchir_aux, gchi0_core, chir_urange, chir_asympt, u, channel):
    '''Note: 1/beta is here included in vrg compared to the old code'''
    u_r = get_ur(u, channel=channel)
    nq = np.shape(gchir_aux)[0]
    niw = np.shape(gchir_aux)[1]
    niv = np.shape(gchir_aux)[2]
    vrg = np.zeros([nq, niw, niv], dtype=complex)
    for iq in range(nq):
        for iwn in range(niw):
            vrg[iq, iwn, :] = 1 / gchi0_core[iq, iwn] * np.sum(gchir_aux[iq, iwn], axis=-1) * (1 - u_r * chir_urange[iq, iwn]) / (
                    1 - u_r * chir_asympt[iq, iwn])
    return vrg


def lam_from_chir_q(gchir, gchi0, channel):
    ''' lambda vertex'''
    sign = get_sign(channel)
    lam = -sign * (1 - np.sum(gchir * (1 / gchi0)[..., None], axis=-1))
    return lam


def lam_tilde(lam_core, chi0q_shell, u, channel):
    u_r = lfp.get_ur(u, channel)
    return (lam_core - u * chi0q_shell[..., None]) / (1 + u_r * chi0q_shell[..., None])


def get_chir_shell(lam_tilde, chi0q_shell, gchi0q_core, beta, u, channel):
    sign = get_sign(channel)
    chir_shell = -sign * u * chi0q_shell ** 2 \
                 + chi0q_shell * (1 - 2 * u * 1 / beta ** 2 * np.sum((lam_tilde + sign) * gchi0q_core, axis=-1))
    return chir_shell


def chir_tilde(chir_core, lam_tilde, chi0q_shell, gchi0q_core, beta, u, channel):
    chir_shell = get_chir_shell(lam_tilde, chi0q_shell, gchi0q_core, beta, u, channel)
    return (chir_core + chir_shell) / (1 - (u * chi0q_shell) ** 2)


def vrg_q_tilde(lam_tilde, chir_q_tilde, u, channel):
    u_r = get_ur(u, channel)
    sign = get_sign(channel)
    return (1 + sign * lam_tilde) / (1 - u_r * chir_q_tilde[..., None])


def get_vrg_and_chir_lad_from_gammar_uasympt_q(gamma_dens: lfp.LocalFourPoint, gamma_magn: lfp.LocalFourPoint,
                                               F_dc, vrg_magn_loc, chi_magn_loc,
                                               bubble_gen: bub.BubbleGenerator, u, my_q_list,
                                               niv_shell=0, logger=None, do_pairing_vertex=False):
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
    if (logger is not None): logger.log_cpu_time(task=' Bubbles constructed. ')

    # double-counting kernel:
    if (logger is not None):
        if (logger.is_root):
            F_dc.plot(pdir=logger.out_dir + '/', name='F_dc')

    kernel_dc = mf.cut_v(get_kernel_dc(F_dc.mat, gchi0_q_urange, u, 'magn'), niv_core, axes=(-1,))
    if (logger is not None): logger.log_cpu_time(task=' DC kernel constructed. ')

    # Density channel:
    gchiq_aux = get_gchir_aux_from_gammar_q(gamma_dens, gchi0_q_core, u)
    chiq_aux = 1 / beta ** 2 * np.sum(gchiq_aux, axis=(-1, -2))
    chi_lad_urange = chi_phys_from_chi_aux_q(chiq_aux, chi0_q_urange, chi0_q_core, u, gamma_dens.channel)
    chi_lad_dens = chi_phys_asympt_q(chi_lad_urange, chi0_q_urange, chi0_q_urange + chi0q_shell)

    vrg_q_dens = vrg_from_gchi_aux(gchiq_aux, gchi0_q_core, chi_lad_urange, chi_lad_dens, u, gamma_dens.channel)

    # if(do_pairing_vertex):

    # Magnetic channel:
    gchiq_aux = get_gchir_aux_from_gammar_q(gamma_magn, gchi0_q_core, u)
    chiq_aux = 1 / beta ** 2 * np.sum(gchiq_aux, axis=(-1, -2))
    chi_lad_urange = chi_phys_from_chi_aux_q(chiq_aux, chi0_q_urange, chi0_q_core, u, gamma_magn.channel)
    chi_lad_magn = chi_phys_asympt_q(chi_lad_urange, chi0_q_urange, chi0_q_urange + chi0q_shell)

    u_r = get_ur(u, gamma_magn.channel)
    # 1/beta**2 since we want F/beta**2
    kernel_dc += u_r / gamma_magn.beta * (1 - u_r * chi_magn_loc[None, :, None]) * vrg_magn_loc.mat[None, :, :] * chi0q_shell_dc[
                                                                                                                  :, :, None]
    vrg_q_magn = vrg_from_gchi_aux(gchiq_aux, gchi0_q_core, chi_lad_urange, chi_lad_magn, u, gamma_magn.channel)

    return vrg_q_dens, vrg_q_magn, chi_lad_dens, chi_lad_magn, kernel_dc


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------
def get_ur(u=1.0, channel='dens'):
    if (channel == 'magn'):
        sign = -1
    elif (channel == 'dens'):
        sign = 1
    elif (channel == 'updo'):
        sign = 1
    elif (channel == 'upup'):
        sign = -1
    else:
        raise ValueError
    return u * sign


def get_sign(channel):
    if (channel == 'magn'):
        return - 1
    if (channel == 'dens'):
        return + 1
    else:
        raise ValueError('Channel not in [dens/magn].')


def chir_from_g2_wn(g2=None, ggv=None, beta=None, iwn=0):
    if (ggv is not None and iwn == 0):
        chir = beta * (g2 - 2. * ggv)
    else:
        chir = beta * g2
    return chir


def gammar_from_gchir_wn(gchir=None, gchi0_urange=None, niv_core=None, beta=1.0, u=1.0):
    full = u / (beta * beta) + np.diag(1. / gchi0_urange)
    inv_full = np.linalg.inv(full)
    inv_core = mf.cut_v(inv_full, niv_core, axes=(-2, -1))
    core = np.linalg.inv(inv_core)
    chigr_inv = np.linalg.inv(gchir)
    return -(core - chigr_inv - u / (beta * beta))


def local_gchi_aux_from_gammar_wn(gammar=None, gchi0=None, beta=1.0, u=1.0):
    gchi0_inv = np.diag(1. / gchi0)
    chi_aux_inv = gchi0_inv + gammar - u / (beta * beta)
    return np.linalg.inv(chi_aux_inv)


def local_vertex_inverse_bse_wn(gamma=None, chi0=None, u_r=None, beta=None):
    niv = np.shape(gamma)[-1] // 2
    niv_u = np.shape(chi0)[-1] // 2
    gamma_u = u_r * np.ones((2 * niv_u, 2 * niv_u), dtype=complex) * 1. / beta ** 2
    gamma_u[niv_u - niv:niv_u + niv, niv_u - niv:niv_u + niv] = gamma  # gamma contains internally 1/beta^2
    return np.matmul(gamma_u, np.linalg.inv(np.eye(2 * niv_u, dtype=complex) + gamma_u * chi0[:, None]))
    # return np.linalg.inv(np.linalg.inv(gamma_u))- np.diag(chi0))


def local_vertex_inverse_bse(gamma=None, chi0=None, u=None):
    u_r = get_ur(u=u, channel=gamma.channel)
    return np.array(
        [local_vertex_inverse_bse_wn(gamma=gamma.mat[wn], chi0=chi0.gchi0[wn], u_r=u_r, beta=gamma.beta) for wn in gamma.wn_lin])


def get_ggv(giw=None, niv_ggv=-1):
    niv = giw.shape[0] // 2
    if (niv_ggv == -1):
        niv_ggv = niv
    return giw[niv - niv_ggv:niv + niv_ggv][:, None] * giw[niv - niv_ggv:niv + niv_ggv][None, :]


def gchi_aux_from_gammar(gammar=None, gchi0=None, beta=1.0, u=1.0):
    gchi0_inv = np.diag(1. / gchi0)
    chi_aux_inv = gchi0_inv + gammar - u / (beta * beta)
    chi_aux = np.linalg.inv(chi_aux_inv)
    return chi_aux
