# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import FourPoint as fp
import Config as conf


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------


def get_lambda_start(chi: fp.LadderSusceptibility = None):
    is_w0 = chi.qiw[:, -1] == 0
    return -np.min(1. / (chi.mat.flatten()[is_w0].real))


def lambda_correction_single(lambda_start=0, chir: fp.LadderSusceptibility = None,
                             chi_loc_sum=None,
                             maxiter=1000, eps=1e-7, delta=0.1, nq=None, mask=None):
    beta = chir.beta
    lambda_old = lambda_start + delta
    lambda_ = lambda_old

    for i in range(maxiter):
        chir_sum = 1. / (beta * nq) * np.sum(1. / (1. / chir.mat[..., mask] + lambda_))
        f_lam = chir_sum - chi_loc_sum
        fp_lam = -1. / (beta * nq) * np.sum((1. / (1. / chir.mat[..., mask] + lambda_)) ** 2.)
        lambda_ = lambda_old - np.real(f_lam / fp_lam)
        if (np.abs(f_lam.real) < eps):
            break

        if (lambda_ < lambda_old):
            delta = delta / 2.
            lambda_old = lambda_start + delta
            lambda_ = lambda_old
        else:
            lambda_old = lambda_
    return lambda_


def lambda_correction_single_use_rpa(lambda_start=0, chir: fp.LadderSusceptibility = None,
                                     chir_rpa: fp.LadderSusceptibility = None, chi_loc_sum=None,
                                     maxiter=1000, eps=1e-7, delta=0.1, nq=None, mask=None):
    beta = chir.beta
    lambda_old = lambda_start + delta
    lambda_ = lambda_old

    for i in range(maxiter):
        chir_sum = 1. / (beta * nq) * np.sum(1. / (1. / chir.mat[..., mask] + lambda_))
        chir_sum += 1. / (beta * nq) * np.sum(1. / (1. / chir_rpa.mat + lambda_))
        f_lam = chir_sum - chi_loc_sum
        fp_lam = -1. / (beta * nq) * np.sum((1. / (1. / chir.mat[..., mask] + lambda_)) ** 2.)
        fp_lam += -1. / (beta * nq) * np.sum((1. / (1. / chir_rpa.mat + lambda_)) ** 2.)
        lambda_ = lambda_old - np.real(f_lam / fp_lam)

        if (np.abs(f_lam.real) < eps):
            break

        if (lambda_ < lambda_old):
            delta = delta / 2.
            lambda_old = lambda_start + delta
            lambda_ = lambda_old
        else:
            lambda_old = lambda_
    return lambda_


def lambda_correction(dga_conf: conf.DgaConfig = None, chi_ladder=None, chi_rpa=None, chi_dmft=None, chi_rpa_loc=None):
    if (dga_conf.opt.lc_use_only_positive):
        mask_dens_loc = chi_dmft['dens'].mat > 0
        mask_magn_loc = chi_dmft['magn'].mat > 0
    else:
        mask_dens_loc = np.ones(np.shape(chi_dmft['dens'].mat), dtype=bool)
        mask_magn_loc = np.ones(np.shape(chi_dmft['magn'].mat), dtype=bool)

    n_dens = np.sum(mask_dens_loc)
    n_magn = np.sum(mask_magn_loc)

    lambda_dens_start = get_lambda_start(chi_ladder['dens'])
    lambda_magn_start = get_lambda_start(chi_ladder['magn'])
    beta = dga_conf.sys.beta
    nq = dga_conf.q_grid.nk_tot
    chi_dens_loc_sum = 1. / beta * np.sum(chi_dmft['dens'].mat[mask_dens_loc])
    chi_magn_loc_sum = 1. / beta * np.sum(chi_dmft['magn'].mat[mask_magn_loc])
    chi_dens_ladder_sum = 1. / (beta * dga_conf.q_grid.nk_tot) * np.sum(chi_ladder['dens'].mat[...,mask_dens_loc])

    if (dga_conf.opt.use_urange_for_lc):
        chi_dens_loc_sum = chi_dens_loc_sum + 1. / beta * np.sum(chi_rpa_loc['dens'].mat)
        lambda_dens_single = lambda_correction_single_use_rpa(lambda_start=lambda_dens_start, chir=chi_ladder['dens'],
                                                              chir_rpa=chi_rpa['dens'], chi_loc_sum=chi_dens_loc_sum,
                                                              nq=nq, mask=mask_dens_loc)

        chi_magn_loc_sum = chi_magn_loc_sum + 1. / beta * np.sum(chi_rpa_loc['magn'].mat)
        lambda_magn_single = lambda_correction_single_use_rpa(lambda_start=lambda_magn_start, chir=chi_ladder['magn'],
                                                              chir_rpa=chi_rpa['magn'], chi_loc_sum=chi_magn_loc_sum,
                                                              nq=nq, mask=mask_magn_loc)

        chi_sum = chi_dens_loc_sum + chi_magn_loc_sum - chi_dens_ladder_sum - 1. / (beta * nq) * np.sum(
            chi_rpa['dens'].mat)
        lambda_magn_totdens = lambda_correction_single_use_rpa(lambda_start=lambda_magn_start, chir=chi_ladder['magn'],
                                                               chir_rpa=chi_rpa['magn'], chi_loc_sum=chi_sum, nq=nq,
                                                               mask=mask_magn_loc)
    else:
        lambda_dens_single = lambda_correction_single(lambda_start=lambda_dens_start, chir=chi_ladder['dens'],
                                                      chi_loc_sum=chi_dens_loc_sum, nq=nq, mask=mask_dens_loc)
        lambda_magn_single = lambda_correction_single(lambda_start=lambda_magn_start, chir=chi_ladder['magn'],
                                                      chi_loc_sum=chi_magn_loc_sum, nq=nq, mask=mask_magn_loc)

        chi_sum = chi_dens_loc_sum + chi_magn_loc_sum - chi_dens_ladder_sum
        lambda_magn_totdens = lambda_correction_single(lambda_start=lambda_magn_start, chir=chi_ladder['magn'],
                                                       chi_loc_sum=chi_sum, nq=nq, mask=mask_magn_loc)

    if (dga_conf.opt.lambda_correction_type == 'spch'):
        lambda_dens = lambda_dens_single
        lambda_magn = lambda_magn_single
    elif (dga_conf.opt.lambda_correction_type == 'sp'):
        lambda_dens = 0.0
        lambda_magn = lambda_magn_totdens
    elif (dga_conf.opt.lambda_correction_type == 'sp_only'):
        lambda_dens = 0.0
        lambda_magn = lambda_magn_single
    elif (dga_conf.opt.lambda_correction_type == 'none'):
        lambda_dens = 0.0
        lambda_magn = 0.0
    else:
        raise ValueError('Unknown value for lambda_correction_type!')

    lambda_ = {
        'dens': lambda_dens,
        'magn': lambda_magn
    }
    n_lambda = {
        'dens': n_dens,
        'magn': n_magn
    }

    return lambda_, n_lambda


def build_chi_lambda(dga_conf: conf.DgaConfig = None, chi_ladder = None, chi_rpa=None, lambda_=None):
    chi_ladder['dens'].mat = use_lambda(chi_ladder=chi_ladder['dens'].mat, lambda_=lambda_['dens'])
    chi_ladder['magn'].mat = use_lambda(chi_ladder=chi_ladder['magn'].mat, lambda_=lambda_['magn'])
    if (dga_conf.opt.use_urange_for_lc):
        if (np.size(dga_conf.box.wn_rpa) > 0):
            chi_rpa['dens'].mat = use_lambda(chi_ladder=chi_rpa['dens'].mat, lambda_=lambda_['dens'])
            chi_rpa['magn'].mat = use_lambda(chi_ladder=chi_rpa['magn'].mat, lambda_=lambda_['magn'])



def use_lambda(chi_ladder=None, lambda_= 0.0):
    return 1. / (1. / chi_ladder + lambda_)
