# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import FourPoint as fp


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------


def get_lambda_start(chi: fp.LadderSusceptibility = None):
    is_w0 = chi.qiw[:, -1] == 0
    return -np.min(1. / (chi.mat[is_w0].real))


def lambda_correction_single(lambda_start=0, chir: fp.LadderSusceptibility = None,
                             chi_loc: fp.LocalSusceptibility = None,
                             maxiter=1000, eps=1e-7, delta=0.1, nq=None):
    beta = chir.beta
    chi_loc_sum = 1. / beta * np.sum(chi_loc.mat)
    lambda_old = lambda_start + delta
    lambda_ = lambda_old

    for i in range(maxiter):
        chir_sum = 1. / (beta * nq) * np.sum(1. / (1. / chir.mat + lambda_))
        f_lam = chir_sum - chi_loc_sum
        fp_lam = -1. / (beta * nq) * np.sum((1. / (1. / chir.mat + lambda_)) ** 2.)
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
                                     chir_rpa: fp.LadderSusceptibility = None,
                                     chi_loc: fp.LocalSusceptibility = None, chi_loc_rpa: fp.LocalSusceptibility = None,
                                     maxiter=1000, eps=1e-7, delta=0.1, nq=None):
    beta = chir.beta
    chi_loc_sum = 1. / beta * np.sum(chi_loc.mat) + 1. / beta * np.sum(chi_loc_rpa.mat)
    lambda_old = lambda_start + delta
    lambda_ = lambda_old

    for i in range(maxiter):
        chir_sum = 1. / (beta * nq) * np.sum(1. / (1. / chir.mat + lambda_))
        chir_sum += 1. / (beta * nq) * np.sum(1. / (1. / chir_rpa.mat + lambda_))
        f_lam = chir_sum - chi_loc_sum
        fp_lam = -1. / (beta * nq) * np.sum((1. / (1. / chir.mat + lambda_)) ** 2.)
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


def lambda_correction_totdens(lambda_start=0, chi_magn: fp.LadderSusceptibility = None,
                              chi_dens: fp.LadderSusceptibility = None,
                              chi_magn_loc: fp.LocalSusceptibility = None, chi_dens_loc: fp.LocalSusceptibility = None,
                              maxiter=1000, eps=1e-7, delta=0.1, nq=None):
    beta = chi_magn.beta
    lambda_old = lambda_start + delta
    lambda_ = lambda_old
    chi_dens_sum = 1. / (beta * nq) * np.sum(1. / (1. / chi_dens.mat))
    chi_magn_loc_sum = 1. / (beta) * np.sum(1. / (1. / chi_magn_loc.mat))
    chi_dens_loc_sum = 1. / (beta) * np.sum(1. / (1. / chi_dens_loc.mat))

    for i in range(maxiter):
        chi_magn_sum = 1. / (beta * nq) * np.sum(1. / (1. / chi_magn.mat + lambda_))
        f_lam = (chi_magn_sum + chi_dens_sum) - (chi_magn_loc_sum + chi_dens_loc_sum)
        fp_lam = -1. / (beta * nq) * np.sum((1. / (1. / chi_magn.mat + lambda_)) ** 2)
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


def lambda_correction_totdens_use_rpa(lambda_start=0, chi_magn: fp.LadderSusceptibility = None,
                                      chi_dens: fp.LadderSusceptibility = None,
                                      chi_dens_rpa: fp.LadderSusceptibility = None,
                                      chi_magn_rpa: fp.LadderSusceptibility = None,
                                      chi_magn_loc: fp.LocalSusceptibility = None,
                                      chi_dens_loc: fp.LocalSusceptibility = None,
                                      chi_dens_rpa_loc: fp.LocalSusceptibility = None,
                                      chi_magn_rpa_loc: fp.LocalSusceptibility = None,
                                      maxiter=1000, eps=1e-7, delta=0.1, nq=None):
    beta = chi_magn.beta
    lambda_old = lambda_start + delta
    lambda_ = lambda_old
    chi_dens_sum = 1. / (beta * nq) * np.sum(1. / (1. / chi_dens.mat)) + 1. / (beta * nq) * np.sum(
        1. / (1. / chi_dens_rpa.mat))
    chi_magn_loc_sum = 1. / (beta) * np.sum(1. / (1. / chi_magn_loc.mat)) + 1. / (beta) * np.sum(
        1. / (1. / chi_magn_rpa_loc.mat))
    chi_dens_loc_sum = 1. / (beta) * np.sum(1. / (1. / chi_dens_loc.mat)) + 1. / (beta) * np.sum(
        1. / (1. / chi_dens_rpa_loc.mat))

    for i in range(maxiter):
        chi_magn_sum = 1. / (beta * nq) * np.sum(1. / (1. / chi_magn.mat + lambda_))
        chi_magn_sum += 1. / (beta * nq) * np.sum(1. / (1. / chi_magn_rpa.mat + lambda_))
        f_lam = (chi_magn_sum + chi_dens_sum) - (chi_magn_loc_sum + chi_dens_loc_sum)
        fp_lam = -1. / (beta * nq) * np.sum((1. / (1. / chi_magn.mat + lambda_)) ** 2)
        fp_lam += -1. / (beta * nq) * np.sum((1. / (1. / chi_magn_rpa.mat + lambda_)) ** 2)
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


def lambda_correction(chi_magn_ladder=None, chi_dens_ladder=None, chi_dens_rpa=None, chi_magn_rpa=None, chi_magn_dmft=None,
                      chi_dens_dmft=None, chi_magn_rpa_loc=None, chi_dens_rpa_loc=None, nq=None, use_rpa_for_lc=False):

    lambda_dens_start = get_lambda_start(chi_dens_ladder)
    lambda_magn_start = get_lambda_start(chi_magn_ladder)

    if (use_rpa_for_lc):
        lambda_dens_single = lambda_correction_single_use_rpa(lambda_start=lambda_dens_start, chir=chi_dens_ladder,
                                                              chir_rpa=chi_dens_rpa, chi_loc=chi_dens_dmft,
                                                              chi_loc_rpa=chi_dens_rpa_loc, nq=nq)
        lambda_magn_single = lambda_correction_single_use_rpa(lambda_start=lambda_magn_start, chir=chi_magn_ladder,
                                                              chir_rpa=chi_magn_rpa, chi_loc=chi_magn_dmft,
                                                              chi_loc_rpa=chi_magn_rpa_loc, nq=nq)
        lambda_magn_totdens = lambda_correction_totdens_use_rpa(lambda_start=lambda_magn_start,
                                                                chi_magn=chi_magn_ladder,
                                                                chi_dens=chi_dens_ladder, chi_magn_rpa=chi_magn_rpa,
                                                                chi_dens_rpa=chi_dens_rpa, chi_magn_loc=chi_magn_dmft,
                                                                chi_dens_loc=chi_dens_dmft,
                                                                chi_dens_rpa_loc=chi_dens_rpa_loc,
                                                                chi_magn_rpa_loc=chi_magn_rpa_loc, nq=nq)

    else:
        lambda_dens_single = lambda_correction_single(lambda_start=lambda_dens_start, chir=chi_dens_ladder,
                                                      chi_loc=chi_dens_dmft, nq=nq)
        lambda_magn_single = lambda_correction_single(lambda_start=lambda_magn_start, chir=chi_magn_ladder,
                                                      chi_loc=chi_magn_dmft, nq=nq)
        lambda_magn_totdens = lambda_correction_totdens(lambda_start=lambda_magn_start, chi_magn=chi_magn_ladder,
                                                        chi_magn_loc=chi_magn_dmft, chi_dens_loc=chi_dens_dmft,
                                                        chi_dens=chi_dens_ladder,
                                                        nq=nq)

    lambda_ = {
        'lambda_dens_single': lambda_dens_single,
        'lambda_magn_single': lambda_magn_single,
        'lambda_magn_totdens': lambda_magn_totdens
    }
    return lambda_
