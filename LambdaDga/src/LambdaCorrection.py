# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import FourPoint as fp


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


def lambda_correction(chi_magn_ladder=None, chi_dens_ladder=None, chi_dens_rpa=None, chi_magn_rpa=None,
                      chi_magn_dmft=None,
                      chi_dens_dmft=None, chi_magn_rpa_loc=None, chi_dens_rpa_loc=None, nq=None, use_rpa_for_lc=False,
                      lc_use_only_positive=True):
    if (lc_use_only_positive):
        mask_dens_loc = chi_dens_dmft.mat > 0
        mask_magn_loc = chi_magn_dmft.mat > 0
    else:
        mask_dens_loc = np.ones(np.shape(chi_dens_dmft.mat), dtype=bool)
        mask_magn_loc = np.ones(np.shape(chi_magn_dmft.mat), dtype=bool)

    n_dens = np.sum(mask_dens_loc)
    n_magn = np.sum(mask_magn_loc)

    # Make sure that the masks have the same size (though I am not sure if this is really needed)
    if(n_dens > n_magn):
        mask_dens_loc = mask_magn_loc
    else:
        mask_magn_loc = mask_dens_loc


    lambda_dens_start = get_lambda_start(chi_dens_ladder)
    lambda_magn_start = get_lambda_start(chi_magn_ladder)
    beta = chi_magn_ladder.beta
    chi_dens_loc_sum = 1. / beta * np.sum(chi_dens_dmft.mat[mask_dens_loc])
    chi_magn_loc_sum = 1. / beta * np.sum(chi_magn_dmft.mat[mask_magn_loc])
    chi_dens_ladder_sum = 1. / (beta*nq) * np.sum(chi_dens_ladder.mat)

    if (use_rpa_for_lc):
        chi_dens_loc_sum = chi_dens_loc_sum + 1. / beta * np.sum(chi_dens_rpa_loc.mat)
        lambda_dens_single = lambda_correction_single_use_rpa(lambda_start=lambda_dens_start, chir=chi_dens_ladder,
                                                              chir_rpa=chi_dens_rpa, chi_loc_sum=chi_dens_loc_sum,
                                                              nq=nq, mask=mask_dens_loc)

        chi_magn_loc_sum = chi_magn_loc_sum + 1. / beta * np.sum(chi_magn_rpa_loc.mat)
        lambda_magn_single = lambda_correction_single_use_rpa(lambda_start=lambda_magn_start, chir=chi_magn_ladder,
                                                              chir_rpa=chi_magn_rpa, chi_loc_sum=chi_magn_loc_sum,
                                                              nq=nq, mask=mask_magn_loc)

        chi_sum = chi_dens_loc_sum + chi_magn_loc_sum - chi_dens_ladder_sum - 1. / (beta*nq) * np.sum(chi_dens_rpa.mat)
        lambda_magn_totdens = lambda_correction_single_use_rpa(lambda_start=lambda_magn_start, chir=chi_magn_ladder,
                                                               chir_rpa=chi_magn_rpa, chi_loc_sum=chi_sum, nq=nq,
                                                               mask=mask_magn_loc)
    else:
        lambda_dens_single = lambda_correction_single(lambda_start=lambda_dens_start, chir=chi_dens_ladder,
                                                      chi_loc_sum=chi_dens_loc_sum, nq=nq, mask=mask_dens_loc)
        lambda_magn_single = lambda_correction_single(lambda_start=lambda_magn_start, chir=chi_magn_ladder,
                                                      chi_loc_sum=chi_magn_loc_sum, nq=nq, mask=mask_magn_loc)

        chi_sum = chi_dens_loc_sum + chi_magn_loc_sum - chi_dens_ladder_sum
        lambda_magn_totdens = lambda_correction_single(lambda_start=lambda_magn_start, chir=chi_magn_ladder,
                                                       chi_loc_sum=chi_sum, nq=nq, mask=mask_magn_loc)

    lambda_ = {
        'lambda_dens_single': lambda_dens_single,
        'lambda_magn_single': lambda_magn_single,
        'lambda_magn_totdens': lambda_magn_totdens,
        'n_sum_dens': n_dens,
        'n_sum_magn': n_magn
    }
    return lambda_
