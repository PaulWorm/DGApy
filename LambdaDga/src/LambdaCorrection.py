# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import FourPoint as fp

# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------


def get_lambda_start(chi: fp.LadderSusceptibility = None):
    is_w0 = chi.qiw[:,-1] == 0
    return -np.min(1. / (chi.mat[is_w0].real))


def lambda_correction(lambda_start = 0, chir: fp.LadderSusceptibility = None, chi_loc: fp.LocalSusceptibility = None,
                      maxiter = 1000, eps=1e-7, delta=0.1, nq = None):
    beta = chir.beta
    chi_loc_sum = 1. / beta * np.sum(chi_loc.mat)
    lambda_old = lambda_start + delta
    lambda_ = lambda_old

    for i in range(maxiter):
        chir_sum = 1. / (beta * nq) * np.sum(1. / (1. / chir.mat + lambda_))
        f_lam = chir_sum - chi_loc_sum
        fp_lam = -1. / (beta * nq) * np.sum((1. / (1. / chir.mat + lambda_)) ** 2)
        lambda_ = lambda_old - np.real(f_lam / fp_lam)
        if (np.abs(f_lam.real) < eps):
            break

        if (lambda_ < lambda_old):
            delta = delta / 2.
            lambda_old = lambda_start + delta
            lambda_ = lambda_old
        else:
            lambda_old = lambda_
    print(f'Lambda correction converged afer {i=} iterations.')
    return lambda_

def lambda_correction_totdens(lambda_start = 0, chi_magn: fp.LadderSusceptibility = None, chi_dens: fp.LadderSusceptibility = None,
                     chi_magn_loc: fp.LocalSusceptibility = None,chi_dens_loc: fp.LocalSusceptibility = None,maxiter = 1000, eps=1e-7, delta=0.1, nq = None):
    beta = chi_magn.beta
    lambda_old = lambda_start + delta
    lambda_ = lambda_old
    chi_dens_sum = 1. / (beta * nq) * np.sum(1. / (1. / chi_dens.mat))
    chi_magn_loc_sum = 1. / (beta) * np.sum(1. / (1. / chi_magn_loc.mat))
    chi_dens_loc_sum = 1. / (beta) * np.sum(1. / (1. / chi_dens_loc.mat))

    for i in range(maxiter):
        chi_magn_sum = 1. / (beta * nq) * np.sum(1. / (1. / chi_magn.mat + lambda_))
        f_lam = (chi_magn_sum + chi_dens_sum) - (chi_magn_loc_sum+chi_dens_loc_sum)
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
    print(f'Lambda correction converged afer {i=} iterations.')
    return lambda_
