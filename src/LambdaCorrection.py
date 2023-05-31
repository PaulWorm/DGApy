# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import FourPoint as fp
import Config as conf


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------


def get_lambda_start(chi_r):
    ''' w needs to be last dimension'''
    w0 = chi_r.shape[-1]//2
    return -np.min(1. / (chi_r[...,w0].real))


def lambda_correction_single(beta,lambda_start=0, chir: fp.LadderSusceptibility = None,
                             chi_loc_sum=None,
                             maxiter=1000, eps=1e-7, delta=0.1):
    lambda_old = lambda_start + delta
    lambda_ = lambda_old

    for i in range(maxiter):
        chir_sum = 1. / (beta) * np.mean(np.sum(1. / (1. / chir + lambda_),axis=-1))
        f_lam = chir_sum - chi_loc_sum
        fp_lam = -1. / (beta) * np.mean(np.sum((1. / (1. / chir + lambda_)) ** 2.,axis=-1))
        lambda_ = lambda_old - np.real(f_lam / fp_lam)
        print('Lambda: ',lambda_,'f_lam: ', f_lam,'fp_lam: ', fp_lam)
        if (np.abs(f_lam.real) < eps):
            break

        if (lambda_ < lambda_old):
            delta = delta / 2.
            lambda_old = lambda_start + delta
            lambda_ = lambda_old
        else:
            lambda_old = lambda_
    return lambda_

def lambda_correction(chi_lad_dens, chi_lad_magn, beta, chi_loc_dens, chi_loc_magn, type='spch'):

    if(type=='spch'):
        lambda_dens_start = get_lambda_start(chi_lad_dens)
        lambda_dens = lambda_correction_single(beta, lambda_start=lambda_dens_start, chir=chi_lad_dens,
                                                  chi_loc_sum=1 / beta * np.sum(chi_loc_dens))
        chi_lad_dens = use_lambda(chi_lad_dens, lambda_dens)

        lambda_magn_start = get_lambda_start(chi_lad_magn)
        lambda_magn = lambda_correction_single(beta, lambda_start=lambda_magn_start, chir=chi_lad_magn,
                                                  chi_loc_sum=1 / beta * np.sum(chi_loc_magn))
        chi_lad_magn = use_lambda(chi_lad_magn, lambda_magn)

        return chi_lad_dens, chi_lad_magn, lambda_dens, lambda_magn

    else:
        raise NotImplementedError('Only spch implemented')




def use_lambda(chi_ladder=None, lambda_= 0.0):
    return 1. / (1. / chi_ladder + lambda_)
