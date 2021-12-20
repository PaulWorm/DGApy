# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Module for the linear Eliashberg equation as for DGA and similar ladder routines.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np

# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def phase1(gammax=None, gk=None, eps = 10**-6, max_count = 10000, norm=1.0):

    delta_old = np.random.random_sample(np.shape(gk))
    lambda_old = 10.
    converged = False
    count = 0
    while(not converged):
        delta_tilde = np.fft.ifftn(delta_old * np.abs(gk) ** 2, axes=(0, 1, 2), norm='ortho')
        delta_new = 1. / norm * np.sum(gammax * delta_tilde[..., None, :], axis=-1)
        delta_new = np.fft.fftn(delta_new, axes=(0, 1, 2))
        lambda_new = np.sum(np.conj(delta_old) * delta_new) / np.sum((np.conj(delta_old) * delta_old))
        delta_new = delta_new / lambda_new
        if (np.abs(lambda_new - lambda_old) < eps or count > max_count):
            converged = True

        lambda_old = lambda_new
        delta_old = delta_new

    return lambda_new, delta_new

def phase2(gammax=None, gk=None, eps = 10**-6, max_count = 10000, norm=1.0, lambda_s = None):

    delta_old = np.random.random_sample(np.shape(gk))
    lambda_old = 10.
    converged = False
    count = 0
    while(not converged):
        count += 1
        Delta_tilde = np.fft.ifftn(delta_old * np.abs(gk) ** 2, axes=(0, 1, 2), norm='ortho')
        delta_new = 1. / (norm) * np.sum(gammax * Delta_tilde[..., None, :], axis=-1)
        delta_new = np.fft.fftn(delta_new, axes=(0, 1, 2))

        delta_new = delta_new - lambda_s * delta_old
        lambda_new = np.sum(np.conj(delta_old) * delta_new) / np.sum((np.conj(delta_old) * delta_old))
        delta_new = delta_new / lambda_new
        delta_old = delta_new
        if (np.abs(lambda_new - lambda_old) < eps or count > max_count):
            converged = True
        lambda_old = lambda_new
        delta_new = delta_old

    return lambda_new, delta_new


def linear_eliashberg(gamma=None, gk=None, eps = 10**-6, max_count = 10000, norm=1.0):
    # Gamma has shape [nkx,nky,nkz,niv,niv]
    gammax = np.fft.fftn(gamma, axes=(0,1,2), norm='ortho')


    lambda_s, delta_s = phase1(gammax=gammax, gk=gk, eps = eps, max_count = max_count, norm=norm)
    lambda_d, delta_d = phase2(gammax=gammax, gk=gk, eps = eps, max_count = max_count, norm=norm, lambda_s = lambda_s)

    lambda_r = [lambda_s, lambda_d+lambda_s]
    delta = [delta_s, delta_d]

    return lambda_r, delta








if __name__ == '__main__':

    pass