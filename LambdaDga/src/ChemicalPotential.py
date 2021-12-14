# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Routines to compute the chemical potential for a given filling of a Green's function.
# Outlay of dimensions is: [iv, kx, ky, kz, spin-band-1, spin-band-2]

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import scipy.linalg
import scipy.optimize


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------
#
# def get_gloc(iv=None, hk=None, siw=None, mu_trial=None, n_spin_band=1):
#     """
#     Calculate local Green's function by momentum integration.
#     The integration is done by trapezoid integration, which amounts
#     to a call of np.mean()
#     """
#     return np.mean(
#         np.linalg.inv(
#             iv[:, None, None, None, None, None] * np.eye(n_spin_band)[None, None, None, None, :, :]
#             + mu_trial * np.eye(n_spin_band)[None, None, None, None, :, :]
#             - hk[None, :, :, :, :, :]
#             - siw), axis=(1, 2, 3))
#
#
# # ==================================================================================================================
# def get_g_model(mu=None, iv=None, hloc=None, smom0=None, n_spin_band=1):
#     """
#     Calculate a model Green's function, needed for
#     the calculation of the electron density by Matsubara summation.
#     References: w2dynamics code, Markus Wallerberger's thesis.
#     """
#     g_model = np.linalg.inv(iv[:, None, None] * np.eye(n_spin_band)[None, :, :]
#                             + mu * np.eye(n_spin_band)[None, :, :]
#                             - hloc.real[None, :, :]
#                             - smom0[None, :, :])
#
#     return g_model
#
#
# # ==================================================================================================================
#
# # ==================================================================================================================
# def get_fill(iv=None, hk=None, siw=None, beta=1.0, smom0=0.0, hloc=None, mu=None, n_spin_band=1):
#     """
#     Calculate the filling from the density matrix.
#     The density matrix is obtained by frequency summation
#     under consideration of the model.
#     """
#     g_model = get_g_model(mu=mu, iv=iv, hloc=hloc, smom0=smom0, n_spin_band=n_spin_band)
#     gloc = get_gloc(iv=iv, hk=hk, siw=siw, mu_trial=mu, n_spin_band=n_spin_band)
#     rho_loc = np.linalg.inv(np.eye(n_spin_band) + scipy.linalg.expm(
#         beta * (smom0 + hloc.real - mu * np.eye(n_spin_band))))
#     rho_new = rho_loc + np.sum(gloc.real - g_model.real, axis=0) / beta
#     n_el = 2. *  np.trace(rho_new)
#     print(n_el, 'electrons at ', mu)
#     return n_el, rho_new
#
#
# # ==================================================================================================================
#
# # ==================================================================================================================
# def root_fun(mu=0.0, target_filling=1.0, iv=None, hk=None, siw=None, beta=1.0, smom0=0.0, hloc=None, n_spin_band=1):
#     """Auxiliary function for the root finding"""
#     return get_fill(iv=iv, hk=hk, siw=siw, beta=beta, smom0=smom0, hloc=hloc, mu=mu, n_spin_band=n_spin_band)[0] - target_filling
#
#
# # ==================================================================================================================
#
# # ==================================================================================================================
# def update_mu(mu0=0.0, target_filling=1.0, iv=None, hk=None, siw=None, beta=1.0, smom0=0.0, hloc=None, n_spin_band=1, tol=1e-6):
#     """
#     Update internal chemical potential (mu) to fix the filling to the target filling with given precision.
#     :return:
#     """
#     print('Update mu...')
#     mu = mu0
#     print(root_fun(mu=mu, target_filling=target_filling, iv=iv, hk=hk, siw=siw, beta=beta, smom0=smom0, hloc=hloc,
#                    n_spin_band=n_spin_band))
#     try:
#         mu = scipy.optimize.newton(root_fun, mu, tol=tol,
#                                    args=(target_filling, iv, hk, siw, beta, smom0, hloc, n_spin_band))
#     except RuntimeError:
#         print('Root finding for chemical potential failed.')
#         print('Using old chemical potential again.')
#     if np.abs(mu.imag) < 1e-8:
#         mu = mu.real
#     else:
#         raise ValueError('In OneParticle.update_mu: Chemical Potential must be real.')
#     return mu
#



def get_gloc(iv=None, hk=None, siwk=None, mu_trial=None):
    """
    Calculate local Green's function by momentum integration.
    The integration is done by trapezoid integration, which amounts
    to a call of np.mean()
    """
    return np.mean(
        1./(
            iv[None, None, None, :]
            + mu_trial
            - hk[:, :, :, None]
            - siwk), axis=(0, 1, 2))


# ==================================================================================================================
def get_g_model(mu=None, iv=None, hloc=None, smom0=None):
    """
    Calculate a model Green's function, needed for
    the calculation of the electron density by Matsubara summation.
    References: w2dynamics code, Markus Wallerberger's thesis.
    """
    g_model =1./(iv
                            + mu
                            - hloc.real
                            - smom0)

    return g_model


# ==================================================================================================================

# ==================================================================================================================
def get_fill(iv=None, hk=None, siwk=None, beta=1.0, smom0=0.0, hloc=None, mu=None):
    """
    Calculate the filling from the density matrix.
    The density matrix is obtained by frequency summation
    under consideration of the model.
    """
    g_model = get_g_model(mu=mu, iv=iv, hloc=hloc, smom0=smom0)
    gloc = get_gloc(iv=iv, hk=hk, siwk=siwk, mu_trial=mu)
    rho_loc = 1./(1. + np.exp(
        beta * (smom0 + hloc.real - mu)))
    rho_new = rho_loc + np.sum(gloc.real - g_model.real, axis=0) / beta
    n_el = 2. *  rho_new
    print(n_el, 'electrons at ', mu)
    return n_el, rho_new


# ==================================================================================================================

# ==================================================================================================================
def root_fun(mu=0.0, target_filling=1.0, iv=None, hk=None, siwk=None, beta=1.0, smom0=0.0, hloc=None):
    """Auxiliary function for the root finding"""
    return get_fill(iv=iv, hk=hk, siwk=siwk, beta=beta, smom0=smom0, hloc=hloc, mu=mu)[0] - target_filling


# ==================================================================================================================

# ==================================================================================================================
def update_mu(mu0=0.0, target_filling=1.0, iv=None, hk=None, siwk=None, beta=1.0, smom0=0.0, tol=1e-6):
    """
    Update internal chemical potential (mu) to fix the filling to the target filling with given precision.
    :return:
    """
    print('Update mu...')
    hloc = hk.mean()
    mu = mu0
    print(root_fun(mu=mu, target_filling=target_filling, iv=iv, hk=hk, siwk=siwk, beta=beta, smom0=smom0, hloc=hloc))
    try:
        mu = scipy.optimize.newton(root_fun, mu, tol=tol,
                                   args=(target_filling, iv, hk, siwk, beta, smom0, hloc))
    except RuntimeError:
        print('Root finding for chemical potential failed.')
        print('Using old chemical potential again.')
    if np.abs(mu.imag) < 1e-8:
        mu = mu.real
    else:
        raise ValueError('In OneParticle.update_mu: Chemical Potential must be real.')
    return mu



# ==================================================================================================================

# ==================================================================================================================
def fit_smom(iv=None, siwk=None):
    """Read or calculate self-energy moments"""
    niv = siwk.shape[3] // 2
    n_freq_fit = int(0.2 * niv)
    if n_freq_fit < 4:
        n_freq_fit = 4
    s_loc = np.mean(siwk, axis=(0, 1, 2))  # moments should not depend on momentum

    iwfit = iv[2 * niv - n_freq_fit:]
    fitdata = s_loc[2 * niv - n_freq_fit:]
    mom0 = np.mean(fitdata.real)
    mom1 = np.mean(fitdata.imag * iwfit.imag) # There is a minus sign in Josef's corresponding code, but this complies with the output from w2dyn.

    return mom0, mom1
# ==================================================================================================================


def get_g(mu=0, ek=None, sigma=None, delta=0.1):
    return 1. / (mu - ek - sigma + 1j * delta)





if __name__ == '__main__':
    nkx = 64
    nky = 64
    nkz = 1
    kx = np.arange(0, nkx) * 2 * np.pi / nkx
    ky = np.arange(0, nky) * 2 * np.pi / nky
    kz = np.arange(0, nkz) * 2 * np.pi / nkz
    t = 1.0 #0.25
    tp = -0.25 * t *0
    tpp = 0.12 * t *0
    beta = 10. / t
    n_spin_band = 1
    target_filling = 1.0
    niv = 200
    iv = 1j * np.pi / beta * (np.arange(-niv, niv) * 2 + 1)
    u = 8 * t
    smom0 = target_filling * u / 2.
    delta = 0.1 * t


    def ek_square(kx=None, ky=None, kz=None, t=1.0, tp=0.0, tpp=0.0):
        return - 2. * t * (np.cos(kx) + np.cos(ky)) - 4. * tp * np.cos(kx) * np.cos(ky) \
               - 2. * tpp * (np.cos(2. * kx) + np.cos(2. * ky)) + np.cos(kz) * 0.0


    ek = ek_square(kx=ky[:, None, None], ky=ky[None, :, None], kz=kz[None, None, :], t=t, tp=tp, tpp=tpp)
    hloc = np.mean(ek)

    mu = update_mu(mu0=smom0/2.,target_filling=target_filling, iv=iv, hk=ek, siw=smom0, beta=beta, smom0=smom0, hloc=hloc)
    print(f'{mu=}')

    g_fs = get_g(mu=mu, ek=ek, sigma=smom0, delta=delta)

    import matplotlib.pyplot as plt

    plt.imshow(g_fs[:, :, 0].real, cmap='RdBu', extent=[0, 2 * np.pi, 0, 2 * np.pi])
    plt.title(r'$\Re G; \beta={}; n={};$'.format(beta, target_filling))
    plt.colorbar()
    plt.show()

    plt.imshow(-1. / np.pi * g_fs[:, :, 0].imag, cmap='RdBu', extent=[0, 2 * np.pi, 0, 2 * np.pi])
    plt.title(r'$ A; \beta={}; n={};$'.format(beta, target_filling))
    plt.colorbar()
    plt.show()

    g_loc = get_gloc(iv=iv,hk=ek,siw=smom0,mu_trial=mu)

    n = get_fill(iv=iv, hk=ek, siw=smom0, beta=beta, smom0=smom0, hloc=hloc, mu=mu)

    plt.plot(iv.imag,g_loc.imag)
    plt.title(r'$ A; \beta={}; n={};$'.format(beta, target_filling))
    plt.show()

    plt.plot(iv.imag,g_loc.real)
    plt.title(r'$ A; \beta={}; n={};$'.format(beta, target_filling))
    plt.show()
