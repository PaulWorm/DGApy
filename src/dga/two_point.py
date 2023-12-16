'''
   Classes to handle self-energies and Green's functions.
   SelfEnergy: For the self-energy tail fitting and asymptotic extrapolation is supported.
   GreenFunction: The Green's function routine can estimate the chemical potential and also support asymptotic extrapolation.
   RealFrequencyGF: Green's function in real frequency.
'''

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
from scipy import optimize as opt

from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import plotting


# -------------------------------------------- SELF ENERGY ----------------------------------------------------------

def get_smom0(u, n):
    '''Return Hartree for the single-band SU(2) symmetric case'''
    return u * n / 2


def get_smom1(u, n):
    ''' return 1/ivu asymptotic prefactor of Im(Sigma) for the single-band SU(2) symmetric case'''
    return -u ** 2 * n / 2 * (1 - n / 2)


def get_sum_chiupup(n):
    ''' return 1/ivu asymptotic prefactor of Im(Sigma) for the single-band SU(2) symmetric case'''
    return n / 2 * (1 - n / 2)


def fit_smom(iv=None, siwk=None):
    '''
        Estimate the moments of the self-energy.
        iv: fermionic matsubara frequency
        siwk: self-energy [nkx,nky,nkz,niv] WARNING: Only positive part should be supplied
        pos: flag whether full range (false) or only positive (true) is supplied.
    '''
    niv = siwk.shape[-1]

    n_freq_fit = int(0.2 * niv)
    if n_freq_fit < 4:
        n_freq_fit = 4

    s_loc = np.mean(siwk, axis=(0, 1, 2))  # moments should not depend on momentum

    iwfit = iv[niv - n_freq_fit:]
    fitdata = s_loc[niv - n_freq_fit:]

    mom0 = np.mean(fitdata.real)
    # asympt_fun = lambda mom0_fit: mom0_fit
    # loss_fun = lambda mom0_fit: np.sum((fitdata.real - asympt_fun(mom0_fit).real) ** 2)
    # mom0 = opt.minimize(loss_fun, x0=np.array((mom0,))).x[0]
    # There is a minus sign in Josef's corresponding code, but this complies with the output from w2dyn.
    mom1 = np.mean(
        fitdata.imag * iwfit.imag)
    # asympt_fun = lambda mom1_fit: -1/iwfit * mom1_fit
    # loss_fun = lambda mom1_fit: np.sum((fitdata.imag - asympt_fun(mom1_fit).imag) ** 2)
    # mom1 = opt.minimize(loss_fun, x0=np.array((mom1,))).x[0]
    return mom0, mom1


def sigma_const(beta, delta, nk, v):
    sigma = np.ones(nk + (len(v),), dtype=complex)
    sigma *= -1j * delta
    return SelfEnergy(sigma, beta, pos=False)


class SelfEnergy():
    ''' class to handle self-energies'''

    niv_core_min = 20

    def __init__(self, sigma, beta, pos=False, smom0=None, smom1=None, err=1e-4, niv_core='estimate'):
        '''

        :param sigma:
        :param beta:
        :param pos: If False sigma is expected in the full range (-niv,niv), otherwise only for positive frequencies.
        :param smom0:
        :param smom1:
        :param err:
        :param niv_core: number of frequencies which are not treated in asymptotic approximation
        '''
        if len(np.shape(sigma)) == 1:
            sigma = sigma[None, None, None, :] # assuming local self-energy
        elif len(np.shape(sigma)) == 4:
            pass
        else:
            raise ValueError('Sigma must be either local [v] or momentum dependent [kx,ky,kz,v].')

        # Cut negative Matsubara frequencies if not provided:
        if not pos:
            niv = sigma.shape[-1] // 2
            sigma = sigma[..., niv:]

        self.sigma = sigma
        self.beta = float(beta)
        iv_fit = 1j * mf.vn(self.beta, self.niv, pos=True)
        fit_mom0, fit_mom1 = fit_smom(iv_fit, sigma)  # Currently moment fitting is local. Non-local parts
        # should decay quickly enough to be negligible.

        self.err = err

        # Set the moments for the asymptotic behaviour:
        if smom0 is None:
            self.smom0 = fit_mom0
        else:
            self.smom0 = smom0

        if smom1 is None:
            self.smom1 = fit_mom1
        else:
            self.smom1 = smom1

        # estimate when the asymptotic behavior is sufficient:
        if niv_core == 'estimate':
            self.niv_core = self.estimate_niv_core()
        else:
            self.niv_core = niv_core

    @property
    def niv(self):
        return self.sigma.shape[-1]

    @property
    def nk(self):
        return self.sigma.shape[:-1]

    @property
    def sigma_core(self):
        return mf.fermionic_full_nu_range(self.sigma[..., :self.niv_core])

    def k_mean(self):
        return np.mean(self.sigma, axis=(0, 1, 2))

    def estimate_niv_core(self):
        '''check when the real and the imaginary part are within error margin of the asymptotic'''
        asympt = self.get_asympt(niv_asympt=self.niv, n_min=0)
        ind_real = np.argmax(np.abs(self.k_mean().real - asympt.real) < self.err)
        ind_imag = np.argmax(np.abs(self.k_mean().imag - asympt.imag) < self.err)
        niv_core = max(ind_real, ind_imag)
        if niv_core < self.niv_core_min:
            return self.niv_core_min
        else:
            return niv_core

    def get_siw(self, niv=-1, pi_shift=False):
        if niv == -1:
            niv = self.niv
        if niv <= self.niv_core:
            sigma = mf.fermionic_full_nu_range(self.sigma[..., :niv])
        else:
            niv_shell = niv - self.niv_core
            iv_asympt = 1j * mf.vn(self.beta, niv_shell, shift=self.niv_core, pos=True)
            asympt = (self.smom0 - 1 / iv_asympt * self.smom1)[None, None, None, :] * np.ones(self.nk)[:, :, :, None]
            sigma_asympt = np.concatenate((self.sigma[..., :self.niv_core], asympt), axis=-1)
            sigma = mf.fermionic_full_nu_range(sigma_asympt)

        return sigma if not pi_shift else bz.shift_mat_by_pi(sigma)

    def get_asympt(self, niv_asympt, n_min=None, pos=True):
        if n_min is None:
            n_min = self.niv_core
        iv_asympt = 1j * mf.vn(self.beta, niv_asympt, shift=n_min, pos=True)
        asympt = (self.smom0 - 1 / iv_asympt * self.smom1)[None, None, None, :] * np.ones(self.nk)[:, :, :, None]
        if pos:
            return asympt
        else:
            return mf.fermionic_full_nu_range(asympt)


def create_dga_siwk_with_dmft_as_asympt(siwk_dga, siwk_dmft: SelfEnergy, niv_shell):
    nk = siwk_dga.shape[:3]
    niv_core = siwk_dga.shape[-1] // 2
    niv_full = niv_core + niv_shell
    siwk_shell = siwk_dmft.get_siw(niv_full)
    siwk_shell = np.ones(nk)[:, :, :, None] * \
                 mf.inv_cut_v(siwk_shell, niv_core=niv_core, niv_shell=niv_shell, axes=-1)[0, 0, 0, :][None, None, None, :]
    siwk_dga = mf.concatenate_core_asmypt(siwk_dga, siwk_shell)
    return SelfEnergy(siwk_dga, siwk_dmft.beta, smom0=siwk_dmft.smom0, smom1=siwk_dmft.smom1, niv_core=niv_full)


# -------------------------------------------- GREENSFUNCTION ----------------------------------------------------------
def get_gloc(iv=None, hk=None, siwk=None, mu_trial=None):
    """
    Calculate local Green's function by momentum integration.
    The integration is done by trapezoid integration, which amounts
    to a call of np.mean()
    """
    return np.mean(
        1. / (
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
    g_model = 1. / (iv
                    + mu
                    - hloc.real
                    - smom0)

    return g_model


# ==================================================================================================================

def get_fill_primitive(g_loc, beta, verbose=False):
    '''
    Primitive way of computing the filling on the matsubara axis.
    giwk: [kx,ky,kz,iv] - currently no easy extension to multi-orbital
    '''
    n = (1 / beta * np.sum(g_loc) + 0.5) * 2
    if verbose: print('n = ', n)
    return n


def root_fun_primitive(mu=0.0, target_filling=1.0, iv=None, hk=None, siwk=None, beta=1.0, verbose=False):
    """Auxiliary function for the root finding"""
    g_loc = get_gloc(iv=iv, hk=hk, siwk=siwk, mu_trial=mu)
    return get_fill_primitive(g_loc, beta, verbose)[0] - target_filling


def update_mu_primitive(mu0=0.0, target_filling=None, iv=None, hk=None, siwk=None, beta=None, smom0=None, tol=1e-6,
                        verbose=False):
    '''
        Find mu to satisfy target_filling.
    '''
    if verbose: print('Update mu...')
    mu = mu0
    hloc = None
    if verbose: print(
        root_fun(mu=mu, target_filling=target_filling, iv=iv, hk=hk, siwk=siwk, beta=beta, smom0=smom0, hloc=hloc))
    try:
        mu = opt.newton(root_fun_primitive, mu, tol=tol,
                        args=(target_filling, iv, hk, siwk, beta, verbose))
    except RuntimeError:
        if verbose: print('Root finding for chemical potential failed.')
        if verbose: print('Using old chemical potential again.')
    if np.abs(mu.imag) < 1e-8:
        mu = mu.real
    else:
        raise ValueError('In OneParticle.update_mu: Chemical Potential must be real.')
    return mu


# ==================================================================================================================
def get_fill(iv=None, hk=None, siwk=None, beta=1.0, smom0=0.0, hloc=None, mu=None, verbose=False):
    """
    Calculate the filling from the density matrix.
    The density matrix is obtained by frequency summation
    under consideration of the model.
    """
    g_model = get_g_model(mu=mu, iv=iv, hloc=hloc, smom0=smom0)
    gloc = get_gloc(iv=iv, hk=hk, siwk=siwk, mu_trial=mu)
    if beta * (smom0 + hloc.real - mu) < 20:
        rho_loc = 1. / (1. + np.exp(beta * (smom0 + hloc.real - mu)))
    else:
        rho_loc = np.exp(-beta * (smom0 + hloc.real - mu))
    rho_new = rho_loc + np.sum(gloc.real - g_model.real, axis=0) / beta
    n_el = 2. * rho_new
    if verbose: print(n_el, 'electrons at ', mu)
    return n_el, rho_new


# ==================================================================================================================

# ==================================================================================================================
def root_fun(mu=0.0, target_filling=1.0, iv=None, hk=None, siwk=None, beta=1.0, smom0=0.0, hloc=None):
    """Auxiliary function for the root finding"""
    return get_fill(iv=iv, hk=hk, siwk=siwk, beta=beta, smom0=smom0, hloc=hloc, mu=mu, verbose=False)[0] - target_filling


# ==================================================================================================================

# ==================================================================================================================
def update_mu(mu0=0.0, target_filling=1.0, iv=None, hk=None, siwk=None, beta=1.0, smom0=0.0, tol=1e-6, verbose=False):
    """
    Update internal chemical potential (mu) to fix the filling to the target filling with given precision.
    :return:
    """
    if verbose: print('Update mu...')
    hloc = hk.mean()
    mu = mu0
    if verbose:
        print(root_fun(mu=mu, target_filling=target_filling, iv=iv, hk=hk, siwk=siwk, beta=beta, smom0=smom0, hloc=hloc))
    try:

        mu = opt.newton(root_fun, mu, tol=tol,
                        args=(target_filling, iv, hk, siwk, beta, smom0, hloc))
    except RuntimeError:
        if verbose: print('Root finding for chemical potential failed.')
        if verbose: print('Using old chemical potential again.')
    if np.abs(mu.imag) < 1e-8:
        mu = mu.real
    else:
        raise ValueError('In OneParticle.update_mu: Chemical Potential must be real.')
    return mu


# ==================================================================================================================

def build_g(v, ek, mu, sigma):
    ''' Build Green's function with [kx,ky,kz,v]'''
    return 1 / (v[None, None, None, :] + mu - ek[..., None] - sigma)


class GreensFunction():
    '''
        Object to build the (imaginary frequency) Green's function from hk and sigma
    '''
    mu0 = 0
    mu_tol = 1e-6

    def __init__(self, sigma: SelfEnergy, ek, mu=None, n=None, niv_asympt=2000):
        self.sigma = sigma
        self.iv_core = 1j * mf.vn(sigma.beta, self.sigma.niv_core)
        self.ek = ek
        if n is not None:
            self._n = n
            niv_mu_find = self.sigma.niv_core #+ niv_asympt
            iv_mu_find = 1j * mf.vn(sigma.beta, niv_mu_find)
            siwk_mu_find = sigma.get_siw(niv_mu_find)
            self._mu = update_mu(mu0=self.mu0, target_filling=self.n, iv=iv_mu_find, hk=ek, siwk=siwk_mu_find,
                                 beta=self.beta, smom0=sigma.smom0,
                                 tol=self.mu_tol)
        elif mu is not None:
            self._mu = mu
            niv_mu_find = self.sigma.niv_core #+ niv_asympt
            iv_mu_find = 1j * mf.vn(sigma.beta, niv_mu_find)
            siwk_mu_find = sigma.get_siw(niv_mu_find)
            self._n = get_fill(iv=iv_mu_find, hk=ek, siwk=siwk_mu_find, beta=self.beta, smom0=sigma.smom0, hloc=np.mean(ek),
                               mu=mu)[0]
        else:
            raise ValueError('Either mu or n, but not both, must be supplied.')

        self.core = self.build_g_core()
        self.asympt = None
        self.niv_asympt = None
        self.full = None
        self.set_g_asympt(niv_asympt)
        self.g_loc = None
        self.set_gloc()

    @property
    def size(self):
        if self.full is None:
            return self.core.size
        else:
            return self.full.size

    @property
    def itemsize(self):
        if self.full is None:
            return self.core.itemsize
        else:
            return self.full.itemsize

    @property
    def mu(self):
        return self._mu

    @property
    def n(self):
        return self._n

    @property
    def v_core(self):
        return mf.vn(self.beta, self.niv_core)

    @property
    def v(self):
        return mf.vn(self.beta, self.niv_core + self.niv_asympt)

    @property
    def vn(self):
        return mf.vn(self.niv_core + self.niv_asympt)

    @property
    def beta(self):
        return self.sigma.beta

    @property
    def niv_core(self):
        return self.sigma.niv_core

    @property
    def niv_full(self):
        return self.niv_core + self.niv_asympt

    @property
    def mem(self):
        ''' returns the memory consumption of the Green's function'''
        if self.full is not None:
            return self.full.size * self.full.itemsize * 1e-9
        else:
            return self.core.size * self.core.itemsize * 1e-9

    def fs(self, pi_shift=False):
        return -self.g_full(pi_shift=pi_shift)[:, :, :, self.niv_full].imag / np.pi

    def g_full(self, pi_shift=False):
        if self.asympt is None:
            return self.core if not pi_shift else bz.shift_mat_by_pi(self.core)
        else:
            return mf.concatenate_core_asmypt(self.core, self.asympt) if not pi_shift else bz.shift_mat_by_pi(
                mf.concatenate_core_asmypt(self.core, self.asympt))

    def build_g_core(self):
        return build_g(self.iv_core, self.ek, self.mu, self.sigma.sigma_core)

    def k_mean(self, iv_range='core'):
        if iv_range == 'core':
            return np.mean(self.core, axis=(0, 1, 2))
        elif iv_range == 'full':
            if self.full is None:
                raise ValueError('Full Greens function has to be set first.')
            return np.mean(self.full, axis=(0, 1, 2))
        else:
            raise ValueError('Range has to be core or full.')

    def set_gloc(self):
        self.g_loc = self.k_mean(iv_range='full')

    def set_g_asympt(self, niv_asympt):
        self.asympt = self.build_asympt(niv_asympt)
        self.niv_asympt = niv_asympt
        self.full = self.g_full()
        self.set_gloc()

    def build_asympt(self, niv_asympt):
        sigma_asympt = self.sigma.get_asympt(niv_asympt)
        iv_asympt = 1j * mf.vn(self.beta, niv_asympt, shift=self.niv_core, pos=True)
        return mf.fermionic_full_nu_range(build_g(iv_asympt, self.ek, self.mu, sigma_asympt))

    def get_sigma_full(self, niv_asympt):
        sigma_asympt = self.sigma.get_asympt(niv_asympt, pos=False)
        sigma_core = self.sigma.sigma_core
        return mf.concatenate_core_asmypt(sigma_core, sigma_asympt)

    @property
    def e_kin(self): # be careful about the on-site energy
        ekin = 1 / self.beta * np.sum(np.mean(self.ek[..., None] * self.g_full(), axis=(0, 1, 2)))
        assert np.abs(ekin.imag) < 1e-8, 'Kinetic energy must be real.'
        return ekin.real

    @property
    def e_pot(self):
        sigma_full = self.get_sigma_full(self.niv_asympt) - self.sigma.smom0 # convergence is bad thats why we subtract the
        # static term
        epot = 1 / (self.beta) * np.sum(np.mean(sigma_full*self.g_full(), axis=(0, 1, 2)))
        # e_pot_core =  1 / self.beta * np.sum(np.mean(self.sigma.sigma_core*self.core, axis=(0, 1, 2)))
        assert np.abs(epot.imag) < 1e-8, 'Potential energy must be real.'
        return epot.real + self.sigma.smom0 * self.n/2


class RealFrequencyGF():
    '''
        Object to handle the real-frequency Green's function
    '''
    mu0 = 0.0

    def __init__(self, w, swk, ek, mu=None, n=None, deltino=0.0):
        if mu is None and n is None:
            raise ValueError('Either mu or n must be supplied.')
        self.w = w
        self.swk = swk - 1j * deltino  # [nkx,nky,nkz,nw] or [1,1,1,nw]
        self.ek = ek

        if n is not None:
            self.n = n
            self.mu = adjust_mu_real_f(self.mu0, self.n, self.swk, self.w, self.ek)
            self._gwk = get_gwk(self.w, self.swk, self.ek, self.mu)
        elif mu is not None:
            self.mu = mu
            self._gwk = get_gwk(self.w, self.swk, self.ek, self.mu)
            self.n = get_filling_real_f(self.w, self.gwk())
        else:
            raise ValueError('Either mu or n, but not both, must be supplied.')

    def awk(self, pi_shift=False):
        return -self.gwk(pi_shift=pi_shift).imag / np.pi

    @property
    def gw(self):
        return np.mean(self.gwk(), axis=(0, 1, 2))

    @property
    def aw(self):
        return -self.gw.imag / np.pi

    def gwk(self, pi_shift=False):
        return self._gwk if not pi_shift else bz.shift_mat_by_pi(self._gwk)

    def fs(self, pi_shift=False):
        ''' Fermi surface'''
        w_ind = np.argmin(np.abs(self.w))
        return self.awk(pi_shift)[..., w_ind]

    def get_fs_surface_slice_indices(self, k_grid: bz.KGrid, k_slice_dim=2, k_slice_val=0, w_val=0, pi_shift=False):
        w_ind = np.argmin(np.abs(self.w - w_val))
        data = k_grid.get_k_slice(self.gwk(pi_shift)[..., w_ind].real, k_slice_dim=k_slice_dim, k_slice_val=k_slice_val)
        return plotting.get_zero_contour(data)

    def get_gwk_tb(self, deltino=0.01):
        ''' Build the non-interacting version of the Green's function'''
        return RealFrequencyGF(self.w, np.zeros_like(self.swk) - 1j * deltino, self.ek, n=self.n)


def get_gwk(w, swk, ek, mu):
    return 1 / (w[None, None, None, :] + mu - ek[..., None] - swk)


def get_filling_real_f(w, gwk):
    ''' gwk: [kx,ky,kz,w] '''
    ind = w <= 0
    return -1 / np.pi * np.trapz(np.mean(gwk.imag, axis=(0, 1, 2))[ind], w[ind]) * 2  # factor 2 for spin


def opt_func(mu, n, swk, w, ek, verbose=False):
    giwk = get_gwk(w, swk, ek, mu)
    fill_current = get_filling_real_f(w, giwk)
    if verbose:
        print('------------------')
        print(f'{mu=}')
        print(f'{fill_current=}')
        print('------------------')
    return (fill_current - n) ** 2


def adjust_mu_real_f(mu0, n_target, swk, w, ek, verbose=False):
    fit = opt.minimize(opt_func, x0=mu0, args=(n_target, swk, w, ek, verbose))
    return fit.x


if __name__ == '__main__':
    pass
