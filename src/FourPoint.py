# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
from scipy.stats import gaussian_kde

import w2dyn_aux
import TwoPoint as tp
import copy
import numpy as np
import Indizes as ind
import MatsubaraFrequencies as mf
import Config as conf
import jax.numpy as jnp
import matplotlib.pyplot as plt


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def cut_iv(mat=None, niv_cut=10):
    niv = mat.shape[-1] // 2
    assert (mat.shape[-1] == mat.shape[-2]), 'Last two dimensions of the array are not consistent'
    return mat[..., niv - niv_cut:niv + niv_cut, niv - niv_cut:niv + niv_cut]


def cut_iw(mat=None, niw_cut=0):
    assert mat.shape > 1, 'Matrix has to be reshaped to [qx,qy,qz,Niw] format.'
    niw = mat.shape[-1] // 2
    mat = mat[..., niw - niw_cut:niw + niw_cut + 1]
    return mat


def get_ur(u=1.0, channel='dens'):
    if (channel == 'magn'):
        sign = -1
    elif (channel == 'dens'):
        sign = 1
    else:
        raise ValueError
    return u * sign


# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------START REIMPLEMENTATION OF THE DGA ROUTINES-----------------------------------
# ----------------------------------------------------------------------------------------------------------------------

KNOWN_CHANNELS = ['dens', 'magn', 'trip', 'sing', 'updo', 'upup']

# ======================================================================================================================
# -------------------------------------------- FOUR POINT PARENT CLASS  ------------------------------------------------
# ======================================================================================================================

FIGSIZE = (7, 3)
ALPHA = 0.7
BASECOLOR = 'cornflowerblue'


def plot_fourpoint_nu_nup(mat, vn, do_save=True, pdir='./', name='NoName', cmap='RdBu', figsize=FIGSIZE):
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    axes = axes.flatten()
    im1 = axes[0].pcolormesh(vn, vn, mat.real, cmap=cmap)
    im2 = axes[1].pcolormesh(vn, vn, mat.imag, cmap=cmap)
    axes[0].set_title('$\Re$')
    axes[1].set_title('$\Im$')
    for ax in axes:
        ax.set_xlabel(r'$\nu_p$')
        ax.set_ylabel(r'$\nu$')
        ax.set_aspect('equal')
    fig.suptitle(name)
    fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location='right',
                 pad=0.05)
    fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location='right',
                 pad=0.05)
    plt.tight_layout()
    if (do_save): plt.savefig(pdir + '/' + name + '.png')
    plt.show()


def plot_omega_dep(x, y, do_save=True, pdir='./', name='NoName', figsize=FIGSIZE):
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    axes[0].plot(x, y.real, '-o', color=BASECOLOR, markeredgecolor='k', alpha=ALPHA)
    axes[1].plot(x, y.imag, '-o', color=BASECOLOR, markeredgecolor='k', alpha=ALPHA)
    for ax in axes: ax.set_xlabel('$\omega_n$')
    axes[0].set_ylabel('$\Re$' + name)
    axes[1].set_ylabel('$\Im$' + name)
    plt.tight_layout()
    if (do_save): plt.savefig(pdir + '/' + name + '.png')
    plt.show()


def get_sign(channel):
    if (channel == 'magn'):
        return - 1
    if (channel == 'dens'):
        return + 1
    else:
        raise ValueError('Channel not in [dens/magn].')


class FourPointBase():
    '''
        Parent class for local four-point objects
    '''

    niv_ind = -1  # Index for (one of the) fermionic frequency indizes

    def __init__(self, matrix=None, beta=None, wn=None, shell=None, mat_tilde=None):
        if wn is None: wn = mf.wn(n=matrix.shape[0] // 2)
        if (matrix is not None):
            assert (matrix.shape[0] == wn.size), f'Size of iw_core ({wn.size}) does not match first dimension of four_point ({matrix.shape[0]}).'
        self._mat = matrix  # [iw, iv, iv']
        self._beta = beta
        self._wn = wn  # bosonic frequency index
        self.shell = shell
        self._mat_tilde = mat_tilde

    @property
    def mat(self):
        return self._mat

    @property
    def wn(self):
        return self._wn

    @property
    def wn_lin(self):
        return np.arange(0, self.wn.size)

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def niv(self) -> int:
        return self._mat.shape[self.niv_ind] // 2

    @property
    def niw(self) -> int:
        return self._wn.size

    @property
    def vn(self):
        return mf.vn(self.niv)

    @property
    def mat_tilde(self):
        if (self._mat_tilde is None):
            return self.mat
        else:
            return self._mat_tilde

    def as_dict(self):
        return self.__dict__


class LocalSusceptibility():

    def __init__(self, matrix=None, channel=None, beta=None, wn=None, shell=None, mat_tilde=None):
        if wn is None: wn = mf.wn(n=matrix.shape[0] // 2)
        if (matrix is not None):
            assert (matrix.shape[0] == wn.size), f'Size of iw_core ({wn.size}) does not match first dimension of four_point ({matrix.shape[0]}).'
        self.mat = matrix  # [iw, iv, iv']
        self.beta = beta
        self.wn = wn  # bosonic frequency index
        self.channel = channel  # dens/magn/trip/sing/bubb
        self.shell = shell
        self._mat_tilde = mat_tilde

    @property
    def mat_tilde(self):
        if (self._mat_tilde is None):
            return self.mat
        else:
            return self._mat_tilde

    @property
    def channel(self) -> str:
        return self._channel

    @channel.setter
    def channel(self, value):
        assert value in KNOWN_CHANNELS, f"Channel with name {value} not known. Kown are {KNOWN_CHANNELS}."
        self._channel = value


class LocalThreePoint(FourPointBase):

    def __init__(self, channel=None, matrix=None, beta=None, wn=None, shell=None, mat_tilde=None):
        super().__init__(matrix=matrix, beta=beta, wn=wn, shell=shell, mat_tilde=mat_tilde)
        self.channel = channel  # dens/magn/trip/sing/bubb

    @property
    def channel(self) -> str:
        return self._channel

    @channel.setter
    def channel(self, value):
        assert value in KNOWN_CHANNELS, f"Channel with name {value} not known. Kown are {KNOWN_CHANNELS}."
        self._channel = value


class LocalFourPoint(FourPointBase):
    ''' Parent class for local four-point correlation
        IMPORTANT: This object does NOT have to contain the full omega range, but is only defined on a set of points wn
    '''

    def __init__(self, channel=None, matrix=None, beta=None, wn=None, shell=None):
        super().__init__(matrix=matrix, beta=beta, wn=wn, shell=shell)
        self.channel = channel  # dens/magn/trip/sing/bubb

    @property
    def channel(self) -> str:
        return self._channel

    @channel.setter
    def channel(self, value):
        assert value in KNOWN_CHANNELS, f"Channel with name {value} not known. Kown are {KNOWN_CHANNELS}."
        self._channel = value

    def cut_iv(self, niv_cut=None):
        self._mat = cut_iv(self.mat, niv_cut)

    def contract_legs(self):
        return LocalSusceptibility(matrix=1. / self.beta ** 2 * np.sum(self.mat, axis=(-2, -1)), channel=self.channel, beta=self.beta, wn=self.wn)

    def plot(self, iwn=0, pdir='./', name=None, do_save=True, niv=-1):
        assert iwn in self.wn, 'omega index not in dataset.'
        iwn_lin = self.wn_lin[self.wn == iwn][0]
        data = mf.cut_iv_2d(self.mat[iwn_lin], niv_cut=niv)
        vn = mf.cut_v_1d(self.vn, niv_cut=niv)
        plot_fourpoint_nu_nup(data, vn, pdir=pdir, name=name, do_save=do_save)


# ======================================================================================================================
# ---------------------------------------------- LOCAL BUBBLE CLASS  ---------------------------------------------------
# ======================================================================================================================

def get_gchi0(giw, niv, beta, iwn):
    niv_giw = np.shape(giw)[0] // 2  # ToDo: Maybe I should introduce a Green's function object, but for now keep it like this
    return - beta * giw[niv_giw - niv:niv_giw + niv] * giw[niv_giw - niv - iwn:niv_giw + niv - iwn]


def vec_get_gchi0(giw, beta, niv, wn):
    return jnp.array([get_gchi0(giw, niv, beta, iwn) for iwn in wn])


def get_gchi0_inv(giw, niv, beta, iwn):
    niv_giw = np.shape(giw)[0] // 2
    return - 1 / beta * 1 / giw[niv_giw - niv:niv_giw + niv] * 1 / giw[niv_giw - niv - iwn:niv_giw + niv - iwn]


def vec_get_gchi0_inv(giw, beta, niv, wn):
    return jnp.array([get_gchi0_inv(giw, niv, beta, iwn) for iwn in wn])


def get_chi0_sum(giw, beta, niv, iwn=0):
    niv_giw = np.shape(giw)[0] // 2
    # iwnd2 = iwn // 2
    # iwnd2mod2 = iwn // 2 + iwn % 2
    iwnd2 = 0
    iwnd2mod2 = -iwn
    return - 1. / beta * np.sum(giw[niv_giw - niv + iwnd2:niv_giw + niv + iwnd2] * giw[niv_giw - niv - iwnd2mod2:niv_giw + niv - iwnd2mod2])


def vec_get_chi0_sum(giw, beta, niv, wn):
    return np.array([get_chi0_sum(giw, beta, niv, iwn=iwn) for iwn in wn])


def chi0_asympt_iwn(beta, hf_denom, niv, iwn):
    # iwnd2 = iwn // 2
    # iwnd2mod2 = iwn // 2 + iwn % 2
    # iv = mf.iv(beta, niv, shift=iwnd2)
    # ivmw = mf.iv(beta, niv, shift=-iwnd2mod2)
    iv = mf.iv(beta, niv, shift=0)
    ivmw = mf.iv(beta, niv, shift=-iwn)
    g_tail = 1 / (iv + hf_denom)
    gshift_tail = 1 / (ivmw + hf_denom)
    chi0_tail = - 1 / beta * np.sum(g_tail * gshift_tail)
    return chi0_tail


def chi0_asympt(beta, hf_denom, niv_asympt, niv, wn):
    full = jnp.array([chi0_asympt_iwn(beta, hf_denom, niv_asympt, iwn) for iwn in wn])
    inner = jnp.array([chi0_asympt_iwn(beta, hf_denom, niv, iwn) for iwn in wn])
    return full - inner


KNOWN_CHI0_METHODS = ['sum']


class LocalBubble():
    ''' Computes the local Bubble suszeptibility \chi_0 = - beta GG '''

    def __init__(self, wn=None, giw: tp.LocalGreensFunction = None, niv=-1, chi0_method='sum', is_inv=False, niv_shell=1000, do_chi0=True,
                 do_shell=True):
        self._wn = wn
        self._giw = giw
        self._niv = niv
        self.is_inv = is_inv
        self._mat = self.get_gchi0()
        self.chi0_method = chi0_method
        self.niv_shell = niv_shell
        self.set_chi0(do_chi0)
        # self.set_asympt()
        self.set_shell(do_shell)

    @property
    def giw(self):
        return self._giw

    @property
    def niv(self) -> int:
        return self._niv

    @niv.setter
    def niv(self, value):
        assert value <= self.niv_giw - np.max(np.abs(self.wn)), f'niv ({value}) exceeds the limit of: {self.niv_giw - np.max(np.abs(self.wn))}'
        if (value == -1):
            value = self.niv_giw - np.max(np.abs(self.wn))
        self._niv = value

    @property
    def mat(self):
        return self._mat

    @property
    def vn(self):
        return mf.vn(self.niv)

    @property
    def wn(self):
        return self._wn

    @property
    def beta(self) -> float:
        return self.giw.beta

    @property
    def niv_giw(self) -> int:
        return self.giw.niv

    @property
    def chi0_tilde(self):
        return self.chi0 + self.shell

    @property
    def niw(self):
        return self._wn.size

    @property
    def wn_lin(self):
        return np.arange(0, self.niw)

    @property
    def chi0(self):
        return self._chi0

    @property
    def gchi0(self):
        return self._mat

    def set_shell(self, do_shell):
        if (do_shell):
            self.shell = chi0_asympt(self.beta, self.giw.hf_denom, self.niv_shell, self.niv, self.wn)
        else:
            self.shell = None

    def get_gchi0(self):
        if (self.is_inv == True):
            gchi0 = vec_get_gchi0_inv(self.giw.mat, self.beta, self.niv, self.wn)
        else:
            gchi0 = vec_get_gchi0(self.giw.mat, self.beta, self.niv, self.wn)
        return gchi0

    def set_chi0(self, do_chi):
        if (do_chi):
            if (self.chi0_method == 'sum'):
                # self._chi0 = vec_get_chi0_sum(self.giw.mat, self.beta, self.niv, self.wn)
                self._chi0 = 1 / self.beta ** 2 * jnp.sum(self.gchi0, axis=1)
            else:
                raise NotImplementedError(f'Method not in {KNOWN_CHI0_METHODS}')
        else:
            self._chi0 = None

    @property
    def gchi0_full(self):
        ''' Return the full qkk' matrix'''
        return jnp.array([jnp.diag(self.mat[iwn_lin]) for iwn_lin in self.wn_lin])

    def plot(self, iwn=0, pdir='./', name=None, do_save=True):
        assert iwn in self.wn, 'omega index not in dataset.'
        iwn_lin = self.wn_lin[self.wn == iwn][0]
        plot_fourpoint_nu_nup(np.diag(self.mat[iwn_lin]), self.vn, pdir=pdir, name=name, do_save=do_save)
        plot_omega_dep(self.wn, self.chi0, pdir=pdir, name=name, do_save=do_save)


def chir_from_g2_wn(g2=None, ggv=None, beta=None, iwn=0):
    if (ggv is not None and iwn == 0):
        chir = beta * (g2 - 2. * ggv)
    else:
        chir = beta * g2
    return chir


def chir_from_g2(g2: LocalFourPoint = None, giw: tp.LocalGreensFunction = None):
    ''' chi_r = beta * (G2 - 2 * GG \delta_dens) '''
    chir_mat = g2.beta * g2.mat
    if (g2.channel == 'dens' and 0 in g2.wn):
        ggv = get_ggv(giw.mat, niv_ggv=g2.niv)
        chir_mat[g2.wn == 0] = g2.beta * (g2.mat[g2.wn == 0] - 2. * ggv)
    chir_mat = jnp.array(chir_mat)
    return LocalFourPoint(matrix=chir_mat, channel=g2.channel, beta=g2.beta, wn=g2.wn)


def g2_from_chir(chir: LocalFourPoint = None, giw=None):
    ''' G2 = 1/beta * chi_r + 2*GG \delta_dens'''
    chir_mat = 1 / chir.beta * chir.mat
    if (chir.channel == 'dens' and 0 in chir.wn):
        ggv = get_ggv(giw, niv_ggv=chir.niv)
        chir_mat[chir.wn == 0] = 1 / chir.beta * chir.mat[chir.wn == 0] + 2. * ggv
    chir_mat = jnp.array(chir_mat)
    return LocalFourPoint(matrix=chir_mat, channel=chir.channel, beta=chir.beta, wn=chir.wn)


def Fob2_from_chir(chir: LocalFourPoint = None, chi0_inv: LocalBubble = None):
    ''' Do not include the beta**2 in F.
        F_r = [chi_0^-1 - chi_0^-1 chi_r chi_0^-1]
    '''
    assert chir.niv == chi0_inv.niv
    F0b2_mat = chi0_inv.gchi0_full - chi0_inv.mat[:, :, None] * chir.mat * chi0_inv.mat[:, None, :]
    return LocalFourPoint(matrix=F0b2_mat, channel=chir.channel, beta=chir.beta, wn=chir.wn)


def chir_from_Fob2(F_r: LocalFourPoint = None, chi0: LocalBubble = None):
    ''' Do not include the beta**2 in F.
        F_r = [chi_0^-1 - chi_0^-1 chi_r chi_0^-1]
    '''
    assert F_r.niv == chi0.niv
    chir_mat = chi0.gchi0_full - chi0.mat[:, :, None] * F_r.mat * chi0.mat[:, None, :]
    return LocalFourPoint(matrix=chir_mat, channel=F_r.channel, beta=F_r.beta, wn=F_r.wn)


def gamob2_from_chir(chir: LocalFourPoint = None, chi0_inv: LocalBubble = None):
    ''' Gamma = - ( chi_0^-1 - chi_r^-1) '''
    gam_r = jnp.array([gamob2_from_chir_wn(chir.mat[iwn_lin], chi0_inv.mat[iwn_lin]) for iwn_lin in chir.wn_lin])
    return LocalFourPoint(matrix=gam_r, channel=chir.channel, beta=chir.beta, wn=chir.wn)


def gamob2_from_chir_wn(chir=None, chi0_inv=None):
    gam_r = -(jnp.diag(chi0_inv) - jnp.linalg.inv(chir))
    return gam_r


def wn_slices(mat=None, n_cut=None, wn=None):
    n = mat.shape[-1] // 2
    mat_grid = jnp.array([mat[n - n_cut - iwn:n + n_cut - iwn] for iwn in wn])
    return mat_grid


def vrg_from_gam(gam: LocalFourPoint = None, chi0_inv: LocalBubble = None, u=None):
    u_r = get_ur(u, channel=gam.channel)
    vrg = jnp.array([vrg_from_gam_wn(gam.mat[iwn_lin], chi0_inv.mat[iwn_lin], gam.beta, u_r) for iwn_lin in gam.wn_lin])
    return LocalThreePoint(matrix=vrg, channel=gam.channel, beta=gam.beta, wn=gam.wn)


def vrg_from_gam_wn(gam, chi0_inv, beta, u_r):
    return 1 / beta * chi0_inv * jnp.sum(jnp.linalg.inv(np.diag(chi0_inv) + gam - u_r / beta ** 2), axis=-1)


def three_leg_from_F(F: LocalFourPoint = None, chi0: LocalBubble = None):
    ''' sg = \sum_k' F \chi_0 '''
    sg = 1 / F.beta * jnp.sum(F.mat * chi0.mat[:, None, :], axis=-1)
    return LocalThreePoint(matrix=sg, channel=F.channel, beta=F.beta, wn=F.wn)


def schwinger_dyson_F(F: LocalFourPoint = None, chi0: LocalBubble = None, giw=None, u=None, totdens=None):
    ''' Sigma = U*n/2 + 1/beta**3 * U \sum_{qk} F \chi_0 G(k-q)'''
    hartree = u * totdens / 2.0
    sigma_F_q = jnp.sum(1 / F.beta * u * F.mat * chi0.mat[:, None, :], axis=-1)
    mat_grid = wn_slices(giw, n_cut=F.niv, wn=F.wn)
    sigma_F = jnp.sum(sigma_F_q * mat_grid, axis=0)
    return hartree + sigma_F


def schwinger_dyson_vrg(vrg: LocalThreePoint = None, chir_phys: LocalSusceptibility = None, giw: tp.LocalGreensFunction = None, u=None,
                        do_tilde=True):
    ''' Sigma = U*n/2 + '''
    u_r = get_ur(u, channel=chir_phys.channel)
    mat_grid = wn_slices(giw.mat, n_cut=vrg.niv, wn=vrg.wn)
    if (do_tilde):
        sigma_F = u_r / 2 * jnp.sum((1 / vrg.beta - (1 - u_r * chir_phys.mat_tilde[:, None]) * vrg.mat_tilde) * mat_grid, axis=0)
    else:
        sigma_F = u_r / 2 * jnp.sum((1 / vrg.beta - (1 - u_r * chir_phys.mat[:, None]) * vrg.mat) * mat_grid, axis=0)
    return sigma_F


def schwinger_dyson_w_asympt(gchi0: LocalBubble = None,giw=None, u=None,niv=None):
    mat_grid = wn_slices(giw.mat, n_cut=niv, wn=gchi0.wn)
    return -u * u/gchi0.beta * jnp.sum(gchi0.chi0_tilde[:,None] * mat_grid, axis=0)


def lam_from_chir_FisUr(chir: LocalFourPoint = None, gchi0: LocalBubble = None, u=None):
    ''' lambda vertex'''
    assert gchi0.shell is not None, "Shell of gchi0 needed for shell of lambda."
    u_r = get_ur(u, chir.channel)
    sign = get_sign(chir.channel)
    lam = jnp.sum(jnp.eye(chir.niv * 2)[None, :, :] - chir.mat * (1 / gchi0.mat)[:, :, None], axis=-1)
    shell = +u_r * gchi0.shell[:, None]
    tilde = (lam + shell)
    return LocalThreePoint(channel=chir.channel, matrix=lam, beta=chir.beta, wn=chir.wn, shell=shell, mat_tilde=tilde)


def chi_phys_tilde_FisUr(chir: LocalFourPoint = None, gchi0: LocalBubble = None, lam: LocalFourPoint = None, u=None):
    u_r = get_ur(u, chir.channel)
    chi_core = chir.contract_legs()
    chi_shell = gchi0.shell - u_r * gchi0.shell ** 2 - 2 * u_r * gchi0.shell * gchi0.chi0
    chi_tilde = (chi_core.mat + chi_shell)
    return LocalSusceptibility(matrix=chi_core.mat, channel=chir.channel, beta=chir.beta, wn=chir.wn, shell=chi_shell, mat_tilde=chi_tilde)


def lam_from_chir(chir: LocalFourPoint = None, gchi0: LocalBubble = None, u=None):
    ''' lambda vertex'''
    assert gchi0.shell is not None, "Shell of gchi0 needed for shell of lambda."
    u_r = get_ur(u, chir.channel)
    sign = get_sign(chir.channel)
    lam = jnp.sum(jnp.eye(chir.niv * 2)[None, :, :] - chir.mat * (1 / gchi0.mat)[:, :, None], axis=-1)
    # shell = +u * gchi0.shell[:, None]
    # tilde = 1 / (1 - u * gchi0.shell[:, None]) * (lam + shell)
    shell = +u_r * gchi0.shell[:, None]  # + u * u_r * gchi0.shell[:, None]
    # tilde = 1 / (1 + u_r * gchi0.shell[:, None]) * (lam + shell)
    tilde = (lam + shell)
    # tilde =  (lam + shell)
    return LocalThreePoint(channel=chir.channel, matrix=lam, beta=chir.beta, wn=chir.wn, shell=shell, mat_tilde=tilde)


def chi_phys_tilde(chir: LocalFourPoint = None, gchi0: LocalBubble = None, lam: LocalFourPoint = None, u=None):
    sign = get_sign(chir.channel)
    u_r = get_ur(u, chir.channel)
    chi_core = chir.contract_legs()
    chi_shell = - u_r * gchi0.shell ** 2  # - gchi0.shell * (1 - 2 *  u/chir.beta**2 * jnp.sum((lam.mat_tilde - 1) * gchi0.mat))#gchi0.shell- 2 * u_r * gchi0.shell * gchi0.chi0
    chi_tilde = 1 / (1 - (u * gchi0.shell) ** 2) * (chi_core.mat + chi_shell)
    chi_shell = gchi0.shell - u_r * gchi0.shell ** 2 - 2 * u_r * gchi0.shell * gchi0.chi0
    chi_tilde = (chi_core.mat + chi_shell)
    return LocalSusceptibility(matrix=chi_core.mat, channel=chir.channel, beta=chir.beta, wn=chir.wn, shell=chi_shell, mat_tilde=chi_tilde)


def vrg_from_lam(chir: LocalSusceptibility = None, lam: LocalFourPoint = None, u=None, do_tilde=True):
    sign = get_sign(chir.channel)
    u_r = get_ur(u, chir.channel)
    vrg_tilde = 1 / chir.beta * (1 - lam.mat_tilde) / (1 - u_r * chir.mat_tilde[:, None])
    vrg = 1 / chir.beta * (1 - lam.mat) / (1 - u_r * chir.mat[:, None])
    return LocalThreePoint(channel=chir.channel, matrix=vrg, beta=chir.beta, wn=chir.wn, mat_tilde=vrg_tilde)


# ------------------------------------------------ OBJECTS -------------------------------------------------------------

# class LocalThreePoint(LocalFourPoint):
#     ''' Class for local three-point objects like the Fermi-bose vertex'''
#
#     def contract_legs(self):
#         return 1. / self.beta * np.sum(self._mat, axis=(-1))


# ======================================================================================================================
# ------------------------------------------- LOCAL SUSCEPTIBILITY CLASS  ----------------------------------------------
# ======================================================================================================================

# class LocalSusceptibility():
#     ''' Parent class for local susceptibilities'''
#
#     asympt = None  # asymptotic correction from Chi = Chi0
#
#     def __init__(self, matrix=None, channel='dens', beta=1.0, iw=None, chi0_urange: LocalBubble = None):
#         assert (matrix.shape[0] == iw.size), 'Size of iw_core does not match first dimension of four_point'
#
#         self._channel = channel
#         self._mat = matrix
#         self._beta = beta
#         self._iw = iw
#         self._niw = np.size(self._iw)
#
#         self.set_asympt(chi0_urange=chi0_urange)
#
#     @property
#     def mat(self):
#         return self._mat
#
#     @property
#     def iw(self):
#         return self._iw
#
#     @property
#     def channel(self):
#         return self._channel
#
#     @property
#     def beta(self) -> float:
#         return self._beta
#
#     @property
#     def niv(self) -> int:
#         return self._niv
#
#     @property
#     def niw(self) -> int:
#         return self._iw.size
#
#     def set_asympt(self, chi0_urange: LocalBubble = None):
#         self.asympt = chi0_urange.chi0_asympt - chi0_urange.chi0
#
#     @property
#     def mat_asympt(self):
#         return self.mat + self.asympt


def local_chi_phys_from_chi_aux(chi_aux=None, chi0_urange: LocalBubble = None, chi0_core: LocalBubble = None, u=None):
    u_r = get_ur(u=u, channel=chi_aux.channel)
    chi = 1. / (1. / (chi_aux.mat + chi0_urange.chi0 - chi0_core.chi0) + u_r)
    return LocalSusceptibility(matrix=chi, channel=chi_aux.channel, beta=chi_aux.beta, wn=chi_aux.wn)


def local_susceptibility_from_four_point(four_point: LocalFourPoint = None, chi0_urange=None):
    return four_point.contract_legs()


def local_rpa_susceptibility(chi0_urange: LocalBubble = None, channel=None, u=None):
    u_r = get_ur(u=u, channel=channel)
    chir = chi0_urange.chi0 / (1 + u_r * chi0_urange.chi0)
    return LocalSusceptibility(matrix=chir, channel=channel, beta=chi0_urange.beta, iw=chi0_urange.iw,
                               chi0_urange=chi0_urange)


# ======================================================================================================================
# ------------------------------------- FREE FUNCTIONS THAT USE OBJECTS AS INPUT ---------------------------------------
# ======================================================================================================================

# ==================================================================================================================
def gammar_from_gchir(gchir: LocalFourPoint = None, gchi0_urange: LocalBubble = None, u=1.0):
    u_r = get_ur(u=u, channel=gchir.channel)
    gammar = np.array(
        [gammar_from_gchir_wn(gchir=gchir.mat[wn], gchi0_urange=gchi0_urange.gchi0[wn], niv_core=gchir.niv,
                              beta=gchir.beta, u=u_r) for wn in gchir.wn_lin])
    return LocalFourPoint(matrix=gammar, channel=gchir.channel, beta=gchir.beta, wn=gchir.wn)


def gammar_from_gchir_wn(gchir=None, gchi0_urange=None, niv_core=None, beta=1.0, u=1.0):
    full = u / (beta * beta) + np.diag(1. / gchi0_urange)
    inv_full = np.linalg.inv(full)
    inv_core = cut_iv(inv_full, niv_core)
    core = np.linalg.inv(inv_core)
    chigr_inv = np.linalg.inv(gchir)
    return -(core - chigr_inv - u / (beta * beta))


# ==================================================================================================================

# ==================================================================================================================
def local_gchi_aux_from_gammar(gammar: LocalFourPoint = None, gchi0_core: LocalBubble = None, u=None):
    u_r = get_ur(u=u, channel=gammar.channel)
    gchi_aux = np.array([local_gchi_aux_from_gammar_wn(gammar=gammar.mat[wn], gchi0=gchi0_core.gchi0[wn],
                                                       beta=gammar.beta, u=u_r) for wn in gammar.wn_lin])
    return LocalFourPoint(matrix=gchi_aux, channel=gammar.channel, beta=gammar.beta, wn=gammar.wn)


def local_gchi_aux_from_gammar_wn(gammar=None, gchi0=None, beta=1.0, u=1.0):
    gchi0_inv = np.diag(1. / gchi0)
    chi_aux_inv = gchi0_inv + gammar - u / (beta * beta)
    return np.linalg.inv(chi_aux_inv)


# ==================================================================================================================


# ==================================================================================================================
def local_fermi_bose_from_chi_aux(gchi_aux: LocalFourPoint = None, gchi0: LocalBubble = None):
    vrg = 1. / gchi0.gchi0 * 1. / gchi0.beta * np.sum(gchi_aux.mat, axis=-1)
    return LocalThreePoint(matrix=vrg, channel=gchi_aux.channel, beta=gchi_aux.beta, wn=gchi_aux.wn)


def local_fermi_bose_urange(vrg: LocalThreePoint = None, niv_urange=-1):
    if (niv_urange == -1):
        niv_urange = vrg.niv
    vrg_urange = 1. / vrg.beta * np.ones((vrg.niw, 2 * niv_urange), dtype=complex)
    vrg_urange[:, niv_urange - vrg.niv:niv_urange + vrg.niv] = vrg.mat
    return LocalThreePoint(matrix=vrg_urange, channel=vrg.channel, beta=vrg.beta, wn=vrg.wn)


def local_fermi_bose_asympt(vrg: LocalThreePoint = None, chi_urange: LocalSusceptibility = None, u=None, niv_core=None):
    u_r = get_ur(u=u, channel=vrg.channel)
    # vrg_asympt = vrg.mat #* (1 - u_r * chi_urange.mat_asympt[:, None]) / (1 - u_r * chi_urange.mat[:, None])
    # vrg_asympt[:, vrg.niv - niv_core:vrg.niv + niv_core] *= (1 - u_r * chi_urange.mat[:, None]) / (1 - u_r * chi_urange.mat_asympt[:, None])
    vrg_asympt = vrg.mat * (1 - u_r * chi_urange.mat[:, None]) / (1 - u_r * chi_urange.mat_asympt[:, None])
    return LocalThreePoint(matrix=vrg_asympt, channel=vrg.channel, beta=vrg.beta, iw=vrg.wn)


def local_fermi_bose_from_chi_aux_urange(gchi_aux: LocalFourPoint = None, gchi0: LocalBubble = None, niv_urange=-1):
    vrg = local_fermi_bose_from_chi_aux(gchi_aux=gchi_aux, gchi0=gchi0)
    vrg = local_fermi_bose_urange(vrg=vrg, niv_urange=niv_urange)
    return vrg


# ======================================================================================================================

# ======================================================================================================================

def local_vertex_urange(gchi_aux: LocalFourPoint = None, gchi0_urange=None, gchi0_core=None, vrg: LocalThreePoint = None,
                        chi=None, u=None):
    u_r = get_ur(u=u, channel=vrg.channel)
    niv_urange = np.shape(gchi0_urange)[-1] // 2
    niv_core = np.shape(gchi_aux.mat)[-1] // 2
    F_urange = u_r * (1 - u_r * chi[:, None, None]) * vrg.mat[:, :, None] * vrg.mat[:, None, :]
    unity = np.eye(np.shape(gchi0_core)[-1], dtype=complex)
    F_urange[:, niv_urange - niv_core:niv_urange + niv_core, niv_urange - niv_core:niv_urange + niv_core] += 1. / gchi0_core[:, :, None] * (
            unity - gchi_aux.mat * 1. / gchi0_core[:, None, :])
    return F_urange


def local_vertex_inverse_bse_wn(gamma=None, chi0=None, u_r=None, beta=None):
    niv = np.shape(gamma)[-1] // 2
    niv_u = np.shape(chi0)[-1] // 2
    gamma_u = u_r * np.ones((2 * niv_u, 2 * niv_u), dtype=complex) * 1. / beta ** 2
    gamma_u[niv_u - niv:niv_u + niv, niv_u - niv:niv_u + niv] = gamma  # gamma contains internally 1/beta^2
    return np.matmul(gamma_u, np.linalg.inv(np.eye(2 * niv_u, dtype=complex) + gamma_u * chi0[:, None]))
    # return np.linalg.inv(np.linalg.inv(gamma_u))- np.diag(chi0))


def local_vertex_inverse_bse(gamma=None, chi0=None, u=None):
    u_r = get_ur(u=u, channel=gamma.channel)
    return np.array([local_vertex_inverse_bse_wn(gamma=gamma.mat[wn], chi0=chi0.gchi0[wn], u_r=u_r, beta=gamma.beta) for wn in gamma.wn_lin])


# ======================================================================================================================
# ----------------------------------------------- NONLOCAL BUBBLE CLASS ------------------------------------------------
# ======================================================================================================================

class Bubble():
    ''' Computes the Bubble suszeptibility \chi_0 = - beta GG
        Layout of gkiw dimensions is: [Nkx,Nky,Nkz,Niv]
    '''

    asympt = None  # Asymptotic correction from G = 1/iv

    def __init__(self, gk=None, gkpq=None, beta=None, wn=None):
        self._gk = gk
        self._gkpq = gkpq
        self._beta = beta
        self.set_chi0()
        self.wn = wn
        self.set_asympt()

    @property
    def gk(self):
        return self._gk

    @property
    def gkpq(self):
        return self._gkpq

    @property
    def niv(self) -> int:
        return self._gk.shape[-1] // 2

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def chi0(self):
        return self._chi0

    @property
    def gchi0(self):
        return self._gchi0

    @property
    def chi0_asympt(self):
        return self.chi0 + self.asympt

    def set_chi0(self):
        self._chi0 = self.get_chi0()

    def get_chi0(self):
        return - 1. / self._beta * np.sum(np.mean(self.gk * self.gkpq, axis=(0, 1, 2)))

    def vec_get_chi0(self):
        return np.array([self.get_chi0(wn=wn) for wn in self._iw])

    def set_asympt(self):
        self.asympt = self.get_asympt_correction(wn=self.wn)

    def get_asympt_correction(self, wn=0):
        vn = (2 * mf.vn(n=self.niv) + 1)
        vpwn = (2 * mf.vn(n=self.niv) + 1) - 2 * wn
        if (wn == 0):
            return self.beta / np.pi ** 2 * (np.pi ** 2 / 8 - 0.5 * np.sum(1. / vn * 1. / vpwn))
        else:
            return self.beta / np.pi ** 2 * (- 0.5 * np.sum(1. / vn * 1. / vpwn))

    def set_gchi0(self):
        self._gchi0 = self.get_gchi0()

    def get_gchi0(self):
        return - self.beta * np.mean(self.gk * self.gkpq, axis=(0, 1, 2))


def get_ggv(giw=None, niv_ggv=-1):
    niv = giw.shape[0] // 2
    if (niv_ggv == -1):
        niv_ggv = niv
    return giw[niv - niv_ggv:niv + niv_ggv][:, None] * giw[niv - niv_ggv:niv + niv_ggv][None, :]


# ======================================================================================================================
# ----------------------------------------- NONLOCAL SUSCEPTIBILITY CLASS ----------------------------------------------
# ======================================================================================================================

class Susceptibility():
    ''' Class for the non-local susceptibility '''

    asympt = None  # asymptotic correction from Chi = Chi0

    def __init__(self, matrix=None, channel='dens', beta=None, u=None, chi0_urange=None):
        self._u = u
        self._channel = channel
        self._mat = matrix
        self._beta = beta

        if (chi0_urange is not None):
            self.set_asympt(chi0_urange=chi0_urange)

    @property
    def channel(self):
        return self._channel

    @property
    def mat(self):
        return self._mat

    @mat.setter
    def mat(self, matrix):
        self._mat = matrix

    @property
    def beta(self):
        return self._beta

    @property
    def u(self):
        return self._u

    @property
    def u_r(self):
        return get_ur(self.u, self.channel)

    @property
    def mat_asympt(self):
        return self.mat + self.asympt

    def set_asympt(self, chi0_urange: Bubble = None):
        self.asympt = chi0_urange.chi0_asympt - chi0_urange.chi0


class FullQ():
    ''' Contains an object on the full {q,w} grid
        Dimension layout is [{q,w},...]
    '''

    def __init__(self, channel='dens', beta=1.0, u=1.0, qiw: ind.qiw = None):
        self._u = u
        self._beta = beta
        self._channel = channel
        self._qiw = qiw
        self._mat = [0] * qiw.size

    @property
    def channel(self):
        return self._channel

    @property
    def mat(self):
        return self._mat

    @mat.setter
    def mat(self, value, index):
        self._mat[index] = value

    @property
    def beta(self):
        return self._beta

    @property
    def u(self):
        return self._u

    @property
    def u_r(self):
        return get_ur(self.u, self.channel)

    @property
    def qiw(self):
        return self._qiw

    def mat_to_array(self):
        self._mat = np.array(self.mat)


class LadderSusceptibility(Susceptibility):
    ''' Class for a ladder susceptibility object. Stores for {q,w} '''

    def __init__(self, qiw=None, **kwargs):
        Susceptibility.__init__(self, **kwargs)
        self.qiw = qiw
        self.mat = [0] * self.nqiw

    @property
    def qiw(self):
        return self._qiw

    @qiw.setter
    def qiw(self, value):
        self._qiw = value

    @property
    def nqiw(self):
        return self.qiw.shape[0]

    def mat_to_array(self):
        self.mat = np.array(self.mat)


# ======================================================================================================================
# ------------------------------------------- NONLOCAL FOUR POINT CLASS ------------------------------------------------
# ======================================================================================================================

class FourPoint():
    ''' Parent class for non-local {iv,iv'} slice of a four-point correlation functions '''

    def __init__(self, matrix=None, channel='dens', beta=1.0, u=1.0):
        self._u = u
        self._channel = channel
        self.mat = matrix
        self._beta = beta

    @property
    def niv(self):
        return self.mat.shape[-1] // 2

    @property
    def channel(self):
        return self._channel

    @property
    def mat(self):
        return self._mat

    @mat.setter
    def mat(self, matrix):
        self._mat = matrix

    @property
    def beta(self):
        return self._beta

    @property
    def u(self):
        return self._u

    @property
    def u_r(self):
        return get_ur(self.u, self.channel)

    def cut_iv(self, niv_cut=10):
        niv = self.niv
        self._mat = self._mat[..., niv - niv_cut:niv + niv_cut, niv - niv_cut:niv + niv_cut]

    def contract_legs(self):
        return 1. / self.beta ** 2 * np.sum(self._mat, axis=(-2, -1))


class LadderFourPoint(FourPoint):
    ''' Class for a ladder four-point object. Stores for {q,w} four-vector {iv,iv'} slices'''

    def __init__(self, qiw=None, **kwargs):
        FourPoint.__init__(self, **kwargs)
        self.qiw = qiw

    @property
    def qiw(self):
        return self._qiw

    @qiw.setter
    def qiw(self, value):
        self._qiw = value

    @property
    def nqiw(self):
        return np.size(self.qiw)


class ThreePoint(FourPoint):
    ''' Class for three-point objects like the Fermi-bose vertex'''

    def contract_legs(self):
        return 1. / self.beta * np.sum(self._mat, axis=(-1))


class LadderThreePoint(ThreePoint):
    ''' Class for a ladder susceptibility object. Stores for {q,w} '''

    def __init__(self, qiw=None, **kwargs):
        ThreePoint.__init__(self, **kwargs)
        self.qiw = qiw
        self.mat = [0] * self.nqiw

    @property
    def qiw(self):
        return self._qiw

    @qiw.setter
    def qiw(self, value):
        self._qiw = value

    @property
    def nqiw(self):
        return np.size(self.qiw)

    def mat_to_array(self):
        self.mat = np.array(self.mat)


class LadderObject():
    ''' Parent class for ladder objects. Contains {q,iw} arrays of basis object'''

    def __init__(self, qiw=None, channel=None, beta=None, u=None):
        self.qiw = qiw
        self._ladder = [0] * self.nqiw
        self._u = u
        self._channel = channel
        self._beta = beta
        self._mat = None

    @property
    def qiw(self):
        return self._qiw

    @qiw.setter
    def qiw(self, value):
        self._qiw = value

    @property
    def ladder(self):
        return self._ladder

    @property
    def nqiw(self):
        return self.qiw.shape[0]

    @property
    def mat(self):
        return self._mat

    @property
    def channel(self):
        return self._channel

    @property
    def beta(self):
        return self._beta

    @property
    def u(self):
        return self._u

    @property
    def u_r(self):
        return get_ur(self.u, self.channel)

    def set_qiw_mat(self):
        other_size = self.ladder[0].mat.shape
        self._mat = np.zeros((self.nqiw,) + other_size, dtype=self.ladder[0].mat.dtype)

        for iqw, qiw in enumerate(self.qiw):
            self._mat[iqw] = self.ladder[iqw].mat

    # ----------------------------------- FREE FUNCTIONS FOR THE NONLOCAL FOUR POINT CLASS ---------------------------------


# ======================================================================================================================

def construct_gchi_aux(gammar: LocalFourPoint = None, gchi0: Bubble = None, u=1.0, wn_lin=0):
    u_r = get_ur(u=u, channel=gammar.channel)
    return FourPoint(matrix=gchi_aux_from_gammar(gammar=gammar.mat[wn_lin], gchi0=gchi0.gchi0, beta=gammar.beta, u=u_r)
                     , channel=gammar.channel, beta=gammar.beta, u=u)


def gchi_aux_from_gammar(gammar=None, gchi0=None, beta=1.0, u=1.0):
    gchi0_inv = np.diag(1. / gchi0)
    chi_aux_inv = gchi0_inv + gammar - u / (beta * beta)
    chi_aux = np.linalg.inv(chi_aux_inv)
    return chi_aux


# ==================================================================================================================
def fermi_bose_from_chi_aux(gchi_aux: FourPoint = None, gchi0: Bubble = None):
    vrg = 1. / gchi0.gchi0 * 1. / gchi0.beta * np.sum(gchi_aux.mat, axis=-1)
    return ThreePoint(matrix=vrg, channel=gchi_aux.channel, beta=gchi_aux.beta, u=gchi_aux.u)


def fermi_bose_urange(vrg: ThreePoint = None, niv_urange=-1):
    if (niv_urange == -1):
        niv_urange = vrg.niv
    vrg_urange = 1. / vrg.beta * np.ones((2 * niv_urange,), dtype=complex)
    vrg_urange[niv_urange - vrg.niv:niv_urange + vrg.niv] = vrg.mat
    return ThreePoint(matrix=vrg_urange, channel=vrg.channel, beta=vrg.beta, u=vrg.u)


def fermi_bose_asympt(vrg: ThreePoint = None, chi_urange: Susceptibility = None):
    vrg_asympt = vrg.mat * (1 - vrg.u_r * chi_urange.mat) / (1 - vrg.u_r * chi_urange.mat_asympt)
    return ThreePoint(matrix=vrg_asympt, channel=vrg.channel, beta=vrg.beta, u=vrg.u)


def fermi_bose_from_chi_aux_urange(gchi_aux: FourPoint = None, gchi0: Bubble = None, niv_urange=-1):
    vrg_core = fermi_bose_from_chi_aux(gchi_aux=gchi_aux, gchi0=gchi0)
    vrg = fermi_bose_urange(vrg=vrg_core, niv_urange=niv_urange)
    return vrg, vrg_core


# ==================================================================================================================


# ------------------------------------- FREE FUNCTIONS FOR NONLOCAL SUSCEPTIBILITY CLASS -------------------------------
# ======================================================================================================================

def chi_rpa(chi0_urange: Bubble = None, channel=None, u=None):
    u_r = get_ur(u=u, channel=channel)
    chi = chi0_urange.chi0 / (1 + u_r * chi0_urange.chi0)
    return Susceptibility(matrix=chi, channel=channel, beta=chi0_urange.beta, u=u, chi0_urange=chi0_urange)


def chi_phys_from_chi_aux(chi_aux: Susceptibility = None, chi0_urange: Bubble = None, chi0_core: Bubble = None):
    chi = 1. / (1. / (chi_aux.mat + chi0_urange.chi0 - chi0_core.chi0) + chi_aux.u_r)
    return Susceptibility(matrix=chi, channel=chi_aux.channel, beta=chi_aux.beta, u=chi_aux.u, chi0_urange=chi0_urange)


def susceptibility_from_four_point(four_point: FourPoint = None):
    return Susceptibility(matrix=four_point.contract_legs(), channel=four_point.channel
                          , beta=four_point.beta, u=four_point.u)


# ------------------------------------- WRAPPER FUNCTIONS FOR NONLOCAL SUSCEPTIBILITY CLASS ----------------------------
# ======================================================================================================================

# -------------------------------------------- DGA SUSCEPTIBILITY ------------------------------------------------------
def rpa_susceptibility(dga_conf: conf.DgaConfig = None, dmft_input=None, qiw_indizes=None, sigma=None):
    beta = dga_conf.sys.beta
    u = dga_conf.sys.u
    mu = dmft_input['mu']
    niv_urange = dga_conf.box.niv_urange

    g_generator = tp.GreensFunctionGenerator(beta=beta, kgrid=dga_conf.k_grid, hr=dga_conf.sys.hr, sigma=sigma)

    gk_urange = g_generator.generate_gk(mu=mu, qiw=[0, 0, 0, 0], niv=niv_urange)

    chi_rpa_dens = LadderSusceptibility(channel='dens', beta=beta, u=u, qiw=qiw_indizes)
    chi_rpa_magn = LadderSusceptibility(channel='magn', beta=beta, u=u, qiw=qiw_indizes)

    for iqw in range(qiw_indizes.shape[0]):
        wn = qiw_indizes[iqw][-1]
        q_ind = qiw_indizes[iqw][0]
        q = dga_conf.q_grid.irr_kmesh[:, q_ind]
        qiw = np.append(q, wn)
        gkpq_urange = g_generator.generate_gk(mu=mu, qiw=qiw, niv=niv_urange)

        chi0q_urange = Bubble(gk=gk_urange.gk, gkpq=gkpq_urange.gk, beta=gk_urange.beta, wn=wn)

        chiq_dens = chi_rpa(chi0_urange=chi0q_urange, channel='dens', u=u)
        chiq_magn = chi_rpa(chi0_urange=chi0q_urange, channel='magn', u=u)

        chi_rpa_dens.mat[iqw] = chiq_dens.mat_asympt
        chi_rpa_magn.mat[iqw] = chiq_magn.mat_asympt

    chi_rpa_dens.mat_to_array()
    chi_rpa_magn.mat_to_array()

    chi = {
        'dens': chi_rpa_dens,
        'magn': chi_rpa_magn
    }

    return chi


def chi_aux_asympt(chi_aux: FourPoint = None, chi: Susceptibility = None):
    # u = chi_aux.u_r
    # u_mat = np.ones(np.shape(chi_aux.mat), dtype=complex) * u
    # chi_u_chi = 1./chi_aux.beta**2 * np.matmul(chi_aux.mat, np.matmul(u_mat, chi_aux.mat))
    # return chi_aux.mat + chi_u_chi * (1 - u * chi.mat) * (
    #             (1 - u * chi.mat) / (1 - u * chi.mat_asympt) - 1.)
    return chi_aux.mat


# -------------------------------------------- DGA SUSCEPTIBILITY ------------------------------------------------------
def dga_susceptibility(dga_conf: conf.DgaConfig = None, dmft_input=None, gamma_dmft=None, qiw_grid=None,
                       file=None, k_grid=None, q_grid=None, hr=None, sigma=None, save_vrg=True):
    '''

    :param dmft_input: Dictionary containing input from DMFT.
    :param local_sde:
    :param hr:
    :param sigma: input self-energy
    :param kgrid:
    :param box_sizes:
    :param qiw_grid: [nqx*nqy*nqz*2*niw,4] flattened meshgrid. Layout: {qx,qy,qz,iw}
    :return:
    '''
    if (dga_conf.opt.do_pairing_vertex):
        import PairingVertex as pv
    beta = dga_conf.sys.beta
    u = dga_conf.sys.u
    mu = dga_conf.sys.mu
    niw = dga_conf.box.niw_core
    niv_core = dga_conf.box.niv_core
    niv_pp = dga_conf.box.niv_pp
    niv_urange = dga_conf.box.niv_urange
    niw_vrg_save = dga_conf.box.niw_vrg_save
    niv_vrg_save = dga_conf.box.niv_vrg_save
    gamma_dens_loc = gamma_dmft['dens']
    gamma_magn_loc = gamma_dmft['magn']

    chi_dens = LadderSusceptibility(channel='dens', beta=beta, u=u, qiw=qiw_grid)
    chi_magn = LadderSusceptibility(channel='magn', beta=beta, u=u, qiw=qiw_grid)

    vrg_dens = LadderObject(qiw=qiw_grid, channel='dens', beta=beta, u=u)
    vrg_magn = LadderObject(qiw=qiw_grid, channel='magn', beta=beta, u=u)

    g_generator = tp.GreensFunctionGenerator(beta=beta, kgrid=k_grid, hr=hr, sigma=sigma)

    gk_urange = g_generator.generate_gk(mu=mu, qiw=[0, 0, 0, 0], niv=niv_urange)
    gk_core = copy.deepcopy(gk_urange)
    gk_core.cut_self_iv(niv_cut=niv_core)

    if (dga_conf.opt.do_pairing_vertex):
        ivn = np.arange(-niv_pp, niv_pp)
        omega = np.zeros((2 * niv_pp, 2 * niv_pp))
        for i, vi in enumerate(ivn):
            for j, vip in enumerate(ivn):
                omega[i, j] = vi - vip

    for iqw in range(qiw_grid.shape[0]):
        wn = qiw_grid[iqw][-1]
        q_ind = qiw_grid[iqw][0]
        q = q_grid.irr_kmesh[:, q_ind]
        qiw = np.append(-q, wn)  # WARNING: Here I am not sure if it should be +q or -q.
        wn_lin = np.array(mf.cen2lin(wn, -niw), dtype=int)
        gkpq_urange = g_generator.generate_gk(mu=mu, qiw=qiw, niv=niv_urange)

        gkpq_core = copy.deepcopy(gkpq_urange)
        gkpq_core.cut_self_iv(niv_cut=niv_core)

        chi0q_core = Bubble(gk=gk_core.gk, gkpq=gkpq_core.gk, beta=gk_core.beta, wn=wn)
        chi0q_urange = Bubble(gk=gk_urange.gk, gkpq=gkpq_urange.gk, beta=gk_urange.beta, wn=wn)

        gchi_aux_dens = construct_gchi_aux(gammar=gamma_dens_loc, gchi0=chi0q_core, u=u, wn_lin=wn_lin)
        gchi_aux_magn = construct_gchi_aux(gammar=gamma_magn_loc, gchi0=chi0q_core, u=u, wn_lin=wn_lin)

        chi_aux_dens = susceptibility_from_four_point(four_point=gchi_aux_dens)
        chi_aux_magn = susceptibility_from_four_point(four_point=gchi_aux_magn)

        chiq_dens = chi_phys_from_chi_aux(chi_aux=chi_aux_dens, chi0_urange=chi0q_urange,
                                          chi0_core=chi0q_core)

        chiq_magn = chi_phys_from_chi_aux(chi_aux=chi_aux_magn, chi0_urange=chi0q_urange,
                                          chi0_core=chi0q_core)

        vrgq_dens, vrgq_dens_core = fermi_bose_from_chi_aux_urange(gchi_aux=gchi_aux_dens, gchi0=chi0q_core,
                                                                   niv_urange=niv_urange)
        vrgq_dens = fermi_bose_asympt(vrg=vrgq_dens, chi_urange=chiq_dens)
        vrgq_magn, vrgq_magn_core = fermi_bose_from_chi_aux_urange(gchi_aux=gchi_aux_magn, gchi0=chi0q_core,
                                                                   niv_urange=niv_urange)
        vrgq_magn = fermi_bose_asympt(vrg=vrgq_magn, chi_urange=chiq_magn)

        chi_dens.mat[iqw] = chiq_dens.mat_asympt
        chi_magn.mat[iqw] = chiq_magn.mat_asympt

        vrg_dens.ladder[iqw] = vrgq_dens
        vrg_magn.ladder[iqw] = vrgq_magn

        if (dga_conf.opt.do_pairing_vertex):
            if (np.abs(wn) < 2 * niv_pp):
                condition = omega == wn

                gchi_aux_magn = chi_aux_asympt(chi_aux=gchi_aux_magn, chi=chiq_magn)
                gchi_aux_dens = chi_aux_asympt(chi_aux=gchi_aux_dens, chi=chiq_dens)
                f1_magn_slice, f2_magn_slice = pv.ladder_vertex_from_chi_aux_components(gchi_aux=gchi_aux_magn,
                                                                                        vrg=vrgq_magn_core.mat,
                                                                                        gchi0=chi0q_core.gchi0,
                                                                                        beta=beta,
                                                                                        u_r=get_ur(u=u, channel='magn'))
                f1_dens_slice, f2_dens_slice = pv.ladder_vertex_from_chi_aux_components(gchi_aux=gchi_aux_dens,
                                                                                        vrg=vrgq_dens_core.mat,
                                                                                        gchi0=chi0q_core.gchi0,
                                                                                        beta=beta,
                                                                                        u_r=get_ur(u=u, channel='dens'))

                group = '/irrq{:03d}wn{:04d}/'.format(*qiw_grid[iqw])
                file[group + 'f1_magn/'] = pv.get_pp_slice_4pt(mat=f1_magn_slice, condition=condition, niv_pp=niv_pp)
                file[group + 'f2_magn/'] = pv.get_pp_slice_4pt(mat=f2_magn_slice, condition=condition, niv_pp=niv_pp)
                file[group + 'f1_dens/'] = pv.get_pp_slice_4pt(mat=f1_dens_slice, condition=condition, niv_pp=niv_pp)
                file[group + 'f2_dens/'] = pv.get_pp_slice_4pt(mat=f2_dens_slice, condition=condition, niv_pp=niv_pp)
                file[group + 'condition/'] = condition

        # Save the lowest 5 frequencies for the spin-fermion vertex::
        if (np.abs(wn) < niw_vrg_save and save_vrg == True):
            group = '/irrq{:03d}wn{:04d}/'.format(*qiw_grid[iqw])
            file[group + 'vrg_magn/'] = beta * vrgq_magn.mat[niv_urange - niv_vrg_save:niv_urange + niv_vrg_save]
            file[group + 'vrg_dens/'] = beta * vrgq_dens.mat[niv_urange - niv_vrg_save:niv_urange + niv_vrg_save]

    chi_dens.mat_to_array()
    chi_magn.mat_to_array()

    vrg_dens.set_qiw_mat()
    vrg_magn.set_qiw_mat()

    dga_chi = {
        'dens': chi_dens,
        'magn': chi_magn
    }
    dga_vrg = {
        'dens': vrg_dens,
        'magn': vrg_magn
    }
    return dga_chi, dga_vrg


def load_spin_fermion(output_path=None, name='Qiw', mpi_size=None, nq=None, niv=None, niw=None):
    '''WARNING: This currently works only if we are using positive matsubaras only. '''
    import h5py
    import re

    # Collect data from subfiles (This is quite ugly, as it is hardcoded to my structure. This should be replaced by a general routine):
    vrg_dens = np.zeros((nq, niw, 2 * niv), dtype=complex)
    vrg_magn = np.zeros((nq, niw, 2 * niv), dtype=complex)

    for ir in range(mpi_size):
        fname = output_path + name + 'Rank{0:05d}'.format(ir) + '.hdf5'
        file_in = h5py.File(fname, 'r')
        for key1 in list(file_in.file.keys()):
            # extract the q indizes from the group name!
            irrq = np.array(re.findall("\d+", key1), dtype=int)[0]
            wn = np.array(re.findall("\d+", key1), dtype=int)[1]
            if (wn < niw):
                # wn_lin = np.array(mf.cen2lin(wn, -niw), dtype=int)
                vrg_magn[irrq, wn, :] = file_in.file[key1 + '/vrg_magn/'][()]
                vrg_dens[irrq, wn, :] = file_in.file[key1 + '/vrg_dens/'][()]

        file_in.close()

    return vrg_dens, vrg_magn


def allgather_qiw_and_build_fbziw(dga_conf=None, mat=None, distributor=None, qiw_grid=None):
    ''' Gather a {q,iw} object and rebuild the full q, iw structure'''
    mat = distributor.allgather(rank_result=mat)
    mat = dga_conf.q_grid.irrk2fbz(mat=qiw_grid.reshape_matrix(mat))
    mat = mf.wplus2wfull(mat=mat)
    return mat


def ladder_susc_allgather_qiw_and_build_fbziw(dga_conf=None, distributor=None, mat=None, qiw_grid=None,
                                              qiw_grid_fbz=None, channel=None):
    ''' Gather a Ladder suszeptibility object and rebuild the full q, iw structure'''
    gathered_qiw = LadderSusceptibility(qiw=qiw_grid_fbz.meshgrid, channel=channel, u=dga_conf.sys.u,
                                        beta=dga_conf.sys.beta)
    gathered_qiw.mat = allgather_qiw_and_build_fbziw(dga_conf=dga_conf, mat=mat, distributor=distributor,
                                                     qiw_grid=qiw_grid)
    return gathered_qiw


def save_and_plot_chi_lambda(dga_conf: conf.DgaConfig = None, chi_lambda=None, name=''):
    np.save(dga_conf.nam.output_path + f'chi_lambda_{name}.npy', chi_lambda, allow_pickle=True)
    string_temp = 'Chi[q=(0,0),iw=0,{}]: {}'
    np.savetxt(dga_conf.nam.output_path + f'Knight_shift_{name}.txt',
               [string_temp.format('magn', chi_lambda['magn'].mat[0, 0, 0, dga_conf.box.niw_core]),
                string_temp.format('dens', chi_lambda['dens'].mat[0, 0, 0, dga_conf.box.niw_core])], delimiter=' ',
               fmt='%s')
    import Plotting as plotting
    plotting.plot_chi_fs(chi=chi_lambda['magn'].mat.real, output_path=dga_conf.nam.output_path, kgrid=dga_conf.q_grid,
                         name=f'magn_w0_{name}')
    plotting.plot_chi_fs(chi=chi_lambda['dens'].mat.real, output_path=dga_conf.nam.output_path, kgrid=dga_conf.q_grid,
                         name=f'dens_w0_{name}')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import TwoPoint as twop
    import Hr as hr_mod
    import BrillouinZone as bz

    hr = hr_mod.one_band_2d_t_tp_tpp(t=1.0, tp=-0.2, tpp=0.1)
    nk = (32, 32, 1)
    k_grid = bz.KGrid(nk=nk)
    beta = 10
    sigma = np.zeros((5000,), dtype=complex)
    g_gen = twop.GreensFunctionGenerator(beta=beta, kgrid=k_grid, hr=hr, sigma=sigma)

    mu = g_gen.adjust_mu(n=0.85, mu0=1)
    giwk = g_gen.generate_gk(mu=mu)
    giw = giwk.k_mean()
    niv_core = 20
    niv_urange = 100
    niv_urange2 = 500
    niv_urange3 = 1000
    niv_urange4 = 2000
    wn = 1
    chi0_core = LocalBubble(giw=giw, beta=beta, niv_sum=niv_core, iw=[wn])
    chi0_urange = LocalBubble(giw=giw, beta=beta, niv_sum=niv_urange, iw=[wn])
    chi0_urange2 = LocalBubble(giw=giw, beta=beta, niv_sum=niv_urange2, iw=[wn])
    chi0_urange3 = LocalBubble(giw=giw, beta=beta, niv_sum=niv_urange3, iw=[wn])
    chi0_urange4 = LocalBubble(giw=giw, beta=beta, niv_sum=niv_urange4, iw=[wn])

    plt.plot([chi0_core.chi0, chi0_urange.chi0, chi0_urange2.chi0, chi0_urange3.chi0, chi0_urange4.chi0])
    plt.plot([chi0_core.chi0_asympt, chi0_urange.chi0_asympt, chi0_urange2.chi0_asympt, chi0_urange3.chi0_asympt,
              chi0_urange4.chi0_asympt])
    plt.show()

    print(f'{chi0_core.chi0=}')
    print(f'{chi0_urange.chi0=}')
    print(f'{chi0_urange2.chi0=}')
    print(f'{chi0_urange3.chi0=}')
    print(f'{chi0_urange4.chi0=}')

    print(f'{chi0_core.chi0_asympt=}')
    print(f'{chi0_urange.chi0_asympt=}')
    print(f'{chi0_urange2.chi0_asympt=}')
    print(f'{chi0_urange3.chi0_asympt=}')
    print(f'{chi0_urange4.chi0_asympt=}')
