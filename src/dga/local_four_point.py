'''
    This module contains the LocalFourPoint class which is used to store the local four-point correlation functions.
    Additionally, it contains routines to compute the various local four- and three-point functions.
'''
import numpy as np
import matplotlib.pyplot as plt

from dga import matsubara_frequencies as mf
from dga import two_point as twop
from dga import bubble as bub


def get_ur(u=1.0, channel='dens'):
    if channel == 'magn':
        sign = -1
    elif channel == 'dens':
        sign = 1
    elif channel == 'updo':
        sign = 1
    elif channel == 'upup':
        sign = -1
    else:
        raise ValueError
    return u * sign


def get_sign(channel):
    if channel == 'magn':
        return - 1
    if channel == 'dens':
        return + 1
    else:
        raise ValueError('Channel not in [dens/magn].')


FIGSIZE = (7, 3)
ALPHA = 0.7
BASECOLOR = 'cornflowerblue'


def plot_fourpoint_nu_nup(mat, do_save=True, pdir='./', name='NoName', cmap='RdBu', figsize=FIGSIZE, show=False):
    fig, axes = plt.subplots(ncols=2, figsize=figsize, dpi=251)
    axes = axes.flatten()
    vn = mf.vn(mat, axis=-1)  # pylint: disable=unexpected-keyword-arg
    im1 = axes[0].pcolormesh(vn, vn, mat.real, cmap=cmap)
    im2 = axes[1].pcolormesh(vn, vn, mat.imag, cmap=cmap)
    axes[0].set_title(r'$\Re$')
    axes[1].set_title(r'$\Im$')
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
    if do_save: plt.savefig(pdir + '/' + name + '.png')
    if show:
        plt.show()
    else:
        plt.close()


def plot_three_point_w_v(mat, do_save=True, pdir='./', name='NoName', cmap='RdBu', figsize=FIGSIZE, show=False):
    fig, axes = plt.subplots(ncols=2, figsize=figsize, dpi=500)
    axes = axes.flatten()
    vn = mf.vn(mat, axis=-1)  # pylint: disable=unexpected-keyword-arg
    wn = mf.wn(mat, axis=0)  # pylint: disable=unexpected-keyword-arg
    im1 = axes[0].pcolormesh(wn, vn, mat.real, cmap=cmap)
    im2 = axes[1].pcolormesh(wn, vn, mat.imag, cmap=cmap)
    axes[0].set_title(r'$\Re$')
    axes[1].set_title(r'$\Im$')
    for ax in axes:
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'$\nu$')
        ax.set_aspect('equal')
    fig.suptitle(name)
    fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location='right',
                 pad=0.05)
    fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location='right',
                 pad=0.05)
    plt.tight_layout()
    if do_save: plt.savefig(pdir + '/' + name + '.png')
    if show:
        plt.show()
    else:
        plt.close()


# ======================================================================================================================
# ---------------------------------------------- LOCAL FOURPOINT CLASS  ------------------------------------------------
# ======================================================================================================================
KNOWN_CHANNELS = ['dens', 'magn', 'trip', 'sing', 'updo', 'upup']


class LocalNPoint():
    ''' Parent class for local n-point correlation
    '''

    def __init__(self, channel, mat, beta, u, wn=None, is_full_w=False):
        assert np.size(mat.shape[0]) % 2 == 1, 'wn is not bosonic. Probably w is not the 0th dimension of the input matrix.'
        self.mat = mat  # [w,v,vp]
        self.beta = beta  # inverse temperature
        self.u = u  # interaction strength
        assert channel in KNOWN_CHANNELS, f'Channel with name {channel} not known. Kown are {KNOWN_CHANNELS}.'
        self.channel = channel  # dens/magn/trip/sing/bubb

        if is_full_w:
            self.is_full_w = is_full_w
        else:
            assert wn is not None, 'wn has to be provided if is_full_w is False.'
            self._wn = wn
            self.is_full_w = is_full_w

    @property
    def size(self):
        return self.mat.size

    @property
    def itemsize(self):
        return self.mat.itemsize

    @property
    def u_r(self):
        return get_ur(self.u, self.channel)

    @property
    def wn_lin(self):
        return np.arange(0, self.wn.size)

    @property
    def wn(self):
        if self.is_full_w:
            return mf.wn(mf.niw_from_mat(self.mat, axis=0))
        else:
            return self._wn

    @property
    def niv(self):
        return mf.niv_from_mat(self.mat, axis=-1)

    @property
    def vn(self):
        return mf.vn(self.niv)

    def cut_iw(self, niw_cut=None):
        assert self.is_full_w, 'Full wn range has to be ensured.'
        self.mat = mf.cut_w(self.mat, niw_cut, (0,))

    def contract_legs(self):
        n = len(self.mat.shape) - 1
        axis = tuple([i for i in range(1, n + 1)])
        return 1. / self.beta ** n * np.sum(self.mat, axis=axis)


class LocalFourPoint(LocalNPoint):
    ''' Parent class for local four-point correlation
        IMPORTANT: This object does NOT have to contain the full omega range, but is only defined on a set of points wn
    '''

    def __init__(self, channel, mat, beta, u, wn=None, is_full_w=False):
        super().__init__(channel, mat, beta, u, wn, is_full_w)
        assert len(mat.shape) == 3, 'Local FourPoint object have a structure (w, v, vp)'

    def cut_iv(self, niv_cut=None):
        self.mat = mf.cut_v(self.mat, niv_cut, axes=(-2, -1))

    def append_shell(self, val, niv_shell):
        ''' Append the value val to the shell region of the lfp'''
        self.mat = mf.append_v_vp_shell(self.mat, val, niv_shell)

    def symmetrize_v_vp(self):
        ''' Symmetrize with respect to (v,vp). This is justified for SU(2) symmetric systems. (Thesis Rohringer p. 72) '''
        self.mat = 0.5 * (self.mat + np.transpose(self.mat, axes=(0, 2, 1)))

    def plot(self, iwn=0, pdir='./', name='', do_save=True, niv=-1, verbose=False):
        assert iwn in self.wn, 'omega index not in dataset.'
        iwn_lin = self.wn_lin[self.wn == iwn][0]
        data = mf.cut_v(self.mat[iwn_lin], niv_cut=niv, axes=(-2, -1))
        plot_fourpoint_nu_nup(data, do_save=do_save, pdir=pdir, name=name + f'_wn{iwn}_niv{niv}', show=verbose)


class LocalThreePoint(LocalNPoint):
    ''' Parent class for local three-point correlation [w,v]
        IMPORTANT: This object does NOT have to contain the full omega range, but is only defined on a set of points wn
    '''

    def __init__(self, channel, mat, beta, u, wn=None, is_full_w=False):
        super().__init__(channel=channel, mat=mat, beta=beta, u=u, wn=wn, is_full_w=is_full_w)
        assert len(mat.shape) == 2, 'Local ThreePoint object have a structure (w, v)'

    def append_shell(self, val, niv_shell):
        raise NotImplementedError('Append shell not implemented for LocalThreePoint.')

    def plot(self, iwn=0, pdir='./', name='', do_save=True, niv=-1, verbose=False):
        assert iwn in self.wn, 'omega index not in dataset.'
        iwn_lin = self.wn_lin[self.wn == iwn][0]
        data = mf.cut_v(self.mat[iwn_lin], niv_cut=niv, axes=(-1,))
        plot_three_point_w_v(data, do_save=do_save, pdir=pdir, name=name + f'_niv{niv}', show=verbose)


def get_g2_from_dmft_input(ddict, channel='dens'):
    ''' Create a LocalFourPoint object from a dictionary containing the necessary information.'''
    return LocalFourPoint(channel=channel, mat=ddict[f'g4iw_{channel}'], beta=ddict['beta'], u=ddict['u'], is_full_w=True)


def construct_lfp_from_lnp(lnp: LocalNPoint, mat):
    ''' Construct a new LocalFourPoint object from an existing one, but with a new matrix mat.'''
    if lnp.is_full_w:
        return LocalFourPoint(channel=lnp.channel, mat=mat, beta=lnp.beta, u=lnp.u, is_full_w=lnp.is_full_w)
    else:
        return LocalFourPoint(channel=lnp.channel, mat=mat, beta=lnp.beta, u=lnp.u, wn=lnp.wn)


def construct_ltp_from_lnp(lnp: LocalNPoint, mat):
    ''' Construct a LocalThreePoint object from an existing one, but with a new matrix mat.'''
    if lnp.is_full_w:
        return LocalThreePoint(channel=lnp.channel, mat=mat, beta=lnp.beta, u=lnp.u, is_full_w=lnp.is_full_w)
    else:
        return LocalThreePoint(channel=lnp.channel, mat=mat, beta=lnp.beta, u=lnp.u, wn=lnp.wn)


def get_ggv(giw=None, niv_ggv=-1):
    ''' ggv(v,vp):1/eV^2 = G(v):1/eV * G(vp):1/eV
        ggv(w,v,vp) = delta(w=0) * ggv(v,vp)
    '''
    niv = giw.shape[0] // 2
    if niv_ggv == -1:
        niv_ggv = niv
    return giw[niv - niv_ggv:niv + niv_ggv][:, None] * giw[niv - niv_ggv:niv + niv_ggv][None, :]


def gchir_from_g2(g2: LocalFourPoint = None, giw=None):
    ''' gchi_r:1/eV^3 = beta:1/eV * (G2:1/eV^2 - 2 * GG:1/eV^2 delta_dens) '''
    chir_mat = g2.beta * g2.mat
    if g2.channel == 'dens' and 0 in g2.wn:
        ggv = get_ggv(giw, niv_ggv=g2.niv)
        chir_mat[g2.wn == 0] = g2.beta * (g2.mat[g2.wn == 0] - 2. * ggv)
    chir_mat = np.array(chir_mat)
    return construct_lfp_from_lnp(g2, chir_mat)


def g2_from_chir(chir: LocalFourPoint = None, giw=None):
    ''' G2 = 1/beta * chi_r + 2*GG delta_dens'''
    g2_mat = 1 / chir.beta * chir.mat
    if chir.channel == 'dens' and 0 in chir.wn:
        ggv = get_ggv(giw, niv_ggv=chir.niv)
        g2_mat[chir.wn == 0] = 1 / chir.beta * chir.mat[chir.wn == 0] + 2. * ggv
    g2_mat = np.array(g2_mat)
    return construct_lfp_from_lnp(chir, g2_mat)


def get_gchi0_full(gchi0):
    ''' Return the full qkk' matrix'''
    return np.array([np.diag(gchi0_i) for gchi0_i in gchi0])


def fob2_from_gchir(gchir: LocalFourPoint, gchi0):
    ''' Do not include the beta**2 in F.
        Fob2_r:eV^3 = [chi_0^-1:eV^3 - chi_0^-1:eV^3 chi_r:1/eV^3 chi_0^-1:eV^3]
    '''
    # assert chir.niv == gchi0.niv
    gchi0_full_inv = get_gchi0_full(1. / gchi0)
    f0b2_mat = gchi0_full_inv - 1 / gchi0[:, :, None] * gchir.mat * 1 / gchi0[:, None, :]
    return construct_lfp_from_lnp(gchir, f0b2_mat)


def gchir_from_fob2(fob2_r: LocalFourPoint, gchi0):
    '''
        chi_r = chi_0 - chi_0 Fob2_r chi_0
    '''
    # assert F_r.niv == gchi0.niv
    gchi0_full = get_gchi0_full(gchi0)
    chir_mat = gchi0_full - gchi0[:, :, None] * fob2_r.mat * gchi0[:, None, :]
    return construct_lfp_from_lnp(fob2_r, chir_mat)


def fob2_from_vrg_and_chir(gchi_aux: LocalFourPoint, vrg_v: LocalThreePoint, vrg_vp: LocalThreePoint, chir, gchi0):
    assert gchi_aux.channel == vrg_v.channel, 'Mixing different channels in gchi_aux and vrg is not allowed.'
    assert gchi_aux.mat.shape[:-1] == gchi0.shape, 'gchi_aux.mat and gchi0 have to have the same shape.'

    f_1 = gchi_aux.beta ** 2 * 1 / gchi0[:, :, None] * (
            np.eye(gchi_aux.mat.shape[-1])[None, :, :] - gchi_aux.mat * 1 / gchi0[:, None, :])
    f_2 = gchi_aux.u_r * (1 - gchi_aux.u_r * chir[:, None, None]) * vrg_v.mat[:, :, None] * vrg_vp.mat[:, None, :]
    f = 1 / gchi_aux.beta ** 2 * (f_1 + f_2)
    return construct_lfp_from_lnp(gchi_aux, f)


def gamob2_from_gchir(gchir: LocalFourPoint, gchi0):
    ''' Gamma/beta^2:eV^3 = - ( chi_0^-1:eV^3 - chi_r^-1:eV^3) '''
    gam_r = np.array([gamob2_from_gchir_wn(gchir.mat[iwn_lin], gchi0[iwn_lin]) for iwn_lin in gchir.wn_lin])
    return construct_lfp_from_lnp(gchir, gam_r)


def gamob2_from_gchir_wn(chir=None, gchi0=None):
    gam_r = -(np.diag(1 / gchi0) - np.linalg.inv(chir))
    return gam_r


def gchir_from_gamob2(gammar: LocalFourPoint, gchi0):
    ''' chi_r = ( chi_0^-1 + 1/beta^2 Gamma_r)^(-1) '''
    chi_r = np.array([gchir_from_gamob2_wn(gammar.mat[iwn_lin], gchi0[iwn_lin]) for iwn_lin in gammar.wn_lin])
    return construct_lfp_from_lnp(gammar, chi_r)


def gchir_from_gamob2_wn(gammar, gchi0):
    return np.linalg.inv(np.diag(1 / gchi0) + gammar)


def get_urange(obj: LocalFourPoint, val, niv_urange):
    ''' Append the value val to the urange of the object obj.'''
    new_mat = mf.append_v_vp_shell(obj.mat, val, niv_urange)
    return construct_lfp_from_lnp(obj, new_mat)


def fob2_from_gamob2_urange(gammar: LocalFourPoint, gchi0):
    ''' F = Gamma [1 - Gamma X_0]^(-1)'''
    niv_core = gammar.niv
    niv_full = np.shape(gchi0)[-1] // 2
    niv_urange = niv_full - niv_core
    gamma_urange = get_urange(gammar, gammar.u_r / gammar.beta ** 2, niv_urange)
    eye = np.eye(niv_full * 2)
    fob2 = np.array(
        [gamma_urange.mat[iwn] @ np.linalg.inv(eye + gamma_urange.mat[iwn] * gchi0[iwn][:, None]) for iwn in gammar.wn_lin])
    return construct_lfp_from_lnp(gammar, fob2)


def lam_from_chir(gchir: LocalFourPoint, gchi0):
    ''' lambda vertex'''
    sign = get_sign(gchir.channel)
    lam = -sign * (1 - np.sum(gchir.mat * (1 / gchi0)[:, :, None], axis=-1))
    return construct_ltp_from_lnp(gchir, lam)


def vrg_from_lam(lam: LocalThreePoint, chir):
    sign = get_sign(lam.channel)
    vrg = (1 + sign * lam.mat) / (1 - lam.u_r * chir[:, None])
    return construct_ltp_from_lnp(lam, vrg)


def get_chir_shell(lam_tilde: LocalThreePoint, chi0_shell, gchi0_core, u):
    sign = get_sign(lam_tilde.channel)
    chir_shell = -sign * u * chi0_shell ** 2 \
                 + chi0_shell * (1 - 2 * u * 1 / lam_tilde.beta ** 2 * np.sum((lam_tilde.mat + sign) * gchi0_core, axis=-1))
    return chir_shell


def get_chir_tilde(lam_tilde: LocalThreePoint, chir_core, chi0_shell, gchi0_core, u):
    chi_r_shell = get_chir_shell(lam_tilde, chi0_shell, gchi0_core, u)
    return (chir_core + chi_r_shell) / (1 - (u * chi0_shell) ** 2)


def get_lam_shell(chi0_shell, u):
    '''Unit of lam_shell is 1'''
    return u * chi0_shell


def get_lam_tilde(lam_core: LocalThreePoint, chi0_shell):
    lam_shell = get_lam_shell(chi0_shell, lam_core.u)
    lam_tilde = (lam_core.mat - lam_shell[:, None]) / (1 + lam_core.u_r * chi0_shell[:, None])
    return construct_ltp_from_lnp(lam_core, lam_tilde)


def get_vrg_and_chir_tilde_from_gammar_urange(gamma_r: LocalFourPoint, gchi0_gen: bub.BubbleGenerator, niv_shell=0, sum_axis=-1):
    '''
        Compute the fermi-bose vertex and susceptibility using urange asymptotics proposed in
        Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005
    '''
    niv_core = gamma_r.niv
    # Create necessary bubbles
    niv_full = niv_core + niv_shell
    chi0_core = gchi0_gen.get_chi0(niv_core)
    chi0_urange = gchi0_gen.get_chi0(niv_full)
    gchi0_core = gchi0_gen.get_gchi0(niv_core)

    # Compute the auxiliary susceptibility:
    gchi_aux = gchi_aux_core_from_gammar(gamma_r, gchi0_core)
    chi_aux_core = gchi_aux.contract_legs()

    # Compute the physical susceptibility:
    chi_urange = chi_phys_urange(chi_aux_core, chi0_core, chi0_urange, gamma_r.u,
                                 gamma_r.channel)

    # Compute the fermion-boson vertex:
    vrg = vrg_from_gchi_aux(gchi_aux, gchi0_core, sum_axis=sum_axis)

    return vrg, chi_urange


def get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_r: LocalFourPoint, gchi0_gen: bub.BubbleGenerator, niv_shell=0,
                                               sum_axis=-1):
    '''
        Compute the fermi-bose vertex and susceptibility using the asymptotics proposed in
        Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005
    '''
    niv_core = gamma_r.niv
    # Create necessary bubbles
    niv_full = niv_core + niv_shell
    chi0_core = gchi0_gen.get_chi0(niv_core)
    chi0_urange = gchi0_gen.get_chi0(niv_full)
    chi0_shell = gchi0_gen.get_asymptotic_correction(niv_full)
    chi0_asympt = chi0_urange + chi0_shell
    gchi0_core = gchi0_gen.get_gchi0(niv_core)

    # Compute the auxiliary susceptibility:
    gchi_aux = gchi_aux_core_from_gammar(gamma_r, gchi0_core)
    chi_aux_core = gchi_aux.contract_legs()

    # Compute the physical susceptibility:
    chi_urange = chi_phys_urange(chi_aux_core, chi0_core, chi0_urange, gamma_r.u,
                                 gamma_r.channel)  # .real  # take real part to avoid numerical noise
    chi_asympt = chi_phys_asympt(chi_urange, chi0_urange, chi0_asympt)  # .real  # take real part to avoid numerical noise

    # Compute the fermion-boson vertex:
    vrg = vrg_from_gchi_aux_asympt(gchi_aux, gchi0_core, chi_urange, chi_asympt, sum_axis=sum_axis)

    return vrg, chi_asympt


def get_f_dc_asympt(vrg: LocalThreePoint, gchi_aux, chi_phys, gchi0, u):
    ''' Has an additional beta**2 compared to Motoharus paper due to different definitions of Chi0. '''
    u_r = get_ur(u, vrg.channel)
    eye = np.eye(vrg.niv * 2)
    mat = np.array([vrg.beta ** 2 * 1. / gchi0[i][:, None] * (eye - gchi_aux.mat[i] * 1 / gchi0[i][None, :])
                    + u_r * (1 - u_r * chi_phys[i]) * vrg.mat[i][:, None] * vrg.mat[i][None, :] for i in vrg.wn_lin])
    return construct_lfp_from_lnp(vrg, mat)


def get_vrg_and_chir_tilde_from_chir_uasympt(chir: LocalFourPoint, gchi0_gen: bub.BubbleGenerator, u, niv_shell=0):
    '''
        Compute the fermi-bose vertex and susceptibility using the asymptotics proposed in
        Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005
    '''
    niv_core = chir.niv
    # Create necessary bubbles
    niv_full = niv_core + niv_shell
    chi0_core = gchi0_gen.get_chi0(niv_core)
    chi0_urange = gchi0_gen.get_chi0(niv_full)
    chi0_shell = gchi0_gen.get_asymptotic_correction(niv_full)
    chi0_asympt = chi0_urange + chi0_shell
    gchi0_core = gchi0_gen.get_gchi0(niv_core)

    # Compute the auxiliary susceptibility:
    gchi_aux = gchi_aux_core(chir)
    chi_aux_core = gchi_aux.contract_legs()

    # Compute the physical susceptibility:
    chi_urange = chi_phys_urange(chi_aux_core, chi0_core, chi0_urange, u, chir.channel)
    chi_asympt = chi_phys_asympt(chi_urange, chi0_urange, chi0_asympt)

    # Compute the fermion-boson vertex:
    vrg = vrg_from_gchi_aux_asympt(gchi_aux, gchi0_core, chi_urange, chi_asympt, u)

    return vrg, chi_asympt


#
#
# ==================================================================================================================
# -------------------------------------- ASYMPTOTIC AS PROPOSED BY MOTOHARU ----------------------------------------
# ==================================================================================================================

def gamob2_from_gchir_urange(gchir: LocalFourPoint, gchi0_urange):
    u_r = gchir.u_r
    gammar = np.array(
        [gamob2_from_gchir_urange_wn(gchir=gchir.mat[wn], gchi0_urange=gchi0_urange[wn], niv_core=gchir.niv, beta=gchir.beta,
                                     u=u_r) for wn in gchir.wn_lin])
    return construct_lfp_from_lnp(gchir, gammar)


def gamob2_from_gchir_urange_wn(gchir=None, gchi0_urange=None, niv_core=None, beta=1.0, u=1.0):
    '''
        See Motoharu et al. (DOI 10.1088/2515-7639/ac7e6d) (Eq. A.4 to A.8)
        Gamma  = X^(-1) - X_tilde^(-1) + U
        X_tilde^(-1) = X_0^(-1) + U
    '''

    # Create chi_tilde in the shell region:
    chi_tilde_shell_inv = u / (beta * beta) + np.diag(1. / gchi0_urange)
    chi_tilde_shell = np.linalg.inv(chi_tilde_shell_inv)

    # Cut frequencies to get chi_tilde in the core region:
    inv_core = mf.cut_v(chi_tilde_shell, niv_core, axes=(-2, -1))
    chi_tilde_core = np.linalg.inv(inv_core)

    # Generate chi^(-1)
    chigr_inv = np.linalg.inv(gchir)
    return chigr_inv - chi_tilde_core + u / (beta * beta)


def chi_phys_urange(chir_aux, chi0_core, chi0_urange, u, channel):
    ''' u-range form of the susceptibility '''
    u_r = get_ur(u, channel)
    return 1. / (1. / (chir_aux + chi0_urange - chi0_core) + u_r)


def chi_phys_asympt(chir_urange, chi0_urange, chi0_asympt):
    ''' asymptotic form of the susceptibility '''
    return chir_urange + chi0_asympt - chi0_urange


def gchi_aux_core(gchir: LocalFourPoint):
    ''' WARNING: this routine is not fully tested yet '''
    mat = np.array([np.linalg.inv((np.linalg.inv(gchir.mat[i]) - gchir.u_r / gchir.beta ** 2)) for i in gchir.wn_lin])
    return construct_lfp_from_lnp(gchir, mat)


def gchi_aux_core_from_gammar(gammar: LocalFourPoint, gchi0_core):
    ''' Create chi_aux from gamma_r in the core region'''
    mat = np.array([np.linalg.inv(np.diag(1. / gchi0_core[i]) + gammar.mat[i] - gammar.u_r / gammar.beta ** 2) for i in
                    gammar.wn_lin])
    return construct_lfp_from_lnp(gammar, mat)


def gchi_aux_core_from_gammar_urange(gammar: LocalFourPoint, gchi0):
    ''' Create chi_aux from gamma_r in the urange region'''
    u_r = gammar.u_r
    niv_core = gammar.niv
    niv_full = mf.niv_from_mat(gchi0, axis=-1)
    niv_shell = niv_full - niv_core
    gamma_urange = get_urange(gammar, u_r / gammar.beta ** 2, niv_shell)
    mat = np.array([np.linalg.inv(np.diag(1. / gchi0[i]) + gamma_urange.mat[i] - u_r / gammar.beta ** 2) for i in gammar.wn_lin])
    return construct_lfp_from_lnp(gammar, mat)


def gchi_aux_asympt(gchi_aux: LocalFourPoint, chi_urange, chi_asympt):
    u_mat = np.ones_like(gchi_aux.mat) * gchi_aux.u
    mat = gchi_aux.mat + gchi_aux.mat @ u_mat @ gchi_aux.mat * (
            -(1 - gchi_aux.u_r * chi_urange) + (1 - gchi_aux.u_r * chi_urange) ** 2 / (1 - gchi_aux.u_r * chi_asympt))
    return construct_lfp_from_lnp(gchi_aux, mat)


def local_gchi_aux_from_gammar(gammar: LocalFourPoint = None, gchi0_core=None):
    gchi_aux = np.array([local_gchi_aux_from_gammar_wn(gammar=gammar.mat[wn], gchi0=gchi0_core[wn],
                                                       beta=gammar.beta, u=gammar.u_r) for wn in gammar.wn_lin])
    return construct_lfp_from_lnp(gammar, gchi_aux)


def local_gchi_aux_from_gammar_wn(gammar=None, gchi0=None, beta=1.0, u=1.0):
    gchi0_inv = np.diag(1. / gchi0)
    chi_aux_inv = gchi0_inv + gammar - u / (beta * beta)
    return np.linalg.inv(chi_aux_inv)


def vrg_from_gchi_aux_asympt(gchir_aux: LocalFourPoint, gchi0_core, chir_urange, chir_asympt, sum_axis=-1):
    '''Note: 1/beta is here included in vrg compared to the old code'''
    gchi_aux_sum = np.squeeze(np.sum(gchir_aux.mat, axis=sum_axis))
    vrg = 1 / gchi0_core * gchi_aux_sum * (1 - gchir_aux.u_r * chir_urange[:, None]) / (1 - gchir_aux.u_r * chir_asympt[:, None])
    return construct_ltp_from_lnp(gchir_aux, vrg)


def vrg_from_gchi_aux(gchir_aux: LocalFourPoint, gchi0_core, sum_axis=-1):
    '''Note: 1/beta is here included in vrg compared to the old code'''
    gchi_aux_sum = np.squeeze(np.sum(gchir_aux.mat, axis=sum_axis))
    vrg = 1 / gchi0_core * gchi_aux_sum
    return construct_ltp_from_lnp(gchir_aux, vrg)


def schwinger_dyson_vrg(vrg: LocalThreePoint, chir_phys, giw):
    '''
        Sigma = U*n/2 +
    '''
    u_r = vrg.u_r
    mat_grid = mf.wn_slices_gen(giw, n_cut=vrg.niv, w=vrg.wn)
    sigma_f = u_r / 2 * 1 / vrg.beta * np.sum((1 - (1 - u_r * chir_phys[:, None]) * vrg.mat) * mat_grid, axis=0)
    return sigma_f


def schwinger_dyson_shell(chir_phys, giw, beta, u, n_shell, n_core, wn):
    ''' u-range asymptotic contribution to the self-energy. '''
    # u_r = get_ur(u, channel=channel)
    mat_grid = mf.wn_slices_shell(giw, n_shell=n_shell, n_core=n_core, w=wn)
    sigma_f = u / 2 * 1 / beta * np.sum((u * chir_phys[:, None]) * mat_grid, axis=0)
    return sigma_f


def schwinger_dyson_full(vrg_dens: LocalThreePoint, vrg_magn: LocalThreePoint, chi_dens, chi_magn,
                         giw, n, niv_shell=0):
    ''' This is the currently used version of the Schwinger-Dyson equation. '''
    siw_dens = schwinger_dyson_core_urange(vrg_dens, chi_dens, giw, niv_shell)
    siw_magn = schwinger_dyson_core_urange(vrg_magn, chi_magn, giw, niv_shell)
    hartree = twop.get_smom0(vrg_dens.u, n)
    return hartree + siw_dens + siw_magn


def schwinger_dyson_core_urange(vrg: LocalThreePoint, chi_phys, giw, niv_shell):
    ''' Wrapper to compute the core and urange part of the SDE. '''
    siw_core = schwinger_dyson_vrg(vrg, chi_phys, giw)
    if niv_shell == 0:
        return siw_core
    else:
        siw_shell = schwinger_dyson_shell(chi_phys, giw, vrg.beta, vrg.u, niv_shell, vrg.niv, vrg.wn)
        return mf.concatenate_core_asmypt(siw_core, siw_shell)


def schwinger_dyson_f(f: LocalFourPoint, gchi0, giw):
    ''' Sigma = 1/beta**3 * U sum_{qk} F chi_0 G(k-q)'''
    sigma_f_q = np.sum(1 / f.beta * f.u * f.mat * gchi0[:, None, :], axis=-1)
    mat_grid = mf.wn_slices_gen(giw, n_cut=f.niv, w=f.wn)
    sigma_f = np.sum(sigma_f_q * mat_grid, axis=0)
    return sigma_f


def get_f_diag(chi_dens, chi_magn, channel='dens'):
    '''Ignore chi_pp'''
    if channel == 'magn':
        return 0.5 * mf.w_to_vmvp(chi_dens) - 0.5 * mf.w_to_vmvp(chi_magn)
    elif channel == 'dens':
        return 0.5 * mf.w_to_vmvp(chi_dens) + 1.5 * mf.w_to_vmvp(chi_magn)
    else:
        raise NotImplementedError('Only Channel magn/dens implemented.')


if __name__ == '__main__':
    pass
