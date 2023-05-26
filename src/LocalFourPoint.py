import numpy as np
import MatsubaraFrequencies as mf
import Bubble as bub
import matplotlib.pyplot as plt
import TwoPoint as twop


def get_ur(u=1.0, channel='dens'):
    if (channel == 'magn'):
        sign = -1
    elif (channel == 'dens'):
        sign = 1
    elif (channel == 'updo'):
        sign = 1
    elif (channel == 'upup'):
        sign = -1
    else:
        raise ValueError
    return u * sign


def get_sign(channel):
    if (channel == 'magn'):
        return - 1
    if (channel == 'dens'):
        return + 1
    else:
        raise ValueError('Channel not in [dens/magn].')


FIGSIZE = (7, 3)
ALPHA = 0.7
BASECOLOR = 'cornflowerblue'


def plot_fourpoint_nu_nup(mat, vn, do_save=True, pdir='./', name='NoName', cmap='RdBu', figsize=FIGSIZE, show=False):
    fig, axes = plt.subplots(ncols=2, figsize=figsize, dpi=500)
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
    if (show):
        plt.show()
    else:
        plt.close()


# ======================================================================================================================
# ---------------------------------------------- LOCAL FOURPOINT CLASS  ---------------------------------------------------
# ======================================================================================================================
KNOWN_CHANNELS = ['dens', 'magn', 'trip', 'sing', 'updo', 'upup']


class LocalThreePoint():
    ''' Parent class for local three-point correlation [w,v]
        IMPORTANT: This object does NOT have to contain the full omega range, but is only defined on a set of points wn
    '''

    def __init__(self, channel=None, matrix=None, beta=None, wn=None):
        self.mat = matrix
        self.beta = beta
        self.wn = wn
        assert channel in KNOWN_CHANNELS, f"Channel with name {channel} not known. Kown are {KNOWN_CHANNELS}."
        self.channel = channel  # dens/magn/trip/sing/bubb

    @property
    def niv(self):
        return np.shape(self.mat)[-1] // 2

    @property
    def wn_lin(self):
        return np.arange(0, self.wn.size)

    @property
    def vn(self):
        return mf.vn(self.niv)


class LocalFourPoint():
    ''' Parent class for local four-point correlation
        IMPORTANT: This object does NOT have to contain the full omega range, but is only defined on a set of points wn
    '''

    def __init__(self, channel=None, matrix=None, beta=None, wn=None):
        self.mat = matrix
        self.beta = beta
        if (wn is None):
            self.is_full_w = True
            assert np.size(matrix.shape[0]) % 2 == 1, 'wn is not bosonic. Probably w is not the 0th dimension of the input matrix.'
        else:
            self._wn = wn
            self.is_full_w = False
        assert channel in KNOWN_CHANNELS, f"Channel with name {channel} not known. Kown are {KNOWN_CHANNELS}."
        self.channel = channel  # dens/magn/trip/sing/bubb

    @property
    def wn_lin(self):
        return np.arange(0, self.wn.size)

    @property
    def wn(self):
        if (self.is_full_w):
            return mf.wn(self.mat.shape[0] // 2)
        else:
            return self._wn

    @property
    def niv(self):
        return np.shape(self.mat)[-1] // 2

    @property
    def vn(self):
        return mf.vn(self.niv)

    def cut_iv(self, niv_cut=None):
        self.mat = mf.cut_iv_fp(self.mat, niv_cut)

    def cut_iw(self, niw_cut=None):
        assert self.is_full_w, 'Full wn range has to be ensured.'
        self.mat = mf.cut_w(self.mat, niw_cut, (0,))

    def contract_legs(self):
        return 1. / self.beta ** 2 * np.sum(self.mat, axis=(-2, -1))

    def contract_legs_centered(self, niv_sum):
        return mf.vn_centered_sum(self.mat, self.wn, self.beta, niv_sum=niv_sum)

    def plot(self, iwn=0, pdir='./', name=None, do_save=True, niv=-1):
        assert iwn in self.wn, 'omega index not in dataset.'
        iwn_lin = self.wn_lin[self.wn == iwn][0]
        data = mf.cut_iv_2d(self.mat[iwn_lin], niv_cut=niv)
        vn = mf.cut_v_1d(self.vn, niv_cut=niv)
        plot_fourpoint_nu_nup(data, vn, pdir=pdir, name=name + f'_wn{iwn}_niv{niv}', do_save=do_save)


def get_ggv(giw=None, niv_ggv=-1):
    niv = giw.shape[0] // 2
    if (niv_ggv == -1):
        niv_ggv = niv
    return giw[niv - niv_ggv:niv + niv_ggv][:, None] * giw[niv - niv_ggv:niv + niv_ggv][None, :]


def chir_from_g2_wn(g2=None, ggv=None, beta=None, iwn=0):
    if (ggv is not None and iwn == 0):
        chir = beta * (g2 - 2. * ggv)
    else:
        chir = beta * g2
    return chir


def gchir_from_g2(g2: LocalFourPoint = None, giw=None):
    ''' chi_r = beta * (G2 - 2 * GG \delta_dens) '''
    chir_mat = g2.beta * g2.mat
    if (g2.channel == 'dens' and 0 in g2.wn):
        ggv = get_ggv(giw, niv_ggv=g2.niv)
        chir_mat[g2.wn == 0] = g2.beta * (g2.mat[g2.wn == 0] - 2. * ggv)
    chir_mat = np.array(chir_mat)
    return LocalFourPoint(matrix=chir_mat, channel=g2.channel, beta=g2.beta, wn=g2.wn)


def gchi0_full(gchi0):
    ''' Return the full qkk' matrix'''
    return np.array([np.diag(gchi0_i) for gchi0_i in gchi0])


def Fob2_from_chir(chir: LocalFourPoint, gchi0):
    ''' Do not include the beta**2 in F.
        F_r = [chi_0^-1 - chi_0^-1 chi_r chi_0^-1]
    '''
    # assert chir.niv == gchi0.niv
    gchi0_full_inv = gchi0_full(1. / gchi0)
    F0b2_mat = gchi0_full_inv - 1 / gchi0[:, :, None] * chir.mat * 1 / gchi0[:, None, :]
    return LocalFourPoint(matrix=F0b2_mat, channel=chir.channel, beta=chir.beta, wn=chir.wn)


def chir_from_Fob2(F_r: LocalFourPoint, gchi0):
    ''' Do not include the beta**2 in F.
        F_r = [chi_0^-1 - chi_0^-1 chi_r chi_0^-1]
    '''
    # assert F_r.niv == gchi0.niv
    gchi0 = gchi0_full(gchi0)
    chir_mat = gchi0 - gchi0[:, :, None] * F_r.mat * gchi0[:, None, :]
    return LocalFourPoint(matrix=chir_mat, channel=F_r.channel, beta=F_r.beta, wn=F_r.wn)


def g2_from_chir(chir: LocalFourPoint = None, giw=None):
    ''' G2 = 1/beta * chi_r + 2*GG \delta_dens'''
    chir_mat = 1 / chir.beta * chir.mat
    if (chir.channel == 'dens' and 0 in chir.wn):
        ggv = get_ggv(giw, niv_ggv=chir.niv)
        chir_mat[chir.wn == 0] = 1 / chir.beta * chir.mat[chir.wn == 0] + 2. * ggv
    chir_mat = np.array(chir_mat)
    return LocalFourPoint(matrix=chir_mat, channel=chir.channel, beta=chir.beta, wn=chir.wn)


#
#
def gamob2_from_chir(chir: LocalFourPoint, gchi0):
    ''' Gamma/beta^2 = - ( chi_0^-1 - chi_r^-1) '''
    gam_r = np.array([gamob2_from_chir_wn(chir.mat[iwn_lin], gchi0[iwn_lin]) for iwn_lin in chir.wn_lin])
    return LocalFourPoint(matrix=gam_r, channel=chir.channel, beta=chir.beta, wn=chir.wn)


def gamob2_from_chir_wn(chir=None, gchi0=None):
    gam_r = -(np.diag(1 / gchi0) - np.linalg.inv(chir))
    return gam_r


def gchir_from_gamob2(gammar: LocalFourPoint, gchi0):
    ''' chi_r = ( chi_0^-1 + 1/beta^2 Gamma_r)^(-1) '''
    chi_r = np.array([gchir_from_gamob2_wn(gammar.mat[iwn_lin], gchi0[iwn_lin]) for iwn_lin in gammar.wn_lin])
    return LocalFourPoint(matrix=chi_r, channel=gammar.channel, beta=gammar.beta, wn=gammar.wn)


def gchir_from_gamob2_wn(gammar, gchi0):
    return np.linalg.inv(np.diag(1 / gchi0) + gammar)


# def vrg_from_gam(gam: LocalFourPoint = None, chi0_inv: LocalBubble = None, u=None):
#     u_r = get_ur(u, channel=gam.channel)
#     vrg = np.array([vrg_from_gam_wn(gam.mat[iwn_lin], chi0_inv.mat[iwn_lin], gam.beta, u_r) for iwn_lin in gam.wn_lin])
#     return LocalThreePoint(matrix=vrg, channel=gam.channel, beta=gam.beta, wn=gam.wn)
#
#
# def vrg_from_gam_wn(gam, chi0_inv, beta, u_r):
#     return 1 / beta * chi0_inv * np.sum(np.linalg.inv(np.diag(chi0_inv) + gam - u_r / beta ** 2), axis=-1)
#
#
# def three_leg_from_F(F: LocalFourPoint = None, chi0: LocalBubble = None):
#     ''' sg = \sum_k' F \chi_0 '''
#     sg = 1 / F.beta * jnp.sum(F.mat * chi0.mat[:, None, :], axis=-1)
#     return LocalThreePoint(matrix=sg, channel=F.channel, beta=F.beta, wn=F.wn)

def lam_from_chir(gchir: LocalFourPoint, gchi0):
    ''' lambda vertex'''
    sign = get_sign(gchir.channel)
    lam = -sign * (1 - np.sum(gchir.mat * (1 / gchi0)[:, :, None], axis=-1))
    return LocalThreePoint(channel=gchir.channel, matrix=lam, beta=gchir.beta, wn=gchir.wn)


def vrg_from_lam(lam: LocalThreePoint, chir, u):
    u_r = get_ur(u, lam.channel)
    sign = get_sign(lam.channel)
    vrg = (1 + sign * lam.mat) / (1 - u_r * chir[:, None])
    return LocalThreePoint(channel=lam.channel, matrix=vrg, beta=lam.beta, wn=lam.wn)


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


def get_lam_tilde(lam_core: LocalThreePoint, chi0_shell, u):
    lam_shell = get_lam_shell(chi0_shell, u)
    sign = get_sign(lam_core.channel)
    lam_tilde = (lam_core.mat - lam_shell[:, None]) / (1 + sign * u * chi0_shell[:, None])
    return LocalThreePoint(channel=lam_core.channel, matrix=lam_tilde, beta=lam_core.beta, wn=lam_core.wn)


def get_vrg_and_chir_tilde_from_chir(gchir: LocalFourPoint, chi0_gen: bub.LocalBubble, u, niv_core=-1, niv_shell=0):
    assert niv_core <= gchir.niv, f'niv_core ({niv_core}) has to be smaller or equal to the niv_g2 ({gchir.niv}).'
    if (niv_core == -1):
        niv_core = gchir.niv
    else:
        import copy
        gchir = copy.deepcopy(gchir)
    gchir.cut_iv(niv_core)
    gchi0_core = chi0_gen.get_gchi0(niv_core)
    lam_core = lam_from_chir(gchir, gchi0_core)
    chir_core = gchir.contract_legs()
    if (niv_shell > 0):
        chi0_shell = chi0_gen.get_chi0_shell(niv_core, niv_shell)
        lam_tilde = get_lam_tilde(lam_core, chi0_shell, u)
        chir_tilde = get_chir_tilde(lam_tilde, chir_core, chi0_shell, gchi0_core, u)
        vrg_tilde = vrg_from_lam(lam_tilde, chir_tilde, u)
        return vrg_tilde, chir_tilde
    else:
        vrg_core = vrg_from_lam(lam_core, chir_core, u)
        return vrg_core, chir_core


def get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_r: LocalFourPoint, gchi0_gen: bub.LocalBubble, u, niv_shell = 0, niv_asympt=None):
    '''
        Compute the fermi-bose vertex and susceptibility using the asymptotics proposed in
        Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005
    '''
    if(niv_asympt is None): niv_asympt = 2*niv_shell
    niv_core = gamma_r.niv
    # Create necessary bubbles
    niv_full = niv_core+niv_shell
    chi0_core = gchi0_gen.get_chi0(niv_core)
    chi0_urange = gchi0_gen.get_chi0(niv_full)
    chi0_shell = gchi0_gen.get_chi0_shell(niv_full, niv_asympt)
    chi0_asympt = chi0_urange + chi0_shell
    gchi0_core = gchi0_gen.get_gchi0(niv_core)

    # Compute the auxiliary susceptibility:
    gchi_aux = gchi_aux_core_from_gammar(gamma_r, gchi0_core, u)
    chi_aux_core = gchi_aux.contract_legs()

    # Compute the physical susceptibility:
    chi_urange = chi_phys_urange(chi_aux_core, chi0_core, chi0_urange, u, gamma_r.channel).real # take real part to avoid numerical noise
    chi_asympt = chi_phys_asympt(chi_urange, chi0_urange, chi0_asympt).real # take real part to avoid numerical noise

    # Compute the fermion-boson vertex:
    vrg = vrg_from_gchi_aux(gchi_aux, gchi0_core, chi_urange, chi_asympt, u)

    return vrg, chi_asympt

def get_vrg_and_chir_tilde_from_chir_uasympt(chir: LocalFourPoint, gchi0_gen: bub.LocalBubble, u, niv_shell = 0, niv_asympt=None):
    '''
        Compute the fermi-bose vertex and susceptibility using the asymptotics proposed in
        Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005
    '''
    if(niv_asympt is None): niv_asympt = 2*niv_shell
    niv_core = chir.niv
    # Create necessary bubbles
    niv_full = niv_core+niv_shell
    chi0_core = gchi0_gen.get_chi0(niv_core)
    chi0_urange = gchi0_gen.get_chi0(niv_full)
    chi0_shell = gchi0_gen.get_chi0_shell(niv_full, niv_asympt)
    chi0_asympt = chi0_urange + chi0_shell
    gchi0_core = gchi0_gen.get_gchi0(niv_core)

    # Compute the auxiliary susceptibility:
    gchi_aux = gchi_aux_core(chir, u)
    chi_aux_core = gchi_aux.contract_legs()

    # Compute the physical susceptibility:
    chi_urange = chi_phys_urange(chi_aux_core, chi0_core, chi0_urange, u, chir.channel)
    chi_asympt = chi_phys_asympt(chi_urange, chi0_urange, chi0_asympt)

    # Compute the fermion-boson vertex:
    vrg = vrg_from_gchi_aux(gchi_aux, gchi0_core, chi_urange, chi_asympt, u)

    return vrg, chi_asympt

#
#
# ==================================================================================================================
# -------------------------------------- ASYMPTOTIC AS PROPOSED BY MOTOHARU ----------------------------------------
# ==================================================================================================================

def gammar_from_gchir(gchir: LocalFourPoint, gchi0_urange, u):
    u_r = get_ur(u=u, channel=gchir.channel)
    gammar = np.array(
        [gammar_from_gchir_wn(gchir=gchir.mat[wn], gchi0_urange=gchi0_urange[wn], niv_core=gchir.niv,
                              beta=gchir.beta, u=u_r) for wn in gchir.wn_lin])
    return LocalFourPoint(matrix=gammar, channel=gchir.channel, beta=gchir.beta, wn=gchir.wn)


def gammar_from_gchir_wn(gchir=None, gchi0_urange=None, niv_core=None, beta=1.0, u=1.0):
    full = u / (beta * beta) + np.diag(1. / gchi0_urange)
    inv_full = np.linalg.inv(full)
    inv_core = mf.cut_v(inv_full, niv_core, axes=(-2, -1))
    # inv_core = mf.cut_iv_2d(inv_full, niv_core)
    core = np.linalg.inv(inv_core)
    chigr_inv = np.linalg.inv(gchir)
    return -(core - chigr_inv - u / (beta * beta))


def chi_phys_urange(chir_aux, chi0_core, chi0_urange, u, channel):
    ''' u-range form of the susceptibility '''
    u_r = get_ur(u, channel)
    return 1. / (1. / (chir_aux + chi0_urange - chi0_core) + u_r)


def chi_phys_asympt(chir_urange, chi0_urange, chi0_asympt):
    ''' asymptotic form of the susceptibility '''
    return chir_urange + chi0_asympt - chi0_urange


def gchi_aux_core(gchir: LocalFourPoint, u):
    ''' WARNING: this routine is not fully tested yet '''
    u_r = get_ur(u, gchir.channel)
    mat = np.array([np.linalg.inv((np.linalg.inv(gchir.mat[i]) - u_r / gchir.beta ** 2)) for i in gchir.wn_lin])
    return LocalFourPoint(matrix=mat, channel=gchir.channel, beta=gchir.beta, wn=gchir.wn)

# def gchi_aux_asympt(gchi_aux,):
    # u_r_mat = np.ones_like(gchir.mat[0]) * u_r
    # chir = gchir.contract_legs()
    # mat = np.array([mat[i] + mat[i] @ u_r_mat @ mat[i]/(1 - u_r * chir[i]) for i in gchir.wn_lin])


def gchi_aux_core_from_gammar(gammar: LocalFourPoint, gchi0_core, u):
    u_r = get_ur(u, gammar.channel)
    mat = np.array([np.linalg.inv(np.diag(1. / gchi0_core[i]) + gammar.mat[i] - u_r / gammar.beta ** 2) for i in gammar.wn_lin])
    return LocalFourPoint(matrix=mat, channel=gammar.channel, beta=gammar.beta, wn=gammar.wn)


def gchi_aux_asympt(gchi_aux: LocalFourPoint, chi_urange, chi_asympt, u):
    u_r = get_ur(u, gchi_aux.channel)
    u_mat = np.ones_like(gchi_aux.mat) * u
    mat = gchi_aux.mat + gchi_aux.mat @ u_mat @ gchi_aux.mat * (-(1 - u_r * chi_urange) + (1 - u_r * chi_urange) ** 2 / (1 - u_r * chi_asympt))
    return LocalFourPoint(matrix=mat, channel=gchi_aux.channel, beta=gchi_aux.beta, wn=gchi_aux.wn)


def local_gchi_aux_from_gammar(gammar: LocalFourPoint = None, gchi0_core=None, u=None):
    u_r = get_ur(u=u, channel=gammar.channel)
    gchi_aux = np.array([local_gchi_aux_from_gammar_wn(gammar=gammar.mat[wn], gchi0=gchi0_core[wn],
                                                       beta=gammar.beta, u=u_r) for wn in gammar.wn_lin])
    return LocalFourPoint(matrix=gchi_aux, channel=gammar.channel, beta=gammar.beta, wn=gammar.wn)


def local_gchi_aux_from_gammar_wn(gammar=None, gchi0=None, beta=1.0, u=1.0):
    gchi0_inv = np.diag(1. / gchi0)
    chi_aux_inv = gchi0_inv + gammar - u / (beta * beta)
    return np.linalg.inv(chi_aux_inv)


def vrg_from_gchi_aux(gchir_aux: LocalFourPoint, gchi0_core, chir_urange, chir_asympt, u):
    '''Note: 1/beta is here included in vrg compared to the old code'''
    u_r = get_ur(u, channel=gchir_aux.channel)
    vrg = np.array([1 / gchi0_core[iwn] * np.sum(gchir_aux.mat[iwn], axis=-1)
                    * (1 - u_r * chir_urange[iwn]) / (1 - u_r * chir_asympt[iwn]) for iwn in gchir_aux.wn_lin])
    return LocalThreePoint(channel=gchir_aux.channel, matrix=vrg, beta=gchir_aux.beta, wn=gchir_aux.wn)


# def local_gchi_aux_from_gammar(gammar: LocalFourPoint = None, gchi0_core: bub.LocalBubble = None, u=None):
#     u_r = get_ur(u=u, channel=gammar.channel)
#     gchi_aux = np.array([local_gchi_aux_from_gammar_wn(gammar=gammar.mat[wn], gchi0=gchi0_core.gchi0[wn],
#                                                        beta=gammar.beta, u=u_r) for wn in gammar.wn_lin])
#     return LocalFourPoint(matrix=gchi_aux, channel=gammar.channel, beta=gammar.beta, wn=gammar.wn)


# # ==================================================================================================================

#
#
# # ==================================================================================================================
#
#
# # ==================================================================================================================
# def local_fermi_bose_from_chi_aux(gchi_aux: LocalFourPoint = None, gchi0: LocalBubble = None):
#     vrg = 1. / gchi0.gchi0 * 1. / gchi0.beta * np.sum(gchi_aux.mat, axis=-1)
#     return LocalThreePoint(matrix=vrg, channel=gchi_aux.channel, beta=gchi_aux.beta, wn=gchi_aux.wn)
#
#
# def local_fermi_bose_urange(vrg: LocalThreePoint = None, niv_urange=-1):
#     if (niv_urange == -1):
#         niv_urange = vrg.niv
#     vrg_urange = 1. / vrg.beta * np.ones((vrg.niw, 2 * niv_urange), dtype=complex)
#     vrg_urange[:, niv_urange - vrg.niv:niv_urange + vrg.niv] = vrg.mat
#     return LocalThreePoint(matrix=vrg_urange, channel=vrg.channel, beta=vrg.beta, wn=vrg.wn)
#
#
# def local_fermi_bose_asympt(vrg: LocalThreePoint = None, chi_urange: LocalSusceptibility = None, u=None, niv_core=None):
#     u_r = get_ur(u=u, channel=vrg.channel)
#     # vrg_asympt = vrg.mat #* (1 - u_r * chi_urange.mat_asympt[:, None]) / (1 - u_r * chi_urange.mat[:, None])
#     # vrg_asympt[:, vrg.niv - niv_core:vrg.niv + niv_core] *= (1 - u_r * chi_urange.mat[:, None]) / (1 - u_r * chi_urange.mat_asympt[:, None])
#     vrg_asympt = vrg.mat * (1 - u_r * chi_urange.mat[:, None]) / (1 - u_r * chi_urange.mat_asympt[:, None])
#     return LocalThreePoint(matrix=vrg_asympt, channel=vrg.channel, beta=vrg.beta, iw=vrg.wn)
#
#
# def local_fermi_bose_from_chi_aux_urange(gchi_aux: LocalFourPoint = None, gchi0: LocalBubble = None, niv_urange=-1):
#     vrg = local_fermi_bose_from_chi_aux(gchi_aux=gchi_aux, gchi0=gchi0)
#     vrg = local_fermi_bose_urange(vrg=vrg, niv_urange=niv_urange)
#     return vrg
#
#
# # ======================================================================================================================
#
# # ======================================================================================================================
#
# def local_vertex_urange(gchi_aux: LocalFourPoint = None, gchi0_urange=None, gchi0_core=None, vrg: LocalThreePoint = None,
#                         chi=None, u=None):
#     u_r = get_ur(u=u, channel=vrg.channel)
#     niv_urange = np.shape(gchi0_urange)[-1] // 2
#     niv_core = np.shape(gchi_aux.mat)[-1] // 2
#     F_urange = u_r * (1 - u_r * chi[:, None, None]) * vrg.mat[:, :, None] * vrg.mat[:, None, :]
#     unity = np.eye(np.shape(gchi0_core)[-1], dtype=complex)
#     F_urange[:, niv_urange - niv_core:niv_urange + niv_core, niv_urange - niv_core:niv_urange + niv_core] += 1. / gchi0_core[:, :, None] * (
#             unity - gchi_aux.mat * 1. / gchi0_core[:, None, :])
#     return F_urange
#
#
# def local_vertex_inverse_bse_wn(gamma=None, chi0=None, u_r=None, beta=None):
#     niv = np.shape(gamma)[-1] // 2
#     niv_u = np.shape(chi0)[-1] // 2
#     gamma_u = u_r * np.ones((2 * niv_u, 2 * niv_u), dtype=complex) * 1. / beta ** 2
#     gamma_u[niv_u - niv:niv_u + niv, niv_u - niv:niv_u + niv] = gamma  # gamma contains internally 1/beta^2
#     return np.matmul(gamma_u, np.linalg.inv(np.eye(2 * niv_u, dtype=complex) + gamma_u * chi0[:, None]))
#     # return np.linalg.inv(np.linalg.inv(gamma_u))- np.diag(chi0))
#
#
# def local_vertex_inverse_bse(gamma=None, chi0=None, u=None):
#     u_r = get_ur(u=u, channel=gamma.channel)
#     return np.array([local_vertex_inverse_bse_wn(gamma=gamma.mat[wn], chi0=chi0.gchi0[wn], u_r=u_r, beta=gamma.beta) for wn in gamma.wn_lin])


def schwinger_dyson_vrg(vrg: LocalThreePoint, chir_phys, giw, u):
    ''' Sigma = U*n/2 + '''
    u_r = get_ur(u, channel=vrg.channel)
    mat_grid = mf.wn_slices(giw, n_cut=vrg.niv, wn=vrg.wn)
    sigma_F = u_r / 2 * 1 / vrg.beta * np.sum((1 - (1 - u_r * chir_phys[:, None]) * vrg.mat) * mat_grid, axis=0)
    return sigma_F


def schwinger_dyson_shell(chir_phys, giw, beta, u, n_shell, n_core, wn):
    mat_grid = mf.wn_slices_shell(giw, n_shell=n_shell, n_core=n_core, wn=wn)
    sigma_F = u ** 2 / 2 * 1 / beta * np.sum((chir_phys[:, None]) * mat_grid, axis=0)
    return sigma_F


def schwinger_dyson_full(vrg_dens: LocalThreePoint, vrg_magn: LocalThreePoint, chi_dens, chi_magn, giw, u, n, niv_shell=0):
    siw_dens_core = schwinger_dyson_vrg(vrg_dens, chi_dens, giw, u)
    siw_magn_core = schwinger_dyson_vrg(vrg_magn, chi_magn, giw, u)
    hartree = twop.get_smom0(u, n)
    if (niv_shell == 0):
        return siw_dens_core + siw_magn_core + hartree
    else:
        siw_magn_shell = schwinger_dyson_shell(chi_magn, giw, vrg_magn.beta, u, niv_shell, vrg_magn.niv, vrg_magn.wn)
        siw_dens_shell = schwinger_dyson_shell(chi_dens, giw, vrg_dens.beta, u, niv_shell, vrg_dens.niv, vrg_dens.wn)
        return hartree + mf.concatenate_core_asmypt(siw_dens_core + siw_magn_core, siw_magn_shell + siw_dens_shell)


def schwinger_dyson_vrg_core_from_g2(g2: LocalFourPoint, chi0_gen: bub.LocalBubble, u, niv_core=-1, niv_shell=0):
    assert niv_core <= g2.niv, f'niv_core ({niv_core}) has to be smaller or equal to the niv_g2 ({g2.niv}).'
    if (niv_core == -1): niv_core = g2.niv
    g2.cut_iv(niv_core)
    gchi = gchir_from_g2(g2, chi0_gen.giw)
    chi_core = gchi.contract_legs()
    gchi0_core = chi0_gen.get_gchi0(niv_core)
    lam_core = lam_from_chir()
    pass


def schwinger_dyson_F(F: LocalFourPoint, chi0, giw, u):
    ''' Sigma = 1/beta**3 * U \sum_{qk} F \chi_0 G(k-q)'''
    sigma_F_q = np.sum(1 / F.beta * u * F.mat * chi0[:, None, :], axis=-1)
    mat_grid = mf.wn_slices(giw, n_cut=F.niv, wn=F.wn)
    sigma_F = np.sum(sigma_F_q * mat_grid, axis=0)
    return sigma_F


def get_F_diag(chi_dens, chi_magn, channel='dens'):
    '''Ignore \chi_pp'''
    if (channel == 'magn'):
        return 0.5 * mf.w_to_vmvp(chi_dens) - 0.5 * mf.w_to_vmvp(chi_magn)
    elif (channel == 'dens'):
        return 0.5 * mf.w_to_vmvp(chi_dens) + 1.5 * mf.w_to_vmvp(chi_magn)
    else:
        raise NotImplementedError('Only Channel magn/dens implemented.')


if __name__ == '__main__':
    import sys, os
    import matplotlib.pyplot as plt

    sys.path.append('../test')
    import TestData as td
    import TwoPoint as tp
    import Hk as hamk
    import BrillouinZone as bz
    import MatsubaraFrequencies as mf

    dmft_input_1 = td.get_data_set_2(load_g2=True)
    siw = dmft_input_1['siw'][None, None, None, :]
    beta = dmft_input_1['beta']
    n = dmft_input_1['n']
    u = dmft_input_1['u']
    hr = dmft_input_1['hr']
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=hr)
    siwk = tp.SelfEnergy(siw, beta, pos=False)  # ,smom1=tp.get_smom1(u,n),smom0=tp.get_smom0(u,n))
    # siwk.niv_core = siw.size//2
    giwk = tp.GreensFunction(siwk, ek, n=n)
    g_loc = giwk.g_loc
    niv_asympt = 6000
    giwk.set_g_asympt(niv_asympt)
    print(giwk.mem)
    g2_file = dmft_input_1['g2_file']
    niw = g2_file.get_niw(channel='dens')

    # niw = 30
    niw = 59
    niv = 59
    wn = mf.wn(niw)
    vn_core = mf.vn(niv)
    bubble_gen = bub.LocalBubble(wn=wn, giw=giwk)
    g2_dens = LocalFourPoint(matrix=g2_file.read_g2_iw(channel='dens', iw=wn), beta=beta, wn=wn, channel='dens')
    g2_magn = LocalFourPoint(matrix=g2_file.read_g2_iw(channel='magn', iw=wn), beta=beta, wn=wn, channel='magn')

    g2_dens.plot(0, '../test/TestPlots/', name='G2_dens')
    g2_magn.plot(0, '../test/TestPlots/', name='G2_magn')

    g2_dens.cut_iv(niv)
    g2_magn.cut_iv(niv)
    g2_dens.plot(0, '../test/TestPlots/', name='G2_dens_cut')
    g2_magn.plot(0, '../test/TestPlots/', name='G2_magn_cut')

    gchi0_core = bubble_gen.get_gchi0(g2_dens.niv)
    chi0_core = bubble_gen.get_chi0(g2_dens.niv)
    gchi_dens = gchir_from_g2(g2_dens, g_loc)
    gchi_magn = gchir_from_g2(g2_magn, g_loc)

    F_dens = Fob2_from_chir(gchi_dens, gchi0_core)
    F_magn = Fob2_from_chir(gchi_magn, gchi0_core)
    F_updo = LocalFourPoint(channel='updo', matrix=0.5 * (F_dens.mat - F_magn.mat), beta=F_dens.beta, wn=F_dens.wn)

    F_magn.plot(10, '../test/TestPlots/', name='F_magn_plot_10')
    # gchi_dens.cut_iv(niw)
    # gchi_magn.cut_iv(niw)
    chi_dens_core = gchi_dens.contract_legs()
    chi_magn_core = gchi_magn.contract_legs()
    # chi_dens_core = gchi_dens.contract_legs_centered(gchi_dens.niv-niw//2-1)
    # chi_magn_core = gchi_magn.contract_legs_centered(gchi_dens.niv-niw//2-1)
    chi_dens_core_centered = gchi_dens.contract_legs_centered(niv_sum=gchi_dens.niv - niw // 2 - 1)
    chi_magn_core_centered = gchi_magn.contract_legs_centered(niv_sum=gchi_magn.niv - niw // 2 - 1)

    # %%
    # fig, ax = plt.subplots(1, 2, figsize=(7, 3), dpi=500)
    # ax[0].plot(wn, chi_dens_core.real, '-o', color='cornflowerblue', markeredgecolor='k', alpha=0.8)
    # ax[0].plot(wn, chi_dens_core_centered.real, '-o', color='firebrick', markeredgecolor='k', alpha=0.8)
    # ax[1].plot(wn, chi_magn_core.real, '-o', color='cornflowerblue', markeredgecolor='k', alpha=0.8)
    # ax[1].plot(wn, chi_magn_core_centered.real, '-o', color='firebrick', markeredgecolor='k', alpha=0.8)
    # plt.show()

    # fig, ax = plt.subplots(1, 2, figsize=(7, 3), dpi=500)
    # ax[0].plot(wn, chi_dens_core.real - chi_dens_core_centered.real, '-o', color='cornflowerblue', markeredgecolor='k', alpha=0.8)
    # ax[1].plot(wn, chi_magn_core.real - chi_magn_core_centered.real, '-o', color='cornflowerblue', markeredgecolor='k', alpha=0.8)
    # plt.show()

    # %%

    lam_dens_core = lam_from_chir(gchi_dens, gchi0_core)
    lam_magn_core = lam_from_chir(gchi_magn, gchi0_core)

    vrg_dens_core = vrg_from_lam(lam_dens_core, chi_dens_core, u)
    vrg_magn_core = vrg_from_lam(lam_magn_core, chi_magn_core, u)

    siw_dens = schwinger_dyson_vrg(vrg_dens_core, chi_dens_core, g_loc, u)
    siw_magn = schwinger_dyson_vrg(vrg_magn_core, chi_magn_core, g_loc, u)

    hartree = tp.get_smom0(u, n)
    siw_sde = hartree + (siw_magn + siw_dens)
    siw_sde_m = hartree + (siw_magn * 2)
    siw_F = hartree + schwinger_dyson_F(F_updo, gchi0_core, g_loc, u)

    # %%
    niv_shell = 1000
    chi0_shell = bubble_gen.get_chi0_shell(niv, niv_shell)
    lam_shell = u * (chi0_shell)

    lam_dens_tilde = get_lam_tilde(lam_dens_core, chi0_shell, u)
    lam_magn_tilde = get_lam_tilde(lam_magn_core, chi0_shell, u)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # fig, ax = plt.subplots(2, 2, figsize=(7, 6), dpi=500)
    # ax = ax.flatten()
    # im1 = ax[0].pcolormesh(vn_core, wn, lam_magn_tilde.mat.real, cmap='RdBu')
    # im2 = ax[1].pcolormesh(vn_core, wn, lam_magn_core.mat.real, cmap='RdBu')
    # im3 = ax[2].pcolormesh(vn_core, wn, lam_dens_tilde.mat.real, cmap='RdBu')
    # im4 = ax[3].pcolormesh(vn_core, wn, lam_dens_core.mat.real, cmap='RdBu')
    # # im3 = ax[2].pcolormesh(vn,wn,lam_magn_tilde.mat.real-lam_magn_core.mat.real,cmap='RdBu')
    # divider = make_axes_locatable(ax[0])
    # cax1 = divider.append_axes('right', size='5%', pad=0.05)
    # divider = make_axes_locatable(ax[1])
    # cax2 = divider.append_axes('right', size='5%', pad=0.05)
    # divider = make_axes_locatable(ax[2])
    # cax3 = divider.append_axes('right', size='5%', pad=0.05)
    # divider = make_axes_locatable(ax[3])
    # cax4 = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im1, cax=cax1, orientation='vertical')
    # fig.colorbar(im2, cax=cax2, orientation='vertical')
    # fig.colorbar(im3, cax=cax3, orientation='vertical')
    # fig.colorbar(im4, cax=cax4, orientation='vertical')
    # plt.tight_layout()
    # plt.show()

    # %%

    chi_dens_tilde = get_chir_tilde(lam_dens_tilde, chi_dens_core, chi0_shell, gchi0_core, u)
    chi_magn_tilde = get_chir_tilde(lam_magn_tilde, chi_magn_core, chi0_shell, gchi0_core, u)

    vrg_dens_tilde = vrg_from_lam(lam_dens_tilde, chi_dens_tilde, u)
    vrg_magn_tilde = vrg_from_lam(lam_magn_tilde, chi_magn_tilde, u)

    siw_dens_tilde = schwinger_dyson_vrg(vrg_dens_tilde, chi_dens_tilde, g_loc, u)
    siw_magn_tilde = schwinger_dyson_vrg(vrg_magn_tilde, chi_magn_tilde, g_loc, u)
    n_shell_giw = 200
    siw_dens_shell = schwinger_dyson_shell(chi_dens_tilde, g_loc, beta, u, n_shell_giw, vrg_dens_tilde.niv, vrg_dens_tilde.wn)
    siw_magn_shell = schwinger_dyson_shell(chi_magn_tilde, g_loc, beta, u, n_shell_giw, vrg_dens_tilde.niv, vrg_dens_tilde.wn)
    siw_tilde = hartree + siw_magn_tilde + siw_dens_tilde
    siw_shell = hartree + siw_dens_shell + siw_magn_shell
    siw_tilde = mf.concatenate_core_asmypt(siw_tilde, siw_shell)
    siw_tilde_m = hartree + siw_magn_tilde * 2

    vrg_magn_2, chi_magn_2 = get_vrg_and_chir_tilde_from_chir(gchi_magn, bubble_gen, u, niv_core=niv, niv_shell=niv_shell)
    vrg_dens_2, chi_dens_2 = get_vrg_and_chir_tilde_from_chir(gchi_dens, bubble_gen, u, niv_core=niv, niv_shell=niv_shell)
    # fig, ax = plt.subplots(1, 2, figsize=(7, 3.5), dpi=500)
    # ax[0].plot(wn, chi_magn_core.real, '-o', color='cornflowerblue')
    # ax[0].plot(wn, chi_magn_tilde.real, '-h', color='firebrick')
    # ax[0].plot(wn, chi_magn_2.real, '-h', color='seagreen')
    # ax[1].plot(wn, chi_dens_core.real, '-o', color='cornflowerblue')
    # ax[1].plot(wn, chi_dens_tilde.real, '-h', color='firebrick')
    # ax[1].plot(wn, chi_dens_2.real, '-h', color='seagreen')
    # ax[0].set_xlim(-10, 10)
    # ax[1].set_xlim(-10, 10)
    # # ax[0].set_ylim(-0.1,0.1)
    # # ax[1].set_ylim(-0.1,0.1)
    # plt.show()

    # plt.figure()
    # plt.plot(wn,chi0_core.real,'-o', color='cornflowerblue')
    # plt.plot(wn,chi0_full.real,'-o', color='firebrick')
    # plt.show()

    # #%%
    # fig,ax = plt.subplots(2, 2, figsize=(7, 7), dpi=500)
    # ax = ax.flatten()
    # ax[0].plot(vn_core,vrg_dens_tilde.mat[niw,:].real, 'o', color='cornflowerblue', ms=4,markeredgecolor='k', alpha=0.8)
    # ax[0].plot(vn_core,vrg_magn_tilde.mat[niw,:].real, 'o', color='firebrick', ms=4,markeredgecolor='k', alpha=0.8)
    # ax[1].plot(vn_core,vrg_dens_tilde.mat[25,:].real, 'o', color='cornflowerblue', ms=4,markeredgecolor='k', alpha=0.8)
    # ax[1].plot(vn_core,vrg_magn_tilde.mat[25,:].real, 'o', color='firebrick', ms=4,markeredgecolor='k', alpha=0.8)
    # ax[2].plot(wn,vrg_dens_tilde.mat[:,niv].real, 'o', color='cornflowerblue', ms=4,markeredgecolor='k', alpha=0.8)
    # ax[2].plot(wn,vrg_magn_tilde.mat[:,niv].real, 'o', color='firebrick', ms=4,markeredgecolor='k', alpha=0.8)
    # ax[3].plot(wn,vrg_dens_tilde.mat[:,25].real, 'o', color='cornflowerblue', ms=4,markeredgecolor='k', alpha=0.8)
    # ax[3].plot(wn,vrg_magn_tilde.mat[:,25].real, 'o', color='firebrick', ms=4,markeredgecolor='k', alpha=0.8)
    # plt.tight_layout()
    # plt.show()
    #
    # #%%
    # F_dens_diag = get_F_diag(chi_dens_tilde,chi_magn_tilde,channel='dens')
    # F_magn_diag = get_F_diag(chi_dens_tilde,chi_magn_tilde,channel='magn')
    # v_diag = mf.vn(F_dens_diag.shape[0]//2)
    # fig, ax = plt.subplots(2, 2, figsize=(7, 7), dpi=500)
    # ax = ax.flatten()
    # ax[0].pcolormesh(v_diag,v_diag,F_dens_diag.real,cmap='RdBu')
    # ax[1].pcolormesh(v_diag,v_diag,F_dens_diag.imag,cmap='RdBu')
    # ax[2].pcolormesh(v_diag,v_diag,F_magn_diag.real,cmap='RdBu')
    # ax[3].pcolormesh(v_diag,v_diag,F_magn_diag.imag,cmap='RdBu')
    # plt.tight_layout()
    # plt.show()

    #
    # #%%
    # niv2 = niv*2
    # gchi0_shell = bubble_gen.get_gchi0(niv=niv2)
    # vn_shell = mf.vn(niv_shell)
    # chi_magn_shell_2 = -u**2/beta**4 *np.sum(gchi0_shell[:,niv2+niv+1:,None] * F_magn_diag[:,:] * gchi0_shell[:,None,niv2+niv+1:],axis=(-2,-1))
    # chi_magn_shell_2 += -u**2/beta**4 *np.sum(gchi0_shell[:,:niv2-niv-1,None] * F_magn_diag[:,:] * gchi0_shell[:,None,:niv2-niv-1],axis=(-2,-1))
    #
    # chi_dens_shell_2 = -u**2/beta**4 *np.sum(gchi0_shell[:,niv2+niv+1:,None] * F_dens_diag[:,:] * gchi0_shell[:,None,niv2+niv+1:],axis=(-2,-1))
    # chi_dens_shell_2 += -u**2/beta**4 *np.sum(gchi0_shell[:,:niv2-niv-1,None] * F_dens_diag[:,:] * gchi0_shell[:,None,:niv2-niv-1],axis=(-2,-1))
    #
    # #%%
    # plt.figure()
    # plt.plot(wn,chi_magn_shell_2.real,'-o',color='cornflowerblue')
    # plt.plot(wn,chi_dens_shell_2.real,'-o',color='firebrick')
    # # plt.plot(np.sum(F_magn_diag.real,-1),'-o',color='firebrick')
    # plt.show()
    #
    # #%%
    # plt.figure()
    # plt.imshow(gchi0_core.real,cmap='RdBu')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    # %%
    wn_asympt = mf.wn_shell(giwk.niv_core, n_core=niw)
    bubble_gen_asypt = bub.LocalBubble(wn=wn_asympt, giw=giwk)
    chi_asmypt = bubble_gen_asypt.get_chi0(niv=niv_shell)
    chi_asmypt = chi_asmypt + bubble_gen_asypt.get_chi0_shell(niv_shell, niv_shell + niv)
    print('--------------')
    print(f'Analytic sum: {n / 2 * (1 - n / 2)}')
    print(f'Numeric sum core: {1 / (2 * beta) * np.sum((chi_dens_core + chi_magn_core)).real}')
    # print(f'Numeric sum core: {1 / (2 * beta) * np.sum((chi_dens_core_centered + chi_magn_core_centered)).real}')
    print(f'Numeric sum tilde: {1 / (2 * beta) * np.sum((chi_dens_tilde + chi_magn_tilde)).real}')
    print(f'Numeric sum shell: {1 / (beta) * np.sum(chi_asmypt).real}')
    print(f'Numeric sum full: {1 / (2 * beta) * np.sum(chi_asmypt).real + 1 / (2 * beta) * np.sum((chi_dens_tilde + chi_magn_tilde)).real}')
    print('--------------')

    plt.loglog(wn, (chi_dens_tilde + chi_magn_tilde).real, '-o', color='firebrick')
    plt.loglog(wn_asympt, chi_asmypt.real / 1, '-o', color='cornflowerblue')
    plt.xlim(30, 500)
    plt.ylim(-0.02, 0.02)
    plt.show()

    print('bla')
    # %%
    vn = mf.vn(1000)
    n_start = -0
    n_plot = 100

    fig, ax = plt.subplots(1, 2, figsize=(7, 3.5), dpi=500)
    ax[0].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(dmft_input_1['siw'], n_plot, n_start).real, '-p', color='k')
    ax[0].plot(mf.cut_v_1d_pos(vn, siw_sde.size // 2, n_start), mf.cut_v_1d_pos(siw_sde, siw_sde.size // 2, n_start).real, 'o',
               color='cornflowerblue', ms=4,
               markeredgecolor='k', alpha=0.8)
    # ax[0].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_sde_m, n_plot, n_start).real, 'o', color='indigo',ms=3,markeredgecolor='k',alpha=0.8)
    ax[0].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_tilde, n_plot, n_start).real, 'h', color='firebrick', ms=3,
               markeredgecolor='firebrick', alpha=0.8)
    # ax[0].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_tilde_m, n_plot, n_start).real, 'p', color='seagreen', ms=1,markeredgecolor='k',alpha=0.8)
    # ax[0].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_F, n_plot, n_start).real, 'h', color='firebrick', ms=4)

    ax[1].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(dmft_input_1['siw'], n_plot, n_start).imag, '-p', color='k')
    ax[1].plot(mf.cut_v_1d_pos(vn, siw_sde.size // 2, n_start), mf.cut_v_1d_pos(siw_sde, siw_sde.size // 2, n_start).imag, 'o',
               color='cornflowerblue', ms=4,
               markeredgecolor='k', alpha=0.8)
    # ax[1].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_sde_m, n_plot, n_start).imag, 'o', color='indigo',ms=3,markeredgecolor='k',alpha=0.8)
    ax[1].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_tilde, n_plot, n_start).imag, 'h', color='firebrick', ms=3,
               markeredgecolor='firebrick', alpha=0.8)
    # ax[1].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_tilde_m, n_plot, n_start).imag, 'p', color='seagreen', ms=1,markeredgecolor='k',alpha=0.8)
    # ax[1].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_F, n_plot, n_start).imag, 'h', color='firebrick', ms=4)
    plt.tight_layout()
    plt.show()

    # %%
    # %%
    n_start = 50
    n_plot = 150
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5), dpi=500)
    ax = np.atleast_1d(ax)
    ax[0].semilogy(mf.cut_v_1d_pos(vn, n_plot, n_start), -mf.cut_v_1d_pos(dmft_input_1['siw'], n_plot, n_start).imag, '-p', color='cornflowerblue')
    ax[0].semilogy(mf.cut_v_1d_pos(vn, n_plot, n_start), -mf.cut_v_1d_pos(siw_tilde, n_plot, n_start).imag, 'h', color='firebrick', ms=3,
                   markeredgecolor='firebrick', alpha=0.8)
    ax[0].semilogy(mf.cut_v_1d_pos(vn, n_plot, n_start), -mf.cut_v_1d_pos(1 / mf.v(beta, n_shell_giw) * tp.get_smom1(u, n), n_plot, n_start), '-',
                   color='k')
    plt.tight_layout()
    plt.show()

    # %%

    # siw_dmft_cut = mf.cut_v_1d_pos(dmft_input_1['siw'], n_plot, n_start)
    # siw_sde_cut = mf.cut_v_1d_pos(siw_sde, n_plot, n_start)
    # siw_sde_tilde_cut = mf.cut_v_1d_pos(siw_tilde, n_plot, n_start)
    # vn_cut = mf.cut_v_1d_pos(vn, n_plot, n_start)
    # fig, ax = plt.subplots(1, 2, figsize=(7, 3.5), dpi=500)
    # ax[0].plot(vn_cut, np.abs((siw_dmft_cut-siw_sde_cut).real), 'o', color='cornflowerblue',ms=3)
    # ax[0].plot(vn_cut, np.abs((siw_dmft_cut-siw_sde_tilde_cut).real), 'h', color='firebrick', ms=2)
    #
    # ax[1].plot(vn_cut, np.abs((siw_dmft_cut-siw_sde_cut).imag), 'o', color='cornflowerblue',ms=3)
    # ax[1].plot(vn_cut, np.abs((siw_dmft_cut-siw_sde_tilde_cut).imag), 'h', color='firebrick', ms=2)
    # plt.tight_layout()
    # plt.show()

    print('Finished')

    # #%%
    # niw = F_magn.wn.size//2
    # mat_tmp = F_magn.mat[20,:,:]
    # mat_tmp_2 = F_magn.mat[-21,:,:]
    # v_tmp = mf.vn(F_magn.niv)
    # w_tmp = F_magn.wn
    # niw_tmp = len(w_tmp)//2
    # shift = w_tmp[10]//2
    # niv_shift = F_magn.niv-shift
    # plt.figure()
    # # plt.pcolormesh(v_tmp,v_tmp,np.roll(mat_tmp.real,(-shift,0),axis=(0,1)),cmap='RdBu')
    # new_mat = np.zeros_like(mat_tmp)
    # new_mat[abs(shift):,abs(shift):] = mat_tmp.real[:shift,:shift]
    # # new_mat[:abs(shift),:] = np.flipud(np.fliplr(mat_tmp_2.real[:shift,:]))
    # fig, ax = plt.subplots(1,2,figsize=[7,3.5],dpi=500)
    # ax[0].imshow(mat_tmp.real,cmap='RdBu')
    # ax[1].imshow(mat_tmp_2.real,cmap='RdBu')
    # # plt.imshow(mat_tmp.real,cmap='RdBu')
    # plt.tight_layout()
    # plt.show()
    #

    # %%

    # F_asympt = lam_magn_tilde.mat[niw,:][:,None]+lam_magn_tilde.mat[niw,:][None,:]
    #
    # plt.imshow(F_asympt.real,cmap='RdBu')
    # plt.show()
