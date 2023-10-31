import numpy as np
import dga.bubble as bub
import dga.two_point as twop
import dga.matsubara_frequencies as mf
import matplotlib.pyplot as plt



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
# ---------------------------------------------- LOCAL FOURPOINT CLASS  ------------------------------------------------
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
            assert np.size(
                matrix.shape[0]) % 2 == 1, 'wn is not bosonic. Probably w is not the 0th dimension of the input matrix.'
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
        self.mat = mf.cut_v(self.mat, niv_cut, axes=(-2,-1))

    def cut_iw(self, niw_cut=None):
        assert self.is_full_w, 'Full wn range has to be ensured.'
        self.mat = mf.cut_w(self.mat, niw_cut, (0,))

    def contract_legs(self):
        return 1. / self.beta ** 2 * np.sum(self.mat, axis=(-2, -1))

    def contract_legs_centered(self, niv_sum):
        return mf.vn_centered_sum(self.mat, self.wn, self.beta, niv_sum=niv_sum)

    def plot(self, iwn=0, pdir='./', name='', do_save=True, niv=-1, verbose=False):
        assert iwn in self.wn, 'omega index not in dataset.'
        iwn_lin = self.wn_lin[self.wn == iwn][0]
        data = mf.cut_v(self.mat[iwn_lin], niv_cut=niv,axes=(-2,-1))
        vn = mf.cut_v_1d(self.vn, niv_cut=niv)
        plot_fourpoint_nu_nup(data, vn, pdir=pdir, name=name + f'_wn{iwn}_niv{niv}', do_save=do_save, show=verbose)


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


def get_urange(obj: LocalFourPoint, val, niv_urange):
    return LocalFourPoint(channel=obj.channel, beta=obj.beta, wn=obj.wn, matrix=mf.v_vp_urange(obj.mat, val, niv_urange))


def Fob2_from_gamob2_urange(gammar: LocalFourPoint, gchi0, u):
    ''' F = Gamma [1 - Gamma X_0]^(-1)'''
    u_r = get_ur(u, channel=gammar.channel)
    niv_core = gammar.niv
    niv_full = np.shape(gchi0)[-1] // 2
    niv_urange = niv_full - niv_core
    gamma_urange = get_urange(gammar, u_r / gammar.beta ** 2, niv_urange)
    eye = np.eye(niv_full * 2)
    Fob2 = np.array(
        [gamma_urange.mat[iwn] @ np.linalg.inv(eye + gamma_urange.mat[iwn] * gchi0[iwn][:, None]) for iwn in gammar.wn_lin])
    return LocalFourPoint(channel=gammar.channel, beta=gammar.beta, wn=gammar.wn, matrix=Fob2)


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


def get_vrg_and_chir_tilde_from_chir(gchir: LocalFourPoint, chi0_gen: bub.BubbleGenerator, u, niv_core=-1, niv_shell=0):
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


def get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_r: LocalFourPoint, gchi0_gen: bub.BubbleGenerator, u, niv_shell=0):
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
    gchi_aux = gchi_aux_core_from_gammar(gamma_r, gchi0_core, u)
    chi_aux_core = gchi_aux.contract_legs()

    # Compute the physical susceptibility:
    chi_urange = chi_phys_urange(chi_aux_core, chi0_core, chi0_urange, u,
                                 gamma_r.channel).real  # take real part to avoid numerical noise
    chi_asympt = chi_phys_asympt(chi_urange, chi0_urange, chi0_asympt).real  # take real part to avoid numerical noise

    # Compute the fermion-boson vertex:
    vrg = vrg_from_gchi_aux(gchi_aux, gchi0_core, chi_urange, chi_asympt, u)


    return vrg, chi_asympt


def get_F_dc_asympt(vrg: LocalThreePoint, gchi_aux, chi_phys, gchi0, u):
    ''' Has an additional beta**2 compared to Motoharus paper due to different definitions of Chi0. '''
    u_r = get_ur(u, vrg.channel)
    eye = np.eye(vrg.niv * 2)
    mat = np.array([vrg.beta ** 2 * 1. / gchi0[i][:, None] * (eye - gchi_aux.mat[i] * 1 / gchi0[i][None, :])
                    + u_r * (1 - u_r * chi_phys[i]) * vrg.mat[i][:, None] * vrg.mat[i][None, :] for i in vrg.wn_lin])
    return LocalFourPoint(channel=vrg.channel, matrix=mat, beta=vrg.beta, wn=vrg.wn)


def get_vrg_and_chir_tilde_from_chir_uasympt(chir: LocalFourPoint, gchi0_gen: bub.BubbleGenerator, u, niv_shell=0, niv_asympt=None):
    '''
        Compute the fermi-bose vertex and susceptibility using the asymptotics proposed in
        Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005
    '''
    if (niv_asympt is None): niv_asympt = 2 * niv_shell
    niv_core = chir.niv
    # Create necessary bubbles
    niv_full = niv_core + niv_shell
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
    return (chigr_inv - chi_tilde_core + u / (beta * beta))


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


def gchi_aux_core_from_gammar(gammar: LocalFourPoint, gchi0_core, u):
    u_r = get_ur(u, gammar.channel)
    mat = np.array([np.linalg.inv(np.diag(1. / gchi0_core[i]) + gammar.mat[i] - u_r / gammar.beta ** 2) for i in gammar.wn_lin])
    return LocalFourPoint(matrix=mat, channel=gammar.channel, beta=gammar.beta, wn=gammar.wn)


def gchi_aux_asympt(gchi_aux: LocalFourPoint, chi_urange, chi_asympt, u):
    u_r = get_ur(u, gchi_aux.channel)
    u_mat = np.ones_like(gchi_aux.mat) * u
    mat = gchi_aux.mat + gchi_aux.mat @ u_mat @ gchi_aux.mat * (
                -(1 - u_r * chi_urange) + (1 - u_r * chi_urange) ** 2 / (1 - u_r * chi_asympt))
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


def schwinger_dyson_vrg(vrg: LocalThreePoint, chir_phys, giw, u):
    ''' Sigma = U*n/2 + '''
    u_r = get_ur(u, channel=vrg.channel)
    mat_grid = mf.wn_slices_gen(giw, n_cut=vrg.niv, wn=vrg.wn)
    sigma_F = u_r / 2 * 1 / vrg.beta * np.sum((1 - (1 - u_r * chir_phys[:, None]) * vrg.mat) * mat_grid, axis=0)
    return sigma_F


def schwinger_dyson_shell(chir_phys, giw, beta, u, n_shell, n_core, wn, channel):
    u_r = get_ur(u, channel=channel)
    mat_grid = mf.wn_slices_shell(giw, n_shell=n_shell, n_core=n_core, wn=wn)
    sigma_F = u / 2 * 1 / beta * np.sum((u * chir_phys[:, None]) * mat_grid, axis=0)
    return sigma_F


def schwinger_dyson_full(vrg_dens: LocalThreePoint, vrg_magn: LocalThreePoint, chi_dens, chi_magn,
                         giw, u, n, niv_shell=0):
    siw_dens = schwinger_dyson_core_urange(vrg_dens, chi_dens, giw, u, niv_shell)
    siw_magn = schwinger_dyson_core_urange(vrg_magn, chi_magn, giw, u, niv_shell)
    hartree = twop.get_smom0(u, n)
    return hartree + siw_dens + siw_magn


def schwinger_dyson_core_urange(vrg: LocalThreePoint, chi_phys, giw, u, niv_shell):
    siw_core = schwinger_dyson_vrg(vrg, chi_phys, giw, u)
    if (niv_shell == 0):
        return siw_core
    else:
        siw_shell = schwinger_dyson_shell(chi_phys, giw, vrg.beta, u, niv_shell, vrg.niv, vrg.wn, vrg.channel)
        return mf.concatenate_core_asmypt(siw_core, siw_shell)


def schwinger_dyson_vrg_core_from_g2(g2: LocalFourPoint, chi0_gen: bub.BubbleGenerator, u, niv_core=-1, niv_shell=0):
    assert niv_core <= g2.niv, f'niv_core ({niv_core}) has to be smaller or equal to the niv_g2 ({g2.niv}).'
    if (niv_core == -1): niv_core = g2.niv
    g2.cut_iv(niv_core)
    gchi = gchir_from_g2(g2, chi0_gen.giw)
    chi_core = gchi.contract_legs()
    gchi0_core = chi0_gen.get_gchi0(niv_core)
    lam_core = lam_from_chir()
    pass


def schwinger_dyson_F(F: LocalFourPoint, gchi0, giw, u):
    ''' Sigma = 1/beta**3 * U \sum_{qk} F \chi_0 G(k-q)'''
    sigma_F_q = np.sum(1 / F.beta * u * F.mat * gchi0[:, None, :], axis=-1)
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
    pass
