import numpy as np
import MatsubaraFrequencies as mf


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
# ---------------------------------------------- LOCAL BUBBLE CLASS  ---------------------------------------------------
# ======================================================================================================================
KNOWN_CHI0_METHODS = ['sum']


def get_gchi0(giw, niv, beta, iwn):
    ''' chi_0[w,v] = - beta G(v) * G(v-w)'''
    niv_giw = np.shape(giw)[0] // 2  # ToDo: Maybe I should introduce a Green's function object, but for now keep it like this
    return - beta * giw[niv_giw - niv:niv_giw + niv] * giw[niv_giw - niv - iwn:niv_giw + niv - iwn]


def vec_get_gchi0(giw, beta, niv, wn):
    return np.array([get_gchi0(giw, niv, beta, iwn) for iwn in wn])


def get_gchi0_inv(giw, niv, beta, iwn):
    niv_giw = np.shape(giw)[0] // 2
    return - 1 / beta * 1 / giw[niv_giw - niv:niv_giw + niv] * 1 / giw[niv_giw - niv - iwn:niv_giw + niv - iwn]


def vec_get_gchi0_inv(giw, beta, niv, wn):
    return np.array([get_gchi0_inv(giw, niv, beta, iwn) for iwn in wn])


def get_chi0_sum(giw, beta, niv, iwn=0):
    niv_giw = np.shape(giw)[0] // 2
    # iwnd2 = iwn // 2
    # iwnd2mod2 = iwn // 2 + iwn % 2
    iwnd2 = 0
    iwnd2mod2 = -iwn
    return - 1. / beta * np.sum(giw[niv_giw - niv + iwnd2:niv_giw + niv + iwnd2] * giw[niv_giw - niv - iwnd2mod2:niv_giw + niv - iwnd2mod2])


def vec_get_chi0_sum(giw, beta, niv, wn):
    return np.array([get_chi0_sum(giw, beta, niv, iwn=iwn) for iwn in wn])


class LocalBubble():
    ''' Computes the local Bubble suszeptibility \chi_0 = - beta GG '''

    def __init__(self, wn, giw, beta, chi0_method='sum'):
        self.wn = wn
        self.giw = giw
        self.beta = beta
        if (chi0_method not in KNOWN_CHI0_METHODS):
            raise ValueError(f'Chi0-method ({chi0_method}) not in {KNOWN_CHI0_METHODS}')
        self.chi0_method = chi0_method

    @property
    def niw(self):
        return self.wn.size

    @property
    def wn_lin(self):
        return np.arange(0, self.niw)

    def get_chi0(self, niv):
        if (self.chi0_method == 'sum'):
            return vec_get_chi0_sum(self.giw, self.beta, niv, self.wn)

    def get_chi0_shell(self, niv_core, niv_shell):
        if (self.chi0_method == 'sum'):
            chi0_core = vec_get_chi0_sum(self.giw, self.beta, niv_core, self.wn)
            chi0_full = vec_get_chi0_sum(self.giw, self.beta, niv_core + niv_shell, self.wn)
            return chi0_full - chi0_core

    def get_gchi0(self, niv):
        return vec_get_gchi0(self.giw, self.beta, niv, self.wn)

    # def plot(self, iwn=0, pdir='./', name=None, do_save=True):
    #     assert iwn in self.wn, 'omega index not in dataset.'
    #     iwn_lin = self.wn_lin[self.wn == iwn][0]
    #     plot_fourpoint_nu_nup(np.diag(self.mat[iwn_lin]), self.vn, pdir=pdir, name=name, do_save=do_save)
    #     plot_omega_dep(self.wn, self.chi0, pdir=pdir, name=name, do_save=do_save)


# ======================================================================================================================
# ---------------------------------------------- LOCAL FOURPOINT CLASS  ---------------------------------------------------
# ======================================================================================================================
KNOWN_CHANNELS = ['dens', 'magn', 'trip', 'sing', 'updo', 'upup']


class LocalThreePoint():
    ''' Parent class for local three-point correlation
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
        self.wn = wn
        assert channel in KNOWN_CHANNELS, f"Channel with name {channel} not known. Kown are {KNOWN_CHANNELS}."
        self.channel = channel  # dens/magn/trip/sing/bubb

    @property
    def wn_lin(self):
        return np.arange(0, self.wn.size)

    @property
    def niv(self):
        return np.shape(self.mat)[-1] // 2

    @property
    def vn(self):
        return mf.vn(self.niv)

    def cut_iv(self, niv_cut=None):
        self.mat = mf.cut_iv_fp(self.mat, niv_cut)

    def contract_legs(self):
        return 1. / self.beta ** 2 * np.sum(self.mat, axis=(-2, -1))

    def plot(self, iwn=0, pdir='./', name=None, do_save=True, niv=-1):
        assert iwn in self.wn, 'omega index not in dataset.'
        iwn_lin = self.wn_lin[self.wn == iwn][0]
        data = mf.cut_iv_2d(self.mat[iwn_lin], niv_cut=niv)
        vn = mf.cut_v_1d(self.vn, niv_cut=niv)
        plot_fourpoint_nu_nup(data, vn, pdir=pdir, name=name, do_save=do_save)


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


#
#
# def g2_from_chir(chir: LocalFourPoint = None, giw=None):
#     ''' G2 = 1/beta * chi_r + 2*GG \delta_dens'''
#     chir_mat = 1 / chir.beta * chir.mat
#     if (chir.channel == 'dens' and 0 in chir.wn):
#         ggv = get_ggv(giw, niv_ggv=chir.niv)
#         chir_mat[chir.wn == 0] = 1 / chir.beta * chir.mat[chir.wn == 0] + 2. * ggv
#     chir_mat = jnp.array(chir_mat)
#     return LocalFourPoint(matrix=chir_mat, channel=chir.channel, beta=chir.beta, wn=chir.wn)

#
#
# def gamob2_from_chir(chir: LocalFourPoint = None, chi0_inv: LocalBubble = None):
#     ''' Gamma = - ( chi_0^-1 - chi_r^-1) '''
#     gam_r = jnp.array([gamob2_from_chir_wn(chir.mat[iwn_lin], chi0_inv.mat[iwn_lin]) for iwn_lin in chir.wn_lin])
#     return LocalFourPoint(matrix=gam_r, channel=chir.channel, beta=chir.beta, wn=chir.wn)
#
#
# def gamob2_from_chir_wn(chir=None, chi0_inv=None):
#     gam_r = -(jnp.diag(chi0_inv) - jnp.linalg.inv(chir))
#     return gam_r
#
#
# def wn_slices(mat=None, n_cut=None, wn=None):
#     n = mat.shape[-1] // 2
#     mat_grid = jnp.array([mat[n - n_cut - iwn:n + n_cut - iwn] for iwn in wn])
#     return mat_grid
#
#
# def vrg_from_gam(gam: LocalFourPoint = None, chi0_inv: LocalBubble = None, u=None):
#     u_r = get_ur(u, channel=gam.channel)
#     vrg = jnp.array([vrg_from_gam_wn(gam.mat[iwn_lin], chi0_inv.mat[iwn_lin], gam.beta, u_r) for iwn_lin in gam.wn_lin])
#     return LocalThreePoint(matrix=vrg, channel=gam.channel, beta=gam.beta, wn=gam.wn)
#
#
# def vrg_from_gam_wn(gam, chi0_inv, beta, u_r):
#     return 1 / beta * chi0_inv * jnp.sum(jnp.linalg.inv(np.diag(chi0_inv) + gam - u_r / beta ** 2), axis=-1)
#
#
# def three_leg_from_F(F: LocalFourPoint = None, chi0: LocalBubble = None):
#     ''' sg = \sum_k' F \chi_0 '''
#     sg = 1 / F.beta * jnp.sum(F.mat * chi0.mat[:, None, :], axis=-1)
#     return LocalThreePoint(matrix=sg, channel=F.channel, beta=F.beta, wn=F.wn)
#
#
# def schwinger_dyson_F(F: LocalFourPoint = None, chi0: LocalBubble = None, giw=None, u=None, totdens=None):
#     ''' Sigma = U*n/2 + 1/beta**3 * U \sum_{qk} F \chi_0 G(k-q)'''
#     hartree = u * totdens / 2.0
#     sigma_F_q = jnp.sum(1 / F.beta * u * F.mat * chi0.mat[:, None, :], axis=-1)
#     mat_grid = wn_slices(giw, n_cut=F.niv, wn=F.wn)
#     sigma_F = jnp.sum(sigma_F_q * mat_grid, axis=0)
#     return hartree + sigma_F
#
#
# def schwinger_dyson_vrg(vrg: LocalThreePoint = None, chir_phys: LocalSusceptibility = None, giw: tp.LocalGreensFunction = None, u=None,
#                         do_tilde=True,scalfac = 1,scalfac_2 = 1,sign=None):
#     ''' Sigma = U*n/2 + '''
#     if(sign is None):
#         u_r = get_ur(u, channel=chir_phys.channel)
#     else:
#         u_r = sign * u
#     mat_grid = wn_slices(giw.mat, n_cut=vrg.niv, wn=vrg.wn)
#     if (do_tilde):
#         sigma_F = u_r / 2 * jnp.sum((1 / vrg.beta*scalfac - (1*scalfac_2 - u_r * chir_phys.mat_tilde[:, None]) * vrg.mat_tilde) * mat_grid, axis=0)
#     else:
#         sigma_F = u_r / 2 * jnp.sum((1 / vrg.beta*scalfac - (1*scalfac_2 - u_r * chir_phys.mat[:, None]) * vrg.mat) * mat_grid, axis=0)
#     return sigma_F
#
# def schwinger_dyson_vrg_updo(vrg: LocalThreePoint = None, chir_phys: LocalSusceptibility = None, giw: tp.LocalGreensFunction = None, u=None,
#                         do_tilde=True):
#     ''' Sigma = U*n/2 + '''
#     mat_grid = wn_slices(giw.mat, n_cut=vrg.niv, wn=vrg.wn)
#     if (do_tilde):
#         sigma_F = u**2 / 2 * jnp.sum(chir_phys.mat_tilde[:, None] * vrg.mat_tilde * mat_grid, axis=0)
#     else:
#         sigma_F = u**2 / 2 * jnp.sum(chir_phys.mat[:, None] * vrg.mat * mat_grid, axis=0)
#     return sigma_F
#
#
# def schwinger_dyson_w_asympt(gchi0: LocalBubble = None,giw=None, u=None,niv=None):
#     mat_grid = wn_slices(giw.mat, n_cut=niv, wn=gchi0.wn)
#     return -u * u/gchi0.beta * jnp.sum(gchi0.chi0_tilde[:,None] * mat_grid, axis=0)
#
#
# def lam_from_chir_FisUr(chir: LocalFourPoint = None, gchi0: LocalBubble = None, u=None):
#     ''' lambda vertex'''
#     assert gchi0.shell is not None, "Shell of gchi0 needed for shell of lambda."
#     u_r = get_ur(u, chir.channel)
#     sign = get_sign(chir.channel)
#     lam = jnp.sum(jnp.eye(chir.niv * 2)[None, :, :] - chir.mat * (1 / gchi0.mat)[:, :, None], axis=-1)
#     shell = +u_r * gchi0.shell[:, None]
#     tilde = (lam + shell)
#     return LocalThreePoint(channel=chir.channel, matrix=lam, beta=chir.beta, wn=chir.wn, shell=shell, mat_tilde=tilde)
#
#
# def chi_phys_tilde_FisUr(chir: LocalFourPoint = None, gchi0: LocalBubble = None, lam: LocalFourPoint = None, u=None):
#     u_r = get_ur(u, chir.channel)
#     chi_core = chir.contract_legs()
#     chi_shell = gchi0.shell - u_r * gchi0.shell ** 2 - 2 * u_r * gchi0.shell * gchi0.chi0
#     chi_tilde = (chi_core.mat + chi_shell)
#     return LocalSusceptibility(matrix=chi_core.mat, channel=chir.channel, beta=chir.beta, wn=chir.wn, shell=chi_shell, mat_tilde=chi_tilde)
#
#
def lam_from_chir(gchir: LocalFourPoint, gchi0):
    ''' lambda vertex'''
    sign = get_sign(gchir.channel)
    lam = -sign * np.sum(np.eye(gchir.niv * 2)[None, :, :] - gchir.mat * (1 / gchi0)[:, :, None], axis=-1)
    return LocalThreePoint(channel=gchir.channel, matrix=lam, beta=gchir.beta, wn=gchir.wn)


def vrg_from_lam(lam: LocalThreePoint, chir, u):
    u_r = get_ur(u, lam.channel)
    sign = get_sign(lam.channel)
    vrg = (1 + sign * lam.mat) / (1 - u_r * chir[:, None])
    return LocalThreePoint(channel=lam.channel, matrix=vrg, beta=lam.beta, wn=lam.wn)


def get_chir_shell(lam_tilde: LocalThreePoint, chi0_shell, gchi0_core, u):
    sign = get_sign(lam_tilde.channel)
    chir_shell = -sign * u * chi0_shell ** 2 + chi0_shell * (
            1 - 2 * u * 1 / lam_tilde.beta ** 2 * np.sum((lam_tilde.mat+sign) * gchi0_core, axis=-1))
    return chir_shell


def get_chir_tilde(lam_tilde: LocalThreePoint, chir_core, chi0_shell, gchi0_core, u):
    chi_r_shell = get_chir_shell(lam_tilde, chi0_shell, gchi0_core, u)
    return (chir_core + chi_r_shell) / (1 - (u * chi0_shell) ** 2)


def get_lam_shell(chi0_shell, u):
    return u * chi0_shell


def get_lam_tilde(lam_core: LocalThreePoint, chi0_shell, u):
    lam_shell = get_lam_shell(chi0_shell, u)
    sign = get_sign(lam_core.channel)
    lam_tilde = (lam_core.mat - lam_shell[:, None]) / (1 + sign * lam_shell[:, None])
    return LocalThreePoint(channel=lam_core.channel, matrix=lam_tilde, beta=lam_core.beta, wn=lam_core.wn)


def get_vrg_and_chir_tilde_from_chir(gchir: LocalFourPoint, chi0_gen: LocalBubble, u, niv_core=-1, niv_shell=0):
    assert niv_core <= gchir.niv, f'niv_core ({niv_core}) has to be smaller or equal to the niv_g2 ({gchir.niv}).'
    if (niv_core == -1): niv_core = gchir.niv
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


#
#
# def chi_phys_tilde(chir: LocalFourPoint = None, gchi0: LocalBubble = None, lam: LocalFourPoint = None, u=None):
#     sign = get_sign(chir.channel)
#     u_r = get_ur(u, chir.channel)
#     chi_core = chir.contract_legs()
#     chi_shell = - u_r * gchi0.shell ** 2  # - gchi0.shell * (1 - 2 *  u/chir.beta**2 * jnp.sum((lam.mat_tilde - 1) * gchi0.mat))#gchi0.shell- 2 * u_r * gchi0.shell * gchi0.chi0
#     chi_tilde = 1 / (1 - (u * gchi0.shell) ** 2) * (chi_core.mat + chi_shell)
#     chi_shell = gchi0.shell - u_r * gchi0.shell ** 2 - 2 * u_r * gchi0.shell * gchi0.chi0
#     chi_tilde = (chi_core.mat + chi_shell) * 1 / (1 - (u * gchi0.shell) ** 2)
#     return LocalSusceptibility(matrix=chi_core.mat, channel=chir.channel, beta=chir.beta, wn=chir.wn, shell=chi_shell, mat_tilde=chi_tilde)
#
#
# def add_chi(chi1: LocalFourPoint = None,chi2: LocalFourPoint = None):
#     return LocalSusceptibility(matrix=chi1.mat+chi2.mat, channel='upup', beta=chi1.beta, wn=chi1.wn, shell=None, mat_tilde=chi1.mat_tilde+chi2.mat_tilde)
#
# def subtract_chi(chi1: LocalFourPoint = None,chi2: LocalFourPoint = None):
#     return LocalSusceptibility(matrix=chi1.mat-chi2.mat, channel='updo', beta=chi1.beta, wn=chi1.wn, shell=None, mat_tilde=chi1.mat_tilde-chi2.mat_tilde)
#
#
# def add_vrg(vrg1: LocalThreePoint = None,vrg2: LocalThreePoint = None):
#     return LocalThreePoint(matrix=vrg1.mat+vrg2.mat, channel='upup', beta=vrg1.beta, wn=vrg1.wn, shell=None, mat_tilde=vrg1.mat_tilde+vrg2.mat_tilde)
#
# def subtract_vrg(vrg1: LocalThreePoint = None,vrg2: LocalThreePoint = None):
#     return LocalThreePoint(matrix=vrg1.mat-vrg2.mat, channel='updo', beta=vrg1.beta, wn=vrg1.wn, shell=None, mat_tilde=vrg1.mat_tilde-vrg2.mat_tilde)
#
#


#
# def local_chi_phys_from_chi_aux(chi_aux=None, chi0_urange: LocalBubble = None, chi0_core: LocalBubble = None, u=None):
#     u_r = get_ur(u=u, channel=chi_aux.channel)
#     chi = 1. / (1. / (chi_aux.mat + chi0_urange.chi0 - chi0_core.chi0) + u_r)
#     return LocalSusceptibility(matrix=chi, channel=chi_aux.channel, beta=chi_aux.beta, wn=chi_aux.wn)
#
#
# def local_susceptibility_from_four_point(four_point: LocalFourPoint = None, chi0_urange=None):
#     return four_point.contract_legs()
#
#
# def local_rpa_susceptibility(chi0_urange: LocalBubble = None, channel=None, u=None):
#     u_r = get_ur(u=u, channel=channel)
#     chir = chi0_urange.chi0 / (1 + u_r * chi0_urange.chi0)
#     return LocalSusceptibility(matrix=chir, channel=channel, beta=chi0_urange.beta, iw=chi0_urange.iw,
#                                chi0_urange=chi0_urange)
#
#
# # ======================================================================================================================
# # ------------------------------------- FREE FUNCTIONS THAT USE OBJECTS AS INPUT ---------------------------------------
# # ======================================================================================================================
#
# # ==================================================================================================================
# def gammar_from_gchir(gchir: LocalFourPoint = None, gchi0_urange: LocalBubble = None, u=1.0):
#     u_r = get_ur(u=u, channel=gchir.channel)
#     gammar = np.array(
#         [gammar_from_gchir_wn(gchir=gchir.mat[wn], gchi0_urange=gchi0_urange.gchi0[wn], niv_core=gchir.niv,
#                               beta=gchir.beta, u=u_r) for wn in gchir.wn_lin])
#     return LocalFourPoint(matrix=gammar, channel=gchir.channel, beta=gchir.beta, wn=gchir.wn)
#
#
# def gammar_from_gchir_wn(gchir=None, gchi0_urange=None, niv_core=None, beta=1.0, u=1.0):
#     full = u / (beta * beta) + np.diag(1. / gchi0_urange)
#     inv_full = np.linalg.inv(full)
#     inv_core = cut_iv(inv_full, niv_core)
#     core = np.linalg.inv(inv_core)
#     chigr_inv = np.linalg.inv(gchir)
#     return -(core - chigr_inv - u / (beta * beta))
#
#
# # ==================================================================================================================
#
# # ==================================================================================================================
# def local_gchi_aux_from_gammar(gammar: LocalFourPoint = None, gchi0_core: LocalBubble = None, u=None):
#     u_r = get_ur(u=u, channel=gammar.channel)
#     gchi_aux = np.array([local_gchi_aux_from_gammar_wn(gammar=gammar.mat[wn], gchi0=gchi0_core.gchi0[wn],
#                                                        beta=gammar.beta, u=u_r) for wn in gammar.wn_lin])
#     return LocalFourPoint(matrix=gchi_aux, channel=gammar.channel, beta=gammar.beta, wn=gammar.wn)
#
#
# def local_gchi_aux_from_gammar_wn(gammar=None, gchi0=None, beta=1.0, u=1.0):
#     gchi0_inv = np.diag(1. / gchi0)
#     chi_aux_inv = gchi0_inv + gammar - u / (beta * beta)
#     return np.linalg.inv(chi_aux_inv)
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


def schwinger_dyson_vrg_core_from_g2(g2: LocalFourPoint, chi0_gen: LocalBubble, u, niv_core=-1, niv_shell=0):
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


if __name__ == '__main__':
    import sys, os
    import matplotlib.pyplot as plt

    sys.path.append('../test')
    import TestData as td
    import TwoPoint as tp
    import Hk as hamk
    import BrillouinZone as bz
    import MatsubaraFrequencies as mf

    dmft_input_1 = td.get_data_set_3(load_g2=True)
    siw = dmft_input_1['siw'][None, None, None, :]
    beta = dmft_input_1['beta']
    n = dmft_input_1['n']
    u = dmft_input_1['u']
    hr = dmft_input_1['hr']
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=hr)
    siwk = tp.SelfEnergy(siw, beta, pos=False)
    giwk = tp.GreensFunction(siwk, ek, n=n)
    niv_asympt = 6000
    giwk.set_g_asympt(niv_asympt)
    g_loc = giwk.k_mean(range='full')
    print(giwk.mem)
    g2_file = dmft_input_1['g2_file']
    niw = g2_file.get_niw(channel='dens')

    niw = 40
    niv = 80
    wn = mf.wn(niw)
    vn_core = mf.vn(niv)
    bubble_gen = LocalBubble(wn=wn, giw=g_loc, beta=beta)
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

    chi_dens_core = gchi_dens.contract_legs()
    chi_magn_core = gchi_magn.contract_legs()

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
    niv_shell = 3000
    chi0_full = bubble_gen.get_chi0(niv + niv_shell)
    chi0_shell = chi0_full - chi0_core
    lam_shell = u * (chi0_shell)

    lam_dens_tilde = get_lam_tilde(lam_dens_core, chi0_shell, u)
    lam_magn_tilde = get_lam_tilde(lam_magn_core, chi0_shell, u)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(2, 2, figsize=(7, 6), dpi=500)
    ax = ax.flatten()
    im1 = ax[0].pcolormesh(vn_core, wn, lam_magn_tilde.mat.real, cmap='RdBu')
    im2 = ax[1].pcolormesh(vn_core, wn, lam_magn_core.mat.real, cmap='RdBu')
    im3 = ax[2].pcolormesh(vn_core, wn, lam_dens_tilde.mat.real, cmap='RdBu')
    im4 = ax[3].pcolormesh(vn_core, wn, lam_dens_core.mat.real, cmap='RdBu')
    # im3 = ax[2].pcolormesh(vn,wn,lam_magn_tilde.mat.real-lam_magn_core.mat.real,cmap='RdBu')
    divider = make_axes_locatable(ax[0])
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    divider = make_axes_locatable(ax[1])
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    divider = make_axes_locatable(ax[2])
    cax3 = divider.append_axes('right', size='5%', pad=0.05)
    divider = make_axes_locatable(ax[3])
    cax4 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax1, orientation='vertical')
    fig.colorbar(im2, cax=cax2, orientation='vertical')
    fig.colorbar(im3, cax=cax3, orientation='vertical')
    fig.colorbar(im4, cax=cax4, orientation='vertical')
    plt.tight_layout()
    plt.show()
    # %%

    chi_dens_tilde = get_chir_tilde(lam_dens_tilde, chi_dens_core, chi0_shell, gchi0_core, u)
    chi_magn_tilde = get_chir_tilde(lam_magn_tilde, chi_magn_core, chi0_shell, gchi0_core, u)

    vrg_dens_tilde = vrg_from_lam(lam_dens_tilde, chi_dens_tilde, u)
    vrg_magn_tilde = vrg_from_lam(lam_magn_tilde, chi_magn_tilde, u)

    siw_dens_tilde = schwinger_dyson_vrg(vrg_dens_tilde, chi_dens_tilde, g_loc, u)
    siw_magn_tilde = schwinger_dyson_vrg(vrg_magn_tilde, chi_magn_tilde, g_loc, u)
    siw_tilde = hartree + siw_magn_tilde + siw_dens_tilde
    siw_tilde_m = hartree + siw_magn_tilde * 2

    vrg_magn_2, chi_magn_2 = get_vrg_and_chir_tilde_from_chir(gchi_magn, bubble_gen, u, niv_core=niv, niv_shell=niv_shell)
    vrg_dens_2, chi_dens_2 = get_vrg_and_chir_tilde_from_chir(gchi_dens, bubble_gen, u, niv_core=niv, niv_shell=niv_shell)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.5), dpi=500)
    ax[0].plot(wn, chi_magn_core.real, '-o', color='cornflowerblue')
    ax[0].plot(wn, chi_magn_tilde.real, '-h', color='firebrick')
    ax[0].plot(wn, chi_magn_2.real, '-h', color='seagreen')
    ax[1].plot(wn, chi_dens_core.real, '-o', color='cornflowerblue')
    ax[1].plot(wn, chi_dens_tilde.real, '-h', color='firebrick')
    ax[1].plot(wn, chi_dens_2.real, '-h', color='seagreen')
    ax[0].set_xlim(-10, 10)
    ax[1].set_xlim(-10, 10)
    # ax[0].set_ylim(-0.1,0.1)
    # ax[1].set_ylim(-0.1,0.1)
    plt.show()

    # plt.figure()
    # plt.plot(wn,chi0_core.real,'-o', color='cornflowerblue')
    # plt.plot(wn,chi0_full.real,'-o', color='firebrick')
    # plt.show()

    # %%
    print('--------------')
    print(f'Analytic sum: {n / 2 * (1 - n / 2)}')
    print(f'Numeric sum core: {1 / (2 * beta) * np.sum((chi_dens_core + chi_magn_core)[35:85]).real}')
    print(f'Numeric sum core: {1 / (2 * beta) * np.sum((chi_dens_core + chi_magn_core)[45:75]).real}')
    print(f'Numeric sum tilde: {1 / (2 * beta) * np.sum((chi_dens_tilde + chi_magn_tilde)[30:90]).real}')
    print(f'Numeric sum tilde: {1 / (2 * beta) * np.sum((chi_dens_tilde + chi_magn_tilde)).real}')
    print('--------------')

    print('bla')
    # %%
    vn = mf.vn(1000)
    n_start = 0
    n_plot = niv

    fig, ax = plt.subplots(1, 2, figsize=(7, 3.5), dpi=500)
    ax[0].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(dmft_input_1['siw'], n_plot, n_start).real, '-p', color='k')
    ax[0].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_sde, n_plot, n_start).real, 'o', color='cornflowerblue', ms=4,
               markeredgecolor='k', alpha=0.8)
    # ax[0].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_sde_m, n_plot, n_start).real, 'o', color='indigo',ms=3,markeredgecolor='k',alpha=0.8)
    ax[0].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_tilde, n_plot, n_start).real, 'h', color='firebrick', ms=3,
               markeredgecolor='k', alpha=0.8)
    # ax[0].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_tilde_m, n_plot, n_start).real, 'p', color='seagreen', ms=1,markeredgecolor='k',alpha=0.8)
    # ax[0].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_F, n_plot, n_start).real, 'h', color='firebrick', ms=4)

    ax[1].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(dmft_input_1['siw'], n_plot, n_start).imag, '-p', color='k')
    ax[1].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_sde, n_plot, n_start).imag, 'o', color='cornflowerblue', ms=4,
               markeredgecolor='k', alpha=0.8)
    # ax[1].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_sde_m, n_plot, n_start).imag, 'o', color='indigo',ms=3,markeredgecolor='k',alpha=0.8)
    ax[1].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_tilde, n_plot, n_start).imag, 'h', color='firebrick', ms=3,
               markeredgecolor='k', alpha=0.8)
    # ax[1].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_tilde_m, n_plot, n_start).imag, 'p', color='seagreen', ms=1,markeredgecolor='k',alpha=0.8)
    # ax[1].plot(mf.cut_v_1d_pos(vn, n_plot, n_start), mf.cut_v_1d_pos(siw_F, n_plot, n_start).imag, 'h', color='firebrick', ms=4)
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
