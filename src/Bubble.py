import numpy as np
import MatsubaraFrequencies as mf
import TwoPoint as tp

# ======================================================================================================================
# ---------------------------------------------- LOCAL BUBBLE CLASS  ---------------------------------------------------
# ======================================================================================================================
KNOWN_CHI0_METHODS = ['sum']
KNOWN_FREQ_SHIFTS = ['plus', 'minus', 'center']


def get_freq_shift(iwn, freq_notation):
    if (freq_notation == 'plus'):  # chi_0[w,v] = - beta G(v) * G(v+w)
        iws, iws2 = 0, iwn
    elif (freq_notation == 'minus'):  # chi_0[w,v] = - beta G(v) * G(v-w)
        iws, iws2 = 0, -iwn
    elif (freq_notation == 'center'):  # chi_0[w,v] = - beta G(v+w//2) * G(v-w//2-w%2)
        iws, iws2 = iwn // 2, -(iwn // 2 + iwn % 2)
    else:
        raise NotImplementedError(f'freq_notation {freq_notation} is not known. Known are: {KNOWN_FREQ_SHIFTS}')
    return iws, iws2


def get_gchi0(giw, niv, beta, iwn, freq_notation='minus'):
    ''' chi_0[w,v] = - beta G(v) * G(v-w) for minus'''
    niv_giw = np.shape(giw)[0] // 2
    iws, iws2 = get_freq_shift(iwn, freq_notation)
    return - beta * giw[niv_giw - niv + iws:niv_giw + niv + iws] * giw[niv_giw - niv + iws2:niv_giw + niv + iws2]


# TODO: Find good way to generate giw(k+q). k-grid objects to that. Maybe pass it?
def get_gchi0_q(giwk, niv, beta, iwn, q, freq_notation='minus'):
    ''' chi_0[w,v] = - beta G(v,k) * G(v-w,k-q) for minus'''
    niv_giw = np.shape(giwk)[0] // 2
    iws, iws2 = get_freq_shift(iwn, freq_notation)
    giwkpq = bz.shift_mat_by_ind(giwk, ind=[-iq for iq in q])
    return - beta * giwk[niv_giw - niv + iws:niv_giw + niv + iws] * giwkpq[niv_giw - niv + iws2:niv_giw + niv + iws2]


def get_chi0_q(giwk, beta, niv, iwn, q, freq_notation='minus'):
    ''' chi_0[w,v] = - beta G(v,k) * G(v-w,k-q) for minus'''
    niv_giw = np.shape(giwk)[-1] // 2
    iws, iws2 = get_freq_shift(iwn, freq_notation)
    giwkpq = bz.shift_mat_by_ind(giwk, ind=[-iq for iq in q])
    return - 1 / beta * np.sum(np.mean((giwk[..., niv_giw - niv + iws:niv_giw + niv + iws] *
                                        giwkpq[..., niv_giw - niv + iws2:niv_giw + niv + iws2]), axis=(0, 1, 2)))


def vec_get_chi0_q(giwk, beta, niv, wn, q_list, freq_notation='minus'):
    chi0_q = np.zeros((len(q_list), len(wn)),dtype=complex)
    for i, q in enumerate(q_list):
        chi0_q[i, :] = np.array([get_chi0_q(giwk, beta,niv, iwn, q, freq_notation) for iwn in wn])
    return chi0_q


def vec_get_gchi0(giw, beta, niv, wn, freq_notation='minus'):
    return np.array([get_gchi0(giw, niv, beta, iwn, freq_notation) for iwn in wn])


def get_chi0_sum(giw, beta, niv, iwn=0, freq_notation='minus'):
    niv_giw = np.shape(giw)[0] // 2
    iws, iws2 = get_freq_shift(iwn, freq_notation)
    return - 1. / beta * np.sum(giw[niv_giw - niv + iws:niv_giw + niv + iws] * giw[niv_giw - niv + iws2:niv_giw + niv + iws2])


def vec_get_chi0_sum(giw, beta, niv, wn, freq_notation='minus'):
    return np.array([get_chi0_sum(giw, beta, niv, iwn=iwn, freq_notation=freq_notation) for iwn in wn])


class LocalBubble():
    ''' Computes the (local) Bubble suszeptibility \chi_0 = - beta GG
        Uses a Greens-function object which has knowledge about the moments of the self-energy and the
        kinetic Hamiltonian.
    '''

    def __init__(self, wn, giw: tp.GreensFunction, chi0_method='sum', freq_notation='minus'):
        self.wn = wn
        self.giw = giw
        self.beta = giw.beta
        if (chi0_method not in KNOWN_CHI0_METHODS):
            raise ValueError(f'Chi0-method ({chi0_method}) not in {KNOWN_CHI0_METHODS}')
        self.chi0_method = chi0_method

        if (freq_notation not in KNOWN_FREQ_SHIFTS):
            raise ValueError(f'freq_notation ({freq_notation}) not in {KNOWN_FREQ_SHIFTS}')
        self.freq_notation = freq_notation

    @property
    def smom1(self):
        return self.giw.sigma.smom1

    @property
    def smom0(self):
        return self.giw.sigma.smom0

    @property
    def ek_mom2(self):
        return np.mean(self.giw.ek ** 2)

    @property
    def ek_mom1(self):
        return np.mean(self.giw.ek)

    @property
    def mu(self):
        return self.giw.mu

    @property
    def niw(self):
        return self.wn.size

    @property
    def g_loc(self):
        return self.giw.g_loc

    @property
    def wn_lin(self):
        return np.arange(0, self.niw)

    def get_chi0(self, niv, freq_notation=None):
        if (freq_notation is None):
            freq_notation = self.freq_notation
        if (self.chi0_method == 'sum'):
            return vec_get_chi0_sum(self.g_loc, self.beta, niv, self.wn, freq_notation)

    def get_chi0_shell(self, niv_core, niv_shell, do_asmypt=True,freq_notation=None):
        if(freq_notation is None):
            freq_notation = self.freq_notation
        if (self.chi0_method == 'sum'):
            if (do_asmypt):
                chi0_core = vec_get_chi0_sum(self.g_loc, self.beta, niv_core, self.wn,freq_notation=freq_notation)
                chi0_shell = vec_get_chi0_sum(self.g_loc, self.beta, niv_core + niv_shell, self.wn,freq_notation=freq_notation)
                chi0_asmypt = self.get_asymptotic_correction(niv_core + niv_shell,freq_notation=freq_notation)
                return chi0_shell - chi0_core + chi0_asmypt
            else:
                chi0_core = vec_get_chi0_sum(self.g_loc, self.beta, niv_core, self.wn,freq_notation=freq_notation)
                chi0_full = vec_get_chi0_sum(self.g_loc, self.beta, niv_core + niv_shell, self.wn,freq_notation=freq_notation)
                return chi0_full - chi0_core

    def get_gchi0(self, niv, freq_notation=None):
        if (freq_notation is None):
            freq_notation = self.freq_notation
        return vec_get_gchi0(self.g_loc, self.beta, niv, self.wn, freq_notation)

    def get_asymptotic_correction(self, niv_core,freq_notation=None):
        chi0_asmypt_sum = self.get_asympt_sum(niv_core,freq_notation=freq_notation)
        chi0_asmypt_exact = self.get_exact_asymptotics()
        return chi0_asmypt_exact - chi0_asmypt_sum

    def get_asympt_prefactors(self):
        fac1 = (self.mu - self.smom0)
        fac2 = (self.mu - self.smom0) ** 2 + self.ek_mom2 - self.smom1
        fac3 = (self.mu - self.smom0) ** 2 + self.ek_mom1**2 # ek_mom1 is zero
        return fac1, fac2, fac3

    def get_asympt_sum(self, niv_sum,freq_notation=None):
        if(freq_notation is None):
            freq_notation = self.freq_notation
        chi0_asympt_sum = np.zeros((self.niw), dtype=complex)
        v = 1 / (1j * mf.v(self.beta, n=niv_sum + self.niw))
        niv = v.size // 2
        fac1, fac2, fac3 = self.get_asympt_prefactors()
        for i, iwn in enumerate(self.wn):
            iws, iws2 = get_freq_shift(iwn, freq_notation)
            v_sum = v[niv - niv_sum + iws:niv + niv_sum + iws]
            vpw_sum = v[niv - niv_sum + iws2:niv + niv_sum + iws2]
            chi0_asympt_sum[i] += np.sum(v_sum * vpw_sum)
            chi0_asympt_sum[i] -= np.sum(v_sum ** 2 * vpw_sum + v_sum * vpw_sum ** 2) * fac1
            chi0_asympt_sum[i] += np.sum(v_sum ** 3 * vpw_sum + v_sum * vpw_sum ** 3) * fac2
            chi0_asympt_sum[i] += np.sum(v_sum ** 2 * vpw_sum ** 2) * fac3
        return chi0_asympt_sum * (-1 / self.beta)

    def get_exact_asymptotics(self):
        chi0_asympt = np.zeros((self.niw), dtype=complex)
        iw = self.wn * 2 * np.pi / self.beta * 1j
        fac1, fac2, fac3 = self.get_asympt_prefactors()
        ind = self.wn != 0
        chi0_asympt[ind] = -self.beta / 2 * 1 / iw[ind] ** 2 * (fac2 - fac3)
        # chi0_asympt[ind] = -1 / iw[ind] ** 2 #* (fac2 - fac3)
        # chi0_asympt[ind] *= 0.5
        ind = self.wn == 0
        chi0_asympt[ind] = self.beta / 4 - self.beta ** 3 / 24 * fac2 - self.beta ** 3 / 48 * fac3
        return chi0_asympt

    def get_chi0_single_q(self, niv, q=(0, 0, 0), freq_notation=None):
        if (freq_notation is None):
            freq_notation = self.freq_notation
        if (self.chi0_method == 'sum'):
            # return get_chi0_q(self.giw.full, self.beta, niv, self.wn, q, freq_notation)
            return None

    def get_chi0_q_list(self, niv, q_list, freq_notation=None):
        if (freq_notation is None):
            freq_notation = self.freq_notation
        if (self.chi0_method == 'sum'):
            return vec_get_chi0_q(self.giw.full, self.beta, niv, self.wn, q_list, freq_notation)


# ======================================================================================================================
# ---------------------------------------------- BUBBLE CLASS  ---------------------------------------------------
# ======================================================================================================================


if __name__ == '__main__':
    import TestData as td
    import BrillouinZone as bz
    import Hk as hamk

    ddict = td.get_data_set_5()
    nk = (16, 16, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    ek_mom0 = np.mean(ek * ek)
    niw_chi0 = 20
    u, n, beta = ddict['u'], ddict['n'], ddict['beta']
    w = mf.w(ddict['beta'], niw_chi0)
    wn = mf.wn(niw_chi0)
    siwk = tp.SelfEnergy(ddict['siw'][None, None, None, :], beta, pos=False, smom0=tp.get_smom0(u, n), smom1=tp.get_smom1(u, n))
    giwk = tp.GreensFunction(siwk, ek, n=n)
    niv_asympt = 3000
    giwk.set_g_asympt(niv_asympt)
    bubble_gen = LocalBubble(wn=wn, giw=giwk, freq_notation='minus')

    niv_core = 100
    niv_shell = 1900
    chi0_core = bubble_gen.get_chi0(niv_core)
    chi0_shell = bubble_gen.get_chi0(niv_shell)
    chi0_asympt = bubble_gen.get_chi0_shell(niv_core, niv_shell)
    chi0_asympt_axact = bubble_gen.get_exact_asymptotics()
    chi0_asympt_sum2 = bubble_gen.get_asympt_sum(niv_shell + niv_core)
    import matplotlib.pyplot as plt

    plt.figure()
    ind = wn > 0
    ind_chi0 = wn > 0
    plt.loglog(wn[ind], chi0_core.real[ind], '-h', color='cornflowerblue', markeredgecolor='cornflowerblue', alpha=0.8)
    plt.loglog(wn[ind_chi0], (chi0_shell[ind_chi0].real), '-h', color='firebrick', markeredgecolor='firebrick', alpha=0.8)
    plt.loglog(wn[ind_chi0], (chi0_core.real[ind_chi0] + chi0_asympt[ind_chi0].real), '-h', color='seagreen', markeredgecolor='seagreen', alpha=0.8)
    plt.loglog(wn[ind_chi0], (chi0_asympt_axact[ind_chi0].real), '-h', color='navy', markeredgecolor='navy', alpha=0.8)
    plt.loglog(wn[ind_chi0], (chi0_asympt_sum2[ind_chi0].real), '-h', color='goldenrod', markeredgecolor='goldenrod', alpha=0.8)
    plt.show()

    #%% Test non-local Bubble:
    niv_core = 100
    q_grid = bz.KGrid(nk=(16, 16, 1), symmetries=bz.two_dimensional_square_symmetries())
    q_list = q_grid.irrk_mesh_ind
    niv_giw = np.shape(giwk.full)[-1] // 2
    iws, iws2 = get_freq_shift(0, 'minus')
    giwkpq = bz.shift_mat_by_ind(giwk.full, ind=[-iq for iq in q_list.T[0]])
    print(- 1 / beta * np.sum(np.mean((giwk.full[..., niv_giw - niv_core + iws:niv_giw + niv_core + iws] *
                                        giwkpq[..., niv_giw - niv_core + iws2:niv_giw + niv_core + iws2]), axis=(0, 1, 2))))
    chi0_q = bubble_gen.get_chi0_q_list(niv_core, q_list.T)
    #%%
    chi0_q_fbz = q_grid.map_fbz2irrk(chi0_q)
    chi0_loc = np.mean(chi0_q_fbz, axis=(0, 1, 2))
    print('Finished!')

    #%%
    plt.figure()
    plt.plot(wn, chi0_loc.real,color='cornflowerblue')
    plt.plot(wn, chi0_core.real,color='firebrick')
    plt.show()

    # chi0_q = vec_get_chi0_q(giwk.full, beta, niv_core, wn, q_list.T, freq_notation='minus')

    # test = get_chi0_q(giwk.full, beta, niv=200, iwn=1, q=(1, 1, 1), freq_notation='minus')

    # plt.figure()
    # plt.plot(wn,chi0_asympt_axact.real-chi0_asympt_sum2.real)
    # plt.show()
