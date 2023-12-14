'''
    This module contains the Bubble class which computes the bubble susceptibility chi_0 = - beta GG
'''

import numpy as np

from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import two_point as twop

# ======================================================================================================================
# ---------------------------------------------- LOCAL BUBBLE CLASS  ---------------------------------------------------
# ======================================================================================================================
KNOWN_CHI0_METHODS = ['sum']


def get_gchi0(giw, beta, niv, iwn, freq_notation='minus'):
    '''
        gchi_0[w,v]:1/eV^3 = - beta:1/eV G(v):1/eV * G(v-w):1/eV for minus
    '''
    niv_giw = np.shape(giw)[0] // 2
    iws, iws2 = mf.get_freq_shift(iwn, freq_notation)
    return - beta * giw[niv_giw - niv + iws:niv_giw + niv + iws] * giw[niv_giw - niv + iws2:niv_giw + niv + iws2]


def vec_get_gchi0(giw, beta, niv, wn, freq_notation='minus', is_full_wn=False):
    if is_full_wn:
        wn_pos = mf.wn(mf.niw_from_mat(wn), pos=True)
        gchi0 = np.array([get_gchi0(giw, beta, niv, iwn=iwn, freq_notation=freq_notation) for iwn in wn_pos])
        return mf.bosonic_full_nu_range(gchi0, axis=0, freq_axes=(0, 1))
    else:
        return np.array([get_gchi0(giw, beta, niv, iwn, freq_notation) for iwn in wn])


# TODO: Find good way to generate giwk_obj(k+q). k-grid objects do that. Maybe pass it?
def get_gchi0_q(giwk, beta, niv, iwn, q, freq_notation='minus'):
    '''
        gchi_0[q,w,v] = - beta sum_k G(v,k) * G(v-w,k-q) for minus
    '''
    niv_giw = np.shape(giwk)[-1] // 2
    iws, iws2 = mf.get_freq_shift(iwn, freq_notation)
    giwkpq = bz.shift_mat_by_ind(giwk, ind=[-iq for iq in q])
    return - beta * np.mean((giwk[..., niv_giw - niv + iws:niv_giw + niv + iws] *
                             giwkpq[..., niv_giw - niv + iws2:niv_giw + niv + iws2]), axis=(0, 1, 2))


def vec_get_gchi0_q(giwk, beta, niv, wn, q_list, freq_notation='minus', is_full_wn=False):
    wn_bub = mf.wn(mf.niw_from_mat(wn), pos=True) if is_full_wn else wn
    gchi0_q = np.zeros((len(q_list), len(wn_bub), niv * 2), dtype=complex)
    giwk = mf.cut_v(giwk, niv_cut=niv + np.max(np.abs(wn_bub)), axes=(-1,))
    for i, q in enumerate(q_list):
        gchi0_q[i, ...] = np.array([get_gchi0_q(giwk, beta, niv, iwn, q, freq_notation) for iwn in wn_bub])
    return mf.bosonic_full_nu_range(gchi0_q, axis=1, freq_axes=(1, 2)) if is_full_wn else gchi0_q


def get_chi0_q(giwk, beta, niv, iwn, q, freq_notation='minus'):
    '''
        chi_0[q,w]:1/eV = - 1/beta:eV sum_{v,k} (G(v,k):1/eV * G(v-w,k-q):1/eV) for minus
    '''
    niv_giw = np.shape(giwk)[-1] // 2
    iws, iws2 = mf.get_freq_shift(iwn, freq_notation)
    giwkpq = bz.shift_mat_by_ind(giwk, ind=[-iq for iq in q])
    return - 1 / beta * np.sum(np.mean((giwk[..., niv_giw - niv + iws:niv_giw + niv + iws] *
                                        giwkpq[..., niv_giw - niv + iws2:niv_giw + niv + iws2]), axis=(0, 1, 2)))


def vec_get_chi0_q(giwk, beta, niv, wn, q_list, freq_notation='minus', is_full_wn=False):
    wn_bub = mf.wn(mf.niw_from_mat(wn), pos=True) if is_full_wn else wn
    chi0_q = np.zeros((len(q_list), len(wn_bub)), dtype=complex)
    giwk = mf.cut_v(giwk, niv_cut=niv + np.max(np.abs(wn_bub)), axes=(-1,))
    for i, q in enumerate(q_list):
        chi0_q[i, :] = np.array([get_chi0_q(giwk, beta, niv, iwn, q, freq_notation) for iwn in wn_bub])
    return mf.bosonic_full_nu_range(chi0_q, axis=1) if is_full_wn else chi0_q


def get_chi0_sum(giw, beta, niv, iwn=0, freq_notation='minus'):
    niv_giw = np.shape(giw)[0] // 2
    iws, iws2 = mf.get_freq_shift(iwn, freq_notation)
    return - 1. / beta * np.sum(giw[niv_giw - niv + iws:niv_giw + niv + iws] * giw[niv_giw - niv + iws2:niv_giw + niv + iws2])


def vec_get_chi0_sum(giw, beta, niv, wn, freq_notation='minus', is_full_wn=False):
    if is_full_wn:
        wn_pos = mf.wn(mf.niw_from_mat(wn), pos=True)
        chi0 = np.array([get_chi0_sum(giw, beta, niv, iwn=iwn, freq_notation=freq_notation) for iwn in wn_pos])
        return mf.bosonic_full_nu_range(chi0)
    else:
        return np.array([get_chi0_sum(giw, beta, niv, iwn=iwn, freq_notation=freq_notation) for iwn in wn])


class BubbleGenerator():
    ''' Computes the (local) Bubble susceptibility chi_0 = - beta GG
        Uses a Greens-function object which has knowledge about the moments of the self-energy and the
        kinetic Hamiltonian.
    '''

    def __init__(self, wn, giwk_obj: twop.GreensFunction, chi0_method='sum', freq_notation='minus', is_full_wn=False):
        self.wn = wn
        self.giwk_obj = giwk_obj
        self.is_full_wn = is_full_wn  # If the full bosonic frequency range is supplied symmetries can be used.
        if is_full_wn: assert np.allclose(wn, -np.flip(wn)), 'wn must be symmetric for is_full_wn=True'
        if chi0_method not in KNOWN_CHI0_METHODS:
            raise ValueError(f'Chi0-method ({chi0_method}) not in {KNOWN_CHI0_METHODS}')
        self.chi0_method = chi0_method

        if freq_notation not in mf.KNOWN_FREQ_SHIFTS:
            raise ValueError(f'freq_notation ({freq_notation}) not in {mf.KNOWN_FREQ_SHIFTS}')
        self.freq_notation = freq_notation

    @property
    def smom1(self):
        return self.giwk_obj.sigma.smom1

    @property
    def smom0(self):
        return self.giwk_obj.sigma.smom0

    @property
    def ek_mom2(self):
        return np.mean(self.ek ** 2)

    @property
    def ek(self):
        return self.giwk_obj.ek

    @property
    def ek_mom1(self):
        return np.mean(self.ek)

    @property
    def mu(self):
        return self.giwk_obj.mu

    @property
    def beta(self):
        return self.giwk_obj.beta

    @property
    def niw(self):
        return self.wn.size

    @property
    def g_loc(self):
        return self.giwk_obj.g_loc

    @property
    def wn_lin(self):
        return np.arange(0, self.niw)

    @property
    def wn_pos(self):
        if not self.is_full_wn:
            raise ValueError('wn_pos is only well defined if the full bosonic frequency range is supplied.')
        else:
            return mf.wn(mf.niw_from_mat(self.wn), pos=True)

    def contract_legs(self, gchi):
        ''' Contract the legs of a bubble susceptibility gchi[v,w] -> chi[w] '''
        return 1/self.beta**2 * np.sum(gchi, axis=(-1,))

    def get_gchi0_vvp_full(self, gchi0):
        ''' expand the last dimension of gchi0[w,v] -> gchi0[w,v,vp] delta(v,vp) '''
        return np.array([np.diag(gchi0_i) for gchi0_i in gchi0])

    def get_chi0(self, niv, freq_notation=None, do_asympt=False):

        if freq_notation is None: freq_notation = self.freq_notation

        if self.chi0_method == 'sum':
            chi0_core = vec_get_chi0_sum(self.g_loc, self.beta, niv, self.wn, freq_notation, is_full_wn=self.is_full_wn)
            if do_asympt:
                chi0_asympt = self.get_asymptotic_correction(niv, freq_notation=freq_notation)
                return chi0_core + chi0_asympt
            else:
                return chi0_core
        else:
            raise NotImplementedError(f'Chi0-method ({self.chi0_method}) not implemented.')

    def get_gchi0(self, niv, freq_notation=None):
        if freq_notation is None:
            freq_notation = self.freq_notation
        return vec_get_gchi0(self.g_loc, self.beta, niv, self.wn, freq_notation, self.is_full_wn)

    def get_asymptotic_correction(self, niv_core, freq_notation=None):
        chi0_asympt_sum = self.get_asympt_sum(niv_core, freq_notation=freq_notation)
        chi0_asympt_exact = self.get_exact_asymptotics()
        return chi0_asympt_exact - chi0_asympt_sum

    def get_asymptotic_correction_q(self, niv_core, q_list, freq_notation=None):
        chi0q_asympt_sum = self.get_asympt_sum_q(niv_core, q_list, freq_notation=freq_notation)
        chi0q_asympt_exact = self.get_exact_asymptotics_q(q_list)
        return chi0q_asympt_exact - chi0q_asympt_sum

    def get_chi0q_shell(self, chi0q_core, niv_core, niv_shell, q_list, freq_notation=None):
        if freq_notation is None:
            freq_notation = self.freq_notation

        chi0q_shell = self.get_chi0_q_list(niv_core + niv_shell, q_list, freq_notation=freq_notation)
        chi0q_asympt = self.get_asymptotic_correction_q(niv_core + niv_shell, q_list, freq_notation=freq_notation)
        return chi0q_shell - chi0q_core + chi0q_asympt

    def get_asympt_prefactors(self):
        '''
            [fac1] = eV
            [fac2] = eV^2
            [fac3] = eV^2
        '''
        fac1 = self.mu - self.smom0
        fac2 = (self.mu - self.smom0) ** 2 + self.ek_mom2 - self.smom1
        fac3 = (self.mu - self.smom0) ** 2 + self.ek_mom1 ** 2  # ek_mom1 is zero
        return fac1, fac2, fac3

    def get_asympt_prefactors_q(self, q_list):
        fac1 = self.mu - self.smom0
        fac2 = (self.mu - self.smom0) ** 2 + self.ek_mom2 - self.smom1
        fac3 = (self.mu - self.smom0) ** 2 + self.get_ek_ekpq(q_list)
        return fac1, fac2, fac3

    def get_ek_ekpq(self, q_list):
        fac_q = []
        for q in q_list:
            ekpq = bz.shift_mat_by_ind(self.ek, ind=[-iq for iq in q])
            fac_q.append(np.mean(self.ek * ekpq))
        return np.array(fac_q)

    def get_asympt_sum(self, niv_sum, freq_notation=None):
        if freq_notation is None:
            freq_notation = self.freq_notation
        chi0_asympt_sum = np.zeros((self.niw), dtype=complex)
        # [v] = 1/eV
        v = 1 / (1j * mf.vn(self.beta, niv_sum + np.max(np.abs(self.wn))))
        niv = mf.niv_from_mat(v)
        fac1, fac2, fac3 = self.get_asympt_prefactors()
        for i, iwn in enumerate(self.wn):
            iws, iws2 = mf.get_freq_shift(iwn, freq_notation)
            v_sum = v[niv - niv_sum + iws:niv + niv_sum + iws]
            vpw_sum = v[niv - niv_sum + iws2:niv + niv_sum + iws2]
            # 1/eV * 1/eV = 1/eV^2
            chi0_asympt_sum[i] += np.sum(v_sum * vpw_sum)
            # 1/eV^3 * eV = 1/eV^2
            chi0_asympt_sum[i] -= np.sum(v_sum ** 2 * vpw_sum + v_sum * vpw_sum ** 2) * fac1
            # 1/eV^4 * eV^2 = 1/eV^2
            chi0_asympt_sum[i] += np.sum(v_sum ** 3 * vpw_sum + v_sum * vpw_sum ** 3) * fac2
            # 1/eV^4 * eV^2 = 1/eV^2
            chi0_asympt_sum[i] += np.sum(v_sum ** 2 * vpw_sum ** 2) * fac3
        return chi0_asympt_sum * (-1 / self.beta)  # 1/eV^2 * eV = 1/eV

    def get_exact_asymptotics(self):
        '''
            [chi0] = 1/eV
            [beta] = 1/eV
            [iw] = eV
        '''
        chi0_asympt = np.zeros((self.niw), dtype=complex)
        iw = self.wn * 2 * np.pi / self.beta * 1j
        _, fac2, fac3 = self.get_asympt_prefactors()
        ind = self.wn != 0
        # 1/eV * 1/eV^2 * eV^2 = 1/eV
        chi0_asympt[ind] = -self.beta / 2 * 1 / iw[ind] ** 2 * (fac2 - fac3)
        ind = self.wn == 0
        # 1/eV - 1/eV^3 * eV^2 - 1/eV^3 * eV^2 = 1/eV
        chi0_asympt[ind] = self.beta / 4 - self.beta ** 3 / 24 * fac2 - self.beta ** 3 / 48 * fac3
        return chi0_asympt

    def get_asympt_sum_q(self, niv_sum, q_list, freq_notation=None):
        if freq_notation is None:
            freq_notation = self.freq_notation
        chi0_asympt_sum = np.zeros((len(q_list), self.niw), dtype=complex)
        v = 1 / (1j * mf.vn(self.beta, niv_sum + self.niw))
        niv = v.size // 2
        fac1, fac2, fac3 = self.get_asympt_prefactors_q(q_list)
        for i, iwn in enumerate(self.wn):
            iws, iws2 = mf.get_freq_shift(iwn, freq_notation)
            v_sum = v[niv - niv_sum + iws:niv + niv_sum + iws]
            vpw_sum = v[niv - niv_sum + iws2:niv + niv_sum + iws2]
            chi0_asympt_sum[:, i] += np.sum(v_sum * vpw_sum)
            chi0_asympt_sum[:, i] -= np.sum(v_sum ** 2 * vpw_sum + v_sum * vpw_sum ** 2) * fac1
            chi0_asympt_sum[:, i] += np.sum(v_sum ** 3 * vpw_sum + v_sum * vpw_sum ** 3) * fac2
            chi0_asympt_sum[:, i] += np.sum(v_sum ** 2 * vpw_sum ** 2) * fac3
        return chi0_asympt_sum * (-1 / self.beta)

    def get_exact_asymptotics_q(self, q_list):
        chi0q_asympt = np.zeros((len(q_list), self.niw), dtype=complex)
        iw = self.wn * 2 * np.pi / self.beta * 1j
        _, fac2, fac3 = self.get_asympt_prefactors_q(q_list)
        ind = self.wn != 0
        chi0q_asympt[:, ind] = -self.beta / 2 * 1 / iw[None, ind] ** 2 * (fac2 - fac3[:, None])
        ind = self.wn == 0
        chi0q_asympt[:, ind] = self.beta / 4 - self.beta ** 3 / 24 * fac2 - self.beta ** 3 / 48 * fac3[:, None]
        return chi0q_asympt

    def get_chi0_q_list(self, niv, q_list, freq_notation=None):
        if freq_notation is None:
            freq_notation = self.freq_notation
        if self.chi0_method == 'sum':
            return vec_get_chi0_q(self.giwk_obj.g_full(), self.beta, niv, self.wn, q_list, freq_notation, self.is_full_wn)

    def get_gchi0_q_list(self, niv, q_list, freq_notation=None):
        if freq_notation is None:
            freq_notation = self.freq_notation
        if self.chi0_method == 'sum':
            return vec_get_gchi0_q(self.giwk_obj.g_full(), self.beta, niv, self.wn, q_list, freq_notation, self.is_full_wn)


if __name__ == '__main__':
    pass
