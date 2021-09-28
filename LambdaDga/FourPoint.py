# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import w2dyn_aux
import numpy as np


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def cen2lin(val=None, start=0):
    return val + start


# ------------------------------------------------ OBJECTS -------------------------------------------------------------


class local_four_point():
    ''' Parent class for local four-point correlation '''

    def __init__(self, giw=None, beta=1.0, u=1.0, channel='dens', box_sizes=None, iw=None):
        self._beta = beta
        self._u = u
        self._channel = channel
        self._box_sizes = box_sizes
        self._iw = iw
        self._niw = np.size(self._iw)
        self.set_giw(giw=giw)

    def set_giw(self, giw=None):
        self._giw = giw
        self._niv_giw = giw.shape[0] // 2


class local_bubble(local_four_point):
    ''' Computes the local Bubble suszeptibility \chi_0 = - beta GG '''

    def __init__(self, giw=None, beta=1.0, box_sizes=None, iw=None):
        local_four_point.__init__(self, giw=giw, beta=beta, box_sizes=box_sizes, iw=iw)
        self._chi0 = {}
        self._gchi0 = {}

    def get_gchi0(self, niv_gchi0=-1, wn=0):
        if (niv_gchi0 == -1):
            niv_gchi0 = self._niv_giw - np.abs(wn)
        return - 1. / self._beta * self._giw[self._niv_giw - niv_gchi0:self._niv_giw + niv_gchi0] \
               * self._giw[self._niv_giw - niv_gchi0 - wn:self._niv_giw + niv_gchi0 - wn]

    def vec_get_gchi0(self, niv_gchi0=-1, iw=None):
        return np.array([self.get_gchi0(niv_gchi0=niv_gchi0, wn=wn) for wn in iw])

    def vec_get_chi0(self, niv_sum=-1, iw=None):
        return np.array([self.get_gchi0(niv_sum=niv_sum, wn=wn) for wn in iw])

    def get_chi0(self, niv_sum=-1, wn=0):
        if (niv_sum == -1):
            niv_sum = self._niv_giw - np.abs(wn)
        return - 1. / self._beta * np.sum(self._giw[self._niv_giw - niv_sum:self._niv_giw + niv_sum]
                                          * self._giw[self._niv_giw - niv_sum - wn:self._niv_giw + niv_sum - wn])

    def vec_get_chi0(self, niv_sum=-1, iw=None):
        return np.array([self.get_chi0(niv_sum=niv_sum, wn=wn) for wn in iw])

    def get_asympt_correction(self, niv_inner=None, niv_outer=None, wn=0):
        niv_full = niv_outer+niv_inner
        v_asympt = 1j * (np.arange(-niv_full, niv_full) * 2. + 1.) * np.pi / self._beta
        vpw_asympt = 1j * (np.arange(-niv_full - wn, niv_full- wn) * 2. + 1.) * np.pi / self._beta
        asympt = np.sum(1. / vpw_asympt * 1. / v_asympt) \
                 - np.sum(1. / vpw_asympt[niv_full - niv_inner:niv_full + niv_inner] *
                          1. / v_asympt[niv_full - niv_inner:niv_full + niv_inner])
        return - 1. / self._beta * asympt

    def vec_get_asympt_correction(self, niv_inner=None, niv_outer=None, iw=None):
        return np.array(
            [self.get_asympt_correction(niv_inner=niv_inner, niv_outer=niv_outer, wn=wn) for wn in iw])

    def set_chi0(self, chi0=None, range='niv_core'):
        if (chi0 is None):
            self._chi0[range] = self.vec_get_chi0(niv_sum=self._box_sizes[range], iw=self._iw)
        else:
            self._chi0[range] = chi0

    def set_gchi0(self, gchi0=None, range='niv_core'):
        if (gchi0 is None):
            self._gchi0[range] = self.vec_get_gchi0(niv_gchi0=self._box_sizes[range], iw=self._iw)
        else:
            self._gchi0[range] = gchi0

    def set_chi0_asympt(self, chi0=None):
        if (chi0 is None):
            self._chi0['niv_asympt'] = self._chi0['niv_urange'] + self.vec_get_asympt_correction(
                niv_inner=self._box_sizes['niv_urange'],
                niv_outer=self._box_sizes['niv_asympt'], iw=self._iw)


# ======================================================================================================================
# ---------------------------------------- local_connected_four_point class --------------------------------------------
# ======================================================================================================================

class local_connected_four_point(local_four_point):
    ''' Class for local four-point functions '''

    def __init__(self, loc_bub=None, u=1.0, channel='dens'):
        self._lb = loc_bub
        local_four_point.__init__(self, giw=loc_bub._giw, beta=loc_bub._beta, u=u, channel=channel,
                                  box_sizes=loc_bub._box_sizes, iw=loc_bub._iw)


    def cut_iv(self, four_point=None, niv_cut=10):
        niv = four_point.shape[-1] // 2
        if (niv_cut > niv):
            raise ValueError('niv_cut > niv')

        if(np.size(four_point.shape) == 3):
            return four_point[:, niv - niv_cut:niv + niv_cut, niv - niv_cut:niv + niv_cut]
        else:
            return four_point[niv - niv_cut:niv + niv_cut, niv - niv_cut:niv + niv_cut]

    def get_ggv(self, niv_ggv=-1):
        niv = self._niv_giw
        if (niv_ggv == -1):
            niv_ggv = niv
        return self._giw[niv - niv_ggv:niv + niv_ggv][:, None] * self._giw[niv - niv_ggv:niv + niv_ggv][None, :]

    def get_g2_from_w2dyn_file(self, fname='g4iw_sym.hdf5', iw='full'):
        g2_file = w2dyn_aux.g4iw_file(fname=fname)
        if (iw == 'full'):
            g2 = g2_file.read_g2_full(channel=self._channel)
        else:
            niw_full = g2_file.get_niw(channel=self._channel)
            iw_lin = cen2lin(val=iw, start=niw_full)
            g2 = g2_file.read_g2_iw(channel=self._channel, iw=iw_lin)
        return g2

    def set_g2_from_w2dyn_file(self, fname='g4iw_sym.hdf5'):
        self._g2 = self.get_g2_from_w2dyn_file(fname=fname, iw=self._iw)
        self._niv_g2 = self._g2.shape[-1] // 2

    def get_gchir_from_g2(self, g2=None, wn=0):
        if (self._channel == 'dens' and wn == 0):
            gchir = self._beta * (g2 - 2. * self.get_ggv(niv_ggv=self._niv_g2))
        else:
            gchir = self._beta * g2
        return gchir

    def vec_get_gchir_from_g2(self, g2=None, iw=None):
        return np.array([self.get_gchir_from_g2(g2=g2, wn=wn) for wn in iw])

    def set_gchir_from_g2(self, g2=None):
        if (g2 is None):
            self._gchir = self.vec_get_gchir_from_g2(g2=self._g2, iw=self._iw)
        else:
            self._gchir = self.vec_get_gchir_from_g2(g2=g2, iw=self._iw)

    def get_gammar_from_gchir(self, gchir=None, gchi0=None):
        full = self._u / (self._beta * self._beta) + np.diag(1. / gchi0)
        inv_full = np.linalg.inv(full)
        inv_core = self.cut_iv(inv_full, self._box_sizes['niv_core'])
        core = np.linalg.inv(inv_core)
        chigr_inv = np.linalg.inv(gchir)
        return -(core - chigr_inv - self._u / (self._beta * self._beta))

    def vec_get_gammar_from_gchir(self, gchir=None, gchi0=None, iw=None):
        return np.array([self.get_gammar_from_gchir(gchir=gchir[wn], gchi0=gchi0[wn]) for wn in iw])

    def set_gammar_from_gchir(self):
        self._gammar = self.vec_get_gammar_from_gchir(gchir=self._gchir, gchi0=self._lb._gchi0['niv_core'], iw=self._iw)

    def get_gchi_aux_from_gammar(self, gammar=None, gchi0=None):
        gchi0_inv = np.diag(1. / gchi0)
        chi_aux_inv = gchi0_inv + gammar - self._u / (self._beta * self._beta)
        return np.linalg.inv(chi_aux_inv)

    def vec_get_gchi_aux_from_gammar(self, gammar=None, gchi0=None, iw=None):
        return np.array([self.get_gchi_aux_from_gammar(self, gammar=gammar[wn], gchi0=gchi0[wn]) for wn in iw])

    def set_gchi_aux_from_gammar(self):
        self._gchi_aux = self.vec_get_gchi_aux_from_gammar(self, gammar=self._gammar, gchi0=self._lb._gchi0['niv_core'], iw=self._iw)

    def set_chi_aux(self):
        self._chi_aux = 1. / (self._beta * self._beta) * np.sum(self._gchi_aux, axis=(-2, -1))

    def set_chir_from_chi_aux(self):
        self._chir['niv_urange'] = 1. / (1. / (self._chi_aux + self._lb.chi0['niv_urange'] - self._lb.chi0['niv_core']) + self._u)
        self._chir['niv_asympt'] = self._chir_urange + self._lb.chi0['niv_asympt'] - self._lb.chi0['niv_urange']

    def set_fermi_bose(self):
        niv_core = self._box_sizes['niv_core']
        niv_urange = self._box_sizes['niv_urange']
        self._vrg['niv_core'] = 1. / self._lb.gchi0['niv_core'] * np.sum(self._chi_aux, axis=-1)
        self._vrg['niv_urange'] = 1. / self._beta * np.ones((self._niw, 2 * niv_urange), dtype=complex)
        self._vrg['niv_urange'][:, niv_urange - niv_core: niv_urange + niv_core] = self._vrg['niv_core']
        self._vrg['niv_asympt'] = self._vrg['niv_urange'] * \
                              ((1 - self._u * self._chi['niv_urange'][:, None]) / (
                                          1 - self._u * self._chi['niv_asympt'][:, None]))

# def set_chi0_asympt(self):
#     if(self._chi0 = )

# class local_generalized_suszeptibility(local_four_point):
#     ''' \chi = \chi_0 + \chi_0 F \chi_0 '''
#
#     def from_g2(self, g2=None):
#
#
#
# class two_particle_greens_function(local_four_point):
#     ''' Two-particle Green's function '''
#
#     def from_w2dyn_file(self, fname='g4iw_sym.hdf5', iw='full'):
#         g2_file = w2dyn_aux.g4iw_file(fname=fname)
#         if (iw == 'full'):
#             self._g2 = g2_file.read_g2_full(channel=self._channel)
#             self._niw = self._g2.shape[0] // 2
#             self._iw = np.arange(-self._niw, self._niw)
#         else:
#             niw_full = g2_file.get_niw(channel=self._channel)
#             self._iw = iw
#             iw_lin = cen2lin(val=self._iw, start=niw_full)
#             self._g2 = g2_file.read_g2_w(channel=self._channel, iw=iw_lin)
