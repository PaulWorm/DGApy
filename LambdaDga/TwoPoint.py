# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import Hk as hk


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

# ------------------------------------------------ OBJECTS -------------------------------------------------------------


# ======================================================================================================================
# ---------------------------------------------- TWO-POINT PARENT CLASS ------------------------------------------------
# ======================================================================================================================

class two_point():
    ''' Parent function for two-point correlation function and derived quantities like self-energy'''

    def __init__(self, beta=1.0, niv=0):
        self.set_beta(beta)
        self.set_niv(niv)
        self.set_vn()
        self.set_iv()

    def set_beta(self, beta):
        self._beta = beta

    def set_niv(self, niv):
        self._niv = niv

    def set_vn(self):
        self._vn = np.arange(-self._niv, self._niv)

    def set_iv(self):
        try:
            self._vn
        except:
            self.set_vn()
        self._iv = (self._vn * 2 + 1) * 1j * np.pi / self._beta


# ======================================================================================================================
# ----------------------------------------------------- GKIW CLASS -----------------------------------------------------
# ======================================================================================================================

class GreensFunction(object):
    ''' Contains routines for the Matsubara one-particle Green's function'''

    def __init__(self, iv=None, beta=1.0, mu=0, ek=None, sigma=None):
        self._iv = iv
        self._beta = beta
        self._mu = mu
        self._ek = ek
        self._sigma = sigma
        self._gk = self.get_gk()

    @property
    def iv(self):
        return self._iv

    @iv.setter
    def iv(self, iv):
        self._iv = iv

    @property
    def niv(self):
        return self.iv.size // 2

    @property
    def beta(self):
        return self._beta

    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    @property
    def niv(self) -> int:
        return self.iv.shape[-1] // 2

    @property
    def gk(self):
        return self._gk

    @gk.setter
    def gk(self, gk):
        self._gk = gk

    def get_gk(self):
        return 1. / (self._iv + self._mu - self._ek - self._sigma)

    def get_giw(self):
        return self._gk.mean(axis=(0, 1, 2))

    def cut_self_iv(self, niv_cut=0):
        self.iv = self.cut_iv(arr=self.iv, niv_cut=niv_cut)
        self.sigma = self.cut_iv(arr=self.sigma, niv_cut=niv_cut)
        self.gk = self.cut_iv(arr=self.gk,niv_cut=niv_cut)

    def cut_iv(self, arr = None, niv_cut=0):
        niv = arr.shape[-1] // 2
        return arr[...,niv-niv_cut:niv+niv_cut]






class GreensFunctionGenerator():
    '''Class that takes the ingredients for a Green's function and return GreensFunction objects'''

    def __init__(self, beta = 1.0, kgrid=None, hr=None, sigma=None):
        self._beta = beta
        self._kgrid = kgrid
        self._hr = hr
        self._sigma = sigma

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def kgrid(self):
        return self._kgrid

    @property
    def hr(self):
        return self._hr

    @property
    def sigma(self):
        return self._sigma

    @property
    def niv_sigma(self):
        return self._sigma.shape[-1] // 2

    def nkx(self):
        return self.kgrid[0].size

    def nky(self):
        return self.kgrid[1].size

    def nkz(self):
        return self.kgrid[2].size

    def generate_gk(self, mu=0, qiw=None, niv=-1):
        q = qiw[:3]
        wn = int(qiw[-1])
        kgrid = self.add_q_to_kgrid(q=q)
        ek = hk.ek_3d(kgrid=kgrid, hr=self.hr)
        sigma = self.cut_sigma(niv_cut=niv, wn=wn)
        iv = self.get_iv(niv=niv, wn=wn)
        return GreensFunction(iv=iv[None,None,None,:], beta=self.beta, mu=mu, ek=ek[:,:,:,None], sigma=sigma[None,None,None,:])

    def get_iv(self, niv=0, wn=0):
        if(niv==-1):
            niv = self.niv_sigma - int(np.abs(wn))
        return 1j * ((np.arange(-niv,niv)-wn) * 2 + 1) * np.pi / self.beta


    def cut_sigma(self, niv_cut=-1, wn=0):
        niv = self.niv_sigma
        if(niv_cut == -1):
            niv_cut = niv - int(np.abs(wn))
        return self.sigma[...,niv-niv_cut-wn:niv+niv_cut-wn]

    def add_q_to_kgrid(self, q=None):
        assert (np.size(self.kgrid) == np.size(q)), 'Kgrid and q have different dimensions.'
        kgrid = []
        for i in range(np.size(q)):
            kgrid.append(self._kgrid[i] + q[i])
        return kgrid















#
