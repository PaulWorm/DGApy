# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
# ------------------------------------------------ OBJECTS -------------------------------------------------------------



class two_point():
    ''' Parent function for two-point correlation function and derived quantities like self-energy'''


    def __init__(self, beta=1.0, u=1.0, niv=0):
        self.set_beta(beta)
        self.set_u(u)
        self.set_niv(niv)
        self.set_vn()
        self.set_iv()

    def set_beta(self, beta):
        self._beta = beta

    def set_u(self,u):
        self._u = u

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

class matsubara_greens_function(two_point):
    ''' Contains routines for the Matsubara one-particle Green's function'''

    def __init__(self, beta=1.0, u=1.0, niv=0, mu=0,  ek=None, sigma=None):
        two_point.__init__(self,beta=beta,u=u, niv=niv)
        self.set_mu(mu)
        self.set_ek(ek)
        self.set_sigma(sigma)
        self.set_gkiw()


    def set_sigma(self,sigma=None):
        self._sigma = sigma

    def set_mu(self,mu=0):
        self._mu = mu

    def set_ek(self, ek=None):
        self._ek = ek

    def get_gkiw(self):
        return 1./(self._iv + self._mu - self._ek - self._sigma)

    def set_gkiw(self):
        self._gkiw = self.get_gkiw()

    def get_giw(self):
        try:
            self._gkiw
        except:
            self.set_gkiw()
        return self._gkiw.mean(axis=(0,1))
