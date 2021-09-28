# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import w2dyn_aux
import numpy as np


# ------------------------------------------------ OBJECTS -------------------------------------------------------------


class local_sde():
    ''' Class to solve the local Schwinger-Dyson equation '''

    def __init__(self, beta=1.0, u=1.0, giw=None, iw=None):
        self._beta = beta
        self._u = u
        self._iw = iw
        self.set_giw(giw=giw)


    def set_giw(self, giw=None):
        self._giw = giw
        self._niv_giw = giw.shape[0] // 2

    def get_gloc_grid(self, niv=-1):
        if(niv == -1):
            niv = self._niv_giw - np.max(np.abs(self._iw))
        return np.array([self._giw[self._niv_giw - niv - wn:self._niv_giw + niv - wn] for wn in self._iw])

    def sde(self, vrg=None, chir=None):
        niv2 = np.shape(vrg)[-1] // 2
        gloc_grid = self.get_gloc_grid(self, niv=niv2)
        return  - np.sum(self._u / (2.0) * (vrg * (1. - self._u * chir[:, None]) - 1./self._beta)* gloc_grid, axis=0)












