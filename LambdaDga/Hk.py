# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np


# ------------------------------------------------ OBJECTS -------------------------------------------------------------

class tight_binding():
    ''' Contains operatons for the kinetic part of the Hamiltonian'''

    #def __init__(self):

    def ek_square(self,kx=None, ky=None, t=1.0, tp=0.0, tpp=0.0):
        return - 2. * t * (np.cos(kx) + np.cos(ky)) - 4. * tp * np.cos(kx) * np.cos(ky) \
               - 2. * tpp * (np.cos(2. * kx) + np.cos(2. * ky))

    def ek_2d(self,kx=None, ky=None, t_mat=None):
        ek = - 2.0 * (t_mat[0, 0] * np.cos(kx) + t_mat[0, 1] * np.cos(ky))
        ek = ek - 2.0 * (t_mat[1, 0] * np.cos(kx + ky) + t_mat[1, 1] * np.cos(kx - ky))
        ek = ek - 2.0 * (t_mat[2, 0] * np.cos(2. * kx) + t_mat[2, 1] * np.cos(2. * ky))
        return ek