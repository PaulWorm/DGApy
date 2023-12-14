# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Extract the correlation length via fitting an OZ function to the susceptibility

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
'''
    Module for fitting the Ornstein-Zernicke function to the susceptibility
'''

import numpy as np
from scipy import optimize as opt


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def oz_spin_w0(qgrid, a, xi):
    qx = np.pi
    qy = np.pi
    qz = 0.0
    oz =  a / (xi ** (-2) + (qgrid.kx[:, None, None] - qx) ** 2 + (qgrid.ky[None, :, None] - qy) ** 2 + (
                qgrid.kz[None, None, :] - qz) ** 2)
    return oz.flatten()


def fit_oz_spin(qgrid, chi):
    initial_guess = (np.max(chi),2.)
    params = opt.curve_fit(oz_spin_w0,qgrid,chi,p0=initial_guess)
    return params
