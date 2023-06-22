# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Extract the correlation length via fitting an OZ function to the susceptibility

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------


import numpy as np
import scipy.optimize as sciopt


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def oz_spin_w0(qgrid, A, xi):
    Qx = np.pi
    Qy = np.pi
    Qz = 0.0
    oz =  A / (xi ** (-2) + (qgrid.kx[:, None, None] - Qx) ** 2 + (qgrid.ky[None, :, None] - Qy) ** 2 + (
                qgrid.kz[None, None, :] - Qz) ** 2)
    return oz.flatten()


def fit_oz_spin(qgrid, chi):
    initial_guess = (np.max(chi),2.)
    params = sciopt.curve_fit(oz_spin_w0,qgrid,chi,p0=initial_guess)
    return params
