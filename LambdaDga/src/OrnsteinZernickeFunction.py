# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Extract the correlation length via fitting an OZ function to the susceptibility

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------


import numpy as np
# import matplotlib.pyplot as plt
import scipy.optimize as sciopt
import BrillouinZone as bz


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def oz_spin_w0(qgrid, A, xi):
    Qx = np.pi
    Qy = np.pi
    Qz = 0.0
    oz =  A / (xi ** (-2) + (qgrid.kx[:, None, None] - Qx) ** 2 + (qgrid.ky[None, :, None] - Qy) ** 2 + (
                qgrid.kz[None, None, :] - Qz) ** 2)
    return oz.flatten()


def fit_oz_spin(qgrid, chi):
    initial_guess = (1,4.)
    params = sciopt.curve_fit(oz_spin_w0,qgrid,chi,p0=initial_guess)
    return params

if __name__ == '__main__':
    nk = (8,8,1)
    qgrid = bz.KGrid(nk=nk)
    chi = oz_spin_w0(qgrid,1,4.)
    chi = chi + 1*np.random.normal(size=chi.shape)
    oz_coeff,_ = fit_oz_spin(qgrid,chi)
    print(f'{oz_coeff=}')

    chi = np.reshape(chi,(nk))

