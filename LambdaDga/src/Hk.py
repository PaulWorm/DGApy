# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np


# ------------------------------------------------ OBJECTS -------------------------------------------------------------

def ek_square(kx=None, ky=None, t=1.0, tp=0.0, tpp=0.0):
    return - 2. * t * (np.cos(kx) + np.cos(ky)) - 4. * tp * np.cos(kx) * np.cos(ky) \
           - 2. * tpp * (np.cos(2. * kx) + np.cos(2. * ky))


def ekpq_3d(kx=None, ky=None, kz=None, qx=0, qy=0, qz=0, t_mat=None):
    kx = kx + qx
    ky = ky + qy
    kz = kz + qz
    ek = - 2.0 * (t_mat[0, 0] * np.cos(kx) + t_mat[0, 1] * np.cos(ky) + t_mat[0, 2] * np.cos(kz))
    ek = ek - 2.0 * (t_mat[1, 0] * np.cos(kx + ky) + t_mat[1, 1] * np.cos(kx - ky))
    ek = ek - 2.0 * (t_mat[2, 0] * np.cos(2. * kx) + t_mat[2, 1] * np.cos(2. * ky) + t_mat[2, 2] * np.cos(kz))
    return ek


def ek_3d(kgrid=None, hr=None):
    kx = kgrid[0][:, None, None]
    ky = kgrid[1][None, :, None]
    kz = kgrid[2][None, None, :]
    ek = - 2.0 * (hr[0, 0] * np.cos(kx) + hr[0, 1] * np.cos(ky) + hr[0, 2] * np.cos(kz))
    ek = ek - 2.0 * (hr[1, 0] * np.cos(kx + ky) + hr[1, 1] * np.cos(kx - ky))
    ek = ek - 2.0 * (hr[2, 0] * np.cos(2. * kx) + hr[2, 1] * np.cos(2. * ky) + hr[2, 2] * np.cos(kz))
    return ek
