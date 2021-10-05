# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import w2dyn_aux
import numpy as np
import TwoPoint as tp
import FourPoint as fp


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------


# def get_chi_aux_asympt(chi_aux: fp.FourPoint = None, chi_r_urange: , chi_r_asympt=None, u=1):
#     niv = np.shape(chi_aux)[-1] // 2
#     u_mat = u * np.ones((2 * niv, 2 * niv), dtype=complex)
#     return chi_aux - np.matmul(chi_aux, np.matmul(u_mat, chi_aux)) * (
#                 (1 - u * chi_r_urange) - (1 - u * chi_r_urange) ** 2 / (1 - u * chi_r_asympt))


# def get_f_ladder_from_chir(chi_r:, chi_aux_r=None, vrg=None, gchi0=None, u=1, beta=1):
#     niv = chi_aux_r.shape[-1] // 2
#     unity = np.eye(2*niv, dtype=complex)
#     gchi0_inv = 1./ gchi0
#     return beta ** 2 * gchi0_inv[:,None] * (unity - chi_aux_r * gchi0_inv[None,:]) + u * (1.0 - u * chi_r) * beta * \
#            vrg[:,None] * beta * vrg[None,:]