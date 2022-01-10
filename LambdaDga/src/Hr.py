import numpy as np


# --------------------------------------- CONSTRUCT REAL SPACE HAMILTONIANS --------------------------------------------

def one_band_2d_t_tp_tpp(t=1.0,tp=0,tpp=0):
    return np.array([[t, t, 0], [tp, tp, 0.], [tpp, tpp, 0]])

def one_band_2d_quasi1D(tx=1.0,ty=0,tppx=0,tppy=0,tpxy=0):
    return np.array([[tx, ty, 0], [tpxy, tpxy, 0.], [tppx, tppy, 0]])

def one_band_2d_triangular_t_tp_tpp(t=1.0,tp=0,tpp=0):
    return np.array([[t, t, 0], [tp, 0, 0.], [tpp, tpp, 0]])

def standard_cuprates(t=1.0):
    tp = -0.2 * t
    tpp = 0.1 * t
    return np.array([[t, t, 0], [tp, tp, 0.], [tpp, tpp, 0]])