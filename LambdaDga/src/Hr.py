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

def motoharu_nickelates(t=0.25):
    tp = -0.25 * t
    tpp = 0.12 * t
    return np.array([[t, t, 0], [tp, tp, 0.], [tpp, tpp, 0]])

def unfrustrated_square(t=1.00):
    tp = 0
    tpp = 0
    return np.array([[t, t, 0], [tp, tp, 0.], [tpp, tpp, 0]])

def Ba2CuO4_plane():
    # Ba2CuO3.25 parameters
    tx = 0.0258
    ty = 0.5181
    tpxy = 0.0119
    tppx = -0.0014
    tppy = 0.0894
    return one_band_2d_quasi1D(tx=tx,ty=ty,tppx=tppx,tppy=tppy,tpxy=tpxy)