import numpy as np

def one_band_2d_t_tp_tpp(t=1.0,tp=0,tpp=0):
    return np.array([[t, t, 0], [tp, tp, 0.], [tpp, tpp, 0]])

def one_band_2d_triangular_t_tp_tpp(t=1.0,tp=0,tpp=0):
    return np.array([[t, t, 0], [tp, 0, 0.], [tpp, tpp, 0]])