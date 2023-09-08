import numpy as np
import pandas as pd

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
    # tx = 0.018545
    # ty = 0.470181
    # tpxy = 0.006765
    # tppx = 0.001255
    # tppy = 0.084597
    tx = 0.0185
    ty = 0.47
    tpxy = 0.0068
    tppx = 0.0013
    tppy = 0.085

    return one_band_2d_quasi1D(tx=tx,ty=ty,tppx=tppx,tppy=tppy,tpxy=tpxy)

def Ba2CuO4_plane_2D_projection():
    # Ba2CuO3.25 2D-projection parameters
    tx = 0.0258
    ty = 0.5181
    tpxy = 0.0119
    tppx = -0.0014
    tppy = 0.0894
    return one_band_2d_quasi1D(tx=tx,ty=ty,tppx=tppx,tppy=tppy,tpxy=tpxy)

# ==================================================================================================================
def read_hr_w2k(fname):
    '''
        Load the H(R) LDA-Hamiltonian from a wien2k hr file.
    '''
    Hr_file = pd.read_csv(fname, header = None,names = np.arange(15), sep = '\s+', dtype = float, engine='python')
    Nbands = Hr_file.values[0].astype(int)[0]
    Nr = Hr_file.values[1].astype(int)[0]

    tmp = np.reshape(Hr_file.values,(np.size(Hr_file.values),1))
    tmp = tmp[~np.isnan(tmp)]

    Rweights = tmp[2:2+Nr].astype(int)
    Rweights = np.reshape(Rweights,(np.size(Rweights),1))
    Ns = 7
    Ntmp = np.size(tmp[2+Nr:]) // Ns
    tmp = np.reshape(tmp[2+Nr:],(Ntmp,Ns))

    Rgrid = np.reshape(tmp[:,0:3],(Nr,Nbands,Nbands,3))
    orbs  = np.reshape(tmp[:,3:5],(Nr,Nbands,Nbands,2))
    Hr = np.reshape(tmp[:, 5] + 1j * tmp[:, 6],(Nr,Nbands,Nbands))
    return Hr, Rgrid, Rweights, orbs
# ==================================================================================================================

def write_hr_w2k(hr, r_grid, r_weights, orbs, fname):
    '''
        Write a real-space Hamiltonian in the format of w2k to a file.
    '''
    pass


# if __name__ == '__main__':

