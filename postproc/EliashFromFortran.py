import numpy as np
import h5py
import matplotlib.pyplot as plt
import Config as conf
import EliashbergEquation as eq
import BrillouinZone as bz
import MatsubaraFrequencies as mf

def load_iqirr(link=None):
    data = []
    for key in link.keys():
        data.append(link[str(key)][()])
    return np.array(data)

def build_fbz(mat=None, nk=None):
    ''' irrk assumed to be 0th dimension of mat. also nkz = 1 adn square lattice symmetry is assumed'''
    mat_fbz = np.zeros(nk+np.shape(mat)[1:], dtype=complex)

    iter = 0
    for ikx in range(nk[0] // 2 - 1, nk[0]):
        for iky in range(nk[1] // 2 - 1, ikx + 1):
            mat_fbz[ikx, iky, 0,...] = mat[iter,...]
            iter = iter + 1

    for ikx in range(nk[0] // 2 - 1, nk[0]):
        for iky in range(ikx, nk[0]):
            mat_fbz[ikx, iky,...] = mat_fbz[iky, ikx,...]

    for ikx in range(nk[0] // 2 - 1):
        for iky in range(nk[1] // 2 - 1, nk[1]):
            mat_fbz[ikx, iky,...] = mat_fbz[nk[0] - 1 - ikx, iky,...]


    for ikx in range(nk[0]):
        for iky in range(nk[1] // 2 - 1):
            mat_fbz[ikx, iky,...] = mat_fbz[ikx, nk[0] - 1 - iky,...]
    mat_fbz = np.roll(mat_fbz,(nk[0]//2-1),0)
    mat_fbz = np.roll(mat_fbz,(nk[1]//2-1),1)
    return mat_fbz




#input_path = '/mnt/d/Research/Ba2CuO4/Ba2CuO4_plane1/U3.0eV_n0.93_b120/Eliashberg/Nw060/'
input_path = '/mnt/c/Users/pworm/Research/Superconductivity/Programs/SuperEigen/testsuit/testdatasets/NdNiO2/Motoharu/INPUT-for-eliashberg/Nw040/'
fname = 'InputEliashberg_v1.hdf5'

file = h5py.File(input_path+fname)
beta = file['.system']['beta'][()]
nk = (file['.system']['Nkx'][()],file['.system']['Nky'][()],file['.system']['Nkz'][()])
gk_dga = file['1P']['Giwk']['irrq'][()].T
gk_dga = build_fbz(mat=gk_dga, nk=nk)[...,0,0]

k_grid = bz.KGrid(nk=nk,ek=None)

gamma_sing = load_iqirr(link=file['2P']['Gamma']['sing']['pp']['iqirr'])
gamma_sing = build_fbz(mat=gamma_sing, nk=nk)[...,0,0]
gk_dga = mf.cut_v(gk_dga,niv_cut=gamma_sing.shape[3]//2,axes=(3,))
niv = gamma_sing.shape[3]//2

plt.figure()
plt.imshow(gk_dga[:,:,0,niv].imag, cmap='RdBu')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(gamma_sing[:,:,0,niv,niv].imag, cmap='RdBu')
plt.colorbar()
plt.show()






el_conf = conf.EliashbergConfig(k_sym='d-wave')
norm = np.prod(nk) * beta

gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=el_conf.gap0_sing['k'], v_type=el_conf.gap0_sing['v'],
                        k_grid=k_grid.grid)

powiter_sing = eq.EliashberPowerIteration(gamma=gamma_sing, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
                                          n_eig=el_conf.n_eig)

lambda_ = powiter_sing.lam[0]
gap = powiter_sing.gap
print(f'{lambda_=}')

plt.figure()
plt.imshow(gap[0,:,:,0,niv].real, cmap='RdBu')
plt.colorbar()
plt.show()