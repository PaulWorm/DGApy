import numpy as np
import matplotlib.pyplot as plt
import Hk as hamk
import BrillouinZone as bz

# path = '/mnt/c/Users/pworm/Research/Susceptibility/La2NiO4/4Paul2/'
path = 'D:/Research/Susceptibility/La2NiO4/4Paul2/'
# path = 'D:/Research/La2NiO4/ForDiana/4Paul3/onSTO/2orb_AF/'
# path = 'D:/Research/TestWien2k/LaNiO2/CheckConvham/'
# name_core = '2orb_AF'
name_core = '1onSTO-2orb'
# name_core = '3onNGO-2orb'
# name_core = '2onLSAT-2orb'
# name_core = 'CheckConvham'
# name_core = '4onLAO-2orb'
# name_core = '4onLAO-2orb'
fname = path + f'{name_core}_hr_no_header.dat'
# nk = (10,32,32)
nk = (64,64,1)
# nk = (10,10,10)
fname_out = path + f'{name_core}_nkx{nk[0]}_nky{nk[1]}_nkz{nk[2]}.hk'
fig_name = name_core

Hr, Rgrid, Rweights, orbs = hamk.read_Hr_w2k(fname)
# nk = (100,100,12)

kgrid = bz.KGrid(nk=nk)
kmesh = kgrid.kmesh.reshape(3,-1)
Hk = hamk.convham(Hr, Rgrid, Rweights, kmesh)

hamk.write_hk_wannier90(Hk, fname_out, kmesh, nk)

##DOS
idelta = 1e-2j / 1
nw = 500
w = np.linspace(-2.5, 2.5, nw)
n_orbs = np.shape(Hk)[-1]
eye_bands = np.eye(n_orbs)
# G = np.sum(np.linalg.inv(w[:, None, None, None]*eye_bands[None,None,:,:] - Hk[None, :, :,:] + idelta*eye_bands[None,None,:,:]), axis=(1, 2,3)) / np.prod(nk)
G_orbs = np.sum(np.linalg.inv(w[:, None, None, None]*eye_bands[None,None,:,:] - Hk[None, :, :,:] + idelta*eye_bands[None,None,:,:]), axis=(1)) / np.prod(nk)
plt.plot(w, -np.trace(G_orbs,axis2=-2,axis1=-1).imag / np.pi)
plt.savefig(path + fig_name + f'_nkx{nk[0]}_nky{nk[1]}_nkz{nk[2]}.jpg')
plt.show()

plt.figure()
for i in range(n_orbs):
    plt.plot(w, -G_orbs[:,i,i].imag / np.pi)
plt.savefig(path + fig_name + f'_orbs_resolved_nkx{nk[0]}_nky{nk[1]}_nkz{nk[2]}.jpg')
plt.show()

n = np.trapz(-1/np.pi*G[:nw//2].imag,w[:nw//2])
n_check = np.trapz(-1/np.pi*G.imag,w)
print(f'{n=}')
print(f'{n_check=}')

