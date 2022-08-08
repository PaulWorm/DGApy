import numpy as np
import matplotlib.pyplot as plt
import Hk as hamk
import BrillouinZone as bz

path = '/mnt/c/Users/pworm/Research/Susceptibility/La2NiO4/4Paul2/'
name_core = '1onSTO-5orb'
fname = path + f'{name_core}_hr_no_header.dat'
fname_out = path + f'{name_core}.hk'
fig_name = name_core

Hr, Rgrid, Rweights, orbs = hamk.read_Hr_w2k(fname)
# nk = (100,100,12)
nk = (100,100,12)
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
G = np.sum(1. / (w[:, None, None, None] - Hk[None, :, :,:] + idelta*eye_bands[None,None,:,:]), axis=(1, 2,3)) / np.prod(nk)
G_orbs = np.sum(1. / (w[:, None, None, None] - Hk[None, :, :,:] + idelta*eye_bands[None,None,:,:]), axis=(1)) / np.prod(nk)
plt.plot(w, -G.imag / np.pi)
plt.savefig(path + fig_name + '.jpg')
plt.show()

plt.figure()
for i in range(n_orbs):
    plt.plot(w, -G_orbs[:,i,i].imag / np.pi)
plt.savefig(path + fig_name + '_orbs_resolved.jpg')
plt.show()

n = np.trapz(-1/np.pi*G[:nw//2].imag,w[:nw//2])
n_check = np.trapz(-1/np.pi*G.imag,w)
print(f'{n=}')
print(f'{n_check=}')

