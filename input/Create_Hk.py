# ---------------------------------------------------------------------------------
# File: create_hk.py
# Date: 11.05.2020
# Author: Paul Worm
# Description: Generates a file for the one-particle part of the Hamiltonian in k-space.
# ---------------------------------------------------------------------------------
import sys, os
sys.path.append('../src/')
sys.path.append(os.environ['HOME'] + "/Programs/dga/LambdaDga/src")

import numpy as np
import matplotlib.pyplot as plt
import Hr as hr_mod
import Hk as hk_mod
import BrillouinZone as bz
# ---------------------------------- START CODE ------------------------------------

# Define Paramters:
# ----------------------------------
nkx = 24
nky = 24
nkz = 1
# t0  = -0.879
t = 1.0
tp = -0.2
tpp = 0.1

hr = hr_mod.Ba2CuO4_plane()
kgrid = bz.KGrid(nk=(nkx,nky,nkz))
Hk = hk_mod.ek_3d(kgrid=kgrid.grid,hr=hr)

# write hamiltonian in the wannier90 format to file
f = open('Ba2CuO4_plane1_nkx{}_nky{}_nkz{}.hk'.format(nkx,nky,nkz), 'w')

# header: no. of k-points, no. of wannier functions(bands), no. of bands (ignored)
print(nkx * nky * nkz, 1, 1, file=f)
for ik in range(nkz):
    for pos, Ed in np.ndenumerate(Hk):
        print(kgrid.kx[pos[0]], kgrid.ky[pos[1]], kgrid.kz[ik], file=f)
        print(Ed, 0, file=f)

f.close()

##DOS
idelta = 5e-2j / 0.25
w = np.linspace(-10, 10, 500)

G = np.sum(1. / (w[:, None, None,None] - Hk[None,:, :,:] + idelta), axis=(1, 2,3)) / kgrid.nk_tot
plt.plot(w, -G.imag / np.pi)
plt.savefig('Ba2CuO4_plane1_nkx{}_nky{}_nkz{}.jpg'.format(nkx,nky,nkz))
plt.show()

# ------------------------------------ END CODE ------------------------------------
