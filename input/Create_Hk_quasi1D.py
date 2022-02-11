# ---------------------------------------------------------------------------------
# File: create_hk.py
# Date: 11.05.2020
# Author: Paul Worm
# Description: Generates a file for the one-particle part of the Hamiltonian in k-space.
# ---------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------- START CODE ------------------------------------

# Define Paramters:
# ----------------------------------
kpoints = 200
Nkz = 1
# t0  = -0.879
t = 1.0
tp = -0.15
tpp = 0.075
# generate non-interacting 2d Hamiltonian
# kmesh = np.linspace(-(kpoints)//2+1, kpoints//2, kpoints, endpoint=True)/float(kpoints)
kmesh = np.linspace(0, kpoints, kpoints, endpoint=False) / float(kpoints)
kz = np.linspace(0, 1, Nkz, endpoint=False)
Hk = -2 * t * (np.cos(2 * np.pi * kmesh)[None, :] + np.cos(2 * np.pi * kmesh)[:, None]) - 4 * tp * t * np.cos(
    2 * np.pi * kmesh)[None, :] * np.cos(2 * np.pi * kmesh)[:, None] - 2 * tpp * t * (
                 np.cos(4 * np.pi * kmesh)[None, :] + np.cos(4 * np.pi * kmesh)[:, None])

# write hamiltonian in the wannier90 format to file
f = open('2dhubbard_tp_{:03}_tpp_{:03}_Nk_{:03}_Nkz{:03}.hk'.format(tp, tpp, kpoints, Nkz), 'w')

# header: no. of k-points, no. of wannier functions(bands), no. of bands (ignored)
print(kpoints ** 2 * Nkz, 1, 1, file=f)
for ik in range(Nkz):
    for pos, Ed in np.ndenumerate(Hk):
        print(kmesh[pos[0]], kmesh[pos[1]], kz[ik], file=f)
        print(Ed, 0, file=f)

f.close()

##DOS
idelta = 5e-2j / 0.25
w = np.linspace(-10, 10, 500)

G = np.sum(1. / (w[:, None, None] - Hk[None, :, :] + idelta), axis=(1, 2)) / kpoints ** 2
plt.plot(w, -G.imag / np.pi)
plt.savefig('2dhubbard_tp_{:03}_tpp_{:03}_Nk_{:03}_Nkz{:03}.jpg'.format(tp, tpp, kpoints, Nkz))
plt.show()

# ------------------------------------ END CODE ------------------------------------
