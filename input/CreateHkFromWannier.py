import numpy as np
import matplotlib.pyplot as plt
import Hk as hamk
import BrillouinZone as bz

path = '/mnt/c/Users/pworm/Research/Susceptibility/La2NiO4/'
name_core = '1onSTO'
fname = path + f'{name_core}_hr_no_header.dat'
fname_out = path + f'{name_core}.hk'
fig_name = name_core

Hr, Rgrid, Rweights, orbs = hamk.read_Hr_w2k(fname)
nk = (200,200,1)
kgrid = bz.KGrid(nk=nk)
kmesh = kgrid.kmesh.reshape(3,-1)
Hk = hamk.convham(Hr, Rgrid, Rweights, kmesh)#.reshape(nk)
# write hamiltonian in the wannier90 format to file
f = open(fname_out, 'w')

# header: no. of k-points, no. of wannier functions(bands), no. of bands (ignored)
print(np.prod(nk), 1, 1, file=f)
for ik in range(nk[-1]):
    for pos, Ed in enumerate(Hk):
        print(kmesh[0,pos], kmesh[1,pos], kmesh[2,pos], file=f)
        print(Ed[0][0].real, 0, file=f)

f.close()

##DOS
idelta = 1e-2j / 1
w = np.linspace(-2.5, 2.5, 500)

G = np.sum(1. / (w[:, None, None, None] - Hk[None, :, :,:] + idelta), axis=(1, 2,3)) / np.prod(nk)
plt.plot(w, -G.imag / np.pi)
plt.savefig(fig_name + '.jpg')
plt.show()

