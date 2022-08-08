import numpy as np
import matplotlib.pyplot as plt
import Hk as hamk
import Hr as hamr
import BrillouinZone as bz

path = '/mnt/c/Users/pworm/Research/Susceptibility/La2NiO4/'
name_core = 'bulk'
fname = path + f'{name_core}_hr_no_header.dat'
fname_out = path + f'{name_core}.hk'
fig_name = name_core


# Wannier Hamiltonian
Hr, Rgrid, Rweights, orbs = hamk.read_Hr_w2k(fname)
nk = (200,200,1)
kgrid = bz.KGrid(nk=nk)
kmesh = kgrid.kmesh.reshape(3,-1)
Hk = hamk.convham(Hr, Rgrid, Rweights, kmesh)


# Tight Binding Hamiltonian
tev = 0.414093
mu = 0.491384
t = 1.00
tp = -0.074917/tev
tpp = 0.045107/tev
hr = hamr.one_band_2d_t_tp_tpp(t=t,tp=tp,tpp=tpp)
kgrid = bz.KGrid(nk)
hk = hamk.ek_3d(kgrid.grid,hr)*tev+mu

##DOS
idelta = 5e-2j
w = np.linspace(-3, 3, 500)

G_w = np.sum(1. / (w[:, None, None, None] - Hk[None, :, :,:] + idelta), axis=(1, 2,3)) / np.prod(nk)
G_tb = np.sum(1. / (w[:, None, None, None] - hk[None, :, :,:] + idelta), axis=(1, 2,3)) / np.prod(nk)
plt.plot(w, -G_w.imag / np.pi,label='Wannier')
plt.plot(w, -G_tb.imag / np.pi,label='Tight Binding')
plt.ylabel('DOS')
plt.xlabel('$\omega$')
plt.legend()
plt.savefig(path + name_core + '_DOS_comparison.jpg')
plt.show()