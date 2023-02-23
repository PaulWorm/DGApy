import numpy as np
import matplotlib.pyplot as plt
import MatsubaraFrequencies as mf
import BrillouinZone as bz
import Hr as hamr
import Hk as hamk
import TwoPoint as tp
import h5py

beta = 50
n = 0.85
u = 2
niv_max = 13000
g0mand = np.loadtxt('./g0mand')
vn = g0mand[:,0][:niv_max]
g0mand = g0mand[:,1] + 1j * g0mand[:,2]
g0mand = g0mand[:niv_max]
gm_wim = np.loadtxt('./gm_wim')
gm_wim = gm_wim[:,1] + 1j * gm_wim[:,2]
gm_wim = gm_wim[:niv_max]
mu = g0mand[-1].real
siw = g0mand - 1/gm_wim


# hr = hamr.one_band_2d_t_tp_tpp(0.25,-0.25*0.25,0.12*0.25)
hr = hamr.one_band_2d_t_tp_tpp(0.25,-0.25*0.25,0.12*0.25)
nk = (42,42,1)
k_grid = bz.KGrid(nk,symmetries=bz.two_dimensional_square_symmetries())
hk = hamk.ek_3d(k_grid.grid,hr=hr)

sigma = tp.SelfEnergy(siw[None,None,None,:],beta=beta,err=1e-6)
smom0 = tp.get_smom0(u,n)
smom1 = tp.get_smom1(u,n)

giwk = tp.GreensFunction(sigma=sigma,ek=hk,mu=mu)#n=0.85)
# giwk = tp.GreensFunction(sigma=sigma,ek=hk,mu=mu)
g_loc = giwk.k_mean()

path = './'
f = h5py.File(path + '1p-data.hdf5', 'w')
f['/dmft-last/ineq-001/giw/value'] = g_loc[None, None, :] * np.ones((1, 2, 1))
f['/dmft-last/ineq-001/siw/value'] = sigma.sigma_core[0,0,0,:][None, None, :] * np.ones((1, 2, 1))
f.create_group('/.config')
f['.config'].attrs['general.beta'] = beta
f['.config'].attrs['atoms.1.udd'] = u
f['.config'].attrs['general.totdens'] = n
f['dmft-last/mu/value'] = mu
f.close()

print('--------')
print(f'Hartree: {smom0}')
print(f'Hartree-fit: {sigma.smom0}')
print(f'Smom1: {smom1}')
print(f'Smom1-fit: {sigma.smom1}')
print('--------')

do_plot_1=False
if(do_plot_1):
    fig, ax = plt.subplots(1,2,figsize=(6,3),dpi=500)
    ax[0].plot(giwk.v_core,g_loc.real,'-o',color='firebrick',ms=5,markeredgecolor='k',alpha=0.8)
    ax[0].plot(vn,gm_wim.real,'-h',color='cornflowerblue',ms=4,markeredgecolor='k',alpha=0.8)
    ax[1].plot(giwk.v_core,g_loc.imag,'-o',color='firebrick',ms=5,markeredgecolor='k',alpha=0.8)
    ax[1].plot(vn,gm_wim.imag,'-h',color='cornflowerblue',ms=4,markeredgecolor='k',alpha=0.8)
    ax[0].set_xlim(0,3)
    ax[1].set_xlim(0,3)
    plt.tight_layout()
    plt.savefig('./G_loc_sanity_check.png')
    plt.show()


#%%
# Vertex:
from mpl_toolkits.axes_grid1 import make_axes_locatable
def add_colorbar(im,ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

Nv = 60
v = mf.vn(Nv)
iw = 59
chi = np.loadtxt('./chi_dir/' + 'chi{:03}'.format(iw))
chid = chi[:, 7].reshape((2 * Nv, 2 * Nv)) + 1j * chi[:, 8].reshape((2 * Nv, 2 * Nv)) + chi[:, 9].reshape(
    (2 * Nv, 2 * Nv)) + 1j * chi[:, 10].reshape((2 * Nv, 2 * Nv))
chim = chi[:, 7].reshape((2 * Nv, 2 * Nv)) + 1j * chi[:, 8].reshape((2 * Nv, 2 * Nv)) - chi[:, 9].reshape(
    (2 * Nv, 2 * Nv)) - 1j * chi[:, 10].reshape((2 * Nv, 2 * Nv))

nv_plot = 10
fig, ax = plt.subplots(1,2,figsize=(7,3),dpi=500)
im = ax[0].pcolormesh(v,v,chid.real,cmap='RdBu')
add_colorbar(im,ax[0])
im = ax[1].pcolormesh(v,v,chim.real,cmap='RdBu')
add_colorbar(im,ax[1])
ax[0].set_xlim(-nv_plot,nv_plot-1)
ax[0].set_ylim(-nv_plot,nv_plot-1)
ax[1].set_xlim(-nv_plot,nv_plot-1)
ax[1].set_ylim(-nv_plot,nv_plot-1)
plt.tight_layout()
plt.savefig(f'./G_loc_sanity_check_w_{iw-59}.png')
plt.show()



# fig, ax = plt.subplots(1,2)
# ax[0].plot(vn,siw.real,'-',color='firebrick')
# ax[1].plot(vn,siw.imag,'-',color='cornflowerblue')
# plt.show()


