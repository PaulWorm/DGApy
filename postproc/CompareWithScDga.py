import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
import BrillouinZone as bz
import TwoPoint_old as twop

# Load SC-DGA data:
path_sc = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta30_n0.95/SCDGA/'
data = h5py.File(path_sc+'1p-data_sc.hdf5')
siwk_sc = data['dmft-last/ineq-001/skiw/value'][()][0,0]
data2P = h5py.File(path_sc+'adga_sc.hdf5')
chi_sc_magn = data2P['/susceptibility/nonloc/magn'][()][0,0]
chi_sc_dens = data2P['/susceptibility/nonloc/dens'][()][0,0]


#ädata.close()

# Load lambda-corrected data:
#input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.95/KonvergenceAnalysis/LambdaDga_lc_sp_Nk576_Nq576_core40_invbse40_vurange40_wurange40/'
input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta30_n0.95/LambdaDga_lc_sp_Nk10000_Nq10000_core60_invbse60_vurange500_wurange60_vsc4/'
siwk = np.load(input_path + 'sigma.npy', allow_pickle=True)

config = np.load(input_path + 'config.npy', allow_pickle=True).item()
chi = np.load(input_path + 'chi_lambda.npy', allow_pickle=True).item()
beta = config.sys.beta
niw = 500
nk = 100
nk_sc = 80

k_grid2 = bz.KGrid(nk=(nk_sc,nk_sc,1))
giwk_sc_gen = twop.GreensFunctionGenerator(beta=beta,kgrid=k_grid2,hr=config.sys.hr,sigma=siwk_sc)
mu_sc = giwk_sc_gen.adjust_mu(n=config.sys.n,mu0=config.sys.mu_dmft)
giwk_sc = giwk_sc_gen.generate_gk(mu=mu_sc)

giwk_gen = twop.GreensFunctionGenerator(beta=beta,kgrid=config.k_grid,hr=config.sys.hr,sigma=siwk)
mu = giwk_gen.adjust_mu(n=config.sys.n,mu0=config.sys.mu_dmft)
giwk = giwk_gen.generate_gk(mu=mu)


siwk_sc = bz.shift_mat_by_pi(mat=siwk_sc,nk=(nk_sc,nk_sc,1))
siwk = bz.shift_mat_by_pi(mat=siwk,nk=(nk,nk,1))

giwk_sc = bz.shift_mat_by_pi(mat=giwk_sc.gk,nk=(nk_sc,nk_sc,1))
giwk = bz.shift_mat_by_pi(mat=giwk.gk,nk=(nk,nk,1))

fig, axes = plt.subplots(nrows=2,ncols=2)
axes = axes.flatten()
im=[]
im.append(axes[0].imshow(siwk_sc.real[:,:,0,120], cmap='RdBu', origin='lower'))
im.append(axes[1].imshow(siwk_sc.imag[:,:,0,120], cmap='RdBu', origin='lower'))
im.append(axes[2].imshow(siwk.real[:,:,0,niw], cmap='RdBu', origin='lower'))
im.append(axes[3].imshow(siwk.imag[:,:,0,niw], cmap='RdBu', origin='lower'))
for i,ax in enumerate(axes):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im[i], cax=cax, orientation='vertical')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2,ncols=2)
axes = axes.flatten()
im=[]
im.append(axes[0].imshow(giwk_sc.real[:,:,0,120], cmap='RdBu', origin='lower'))
im.append(axes[1].imshow(giwk_sc.imag[:,:,0,120], cmap='RdBu', origin='lower'))
im.append(axes[2].imshow(giwk.real[:,:,0,niw], cmap='RdBu', origin='lower'))
im.append(axes[3].imshow(giwk.imag[:,:,0,niw], cmap='RdBu', origin='lower'))
for i,ax in enumerate(axes):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im[i], cax=cax, orientation='vertical')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2,ncols=2)
axes = axes.flatten()
im=[]
im.append(axes[0].imshow(chi_sc_magn.real[:,:,0,60], cmap='RdBu', origin='lower'))
im.append(axes[1].imshow(chi_sc_dens.real[:,:,0,60], cmap='RdBu', origin='lower'))
im.append(axes[2].imshow(chi['magn'].mat.real[:,:,0,60], cmap='RdBu', origin='lower'))
im.append(axes[3].imshow(chi['dens'].mat.real[:,:,0,60], cmap='RdBu', origin='lower'))
for i,ax in enumerate(axes):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im[i], cax=cax, orientation='vertical')
plt.tight_layout()
plt.show()