import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import BrillouinZone as bz

# Load Data from Thomas Schäfer:
path1 =  '/mnt/d/Research/U2BenchmarkData/BenchmarkSchaefer_beta_15/ladder_spin_only/'

data1 = h5py.File(path1 + 'Siwk.hdf5')
sigma1 = data1['siwk'][()]
sigma1_d = data1['siwk_d'][()]
sigma1_m = data1['siwk_m'][()]
sigma1_rest = data1['siwk_rest'][()]
data1.close()

# chi1_ch = np.loadtxt(path1 + '/chich_omega/chi079')
# chi1_ch = chi1_ch[:,2]
sigma1_loc = np.loadtxt(path1 + '/klist/SELF_LOC_parallel')
niws = 80
#%%
# Load my Data:

path2 = '/mnt/d/Research/U2BenchmarkData/BenchmarkSchaefer_beta_15/LambdaDgaPython/LambdaDga_lc_sp_Nk10000_Nq10000_core79_invbse80_vurange80_wurange79/'

niw = 80
sigma = np.load(path2+'sigma.npy', allow_pickle=True)
sigma = bz.shift_mat_by_pi(mat=sigma,nk=(100,100,1))
sigma_dga = np.load(path2+'sigma_dga.npy', allow_pickle=True).item()
sigma_d = bz.shift_mat_by_pi(mat=sigma_dga['dens'],nk=(100,100,1))
sigma_m = bz.shift_mat_by_pi(mat=sigma_dga['magn'],nk=(100,100,1))
dmft_sde = np.load(path2+'dmft_sde.npy', allow_pickle=True).item()

vrg = np.load(path2+'/SpinFermion/spin_fermion_vertex.npy', allow_pickle=True).item()

#%%
fig, axes = plt.subplots(nrows=2,ncols=2)
axes = axes.flatten()
im=[]
im.append(axes[0].imshow(0.25+sigma1.real[0,:,:], cmap='RdBu', origin='lower'))
im.append(axes[1].imshow(sigma1.imag[0,:,:], cmap='RdBu', origin='lower'))
im.append(axes[2].imshow(sigma.real[:,:,0,niw], cmap='RdBu', origin='lower'))
im.append(axes[3].imshow(sigma.imag[:,:,0,niw], cmap='RdBu', origin='lower'))
for i,ax in enumerate(axes):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im[i], cax=cax, orientation='vertical')
plt.tight_layout()
plt.show()
#%%
fig, axes = plt.subplots(nrows=2,ncols=2)
axes = axes.flatten()
im=[]
im.append(axes[0].imshow(sigma1_d.real[0,:,:], cmap='RdBu', origin='lower'))
im.append(axes[1].imshow(sigma1_d.imag[0,:,:], cmap='RdBu', origin='lower'))
im.append(axes[2].imshow(-sigma_d.real[:,:,0,niw], cmap='RdBu', origin='lower'))
im.append(axes[3].imshow(-sigma_d.imag[:,:,0,niw], cmap='RdBu', origin='lower'))
for i,ax in enumerate(axes):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im[i], cax=cax, orientation='vertical')
plt.tight_layout()
plt.show()

#%%
fig, axes = plt.subplots(nrows=2,ncols=2)
axes = axes.flatten()
im=[]
im.append(axes[0].imshow(sigma1_m.real[0,:,:], cmap='RdBu', origin='lower'))
im.append(axes[1].imshow(sigma1_m.imag[0,:,:], cmap='RdBu', origin='lower'))
im.append(axes[2].imshow(sigma_m.real[:,:,0,niw], cmap='RdBu', origin='lower'))
im.append(axes[3].imshow(sigma_m.imag[:,:,0,niw], cmap='RdBu', origin='lower'))
for i,ax in enumerate(axes):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im[i], cax=cax, orientation='vertical')
plt.tight_layout()
plt.show()

#%%
plt.figure()
plt.plot(sigma1_loc[:,2],'-o', label='DMFT')
plt.plot(sigma1_loc[:,4],'-o', label='SDE-LOC-TS')
plt.plot(dmft_sde['siw'][80:].imag,'-o', label='SDE-LOC')
plt.plot(3*dmft_sde['magn'][80:].imag-dmft_sde['dens'][80:].imag,'-o', label='SDE-LOC2')
plt.legend()
plt.show()

