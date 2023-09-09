'''
       Simple post-processing plotscript to plot the magnon dispersion from the continued magnetic susceptibility.
'''
import numpy as np
import dga.brillouin_zone as bz
import matplotlib.pyplot as plt
import dga.plot_specs



#%% Load Data
path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.70/LDGA_spch_Nk19600_Nq19600_wc40_vc40_vs200' \
       '/MaxEntChiMagn/'
fname = 'chi_magn_cont_fbz.npy'

chi = np.load(path+fname, allow_pickle=True)
w = np.load(path+'w.npy', allow_pickle=True)

nk = np.shape(chi)[:-1]
bz_k_path = 'Gamma-X-M2-Gamma'
q_path = bz.KPath(nk=nk,path=bz_k_path)

chi_qpath = chi[q_path.ikx, q_path.iky, 0]

#%%
plt.figure()
plt.pcolormesh(q_path.k_axis,w,w[:,None]*chi_qpath.real.T,cmap='terrain',vmax=0.3,vmin=0)
# plt.xlabel('q')
plt.ylabel('$\omega$')
ax = plt.gca()
ax.set_xticks(q_path.x_ticks,q_path.labels)
plt.colorbar()
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(path + 'chi_cont_q_path.png')
plt.show()



