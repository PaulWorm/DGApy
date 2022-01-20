# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Performs the analytical continuation on a Lambda-Dga output
# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import sys, os
import BrillouinZone as bz
import AnalyticContinuation as a_cont
import matplotlib.pyplot as plt
import PadeAux as pade_aux
from Plotting import MidpointNormalize
import Hk as hamk
import Output as output

# Define input path:
#input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.2_tpp0.1_beta50_n0.95/LambdaDga_lc_sp_Nk19600_Nq19600_core60_invbse60_vurange150_wurange60/'
#input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/Testset1/LambdaDga_Python/LambdaDga_lc_sp_Nk6400_Nq6400_core59_invbse100_vurange100_wurange99/'
#input_path = '/mnt/c/users/pworm/Research/U2BenchmarkData/BenchmarkSchaefer_beta_15/LambdaDgaPython/LambdaDga_lc_sp_Nk4096_Nq4096_core27_invbse80_vurange80_wurange80/'
input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.875/LambdaDga_lc_sp_Nk10000_Nq10000_core59_invbse60_vurange500_wurange59/'

output_path = input_path


output_path = output.uniquify(output_path + 'AnaCont') + '/'
os.mkdir(output_path)

# Load config:
config = np.load(input_path + 'config.npy', allow_pickle=True).item()
#k_grid = config['grids']['k_grid']
dmft1p = config['dmft1p']
hr = config['system']['hr']
niv_urange = config['box_sizes']['niv_urange']
niv_core = config['box_sizes']['niv_core']
nk = config['box_sizes']['nk']
t = 1.00
k_grid = bz.KGrid(nk=nk)
ek = hamk.ek_3d(kgrid=k_grid.grid, hr=hr)
k_grid.get_irrk_from_ek(ek=ek, dec=9)
# Load Green's function:
gk_dict = np.load(input_path + 'gk_dga.npy', allow_pickle=True).item()
try:
    gk = gk_dict['gk']
except:
    gk = gk_dict.gk
qpd = np.real(1./gk)[:,:,:,niv_urange]
gr = gk.real[:,:,:,niv_urange]
ak_fs = -1./np.pi * gk.imag[:,:,:,niv_urange]
# Find location of Fermi-surface:

shift = 0
ind_node = (nk[0]//4+shift,nk[0]//4+shift,0)
ind_anti_node = (nk[0]//2-10,0,0)

ind3 = (nk[0]//4+1,nk[0]//4+1,0)
ind4 = (nk[0]//2-13,0,0)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(ak_fs, cmap='RdBu_r', extent=bz.get_extent(kgrid=k_grid))
ax[0].plot(k_grid.kx[ind_node[0]], k_grid.ky[ind_node[1]], 'o')
ax[0].plot(k_grid.kx[ind_anti_node[0]], k_grid.ky[ind_anti_node[1]], 'o')
ax[0].plot(k_grid.kx[ind3[0]], k_grid.ky[ind3[1]], 'o')
ax[0].plot(k_grid.kx[ind4[0]], k_grid.ky[ind4[1]], 'o')

norm = MidpointNormalize(midpoint=0, vmin=gr.min(), vmax=gr.max())
ax[1].imshow(gr, cmap='RdBu', extent=bz.get_extent(kgrid=k_grid))
ax[1].plot(k_grid.kx[ind_node[0]], k_grid.ky[ind_node[1]], 'o')
ax[1].plot(k_grid.kx[ind_anti_node[0]], k_grid.ky[ind_anti_node[1]], 'o')
ax[1].plot(k_grid.kx[ind3[0]], k_grid.ky[ind3[1]], 'o')
ax[1].plot(k_grid.kx[ind4[0]], k_grid.ky[ind4[1]], 'o')
plt.tight_layout()
plt.savefig(output_path + 'location_of_special_points.png')
plt.show()


ind_cont = [ind_node,ind_anti_node, ind3, ind4]



# # Pade approximation of the Green's function:
nw = 1501
wmax = 15*t
v_real = a_cont.v_real_tan(wmax=wmax, nw=nw)
n_pade = 10#np.min((niv_urange, 10))
delta = 0.  # 1./dmft1p['beta']
n_fit = 30
gk_arc_thiele = a_cont.do_pade_on_ind(mat=gk, ind_list=ind_cont, v_real=v_real, beta=dmft1p['beta'],
                                      method='thiele', n_fit=n_fit, n_pade=n_pade, delta=delta)


gk_arc_thiele = a_cont.do_pade_on_ind(mat=gk.reshape((k_grid.nk_tot,-1)), ind_list=k_grid.irrk_ind, v_real=v_real, beta=dmft1p['beta'],
                                      method='thiele', n_fit=n_fit, n_pade=n_pade, delta=delta)

v_real_max_ent = a_cont.v_real_tan(wmax=wmax, nw=501)
gk_arc_max_ent = a_cont.do_max_ent_on_ind(mat=gk, ind_list=ind_cont, v_real=v_real_max_ent, beta=dmft1p['beta'],
                                      n_fit=60,err=1e-2)

# gk_arc_max_ent_fbz = a_cont.do_max_ent_on_ind(mat=gk.reshape((k_grid.nk_tot,-1)), ind_list=k_grid.irrk_ind, v_real=v_real_max_ent, beta=dmft1p['beta'],
#                                       n_fit=60,err=1e-2)

gk_arc_thiele_fbz = k_grid.irrk2fbz(mat=gk_arc_thiele.T)

plt.figure()
for i, i_arc in enumerate(ind_cont):
    plt.plot(v_real, -1. / np.pi * gk_arc_thiele[:, i].imag, label='{}'.format(i_arc))
plt.legend()
plt.xlim(-t, t)
plt.ylim([0, 1])
plt.savefig(output_path + 'aw_arc_thiele.png')
plt.close()

plt.figure()
for i, i_arc in enumerate(ind_cont):
    plt.plot(v_real_max_ent, -1. / np.pi * gk_arc_max_ent[:, i].imag, label='{}'.format(i_arc))
plt.legend()
plt.xlim(-t, t)
plt.ylim([0, 1])
plt.savefig(output_path + 'aw_arc_max_ent.png')
plt.close()

plt.figure()
for i, i_arc in enumerate(ind_cont):
    plt.plot(v_real, gk_arc_thiele[:, i].real, label='{}'.format(i_arc))
plt.legend()
plt.xlim(-t, t)
#plt.ylim([0, np.max(-1. / np.pi * gk_arc_thiele.imag)])
plt.savefig(output_path + 'g_real_arc_thiele.png')
plt.close()

#
# awk_fs = bz.shift_mat_by_pi(mat=-1./np.pi * gk_arc_thiele_fbz[:,:,0,nw//2].imag, nk=k_grid.nk)
# plt.figure()
# plt.imshow(awk_fs, cmap='terrain', extent=bz.get_extent_pi_shift(kgrid=k_grid))#, vmin=0.05)
# plt.colorbar()
# plt.savefig(output_path + 'aw_cont_fs.png')
# plt.show()

wm1 = np.argwhere(np.isclose(v_real-0.1,0., atol=1e-06))
awk_fs = bz.shift_mat_by_pi(mat=-1./np.pi * gk_arc_thiele_fbz[:,:,0,nw//2-10].imag, nk=k_grid.nk)
plt.figure()
plt.imshow(awk_fs, cmap='terrain', extent=bz.get_extent_pi_shift(kgrid=k_grid), vmin=0.02)
plt.colorbar()
plt.savefig(output_path + 'aw_cont_fs.png')
plt.show()

wm1 = np.argwhere(np.isclose(v_real-0.1,0., atol=1e-06))
awk_fs = bz.shift_mat_by_pi(mat=-1./np.pi * gk_arc_thiele_fbz[:,:,0,nw//2].imag, nk=k_grid.nk)
plt.figure()
plt.imshow(awk_fs, cmap='terrain', extent=bz.get_extent_pi_shift(kgrid=k_grid), vmin=0.02)
plt.colorbar()
plt.savefig(output_path + 'aw_cont_fs_max_ent.png')
plt.show()


plt.figure()
plt.plot(v_real, -1./np.pi * gk_arc_thiele_fbz.imag.mean(axis=(0,1,2)))
plt.savefig(output_path + 'aw_cont_dga.png')
plt.show()







