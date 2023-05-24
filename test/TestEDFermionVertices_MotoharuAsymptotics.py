import copy

import numpy as np
import matplotlib.pyplot as plt
import MatsubaraFrequencies as mf
import Bubble as bub
import LocalFourPoint as lfp
import h5py
import TwoPoint as twop
import Hk as hamk
import Hr as hamr
import BrillouinZone as bz
import w2dyn_aux_dga as w2dyn_aux
import PlotSpecs as ps
import sys,os

# Load the ED data:
path = './2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
pdir = path +'PlotsMotoharuAsymptotics/'
if(not os.path.exists(pdir)):
    os.mkdir(pdir)
# Define the frequency grid range:
niw_core = 15
niv_core = 15
niv_shell = 2*niv_core
niv_full = niv_core + niv_shell

# Load the single-particle quantities:
n_bath = 3
fname = f'EDFermion_1p-data.hdf5'
file = h5py.File(path + fname, 'r')
beta = file['config/beta'][()]
mu_dmft = file['dmft/mu'][()]
giw_input = file['giw'][()]
siw_input = file['siw_hat'][()] # important to use the siw_hat since this is the one fitted in ED!!!!
giw_bm = file['dmft/giw'][()]
siw_bm = file['dmft/siw'][()]
vn_bm = mf.vn(np.size(giw_bm) // 2)
u = file['config/U'][()]
n = file['config/totdens'][()]
vn_input = mf.vn(np.size(giw_input) // 2)

# Load the physical susceptibilities:
fname_chi = 'EDFermion_chi.hdf5'
file_chi = h5py.File(path + fname_chi, 'r')
chi_dens_input = file_chi['chi_dens'][()]
chi_magn_input= file_chi['chi_magn'][()]
niw_chi_input = chi_dens_input.size//2
wn_chi_input = mf.wn(np.size(chi_dens_input) // 2)

# Load the two-particle quantities:
fname_g4iw = 'EDFermion_g4iw_sym.hdf5'
file_g4iw = h5py.File(path + fname_g4iw, 'r')


g2_dens = lfp.LocalFourPoint(matrix=file_g4iw['g4iw_dens'][()], beta=beta, wn=None, channel='dens')
g2_magn = lfp.LocalFourPoint(matrix=file_g4iw['g4iw_magn'][()], beta=beta, wn=None, channel='magn')

g2_dens.cut_iw(niw_core)
g2_magn.cut_iw(niw_core)

g2_dens.cut_iv(niv_core)
g2_magn.cut_iv(niv_core)
niv_g2 = g2_dens.niv
niw_g2 = niw_core
wn_g2 = mf.wn(niw_g2)

# Build the Green's function:
hr = hamr.one_band_2d_t_tp_tpp(1, -0.2, 0.1)
nk = (8, 8, 1)
nq = nk


k_grid = bz.KGrid(nk=nk)
ek = hamk.ek_3d(k_grid.grid, hr)
sigma = twop.SelfEnergy(siw_input[None, None, None, :], beta)
green = twop.GreensFunction(sigma, ek, mu=mu_dmft,niv_asympt=4000)
n_giw = green.n
n_test = (1 / beta * np.sum(green.g_loc.real) + 0.5) * 2

print('------------------------')
print('n_giw = ', n_giw)
print('n_test = ', n_test)
print('n_input = ', n)
print('------------------------')

#%%
# Check the quality of the self-energy and the Green's function:
import PlotSpecs as ps
fig, axes = plt.subplots(ncols=2,nrows=2, figsize=(7,5), dpi=500)
axes = axes.flatten()

axes[0].plot(green.vn, green.g_loc.real, label='Build')
axes[0].plot(vn_input, giw_input.real, label='Input')
axes[0].plot(vn_bm, giw_bm.real, label='BM')
axes[0].set_ylabel('$\Re G(i\omega_n)$')

axes[1].plot(green.vn, green.g_loc.imag, label='Build')
axes[1].plot(vn_input, giw_input.imag, label='Input')
axes[1].plot(vn_bm, giw_bm.imag, label='BM')
axes[1].set_ylabel('$\Im G(i\omega_n)$')

axes[2].plot(mf.vn(green.sigma.niv), green.sigma.get_siw()[0,0,0,:].real, label='Build')
axes[2].plot(vn_input, siw_input.real, label='Input')
axes[2].plot(vn_bm, siw_bm.real, label='BM')
axes[2].set_ylabel('$\Re \Sigma(i\omega_n)$')

axes[3].plot(mf.vn(green.sigma.niv), green.sigma.get_siw()[0,0,0,:].imag, label='Build')
axes[3].plot(vn_input, siw_input.imag, label='Input')
axes[3].plot(vn_bm, siw_bm.imag, label='BM')
axes[3].set_ylabel('$\Im \Sigma(i\omega_n)$')

for ax in axes:
    ax.set_xlabel(r'$i\nu_n$')
    ax.set_xlim(0,5+2*beta)

    ax.vlines(0, ax.get_ylim()[0], ax.get_ylim()[1], linestyle='--', color='grey')

axes[1].set_ylim(None,0)
axes[3].set_ylim(None,0)
plt.legend()
plt.tight_layout()
plt.savefig(pdir+f'Check_Giw_and_Siw_niv{np.size(vn_input)//2}_nbath_{n_bath}.png')
plt.show()

#%% Plot the two-particle Green's function and vertices:

g2_dens.plot(0, pdir=pdir, name='G2_dens')
g2_dens.plot(min(niw_g2//2,10), pdir=pdir, name=f'G2_dens')
g2_dens.plot(min(niw_g2,40), pdir=pdir, name=f'G2_dens')

g2_magn.plot(0, pdir=pdir, name='G2_magn')
g2_magn.plot(min(niw_g2//2,10), pdir=pdir, name=f'G2_magn')
g2_magn.plot(min(niw_g2,40), pdir=pdir, name=f'G2_magn')

# Compute the vertex:
niv_core = g2_dens.niv
gchi_dens = lfp.gchir_from_g2(g2_dens, green.g_loc)
gchi_magn = lfp.gchir_from_g2(g2_magn, green.g_loc)
gchi_dens.plot(0,pdir=pdir,name='Gchi_dens')
gchi_magn.plot(0,pdir=pdir,name='Gchi_magn')
gchi_magn.plot(niw_core,pdir=pdir,name='Gchi_magn')




gchi0_gen = bub.LocalBubble(wn=wn_g2, giw=green)
gchi0_core = gchi0_gen.get_gchi0(niv_core)
gchi0_urange = gchi0_gen.get_gchi0(niv_full)
chi0_core = gchi0_gen.get_chi0(niv_core)
chi0_urange = gchi0_gen.get_chi0(niv_full)
chi0_shell = gchi0_gen.get_chi0_shell(niv_full,2*niv_full)
chi0_asympt = chi0_urange+chi0_shell

F_dens = lfp.Fob2_from_chir(gchi_dens, gchi0_core)
F_magn = lfp.Fob2_from_chir(gchi_magn, gchi0_core)

F_dens.plot(0, pdir=pdir, name='F_dens')
F_magn.plot(0, pdir=pdir, name='F_magn')

F_dens.plot(0, pdir=pdir, name='F_dens', niv=30)
F_magn.plot(0, pdir=pdir, name='F_magn', niv=30)

F_dens.plot(-10, pdir=pdir, name='F_dens')
F_magn.plot(-10, pdir=pdir, name='F_magn')

#%%
# Compute the Physical susceptibilities:

gamma_dens = lfp.gammar_from_gchir(gchi_dens,gchi0_urange,u)
gamma_magn = lfp.gammar_from_gchir(gchi_magn,gchi0_urange,u)

gamma_dens.plot(0,pdir=pdir,name='Gamma_dens')
gamma_magn.plot(0,pdir=pdir,name='Gamma_magn')
#
# #%%
# plt.figure()
# plt.imshow(gamma_magn.mat[niw_core].real,cmap='RdBu')
# plt.colorbar()
# plt.show()
# #%%

chi_dens_core = gchi_dens.contract_legs()
chi_magn_core = gchi_magn.contract_legs()

# gchi_aux_core_dens = lfp.gchi_aux_core(gchi_dens,u)
# gchi_aux_core_magn = lfp.gchi_aux_core(gchi_magn,u)

gchi_aux_core_dens = lfp.gchi_aux_core_from_gammar(gamma_dens,gchi0_core,u)
gchi_aux_core_magn = lfp.gchi_aux_core_from_gammar(gamma_magn,gchi0_core,u)

gchi_aux_core_magn.plot(0,pdir=pdir,name='Chi_aux_magn')
gchi_aux_core_dens.plot(0,pdir=pdir,name='Chi_aux_dens')

# gchi_aux_core_dens_v2 = lfp.local_gchi_aux_from_gammar(gamma_dens,gchi0_core,u)
# gchi_aux_core_magn_v2 = lfp.local_gchi_aux_from_gammar(gamma_magn,gchi0_core,u)

# gchi_aux_core_magn_v2.plot(0,pdir=pdir,name='Chi_aux_magn_v2')
# gchi_aux_core_dens_v2.plot(0,pdir=pdir,name='Chi_aux_dens_v2')

chi_aux_core_dens = gchi_aux_core_dens.contract_legs()
chi_aux_core_magn = gchi_aux_core_magn.contract_legs()

chi_dens_urange = lfp.chi_phys_urange(chi_aux_core_dens,chi0_core,chi0_urange,u,'dens')
chi_magn_urange = lfp.chi_phys_urange(chi_aux_core_magn,chi0_core,chi0_urange,u,'magn')

chi_dens = lfp.chi_phys_asympt(chi_dens_urange,chi0_urange,chi0_asympt)
chi_magn = lfp.chi_phys_asympt(chi_magn_urange,chi0_urange,chi0_asympt)


#%%
tmp = 1. / (1. / (chi_aux_core_magn + 0*chi0_urange - 0*chi0_core) -u)
print(tmp[niw_core])

#%%
fig, axes = plt.subplots(ncols=2,nrows=3, figsize=(8,9), dpi=500)
axes = axes.flatten()

axes[0].plot(mf.wn(len(chi_dens_core)//2), chi_dens_core.real, label='Core')
axes[0].plot(mf.wn(len(chi_dens_urange)//2), chi_dens_urange.real, label='Urange')
axes[0].plot(mf.wn(len(chi_dens)//2), chi_dens.real, label='Tilde')
axes[0].plot(mf.wn(len(chi_dens_input)//2), chi_dens_input.real, label='Input')
axes[0].set_ylabel('$\Re \chi(i\omega_n)_{dens}$')
axes[0].legend()

axes[1].plot(mf.wn(len(chi_magn_core)//2), chi_magn_core.real, label='Core')
axes[1].plot(mf.wn(len(chi_magn_urange)//2), chi_magn_urange.real, label='Urange')
axes[1].plot(mf.wn(len(chi_magn)//2), chi_magn.real, label='Tilde')
axes[1].plot(mf.wn(len(chi_magn_input)//2), chi_magn_input.real, label='Input')
axes[1].set_ylabel('$\Re \chi(i\omega_n)_{magn}$')
axes[1].legend()

axes[2].loglog(mf.wn(len(chi_dens_core)//2), chi_dens_core.real, label='Core',ms=0)
axes[2].loglog(mf.wn(len(chi_dens_urange)//2), chi_dens_urange.real, label='Urange',ms=0)
axes[2].loglog(mf.wn(len(chi_dens)//2), chi_dens.real, label='Tilde',ms=0)
axes[2].loglog(mf.wn(len(chi_dens_input)//2), chi_dens_input.real, label='Input',ms=0)
axes[2].loglog(mf.wn(niw_chi_input), np.real(1 / (mf.iw(beta,niw_chi_input)+0.000001) ** 2 * green.e_kin) * 2,ls='--', label='Asympt',ms=0)
axes[2].set_ylabel('$\Re \chi(i\omega_n)_{dens}$')
axes[2].legend()

axes[3].loglog(mf.wn(len(chi_magn_core)//2), chi_magn_core.real, label='Core',ms=0)
axes[3].loglog(mf.wn(len(chi_magn_urange)//2), chi_magn_urange.real, label='Urange')
axes[3].loglog(mf.wn(len(chi_magn)//2), chi_magn.real, label='Tilde',ms=0)
axes[3].loglog(mf.wn(len(chi_magn_input)//2), chi_magn_input.real, label='Input',ms=0)
axes[3].loglog(mf.wn(niw_chi_input), np.real(1 / (mf.iw(beta,niw_chi_input)+0.000001) ** 2 * green.e_kin) * 2,'--', label='Asympt',ms=0)
axes[3].set_ylabel('$\Re \chi(i\omega_n)_{magn}$')
axes[3].legend()

axes[4].loglog(1+mf.wn(len(chi_dens_core)//2), np.abs(chi_dens.real-mf.cut_w(chi_dens_input,niw_core)), label='Diff()',ms=0)
axes[4].set_ylabel('$Diff \Re \chi(i\omega_n)_{dens}$')
axes[4].legend()

axes[5].loglog(1+mf.wn(len(chi_magn_core)//2), np.abs(chi_magn.real-mf.cut_w(chi_magn_input,niw_core)), label='Diff()',ms=0)
axes[5].set_ylabel('$Diff \Re \chi(i\omega_n)_{magn}$')
axes[5].legend()

axes[0].set_xlim(-1,10)
axes[1].set_xlim(-1,10)
plt.tight_layout()
plt.savefig(pdir+f'chi_dens_magn_nbath{n_bath}.png')
plt.show()

#%% Get the vrg vertices:

# vrg_dens = lfp.vrg_from_gchi_aux(gchi_aux_core_dens,gchi0_core,chi_dens_urange,chi_dens,u)
# vrg_magn = lfp.vrg_from_gchi_aux(gchi_aux_core_magn,gchi0_core,chi_magn_urange,chi_magn,u)

vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir_uasympt(gamma_dens, gchi0_gen, u, niv_shell=niv_shell)
vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir_uasympt(gamma_magn, gchi0_gen, u, niv_shell=niv_shell)


#%% Check the Schwinger-Dyson equation:
# siw_sde_full = lfp.schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, green.g_loc, u, n, niv_shell=niv_shell)
siw_sde_full = lfp.schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, green.g_loc, u, n, niv_shell=niv_shell)
vn_full = mf.vn(niv_full)


fig, axes = plt.subplots(ncols=2,nrows=2, figsize=(8,5), dpi=500)
axes = axes.flatten()

axes[0].plot(vn_full, siw_sde_full.real, '-o', color='cornflowerblue', markeredgecolor='k', label='SDE', lw=4)
axes[0].plot(vn_full, green.sigma.get_siw(niv=niv_full)[0, 0, 0, :].real, '-p', color='firebrick', label='Input', markeredgecolor='k', ms=2)
axes[0].set_ylabel(r'$\Re \Sigma(i\nu_n)$')
axes[0].set_xlabel(r'$\nu_n$')

axes[1].plot(vn_full, siw_sde_full.imag, '-o', color='cornflowerblue', markeredgecolor='k', label='SDE', lw=4)
axes[1].plot(vn_full, green.sigma.get_siw(niv=niv_full)[0, 0, 0, :].imag, '-p', color='firebrick', label='Input', markeredgecolor='k', ms=2)
axes[1].set_ylabel(r'$\Im \Sigma(i\nu_n)$')
axes[1].set_xlabel(r'$\nu_n$')


axes[2].loglog(vn_full, siw_sde_full.real, '-o', color='cornflowerblue', markeredgecolor='k', label='SDE', lw=4)
axes[2].loglog(vn_full, green.sigma.get_siw(niv=niv_full)[0, 0, 0, :].real, '-p', color='firebrick', label='Input', markeredgecolor='k', ms=2)
axes[2].set_ylabel(r'$\Re \Sigma(i\nu_n)$')
axes[2].set_xlabel(r'$\nu_n$')

axes[3].loglog(vn_full, np.abs(siw_sde_full.imag), '-o', color='cornflowerblue', markeredgecolor='k', label='SDE', lw=4)
axes[3].loglog(vn_full, np.abs(green.sigma.get_siw(niv=niv_full)[0, 0, 0, :].imag), '-p', color='firebrick', label='Input', markeredgecolor='k', ms=2)
axes[3].set_ylabel(r'$|\Im \Sigma(i\nu_n)|$')
axes[3].set_xlabel(r'$\nu_n$')


axes[0].set_xlim(0,niv_full)
axes[1].set_xlim(0,niv_full)
plt.legend()
axes[1].set_ylim(None,0)
plt.tight_layout()
plt.savefig(pdir + f'sde_vs_input_nbath{n_bath}_niv_{niv_core}.png')
plt.show()

#
#%% Test non-local part:
symmetries = bz.two_dimensional_square_symmetries()
# symmetries=None
q_grid = bz.KGrid(nk=nq,symmetries=symmetries)
q_list =  q_grid.irrk_mesh_ind.T
# q_list_full = np.array([q_grid.kmesh_ind[i].flatten() for i in range(3)]).T
chi0_q = gchi0_gen.get_chi0_q_list(niv_core,q_list)
chi0_q_urange = gchi0_gen.get_chi0_q_list(niv_full,q_list)
# chi0_q_list_full = gchi0_gen.get_chi0_q_list(niv_core,q_list)
chi0_q_shell = gchi0_gen.get_chi0q_shell(chi0_q_urange,niv_full,niv_full*2,q_list)

#%%
chi0_q = q_grid.map_irrk2fbz(chi0_q)
chi0_q_urange = q_grid.map_irrk2fbz(chi0_q_urange)
chi0_q_full = chi0_q_urange + q_grid.map_irrk2fbz(chi0_q_shell)
# chi0_q = chi0_q
# chi0_q_full = chi0_q + chi0_q_shell
#%%
chi0_sum_core = np.mean(chi0_q,axis=(0,1,2))
chi0_sum_tilde = np.mean(chi0_q_full,axis=(0,1,2))
# chi0_sum_core = np.mean(chi0_q,axis=(0,))
# chi0_sum_tilde = np.mean(chi0_q_full,axis=(0,))
wn_core = gchi0_gen.wn

fig, axes = plt.subplots(ncols=2,nrows=2, figsize=(8,5), dpi=500)
axes = axes.flatten()

axes[0].plot(wn_core,chi0_sum_core.real,label='sum')
axes[0].plot(wn_core,chi0_sum_tilde.real,label='sum-tilde')
axes[0].plot(wn_core,chi0_core.real,label='loc-core')
axes[0].plot(wn_core,(chi0_asympt).real,label='loc-tilde')
axes[0].legend()

axes[1].loglog(wn_core,chi0_sum_core.real,label='sum')
axes[1].loglog(wn_core,chi0_sum_tilde.real,label='sum-tilde')
axes[1].loglog(wn_core,chi0_core.real,label='loc-core')
axes[1].loglog(wn_core,(chi0_asympt).real,label='loc-tilde')

axes[3].loglog(wn_core,np.abs(chi0_sum_core.real-chi0_core.real),label='diff-core')
axes[3].loglog(wn_core,np.abs(chi0_sum_tilde.real-chi0_asympt.real),label='diff-tilde')
plt.legend()

plt.show()


#%%
q_list_full = np.array([q_grid.kmesh_ind[i].flatten() for i in range(3)]).T
f1,f2,f3 = gchi0_gen.get_asympt_prefactors_q(q_list_full)
f1l,f2l,f3l = gchi0_gen.get_asympt_prefactors()

print(np.mean(f3))
print(f3l)


# #%%
#
# plt.figure()
# plt.plot(wn_core,chi_aux_core_dens.real,'-o')
# plt.plot(wn_core,chi_aux_core_magn.real,'-o')
# plt.show()
#
#
# #%%
# plt.figure()
# plt.imshow(vrg_magn.mat.real,cmap='RdBu')
# plt.colorbar()
# plt.show()
#
# #%%
# plt.figure()
# plt.imshow(vrg_dens.mat.real,cmap='RdBu')
# plt.colorbar()
# plt.show()
#
#
# #%%
# plt.figure()
# plt.imshow(gchi_aux_core_magn.mat[niw_core].real,cmap='RdBu')
# plt.colorbar()
# plt.show()
#
# #%%
# plt.figure()
# plt.imshow(gchi_magn.mat[niw_core].real,cmap='RdBu')
# plt.colorbar()
# plt.show()
