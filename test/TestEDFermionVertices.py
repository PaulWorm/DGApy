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

# Load the ED data:
path = './2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'

# Load the single-particle quantities:
n_bath = 3
fname = f'EDFermion_GreenFunctions_nbath_{n_bath}_niv_10000.hdf5'
file = h5py.File(path + fname, 'r')
beta = file['config/beta'][()]
mu_dmft = file['dmft/mu'][()]
giw_input = file['giw_hat'][()]
siw_input = file['siw_hat'][()] # important to use the siw_hat since this is the one fitted in ED!!!!
giw_bm = file['dmft/giw'][()]
siw_bm = file['dmft/siw'][()]
vn_bm = mf.vn(np.size(giw_bm) // 2)
u = file['config/U'][()]
n = file['config/totdens'][()]
vn_input = mf.vn(np.size(giw_input) // 2)

# Load the physical susceptibilities:
fname_chi = 'EDFermion_ChiPhys_nbath_3_niw_10000.hdf5'
file_chi = h5py.File(path + fname_chi, 'r')
chi_dens_input = file_chi['chi_dens'][()]
chi_magn_input= file_chi['chi_magn'][()]
niw_chi_input = chi_dens_input.size//2
wn_chi_input = mf.wn(np.size(chi_dens_input) // 2)

# Load the two-particle quantities:
fname_g4iw = 'EDFermion_GreenFunctions_nbath_3_niw_250_niv_249.hdf5'
file_g4iw = h5py.File(path + fname_g4iw, 'r')
niw_g2 = np.shape(file_g4iw['g4iw_dens'][()])[0]//2

wn_g2 = mf.wn(niw_g2)

g2_dens = lfp.LocalFourPoint(matrix=file_g4iw['g4iw_dens'][()], beta=beta, wn=wn_g2, channel='dens')
g2_magn = lfp.LocalFourPoint(matrix=file_g4iw['g4iw_magn'][()], beta=beta, wn=wn_g2, channel='magn')

niv_g2 = g2_dens.niv
niv_shell = 2*niv_g2
niv_full = niv_g2 + niv_shell
# Build the Green's function:
hr = hamr.one_band_2d_t_tp_tpp(1, -0.2, 0.1)
nk = (64, 64, 1)
k_grid = bz.KGrid(nk=nk)
ek = hamk.ek_3d(k_grid.grid, hr)
sigma = twop.SelfEnergy(siw_input[None, None, None, :], beta)
green = twop.GreensFunction(sigma, ek, mu=mu_dmft)
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
plt.savefig(path+f'Check_Giw_and_Siw_niv{np.size(vn_input)//2}_nbath_{n_bath}.png')
plt.show()

#%% Plot the two-particle Green's function and vertices:

g2_dens.plot(0, pdir=path, name='G2_dens')
g2_dens.plot(max(niw_g2//2,10), pdir=path, name=f'G2_dens')
g2_dens.plot(max(niw_g2,40), pdir=path, name=f'G2_dens')

g2_magn.plot(0, pdir=path, name='G2_magn')
g2_magn.plot(max(niw_g2//2,10), pdir=path, name=f'G2_magn')
g2_magn.plot(max(niw_g2,40), pdir=path, name=f'G2_magn')

# Compute the vertex:
niv_core = g2_dens.niv
gchi_dens = lfp.gchir_from_g2(g2_dens, green.g_loc)
gchi_magn = lfp.gchir_from_g2(g2_magn, green.g_loc)
gchi_dens.plot(0,pdir=path,name='Gchi_dens', niv=30)
gchi_magn.plot(0,pdir=path,name='Gchi_magn', niv=30)


gchi0_gen = bub.LocalBubble(wn=wn_g2, giw=green)
gchi0_core = gchi0_gen.get_gchi0(niv_core)
chi0_core = gchi0_gen.get_chi0(niv_core)
chi0_shell = gchi0_gen.get_asymptotic_correction(niv_core)

F_dens = lfp.Fob2_from_chir(gchi_dens, gchi0_core)
F_magn = lfp.Fob2_from_chir(gchi_magn, gchi0_core)

F_dens.plot(0, pdir=path, name='F_dens')
F_magn.plot(0, pdir=path, name='F_magn')

F_dens.plot(0, pdir=path, name='F_dens', niv=30)
F_magn.plot(0, pdir=path, name='F_magn', niv=30)

F_dens.plot(-10, pdir=path, name='F_dens')
F_magn.plot(-10, pdir=path, name='F_magn')

#%%
# Compute the Physical susceptibilities:

chi_dens_core = gchi_dens.contract_legs()
chi_magn_core = gchi_magn.contract_legs()


vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, gchi0_gen, u, niv_core=niv_core, niv_shell=niv_shell)
vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir(gchi_magn, gchi0_gen, u, niv_core=niv_core, niv_shell=niv_shell)


fig, axes = plt.subplots(ncols=2,nrows=2, figsize=(8,5), dpi=500)
axes = axes.flatten()

axes[0].plot(mf.wn(len(chi_dens_core)//2), chi_dens_core.real, label='Core')
axes[0].plot(mf.wn(len(chi_dens)//2), chi_dens.real, label='Tilde')
axes[0].plot(mf.wn(len(chi_dens_input)//2), chi_dens_input.real, label='Input')
axes[0].set_ylabel('$\Re \chi(i\omega_n)_{dens}$')
axes[0].legend()

axes[1].plot(mf.wn(len(chi_magn_core)//2), chi_magn_core.real, label='Core')
axes[1].plot(mf.wn(len(chi_magn)//2), chi_magn.real, label='Tilde')
axes[1].plot(mf.wn(len(chi_magn_input)//2), chi_magn_input.real, label='Input')
axes[1].set_ylabel('$\Re \chi(i\omega_n)_{magn}$')
axes[1].legend()

axes[2].loglog(mf.wn(len(chi_dens_core)//2), chi_dens_core.real, label='Core',ms=0)
axes[2].loglog(mf.wn(len(chi_dens)//2), chi_dens.real, label='Tilde',ms=0)
axes[2].loglog(mf.wn(len(chi_dens_input)//2), chi_dens_input.real, label='Input',ms=0)
axes[2].loglog(mf.wn(niw_chi_input), np.real(1 / (mf.iw(beta,niw_chi_input)+0.000001) ** 2 * green.e_kin) * 2,ls='--', label='Asympt',ms=0)
axes[2].set_ylabel('$\Re \chi(i\omega_n)_{dens}$')
axes[2].legend()

axes[3].loglog(mf.wn(len(chi_magn_core)//2), chi_magn_core.real, label='Core',ms=0)
axes[3].loglog(mf.wn(len(chi_magn)//2), chi_magn.real, label='Tilde',ms=0)
axes[3].loglog(mf.wn(len(chi_magn_input)//2), chi_magn_input.real, label='Input',ms=0)
axes[3].loglog(mf.wn(niw_chi_input), np.real(1 / (mf.iw(beta,niw_chi_input)+0.000001) ** 2 * green.e_kin) * 2,'--', label='Asympt',ms=0)
axes[3].set_ylabel('$\Re \chi(i\omega_n)_{magn}$')
axes[3].legend()

axes[0].set_xlim(-1,10)
axes[1].set_xlim(-1,10)
plt.tight_layout()
plt.savefig(path+f'chi_dens_magn_nbath{n_bath}.png')
plt.show()

#%% Check the Schwinger-Dyson equation:

siw_sde_full = lfp.schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, green.g_loc, u, n, niv_shell=niv_shell)
vn_full = mf.vn(niv_full)


plt.figure(dpi=500)
plt.plot(vn_full, siw_sde_full.imag, '-o', color='cornflowerblue', markeredgecolor='k', label='SDE', lw=4)
plt.plot(vn_full, green.sigma.get_siw(niv=niv_full)[0, 0, 0, :].imag, '-p', color='firebrick', label='Input', markeredgecolor='k', ms=2)
# plt.plot(vn_full, siw_shell_only.imag, '-h', color='forestgreen', label='Asympt', markeredgecolor='k', ms=1)
plt.legend()
plt.xlim(0, 100)
# plt.ylim(None,0)
plt.savefig(path + f'sde_vs_input_nbath{n_bath}_niv_{niv_core}.png')
plt.show()



#%%
gchi0_core = gchi0_gen.get_gchi0(niv_shell)
chi0_shell = gchi0_gen.get_chi0(niv_shell)
prefac_magn = 1 / (1 - u * chi0_shell)
prefac_dens = 1 / (1 + u * chi0_shell)
import copy

vrg_magn_old = copy.deepcopy(vrg_magn)
vrg_dens_old = copy.deepcopy(vrg_dens)

tmp = prefac_dens[:, None] * u ** 2 * np.sum(
    F_diag_dens[None, :, :niv_shell-niv_core  ] * gchi0_core[:, None, :niv_shell-niv_core  ]
    + F_diag_dens[None, :,niv_shell+niv_core  :] * gchi0_core[:, None,niv_shell +niv_core :],
    axis=-1)
tmp = mf.cut_v(tmp, niv_core, (-1,))
plt.figure()
plt.plot(tmp[niw_core//2,:].real,'-o')
plt.show()
vrg_dens.mat = 1 / (1 - u * chi_dens[:, None]) * (vrg_dens.mat * (1 - u * chi_dens[:, None]) + tmp)

tmp = prefac_magn[:, None] * u ** 2 * np.sum(
    F_diag_magn[None, :, :niv_shell - niv_core] * gchi0_core[:, None, :niv_shell - niv_core]
    + F_diag_magn[None, :,niv_shell + niv_core:] * gchi0_core[:, None,niv_shell + niv_core:],
    axis=-1)
tmp = mf.cut_v(tmp, niv_core, (-1,))
vrg_magn.mat = 1 / (1 + u * chi_magn[:, None]) * (vrg_magn.mat * (1 + u * chi_magn[:, None]) - tmp)
# %%
plt.figure()
plt.plot(vrg_magn.vn, vrg_magn.mat[30, :].imag, '-o', color='cornflowerblue', label='Tilde', markeredgecolor='k')
plt.plot(vrg_magn.vn, vrg_magn.mat[30, :].real - 1, '-o', color='firebrick', label='Tilde', markeredgecolor='k')
plt.plot(vrg_magn.vn, vrg_magn_old.mat[30, :].real - 1, '-o', color='goldenrod', label='Core', markeredgecolor='k')
plt.plot(vrg_magn.vn, vrg_magn_old.mat[30, :].imag, '-o', color='seagreen', label='Core', markeredgecolor='k')
plt.legend()
plt.xlim(0, 100)
plt.show()

# %%
siw_dens = lfp.schwinger_dyson_vrg(vrg_dens, chi_dens, green.g_loc, u)
siw_magn = lfp.schwinger_dyson_vrg(vrg_magn, chi_magn, green.g_loc, u)

n_shell = 800
niv_full = n_shell + niv_core
siw_sde = siw_dens + siw_magn
siw_shell_dens = lfp.schwinger_dyson_shell(chi_dens, green.g_loc, beta, u, n_shell=n_shell, n_core=niv_core, wn=g2_dens.wn)
siw_shell_magn = lfp.schwinger_dyson_shell(chi_magn, green.g_loc, beta, u, n_shell=n_shell, n_core=niv_core, wn=g2_dens.wn)
siw_shell = siw_shell_dens + siw_shell_magn

siw_sde_full = mf.concatenate_core_asmypt(siw_sde, siw_shell)
vn_full = mf.vn(niv_full)

siw_shell_dens_only = lfp.schwinger_dyson_shell(chi_dens, green.g_loc, beta, u, n_shell=niv_full, n_core=0, wn=g2_dens.wn)
siw_shell_magn_only = lfp.schwinger_dyson_shell(chi_magn, green.g_loc, beta, u, n_shell=niv_full, n_core=0, wn=g2_dens.wn)
siw_shell_only = siw_shell_dens_only + siw_shell_magn_only

plt.figure(dpi=500)
plt.plot(vn_full, siw_sde_full.imag, '-o', color='cornflowerblue', markeredgecolor='k', label='SDE', lw=4)
plt.plot(vn_full, green.sigma.get_siw(niv=niv_full)[0, 0, 0, :].imag, '-p', color='firebrick', label='Input', markeredgecolor='k', ms=2)
plt.plot(vn_full, siw_shell_only.imag, '-h', color='forestgreen', label='Asympt', markeredgecolor='k', ms=1)
plt.legend()
plt.xlim(0, 60)
# plt.ylim(None,0)
plt.savefig(path + f'sde_vs_input_niv_{niv_core}.png')
plt.show()

# %%
plt.figure()
plt.plot(g2_dens.wn, chi_magn.real, '-o', color='cornflowerblue', label='Tilde')
plt.plot(g2_dens.wn, chi_magn_core.real, '-p', color='firebrick', label='Core')
plt.plot(wn_chi_ed, chi_magn_ed.real, '-p', color='goldenrod', label='ED')
plt.legend()
plt.savefig(path + 'chi_magn.png')
plt.show()

p2_dens = g2_dens.contract_legs()

plt.figure()
plt.plot(g2_dens.wn, chi_dens.real, '-o', color='cornflowerblue', label='Tilde')
plt.plot(g2_dens.wn, chi_dens_core.real, '-p', color='firebrick', label='Core')
# chi_dens_ed[np.size(chi_dens_ed)//2] -= (n/2)**2*beta*beta/1.675*2

plt.plot(wn_chi_ed, chi_dens_ed.real, '-p', color='goldenrod', label='ED')
# p2_dens[np.size(p2_dens)//2] -= (n/2)**2*1/beta**2*2*1/4
# p2_dens[np.size(p2_dens)//2] -= (n/2-0.5)**2
# # p2_dens[np.size(p2_dens)//2] -= np.sum(ggv)*2*1/beta**2
# plt.plot(g2_dens.wn,p2_dens.real*beta,'-p',color='forestgreen',label='ED-sum')
plt.legend()
plt.xlim(0,10)
plt.savefig(path + 'chi_dens.png')
plt.show()

ggv = lfp.get_ggv(green.g_loc, g2_dens.niv)

print(f'Sum ggv: {np.sum(ggv) * 2 * 1 / beta ** 2}')
print(f'(n/2)^2*2: {(n / 2) ** 2 * 1 / beta * 1 / 4 * 1 / 4}')
print(f'Sum ggv: {np.sum(ggv)}')
print(f'Sum giw: {(1 / beta * np.sum(mf.cut_v_1d(giw_dmft.real, 25))) ** 2}')
print(f'Sum giw: {(1 / beta * np.sum(mf.cut_v_1d(giw_dmft.real, 1000))) ** 2}')
print(f'(n/2)^2*2: {(n / 2 - 0.5) ** 2}')

print(f'Sum chi_upup: {1 / beta * 0.5 * np.sum(chi_dens.real + chi_magn.real)}')
print(f'Sum chi_upup_ed: {1 / beta * 0.5 * np.sum(chi_dens_ed.real + chi_magn_ed.real)}')
print(f'True value: {twop.get_sum_chiupup(n)}')

# plt.figure()
# plt.semilogy(g2_dens.wn,np.abs(chi_magn.real-chi_magn_ed.real),'-o',color='cornflowerblue',label='Tilde-ED')
# plt.legend()
# plt.xlim(0,30)
# plt.show()


print(1 / beta * np.sum(chi_dens.real + chi_magn.real))
print(1 / beta * np.sum(chi_dens_ed.real + chi_magn_ed.real))
print(twop.get_sum_chiupup(n) * 2)

# %%
siw_dens = lfp.schwinger_dyson_vrg(vrg_dens, chi_dens, green.g_loc, u)
siw_magn = lfp.schwinger_dyson_vrg(vrg_magn, chi_magn, green.g_loc, u)

n_shell = 800
niv_full = n_shell + niv_core
siw_sde = siw_dens + siw_magn
siw_shell_dens = lfp.schwinger_dyson_shell(chi_dens, green.g_loc, beta, u, n_shell=n_shell, n_core=niv_core, wn=g2_dens.wn)
siw_shell_magn = lfp.schwinger_dyson_shell(chi_magn, green.g_loc, beta, u, n_shell=n_shell, n_core=niv_core, wn=g2_dens.wn)
siw_shell = siw_shell_dens + siw_shell_magn

siw_sde_full = mf.concatenate_core_asmypt(siw_sde, siw_shell)
vn_full = mf.vn(niv_full)

siw_shell_dens_only = lfp.schwinger_dyson_shell(chi_dens, green.g_loc, beta, u, n_shell=niv_full, n_core=0, wn=g2_dens.wn)
siw_shell_magn_only = lfp.schwinger_dyson_shell(chi_magn, green.g_loc, beta, u, n_shell=niv_full, n_core=0, wn=g2_dens.wn)
siw_shell_only = siw_shell_dens_only + siw_shell_magn_only

plt.figure(dpi=500)
plt.plot(vn_full, siw_sde_full.imag, '-o', color='cornflowerblue', markeredgecolor='k', label='SDE', lw=4)
plt.plot(vn_full, green.sigma.get_siw(niv=niv_full)[0, 0, 0, :].imag, '-p', color='firebrick', label='Input', markeredgecolor='k', ms=2)
plt.plot(vn_full, siw_shell_only.imag, '-h', color='forestgreen', label='Asympt', markeredgecolor='k', ms=1)
plt.legend()
plt.xlim(0, 100)
# plt.ylim(None,0)
plt.savefig(path + f'sde_vs_input_niv_{niv_core}.png')
plt.show()
# %%
plt.figure()
plt.semilogy(vn_full, np.abs(siw_sde_full.imag - green.sigma.get_siw(niv=niv_full)[0, 0, 0, :].imag), '-', color='cornflowerblue',
             markeredgecolor='k', label='SDE')
plt.vlines(vn_full[niv_full + niv_core], 1e-4, 1e-2, linestyles='dashed', color='k')
plt.xlim(0, n_shell)
plt.legend()
plt.show()

plt.figure()
plt.semilogy(vn_full, np.abs(siw_shell_only.imag - green.sigma.get_siw(niv=niv_full)[0, 0, 0, :].imag), '-', color='cornflowerblue',
             markeredgecolor='k', label='SDE')
plt.vlines(vn_full[niv_full + niv_core], 1e-4, 1e-2, linestyles='dashed', color='k')
plt.xlim(0, n_shell)
plt.legend()
plt.show()

# %%
plt.figure()
plt.semilogy(mf.cut_v_1d(g2_dens.vn, niv_core), np.abs(siw_sde.imag - mf.cut_v_1d(green.sigma.sigma_core[0, 0, 0, :].imag, niv_cut=niv_core)), '-o',
             color='cornflowerblue', markeredgecolor='k', label='SDE')
# plt.plot(mf.vn(green.sigma.niv_core), green.sigma.sigma_core[0,0,0,:].imag,'-o',color='firebrick',label='Input',markeredgecolor='k')
plt.legend()
plt.xlim(0, 100)
plt.savefig(path + f'sde_minus_input_niv_{niv_core}.png')
plt.show()

# #%%
# g_udb = file['g4iw_udbar'][()]
# g_magn = file['g4iw_magn'][()]
#
# print(np.max(g_udb-g_magn))


# Load the w2dynamics vertex:
# w2dyn_file = w2dyn_aux.g4iw_file(fname=path+'g4iw_sym.hdf5')
# tmp  = w2dyn_file.read_g2_full(channel='dens')
# wn_w2dyn = mf.wn(np.shape(tmp)[0]//2)
# g2_dens_w2dyn = lfp.LocalFourPoint(matrix=tmp, beta=beta, wn=wn_w2dyn, channel='dens')
# g2_magn_w2dyn = lfp.LocalFourPoint(matrix=w2dyn_file.read_g2_full(channel='magn'), beta=beta, wn=wn_w2dyn, channel='magn')
# g2_dens_w2dyn.cut_iv(niv_cut=g2_dens.niv)
# g2_magn_w2dyn.cut_iv(niv_cut=g2_dens.niv)
# g2_dens_w2dyn.plot(0,pdir=path,name='G2_dens_w2dyn_wn0')
# g2_magn_w2dyn.plot(0,pdir=path,name='G2_magn_w2dyn_wn0')
#
# gchi_dens_w2dyn = lfp.gchir_from_g2(g2_dens_w2dyn,green.g_loc)
# gchi_dens_w2dyn.plot(0,pdir=path,name='Gchi_dens_w2dyn_wn0')

# #%%
# plt.figure(figsize=(5,4))
# plt.pcolormesh(g2_dens.vn,g2_dens.vn,g2_dens.mat[np.size(g2_dens.wn)//2,:,:].real/2.5-g2_dens_w2dyn.mat[np.size(g2_dens_w2dyn.wn)//2,:,:].real,cmap='RdBu')
# plt.colorbar()
# plt.show()
#
# # plt.figure(figsize=(5,4))
# # plt.pcolormesh(g2_dens.vn,g2_dens.vn,g2_dens.mat[np.size(g2_dens.wn)//2,:,:].real/g2_dens_w2dyn.mat[np.size(g2_dens_w2dyn.wn)//2,:,:].real,cmap='RdBu')
# # plt.colorbar()
# # plt.show()
# #
# ggv = lfp.get_ggv(green.g_loc,g2_dens.niv)
# #
# plt.figure(figsize=(5,4))
# # plt.pcolormesh(mf.vn(g2_dens.niv),mf.vn(g2_dens.niv),(g2_dens.mat[np.size(g2_dens.wn)//2,:,:].real*1/np.pi-ggv.real*2),cmap='RdBu')
# plt.pcolormesh(mf.vn(g2_dens.niv),mf.vn(g2_dens.niv),(g2_dens.mat[np.size(g2_dens.wn)//2,:,:].real*1/2.5-ggv.real*2),cmap='RdBu')
# # plt.pcolormesh(mf.vn(g2_dens.niv),mf.vn(g2_dens.niv),(g2_dens_w2dyn.mat[np.size(g2_dens_w2dyn.wn)//2,:,:].real-ggv.real*2),cmap='RdBu')
# plt.colorbar()
# plt.show()
# # plt.figure(figsize=(5,4))
# # plt.pcolormesh(g2_dens.vn,g2_dens.vn,g2_dens.mat[np.size(g2_dens.wn)//2,:,:].real,cmap='RdBu')
# # plt.colorbar()
# # plt.show()
# #
# # plt.figure(figsize=(5,4))
# # plt.pcolormesh(g2_dens.vn,g2_dens.vn,g2_dens_w2dyn.mat[np.size(g2_dens_w2dyn.wn)//2,:,:].real,cmap='RdBu')
# # plt.colorbar()
# # plt.show()
