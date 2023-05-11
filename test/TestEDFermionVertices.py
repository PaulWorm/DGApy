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

# Load the vertex:
path = './2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
# path = './2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
# fname = 'EDFermion_GreenFunctions_nbath_3_niw_50_niv_49.hdf5'
fname = 'EDFermion_GreenFunctions_nbath_3_new.hdf5'
# fname = 'EDFermion_GreenFunctions_nbath_3_niw_200_niv_199.hdf5'
fname = 'EDFermion_GreenFunctions_nbath_3_niw_250_niv_249.hdf5'
file = h5py.File(path + fname, 'r')
beta = file['config/beta'][()]
giw_dmft = file['dmft/giw'][()]
giw_ed = file['giw'][()]
mu_dmft = file['dmft/mu'][()]
iv_ed = mf.iv(beta, np.size(giw_ed) // 2)
n_ed = (1 / beta * np.sum(giw_dmft.real - iv_ed) + 0.5) * 2
print(n_ed)

# % Load the Susceptibility:

fname_chi = 'EDFermion_GreenFunctions_nbath_3_niw_4000_niv_49.hdf5'
# fname_chi = 'EDFermion_GreenFunctions_nbath_chi_giw.hdf5'
file_chi = h5py.File(path + fname_chi, 'r')
chi_dens_ed = file_chi['chi_dens'][()]
n = file['config/totdens'][()]
chi_dens_ed[np.size(chi_dens_ed) // 2] -= (n / 2 - 1) ** 2 * beta * 2  # * 1.9964
chi_magn_ed = file_chi['chi_magn'][()]
wn_chi_ed = mf.wn(np.size(chi_dens_ed) // 2)

# Load fiw:
fname_fiw = h5py.File(path + 'bath_values_nbath_3.hdf5', 'r')
fiw_ed = fname_fiw['fiw'][()]
fiw_dmft = fname_fiw['fiw_input'][()]
fname_fiw.close()
# %%
g0iw_ed = (1 / (iv_ed + mu_dmft - np.conj(fiw_ed)))
# g0iw_ed = (1/(iv_ed + mu_dmft - np.conj(fiw_dmft)))
plt.figure(dpi=200)
plt.plot(iv_ed.imag, g0iw_ed.real, '-o', color='cornflowerblue', markeredgecolor='k', ms=12)
plt.plot(iv_ed.imag, file['dmft/g0iw'][()].real, '-h', color='firebrick', markeredgecolor='k')
plt.xlim(-2, 10)
plt.show()

plt.figure(dpi=200)
plt.plot(iv_ed.imag, g0iw_ed.imag, '-o', color='cornflowerblue', markeredgecolor='k', ms=12)
plt.plot(iv_ed.imag, file['dmft/g0iw'][()].imag, '-h', color='firebrick', markeredgecolor='k')
plt.xlim(-2, 10)
plt.show()
# %%
mu = file['dmft/mu'][()]
wn = mf.wn(np.size(file['iw4'][()]) // 2)
# wn_ed = mf.wn(np.size(file['iw4'][()]) // 2)

u = file['config/U'][()]
hr = hamr.one_band_2d_t_tp_tpp(1, -0.2, 0.1)
nk = (64, 64, 1)
k_grid = bz.KGrid(nk=nk)
ek = hamk.ek_3d(k_grid.grid, hr)
# siw = 1 / file['dmft/g0iw'][()] - 1 / file['giw'][()]
siw = 1 / g0iw_ed - 1 / file['giw'][()]
sigma = twop.SelfEnergy(siw[None, None, None, :], beta)
green = twop.GreensFunction(sigma, ek, n=n)
n_test = (1 / beta * np.sum(green.g_loc.real) + 0.5) * 2
print(green.e_kin)
print(n_test)
# %%
# chi_dens_ed = file['chi_dens'][()]

v_dmft = mf.v(beta, np.size(giw_dmft) // 2)
v_ed = mf.v(beta, np.size(giw_ed) // 2)
plt.figure(dpi=500)
plt.plot(green.v, green.g_loc.imag, '-o', color='cornflowerblue', label='Test', ms=8)
plt.plot(v_dmft, giw_dmft.imag, '-h', color='firebrick', label='DMFT', markeredgecolor='k', ms=6)
plt.plot(v_ed, giw_ed.imag, '-p', color='goldenrod', label='ED', markeredgecolor='k', ms=4)
plt.legend()
plt.xlim(-10, 30)
plt.savefig(path + 'Giw_imag.png')
plt.close()

plt.figure(dpi=500)
plt.plot(green.v, green.g_loc.real, '-o', color='cornflowerblue', label='Test', ms=8)
plt.plot(v_dmft, giw_dmft.real, '-h', color='firebrick', label='DMFT', markeredgecolor='k', ms=6)
plt.plot(v_ed, giw_ed.real, '-p', color='goldenrod', label='ED', markeredgecolor='k', ms=4)
plt.legend()
plt.xlim(-10, 30)
plt.savefig(path + 'Giw_real.png')
plt.close()

# %%
# Load the ED vertex:
niw_core = 50
g2_dens = lfp.LocalFourPoint(matrix=file['g4iw_dens'][()], beta=beta, wn=wn, channel='dens')
g2_dens.mat = mf.cut_w(g2_dens.mat, niw_core, (0,))
g2_dens.wn = mf.cut_w(g2_dens.wn, niw_core, (0,))
g2_magn = lfp.LocalFourPoint(matrix=file['g4iw_magn'][()], beta=beta, wn=wn, channel='magn')
g2_magn.mat = mf.cut_w(g2_magn.mat, niw_core, (0,))
g2_magn.wn = mf.cut_w(g2_magn.wn, niw_core, (0,))

wn = mf.wn(niw_core)
# g2_dens.plot(0,pdir=path,name='G2_dens_wn0')
# g2_magn.plot(0,pdir=path,name='G2_magn_wn0')

g2_dens.plot(10, pdir=path, name='G2_dens_wn10')
g2_dens.plot(10, pdir=path, name='G2_dens_wn30')
# g2_magn.plot(10,pdir=path,name='G2_magn_wn10')

# Compute the vertex:
niv_core = g2_dens.niv
gchi_dens = lfp.gchir_from_g2(g2_dens, green.g_loc)
gchi_magn = lfp.gchir_from_g2(g2_magn, green.g_loc)
# gchi_dens.plot(0,pdir=path,name='Gchi_dens_wn0')
# gchi_magn.plot(0,pdir=path,name='Gchi_magn_wn0')


gchi0_gen = bub.LocalBubble(wn=wn, giw=green)
gchi0_core = gchi0_gen.get_gchi0(niv_core)
chi0_core = gchi0_gen.get_chi0(niv_core)
chi0_shell = gchi0_gen.get_asymptotic_correction(niv_core)

F_dens = lfp.Fob2_from_chir(gchi_dens, gchi0_core)
F_magn = lfp.Fob2_from_chir(gchi_magn, gchi0_core)

F_dens.plot(0, pdir=path, name='F_dens_wn0')
F_magn.plot(0, pdir=path, name='F_magn_wn0')

F_dens.plot(0, pdir=path, name='F_dens_wn0_niv30', niv=30)
F_magn.plot(0, pdir=path, name='F_magn_wn0_niv30', niv=30)

F_dens.plot(-10, pdir=path, name='F_dens_wnm10')
F_magn.plot(-10, pdir=path, name='F_magn_wnm10')

# lam_core_dens = lfp.lam_from_chir(gchi_dens,gchi0_core)
# lam_tilde_dens = lfp.get_lam_tilde(lam_core_dens,chi0_shell=chi0_shell,u=u)
#
chi_dens_core = gchi_dens.contract_legs()
chi_magn_core = gchi_magn.contract_legs()
# chi_dens_tilde = lfp.get_chir_tilde(lam_tilde_dens,chi0_core,chi0_shell, gchi0_core, u)

# vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, gchi0_gen, u, niv_core=g2_dens.niv, niv_shell=500)
# vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir(gchi_magn, gchi0_gen, u, niv_core=g2_dens.niv, niv_shell=500)

# plt.figure()
# colors_d = plt.cm.Blues(np.linspace(0.5, 1.0, 5))[::-1]
# for i,niv_core in enumerate([10,20,30,40,50]):
#     _, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, gchi0_gen, u, niv_core=niv_core, niv_shell=500)
#     plt.semilogy(g2_dens.wn,chi_dens.real,'-o',color=colors_d[i],label='Tilde')
# plt.semilogy(g2_dens.wn,chi_dens_core.real,'-p',color='firebrick',label='Core')
# plt.legend()
# plt.show()
# %%
niv_core = 50
niv_shell = 500
vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, gchi0_gen, u, niv_core=niv_core, niv_shell=500)
vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir(gchi_magn, gchi0_gen, u, niv_core=niv_core, niv_shell=500)

niw_shell = 400
wn_shell = mf.wn(niw_shell)
gchi0_gen2 = bub.LocalBubble(wn=wn_shell, giw=green, freq_notation='minus')
chi0_core2 = gchi0_gen2.get_chi0(niv_shell, freq_notation='minus')
# chi0_asympt = gchi0_gen2.get_chi0_shell(niv_shell,niv_shell+100,freq_notation='center')
# chi0_asympt = gchi0_gen2.get_asymptotic_correction(niv_shell)
chi0_asympt = gchi0_gen2.get_asympt_sum(10 * niv_shell) - gchi0_gen2.get_asympt_sum(niv_shell)
chi0_asympt_exact = gchi0_gen2.get_chi0_shell(niv_shell, 2 * niv_shell, freq_notation='minus')
chi0_full = chi0_core2 + chi0_asympt_exact
chi_dens_full = (chi0_core2 + chi0_asympt_exact)  # /np.pi
chi_dens_full[niw_shell - niw_core:niw_shell + niw_core + 1] = chi_dens  # *np.pi

chi_magn_full = (chi0_core2 + chi0_asympt_exact)  # /np.pi
chi_magn_full[niw_shell - niw_core:niw_shell + niw_core + 1] = chi_magn  # *np.pi

chi_magn_rpa = (chi0_core2 + chi0_asympt) / (1 - u * (chi0_core2 + chi0_asympt))
iv = mf.iv(beta, green.niv_full)
f1, f2, f3 = gchi0_gen2.get_asympt_prefactors()
g_asympt = 1 / (iv) - 1 / (iv ** 2) * f1 + 1 / (iv ** 3) * (f2)
g_asympt_simple = 1 / (iv)
import scipy

tmp = scipy.signal.convolve(green.g_loc, green.g_loc, method='fft')
tmp2 = scipy.signal.convolve(g_asympt, g_asympt, method='fft')
wn_tmp = mf.wn(np.size(tmp) // 2)
asympt_exact = gchi0_gen2.get_exact_asymptotics()
asympt_correction = gchi0_gen2.get_asymptotic_correction(niv_shell * 10)
asympt_sum = gchi0_gen2.get_asympt_sum(niv_shell * 10)

# %% load and test primitive bubble:
test_file = h5py.File(path + 'EDFermion_GreenFunctions_nbath_3_niw_50_niv_49.hdf5')
# test_file = h5py.File(path+'EDFermion_GreenFunctions_nbath_chi_giw.hdf5')
giw_test = test_file['giw'][()]
vn_test = mf.vn(np.size(giw_test))
plt.figure(dpi=500)

bubble_test = np.zeros((len(wn_chi_ed),), dtype=complex)
niv_dmft = np.size(giw_test) // 2
niv_sum = 20000  # niv_dmft-np.size(wn)
for i, wn in enumerate(wn_chi_ed):
    bubble_test[i] = -1 / beta * np.sum(giw_test[niv_dmft - niv_sum:niv_dmft + niv_sum] * giw_test[niv_dmft - niv_sum - wn:niv_dmft + niv_sum - wn])

plt.loglog(wn_chi_ed, chi_magn_ed.real, color='goldenrod')
plt.loglog(wn_shell, chi_magn_full.real, color='cornflowerblue')
plt.loglog(wn_chi_ed, bubble_test.real, color='firebrick')
plt.loglog(wn_shell, chi0_full.real, color='seagreen')
plt.loglog(wn_shell, np.real(1 / (mf.iw(beta, niw_shell)) ** 2 * green.e_kin) * 2, '--', color='tab:orange')
# plt.semilogy(wn_shell,chi0_asympt.real,'seagreen')
# plt.semilogy(wn_shell,chi0_asympt_exact.real,'firebrick')
# plt.semilogy(wn_shell,chi_magn_rpa.real,color='brown')
# plt.semilogy(wn_shell,chi0_core2 + chi0_asympt,color='k')
# plt.semilogy(wn_shell,chi0_core2,color='gray')
# plt.semilogy(wn_tmp,tmp/beta,color='gray')
# plt.semilogy(wn_tmp,tmp2/beta,color='k')
# plt.semilogy(wn_chi_ed,chi_magn_ed.real,color='navy')
# plt.semilogy(wn_shell,asympt_exact.real,color='forestgreen')
# plt.semilogy(wn_shell,asympt_correction.real,color='crimson')
# plt.semilogy(wn_shell,asympt_sum.real,color='firebrick')
# plt.semilogy(wn_shell,1/(wn_shell*2*np.pi/beta)**2,color='navy')
# plt.semilogy(wn,chi0_shell.real)
# plt.semilogy(wn,chi0_core.real+chi0_shell.real)
plt.xlim(0, 1000)
plt.show()

# %%

plt.figure()
plt.semilogy(green.v, np.abs(green.g_loc.imag), color='cornflowerblue')
plt.semilogy(green.v, np.abs(g_asympt.imag), color='firebrick')
plt.xlim(0, None)
plt.show()
# %%
plt.figure()
plt.semilogy(green.v, np.abs(green.g_loc.imag - g_asympt.imag), color='cornflowerblue')
plt.semilogy(green.v, np.abs(green.g_loc.real - g_asympt.real), color='firebrick')
plt.semilogy(green.v, np.abs(green.g_loc.real - g_asympt_simple.real), color='goldenrod')
plt.xlim(0, 1000)
plt.show()
# %%
vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, gchi0_gen, u, niv_core=niv_core, niv_shell=niv_shell)
vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir(gchi_magn, gchi0_gen, u, niv_core=niv_core, niv_shell=niv_shell)
F_diag_dens = lfp.get_F_diag(chi_dens_ed, chi_magn_ed, channel='dens')
F_diag_magn = lfp.get_F_diag(chi_dens_ed, chi_magn_ed, channel='magn')



F_diag_dens = mf.cut_v(F_diag_dens, niv_shell, (0, 1))
F_diag_magn = mf.cut_v(F_diag_magn, niv_shell, (0, 1))

plt.figure()
plt.pcolormesh(F_diag_magn.real, cmap='RdBu_r')
plt.show()


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
