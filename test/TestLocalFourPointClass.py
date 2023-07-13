import numpy as np
import matplotlib.pyplot as plt

import TestData as td
import dga.two_point as tp
import dga.local_four_point as lfp
import dga.bubble as bub
import dga.matsubara_frequencies as mf
import dga.brillouin_zone as bz
import dga.hk as hamk

ddict = td.get_data_set_6(True, True)

siw = ddict['siw']
beta = ddict['beta']
u = ddict['u']
n = ddict['n']
g2_dens = ddict['g2_dens']
g2_magn = ddict['g2_magn']

chi_dens_dmft = ddict['chi_dens']
chi_magn_dmft = ddict['chi_magn']

niv_core = 100
niw_core = 100
niv_shell = 400

nk = (42, 42, 1)
k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])

sigma = tp.SelfEnergy(siw[None, None, None, :], beta, pos=False)
giwk = tp.GreensFunction(sigma, ek, n=n)
giwk.set_g_asympt(niv_asympt=np.max(niv_core) + 1000)
g_loc = giwk.k_mean(iv_range='full')

wn = g2_dens.wn
gchi_dens = lfp.gchir_from_g2(g2_dens, g_loc)
gchi_magn = lfp.gchir_from_g2(g2_magn, g_loc)


gchi_magn.is_full_w = True
gchi_dens.is_full_w = True
gchi_magn.cut_iw(niw_core)
gchi_dens.cut_iw(niw_core)
chi0_gen = bub.BubbleGenerator(gchi_magn.wn, giwk)
# niv_core = np.arange(10, niw)[::10][::-1]

chi_dens_core, chi_magn_core = [], []
wn_list = []

gchi_dens.cut_iv(niv_core)
gchi_magn.cut_iv(niv_core)
gchi0_shell = chi0_gen.get_gchi0(niv_core + niv_shell)
gamma_dens = lfp.gammar_from_gchir(gchi_dens, gchi0_shell, u)
gamma_magn = lfp.gammar_from_gchir(gchi_magn, gchi0_shell, u)
#
_, chi_dens = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_dens, chi0_gen, u, niv_shell)
_, chi_magn = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, chi0_gen, u, niv_shell)
#
# # gamma_magn = lfp.gammar_from_gchir(gchi_magn, gchi0_core, u)
# # _, chi_magn_core = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, chi0_gen, u, 0)
#
# wn = mf.wn(len(chi_magn) // 2)
#
# # %%
#
#
gchi0_core = chi0_gen.get_gchi0(niv_core)
chi0_asympt = chi0_gen.get_asymptotic_correction(niv_core+niv_shell)
gchi_magn_core = lfp.gchir_from_gamob2(gamma_magn,gchi0_core)
_, chi_magn_simple = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, chi0_gen, u, 0)
#
#
# %%
wn_dmft = mf.wn(len(chi_magn_dmft) // 2)
plt.figure()
plt.loglog(gchi_magn.wn, chi_magn.real, '-o', color='firebrick')
plt.loglog(gchi_magn.wn, chi_magn.real-chi0_asympt, '-o', color='k')
plt.loglog(gchi_magn.wn, gchi_magn.contract_legs().real, '-o', color='cornflowerblue')
plt.loglog(gchi_magn.wn, gchi_magn_core.contract_legs().real, '-o', color='navy')
plt.loglog(gchi_magn.wn, chi_magn_simple.real, '-o', color='forestgreen')
plt.loglog(wn_dmft, chi_magn_dmft.real, '-o', color='tab:orange')
plt.xlim(0, None)
plt.show()
#%%
wn_dmft = mf.wn(len(chi_dens_dmft) // 2)
plt.figure()
plt.loglog(gchi_dens.wn, chi_dens.real, '-o', color='firebrick')
plt.loglog(gchi_dens.wn, chi_dens.real-chi0_asympt, '-o', color='k')
plt.loglog(gchi_dens.wn, gchi_dens.contract_legs().real, '-o', color='cornflowerblue')
# plt.loglog(gchi_dens.wn, gchi_dens_core.contract_legs().real, '-o', color='navy')
# plt.loglog(gchi_dens.wn, chi_dens_simple.real, '-o', color='forestgreen')
plt.loglog(wn_dmft, chi_dens_dmft.real, '-o', color='tab:orange')
plt.xlim(0, None)
plt.show()

#%%
wn_dmft = mf.wn(len(chi_dens_dmft) // 2)
plt.figure()
plt.plot(gchi_dens.wn, chi_dens.real, '-o', color='firebrick')
plt.plot(gchi_dens.wn, chi_dens.real-chi0_asympt, '-o', color='k')
plt.plot(gchi_dens.wn, gchi_dens.contract_legs().real, '-o', color='cornflowerblue')
# plt.loglog(gchi_dens.wn, gchi_dens_core.contract_legs().real, '-o', color='navy')
# plt.loglog(gchi_dens.wn, chi_dens_simple.real, '-o', color='forestgreen')
plt.plot(wn_dmft, chi_dens_dmft.real, '-o', color='tab:orange')
plt.xlim(-10, 10)
plt.ylim(0,0.06)
plt.show()

#%%
wn_dmft = mf.wn(len(chi_dens_dmft) // 2)
plt.figure()
plt.plot(gchi_magn.wn, chi_magn.real, '-o', color='firebrick',ms=8)
plt.plot(gchi_magn.wn, chi_magn.real-chi0_asympt, '-o', color='k')
plt.plot(gchi_magn.wn, gchi_magn.contract_legs().real, '-o', color='cornflowerblue')
# plt.loglog(gchi_magn.wn, gchi_magn_core.contract_legs().real, '-o', color='navy')
# plt.loglog(gchi_magn.wn, chi_magn_simple.real, '-o', color='forestgreen')
plt.plot(wn_dmft, chi_magn_dmft.real, '-o', color='tab:orange')
plt.xlim(-10, 10)
plt.ylim(0,2)
plt.show()


#%%

plt.figure()
plt.plot(wn_dmft, chi_dens_dmft.real, '-o', color='tab:orange')
plt.xlim(-10,10)
plt.show()

#%%
print(f'Chi-dens-sum: {1/beta*np.sum(chi_dens_dmft.real)}') # -(1-n/2)*n/2*2+(1-n/2)**2*2
print(f'Chi-dens-sum-sde: {1/beta*np.sum(chi_dens.real)}')
print(f'Chi-dens-sum-sde-urange: {1/beta*np.sum(chi_dens.real-chi0_asympt.real)}')
gchi_dens_full = lfp.gchir_from_g2(g2_dens, g_loc)
gchi_magn_full = lfp.gchir_from_g2(g2_magn, g_loc)
print(f'Chi-dens-sum-gchi: {1/beta**3*np.sum(gchi_dens_full.mat.real)}')
print(f'Chi-magn-sum: {1/beta*np.sum(chi_magn_dmft.real)}')
print(f'Chi-magn-sum-sde: {1/beta*np.sum(chi_magn.real)}')
print(f'Chi-magn-sum-gchi: {1/beta**3*np.sum(gchi_magn_full.mat.real)}')
print(f'Chi-upup-sum: {1/beta*0.5*np.sum(chi_magn_dmft.real+chi_dens_dmft.real)}')
print(f'Chi-upup-sum-sde: {1/beta*0.5*np.sum(chi_magn.real+chi_dens.real)}')
print(f'Chi-upup-sum-sde-urange: {1/beta*0.5*np.sum(chi_magn.real+chi_dens.real)-1/beta*np.sum(chi0_asympt.real)}')
print(f'Chi-upup-sum-gchi: '
      f'{1/beta**3*0.5*np.sum(gchi_magn_full.mat.real+gchi_dens_full.mat.real)+1/beta*np.sum(chi0_asympt.real)}')
print(f'Chi-upup-sum-gchi: {1/beta**3*0.5*np.sum(gchi_magn_full.mat.real+gchi_dens_full.mat.real)}')
print(f'Chi-upup-sum-anal: {n/2*(1-n/2)}')
print(f'Chi-upup-sum-anal: {1/beta*np.sum(chi0_asympt.real)}')



# # %%
# wn_dmft = mf.wn(len(chi_magn_dmft) // 2)
# plt.figure()
# plt.semilogx(gchi_magn.wn, chi_magn.real, '-o', color='firebrick')
# plt.semilogx(gchi_magn.wn, chi_magn.real-chi0_asympt, '-o', color='k')
# plt.semilogx(gchi_magn.wn, gchi_magn.contract_legs().real, '-o', color='cornflowerblue')
# plt.semilogx(gchi_magn.wn, gchi_magn_core.contract_legs().real, '-o', color='navy')
# plt.semilogx(gchi_magn.wn, chi_magn_simple.real, '-o', color='forestgreen')
# plt.semilogx(wn_dmft, chi_magn_dmft.real, '-o', color='tab:orange')
# plt.xlim(0, None)
# plt.ylim(0, 0.7)
# plt.show()

