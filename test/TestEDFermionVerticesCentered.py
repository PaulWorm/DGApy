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
path = './2DSquare_U8_tp-0.2_tpp0.1_beta2_n0.90/'
fname = 'EDFermion_GreenFunctions_centered_nbath_3_niw_50_niv_49.hdf5'
file = h5py.File(path + fname, 'r')
beta = file['config/beta'][()]
giw_dmft = file['dmft/giw'][()]
giw_ed = file['giw'][()]
chi_dens_ed = file['chi_dens'][()]
chi_magn_ed = file['chi_magn'][()]
iv_ed = mf.iv(beta,np.size(giw_ed)//2)
n_ed = (1/beta*np.sum(giw_dmft.real-iv_ed)+0.5)*2
# n_ed = (np.sum(giw_ed.real-iv_ed)-0.5)
print(n_ed)
#%%
mu = file['dmft/mu'][()]
wn = mf.wn(np.size(file['iw4'][()])//2)
n = file['config/totdens'][()]
u = file['config/U'][()]
hr = hamr.one_band_2d_t_tp_tpp(1,-0.2,0.1)
nk = (64,64,1)
k_grid = bz.KGrid(nk=nk)
ek = hamk.ek_3d(k_grid.grid,hr)
siw = 1/file['dmft/g0iw'][()] - 1/file['giw'][()]
sigma = twop.SelfEnergy(siw[None,None,None,:],beta)
green = twop.GreensFunction(sigma,ek,n=n)
n_test = (1/beta*np.sum(green.g_loc.real)+0.5)*2
print(n_test)

#%%
# Load the ED vertex:
g2_dens = lfp.LocalFourPoint(matrix=file['g4iw_dens'][()], beta=beta, wn=wn, channel='dens')
g2_magn = lfp.LocalFourPoint(matrix=file['g4iw_magn'][()], beta=beta, wn=wn, channel='magn')

g2_dens.plot(10,pdir=path,name='G2_dens_wn10_centered')
g2_magn.plot(10,pdir=path,name='G2_magn_wn10_centered')

g2_dens.plot(0,pdir=path,name='G2_dens_wn0_centered')
g2_magn.plot(0,pdir=path,name='G2_magn_wn0_centered')

niv_core = g2_dens.niv
gchi_dens = lfp.gchir_from_g2(g2_dens,green.g_loc)
gchi_magn = lfp.gchir_from_g2(g2_magn,green.g_loc)

gchi_dens.plot(10,pdir=path,name='Gchi_dens_wn10_centered')
gchi_magn.plot(10,pdir=path,name='Gchi_magn_wn10_centered')


#%%
gchi0_gen = bub.LocalBubble(wn=wn,giw=green,freq_notation='center')
gchi0_core = gchi0_gen.get_gchi0(niv_core)
chi0_core = gchi0_gen.get_chi0(niv_core)
chi0_shell = gchi0_gen.get_asymptotic_correction(niv_core)

F_dens = lfp.Fob2_from_chir(gchi_dens,gchi0_core)
F_magn = lfp.Fob2_from_chir(gchi_magn,gchi0_core)

F_dens.plot(-10,pdir=path,name='F_dens_wnm10_centered')
F_magn.plot(-10,pdir=path,name='F_magn_wnm10_centered')

F_dens.plot(0,pdir=path,name='F_dens_wn0_centered')
F_magn.plot(0,pdir=path,name='F_magn_wn0_centered')

#%%
chi_dens_core = gchi_dens.contract_legs()
chi_magn_core = gchi_magn.contract_legs()

niv_core = 25
vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, gchi0_gen, u, niv_core=niv_core, niv_shell=500)
vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir(gchi_magn, gchi0_gen, u, niv_core=niv_core, niv_shell=500)

plt.figure()
plt.plot(g2_dens.wn,chi_magn.real,'-o',color='cornflowerblue',label='Tilde')
plt.plot(g2_dens.wn,chi_magn_core.real,'-p',color='firebrick',label='Core')
plt.plot(g2_dens.wn,chi_magn_ed.real,'-p',color='goldenrod',label='ED')
plt.legend()
plt.savefig(path+'chi_magn.png')
plt.show()

p2_dens = g2_dens.contract_legs()

chi_dens_ed = file['chi_dens'][()]
plt.figure()
plt.plot(g2_dens.wn,chi_dens.real,'-o',color='cornflowerblue',label='Tilde')
plt.plot(g2_dens.wn,chi_dens_core.real,'-p',color='firebrick',label='Core')
chi_dens_ed[np.size(chi_dens_ed)//2] -= (n/2-1)**2*beta*2
plt.plot(g2_dens.wn,chi_dens_ed.real,'-p',color='goldenrod',label='ED')
plt.legend()
plt.show()


print(f'Sum chi_upup: {1/beta*0.5*np.sum(chi_dens.real+chi_magn.real)}')
print(f'Sum chi_upup_ed: {1/beta*0.5*np.sum(chi_dens_ed.real+chi_magn_ed.real)}')
print(f'True value: {twop.get_sum_chiupup(n)}')


