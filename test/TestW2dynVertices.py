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
path = 'D:/Research/From_Simone/PrNiO2/0_GPa_d_0.18_beta_15/'
path = './2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
fname = 'g4iw_sym.hdf5'
fname_1p = '1p-data.hdf5'


dmft_file = w2dyn_aux.w2dyn_file(fname=path+fname_1p)

u = dmft_file.get_udd()
n = dmft_file.get_totdens()
giw_dmft = dmft_file.get_giw()[0,0,:]
beta = dmft_file.get_beta()
siw = dmft_file.get_siw()[0,0,:]
mu = dmft_file.get_mu()

w2dyn_file = w2dyn_aux.g4iw_file(fname=path+'g4iw_sym.hdf5')
tmp  = w2dyn_file.read_g2_full(channel='dens')
wn_w2dyn = mf.wn(np.shape(tmp)[0]//2)
g2_dens = lfp.LocalFourPoint(matrix=tmp, beta=beta, wn=wn_w2dyn, channel='dens')
g2_magn = lfp.LocalFourPoint(matrix=w2dyn_file.read_g2_full(channel='magn'), beta=beta, wn=wn_w2dyn, channel='magn')
wn = g2_dens.wn

n_dmft = (1/beta*np.sum(giw_dmft.real)+0.5)*2
print(n_dmft)

# Load the susceptibility:
fname_chi = 'chi.hdf5'
chi_file = h5py.File(path+fname_chi,'r')
chi_uu_w2dyn = chi_file['worm-001/ineq-001/p2iw-worm/00001/value'][()]
chi_ud_w2dyn = chi_file['worm-001/ineq-001/p2iw-worm/00004/value'][()]
chi_dens_w2dyn = chi_uu_w2dyn + chi_ud_w2dyn
chi_dens_w2dyn[np.size(chi_dens_w2dyn)//2] -= (n / 2 - 1) ** 2 * beta *2
chi_magn_w2dyn = chi_uu_w2dyn - chi_ud_w2dyn
wn_chi = mf.wn(np.shape(chi_dens_w2dyn)[0]//2)
#%%
hr = hamr.one_band_2d_t_tp_tpp(1,-0.2,0.1)
nk = (64,64,1)
k_grid = bz.KGrid(nk=nk)
ek = hamk.ek_3d(k_grid.grid,hr)
sigma = twop.SelfEnergy(siw[None,None,None,:],beta)
green = twop.GreensFunction(sigma,ek,n=n)
n_test = (1/beta*np.sum(green.g_loc.real)+0.5)*2
print(n_test)
#%%
# chi_dens_ed = file['chi_dens'][()]

v_dmft = mf.v(beta,np.size(giw_dmft)//2)
plt.figure()
plt.plot(green.v,green.g_loc.imag,'-o',color='cornflowerblue',label='Test')
plt.plot(v_dmft,giw_dmft.imag,'-o',color='firebrick',label='DMFT')
plt.xlim(-10,30)
plt.close()

plt.figure()
plt.plot(green.v,green.g_loc.real,'-o',color='cornflowerblue',label='Test')
plt.plot(v_dmft,giw_dmft.real,'-o',color='firebrick',label='DMFT')
plt.xlim(-10,30)
plt.close()

g2_dens.plot(0,pdir=path,name='G2_dens_wn0')
g2_magn.plot(0,pdir=path,name='G2_magn_wn0')

g2_dens.plot(10,pdir=path,name='G2_dens_wn10')
g2_dens.plot(50,pdir=path,name='G2_dens_wn50')
g2_magn.plot(10,pdir=path,name='G2_magn_wn10')

# Compute the vertex:
niv_core = g2_dens.niv
gchi_dens = lfp.gchir_from_g2(g2_dens,green.g_loc)
gchi_magn = lfp.gchir_from_g2(g2_magn,green.g_loc)
gchi_dens.plot(0,pdir=path,name='Gchi_dens_wn0')
gchi_magn.plot(0,pdir=path,name='Gchi_magn_wn0')



gchi0_gen = bub.LocalBubble(wn=wn,giw=green)
gchi0_core = gchi0_gen.get_gchi0(niv_core)
chi0_core = gchi0_gen.get_chi0(niv_core)
# chi0_shell = gchi0_gen.get_asymptotic_correction(niv_core)
chi0_shell = gchi0_gen.get_chi0_shell(niv_core,500)

F_dens = lfp.Fob2_from_chir(gchi_dens,gchi0_core)
F_magn = lfp.Fob2_from_chir(gchi_magn,gchi0_core)

F_dens.plot(0,pdir=path,name='F_dens_wn0_w2dyn')
F_dens.plot(0,pdir=path,name='F_dens_wn0_niv30_w2dyn',niv=30)
F_magn.plot(0,pdir=path,name='F_magn_wn0_niv30_w2dyn',niv=30)
F_magn.plot(0,pdir=path,name='F_magn_wn0_w2dyn')

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
#

#%%
niv_core = 50
niv_shell = 500
vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, gchi0_gen, u, niv_core=niv_core, niv_shell=niv_shell)
vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir(gchi_magn, gchi0_gen, u, niv_core=niv_core, niv_shell=niv_shell)

gchi0_gen2 = bub.LocalBubble(wn_chi,green,freq_notation='center')
chi0_core = gchi0_gen2.get_chi0(niv_shell)
chi0_shell = gchi0_gen2.get_asymptotic_correction(niv_shell)
chi0_shell = gchi0_gen2.get_chi0_shell(niv_shell,2*niv_shell)
chi0_full = chi0_core+chi0_shell
bubble_test = np.zeros((len(wn_chi),),dtype=complex)
niv_dmft = np.size(giw_dmft)//2
niv_sum = 500#niv_dmft-np.size(wn)
for i,wn in enumerate(wn_chi):
    bubble_test[i] = -1/beta * np.sum(giw_dmft[niv_dmft-niv_sum:niv_dmft+niv_sum] * giw_dmft[niv_dmft-niv_sum-wn:niv_dmft+niv_sum-wn])

plt.figure()
plt.semilogy(g2_dens.wn,chi_magn.real,'-',color='cornflowerblue',label='Tilde')
plt.semilogy(g2_dens.wn,chi_magn_core.real,'-',color='firebrick',label='Core')
plt.semilogy(wn_chi,chi0_full.real,'-',color='goldenrod',label='Chi0')
plt.semilogy(wn_chi,chi0_core.real,'-',color='navy',label='Chi0')
plt.semilogy(wn_chi,chi_magn_w2dyn.real,'-',color='forestgreen',label='Chi-sample')
plt.semilogy(wn_chi,bubble_test.real,'-',color='tab:orange',label='Chi0-test')
plt.legend()
plt.show()

#%%
plt.figure()
plt.semilogy(g2_dens.wn,chi_dens.real,'-',color='cornflowerblue',label='Tilde')
plt.semilogy(g2_dens.wn,chi_dens_core.real,'-',color='firebrick',label='Core')
plt.semilogy(g2_dens.wn,chi0_core.real+chi0_shell.real,'-',color='goldenrod',label='Chi0')
plt.semilogy(wn_chi,chi_dens_w2dyn.real,'-',color='forestgreen',label='Chi-sample')
plt.legend()
plt.show()

#%%
siw_dens = lfp.schwinger_dyson_vrg(vrg_dens,chi_dens,green.g_loc,u)
siw_magn = lfp.schwinger_dyson_vrg(vrg_magn,chi_magn,green.g_loc,u)



siw_sde = siw_dens+siw_magn

plt.figure()
plt.plot(mf.cut_v_1d(g2_dens.vn,niv_core),siw_sde.imag,'-o',color='cornflowerblue',markeredgecolor='k',label='SDE')
plt.plot(mf.vn(green.sigma.niv_core), green.sigma.sigma_core[0,0,0,:].imag,'-o',color='firebrick',label='Input',markeredgecolor='k',ms=1)
plt.legend()
plt.xlim(0,110)
plt.savefig(path+f'sde_vs_input_niv_{niv_core}_w2dyn.png')
plt.show()


plt.figure()
plt.semilogy(mf.cut_v_1d(g2_dens.vn,niv_core),np.abs(siw_sde.imag-mf.cut_v_1d(green.sigma.sigma_core[0,0,0,:].imag,niv_cut=niv_core)),'-o',color='cornflowerblue',markeredgecolor='k',label='SDE')
# plt.plot(mf.vn(green.sigma.niv_core), green.sigma.sigma_core[0,0,0,:].imag,'-o',color='firebrick',label='Input',markeredgecolor='k')
plt.legend()
plt.xlim(0,110)
plt.savefig(path+f'sde_minus_input_{niv_core}_w2dyn.png')
plt.show()







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