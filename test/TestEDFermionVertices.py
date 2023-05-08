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
# fname = 'EDFermion_GreenFunctions_nbath_3_niw_50_niv_49.hdf5'
fname = 'EDFermion_GreenFunctions_nbath_3_new.hdf5'
fname = 'EDFermion_GreenFunctions_nbath_3_niw_200_niv_199.hdf5'
fname = 'EDFermion_GreenFunctions_nbath_3_niw_50_niv_49.hdf5'
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
# chi_dens_ed = file['chi_dens'][()]

v_dmft = mf.v(beta,np.size(giw_dmft)//2)
v_ed = mf.v(beta,np.size(giw_ed)//2)
plt.figure()
plt.plot(green.v,green.g_loc.imag,'-o',color='cornflowerblue',label='Test')
plt.plot(v_dmft,giw_dmft.imag,'-o',color='firebrick',label='DMFT')
plt.plot(v_ed,giw_ed.imag,'-o',color='goldenrod',label='ED')
plt.xlim(-10,30)
plt.close()

plt.figure()
plt.plot(green.v,green.g_loc.real,'-o',color='cornflowerblue',label='Test')
plt.plot(v_dmft,giw_dmft.real,'-o',color='firebrick',label='DMFT')
plt.plot(v_ed,giw_ed.real,'-o',color='goldenrod',label='ED')
plt.xlim(-10,30)
plt.close()



# Load the ED vertex:
g2_dens = lfp.LocalFourPoint(matrix=file['g4iw_dens'][()], beta=beta, wn=wn, channel='dens')
g2_magn = lfp.LocalFourPoint(matrix=file['g4iw_magn'][()], beta=beta, wn=wn, channel='magn')



# g2_dens.plot(0,pdir=path,name='G2_dens_wn0')
# g2_magn.plot(0,pdir=path,name='G2_magn_wn0')

g2_dens.plot(10,pdir=path,name='G2_dens_wn10')
g2_dens.plot(10,pdir=path,name='G2_dens_wn30')
# g2_magn.plot(10,pdir=path,name='G2_magn_wn10')

# Compute the vertex:
niv_core = g2_dens.niv
gchi_dens = lfp.gchir_from_g2(g2_dens,green.g_loc)
gchi_magn = lfp.gchir_from_g2(g2_magn,green.g_loc)
# gchi_dens.plot(0,pdir=path,name='Gchi_dens_wn0')
# gchi_magn.plot(0,pdir=path,name='Gchi_magn_wn0')



gchi0_gen = bub.LocalBubble(wn=wn,giw=green)
gchi0_core = gchi0_gen.get_gchi0(niv_core)
chi0_core = gchi0_gen.get_chi0(niv_core)
chi0_shell = gchi0_gen.get_asymptotic_correction(niv_core)

F_dens = lfp.Fob2_from_chir(gchi_dens,gchi0_core)
F_magn = lfp.Fob2_from_chir(gchi_magn,gchi0_core)

F_dens.plot(0,pdir=path,name='F_dens_wn0')
F_magn.plot(0,pdir=path,name='F_magn_wn0')

F_dens.plot(-10,pdir=path,name='F_dens_wnm10')
F_magn.plot(-10,pdir=path,name='F_magn_wnm10')

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
#%%
niv_core = 25
vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, gchi0_gen, u, niv_core=niv_core, niv_shell=500)
vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir(gchi_magn, gchi0_gen, u, niv_core=niv_core, niv_shell=500)


# plt.figure()
# plt.plot(vrg_magn.vn,vrg_magn.mat[100,:].imag,'-o',color='cornflowerblue',label='Tilde',markeredgecolor='k')
# plt.plot(vrg_magn.vn,vrg_magn.mat[100,:].real-1,'-o',color='firebrick',label='Tilde',markeredgecolor='k')
# plt.legend()
# plt.xlim(0,10)
# plt.show()

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
# chi_dens_ed[np.size(chi_dens_ed)//2] -= (n/2)**2*beta*beta/1.675*2
chi_dens_ed[np.size(chi_dens_ed)//2] -= (n/2-1)**2*beta*2
plt.plot(g2_dens.wn,chi_dens_ed.real,'-p',color='goldenrod',label='ED')
# p2_dens[np.size(p2_dens)//2] -= (n/2)**2*1/beta**2*2*1/4
p2_dens[np.size(p2_dens)//2] -= (n/2-0.5)**2
# p2_dens[np.size(p2_dens)//2] -= np.sum(ggv)*2*1/beta**2
plt.plot(g2_dens.wn,p2_dens.real*beta,'-p',color='forestgreen',label='ED-sum')
plt.legend()
plt.show()

ggv = lfp.get_ggv(green.g_loc,g2_dens.niv)

print(f'Sum ggv: {np.sum(ggv)*2*1/beta**2}')
print(f'(n/2)^2*2: {(n/2)**2*1/beta*1/4*1/4}')
print(f'Sum ggv: {np.sum(ggv)}')
print(f'Sum giw: {(1/beta*np.sum(mf.cut_v_1d(giw_dmft.real,25)))**2}')
print(f'Sum giw: {(1/beta*np.sum(mf.cut_v_1d(giw_dmft.real,1000)))**2}')
print(f'(n/2)^2*2: {(n/2-0.5)**2}')


print(f'Sum chi_upup: {1/beta*0.5*np.sum(chi_dens.real+chi_magn.real)}')
print(f'Sum chi_upup_ed: {1/beta*0.5*np.sum(chi_dens_ed.real+chi_magn_ed.real)}')
print(f'True value: {twop.get_sum_chiupup(n)}')


# plt.figure()
# plt.semilogy(g2_dens.wn,np.abs(chi_magn.real-chi_magn_ed.real),'-o',color='cornflowerblue',label='Tilde-ED')
# plt.legend()
# plt.xlim(0,30)
# plt.show()


print(1/beta*np.sum(chi_dens.real+chi_magn.real))
print(1/beta*np.sum(chi_dens_ed.real+chi_magn_ed.real))
print(twop.get_sum_chiupup(n)*2)

siw_dens = lfp.schwinger_dyson_vrg(vrg_dens,chi_dens,green.g_loc,u)
siw_magn = lfp.schwinger_dyson_vrg(vrg_magn,chi_magn,green.g_loc,u)

n_shell = 800
niv_full = n_shell + niv_core
siw_sde = siw_dens+siw_magn
siw_shell_dens = lfp.schwinger_dyson_shell(chi_dens,green.g_loc,beta,u,n_shell=n_shell,n_core=niv_core,wn=g2_dens.wn)
siw_shell_magn = lfp.schwinger_dyson_shell(chi_magn,green.g_loc,beta,u,n_shell=n_shell,n_core=niv_core,wn=g2_dens.wn)
siw_shell = siw_shell_dens+siw_shell_magn

siw_sde_full = mf.concatenate_core_asmypt(siw_sde,siw_shell)
vn_full = mf.vn(niv_full)

siw_shell_dens_only = lfp.schwinger_dyson_shell(chi_dens,green.g_loc,beta,u,n_shell=niv_full,n_core=0,wn=g2_dens.wn)
siw_shell_magn_only = lfp.schwinger_dyson_shell(chi_magn,green.g_loc,beta,u,n_shell=niv_full,n_core=0,wn=g2_dens.wn)
siw_shell_only = siw_shell_dens_only+siw_shell_magn_only


plt.figure(dpi=500)
plt.plot(vn_full,siw_sde_full.imag,'-o',color='cornflowerblue',markeredgecolor='k',label='SDE',lw=4)
plt.plot(vn_full, green.sigma.get_siw(niv=niv_full)[0,0,0,:].imag,'-p',color='firebrick',label='Input',markeredgecolor='k',ms=2)
plt.plot(vn_full, siw_shell_only.imag,'-h',color='forestgreen',label='Asympt',markeredgecolor='k',ms=1)
plt.legend()
plt.xlim(0,40)
# plt.ylim(None,0)
plt.savefig(path+f'sde_vs_input_niv_{niv_core}.png')
plt.show()

plt.figure()
plt.semilogy(vn_full,np.abs(siw_sde_full.imag-green.sigma.get_siw(niv=niv_full)[0,0,0,:].imag),'-',color='cornflowerblue',markeredgecolor='k',label='SDE')
plt.vlines(vn_full[niv_full+niv_core],1e-4,1e-2,linestyles='dashed',color='k')
plt.xlim(0,n_shell)
plt.legend()
plt.show()

plt.figure()
plt.semilogy(vn_full,np.abs(siw_shell_only.imag-green.sigma.get_siw(niv=niv_full)[0,0,0,:].imag),'-',color='cornflowerblue',markeredgecolor='k',label='SDE')
plt.vlines(vn_full[niv_full+niv_core],1e-4,1e-2,linestyles='dashed',color='k')
plt.xlim(0,n_shell)
plt.legend()
plt.show()

#%%
plt.figure()
plt.semilogy(mf.cut_v_1d(g2_dens.vn,niv_core),np.abs(siw_sde.imag-mf.cut_v_1d(green.sigma.sigma_core[0,0,0,:].imag,niv_cut=niv_core)),'-o',color='cornflowerblue',markeredgecolor='k',label='SDE')
# plt.plot(mf.vn(green.sigma.niv_core), green.sigma.sigma_core[0,0,0,:].imag,'-o',color='firebrick',label='Input',markeredgecolor='k')
plt.legend()
plt.xlim(0,100)
plt.savefig(path+f'sde_minus_input_niv_{niv_core}.png')
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