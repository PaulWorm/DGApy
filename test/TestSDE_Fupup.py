import numpy as np
import matplotlib.pyplot as plt
import FourPoint as fp
import TwoPoint_old as tp
import w2dyn_aux_dga
import MatsubaraFrequencies as mf


path = './2DSquare_U8_tp-0.25_tpp0.12_beta12.5_n0.85/'
path = './2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.85/'
path = './2DSquare_U8_tp-0.2_tpp0.1_beta30_n0.85/'
path = './2DSquare_U8_tp-0.2_tpp0.1_beta17.5_n0.90/'
path = './2DSquare_U8_tp-0.25_tpp0.12_beta12.5_n0.85/'
# path = './2DSquare_U8_tp-0.2_tpp0.1_beta2_n0.70/'

giw_file = w2dyn_aux_dga.w2dyn_file(fname=path + '1p-data.hdf5')
beta = giw_file.get_beta()
u = giw_file.get_udd()
totdens = giw_file.get_totdens()
mu = giw_file.get_mu()
hartree = u*totdens/2
giw = giw_file.get_giw()[0, 0, :]
siw_dmft = giw_file.get_siw()[0, 0, :]
giw_obj = tp.LocalGreensFunction(mat=giw, beta=beta, hf_denom=mu - u * totdens / 2)
g2iw_file = w2dyn_aux_dga.g4iw_file(fname=path + 'g4iw_sym.hdf5')
niw = g2iw_file.get_niw(channel='dens')
niv = niw + 1
# niv = niw
wn = mf.wn(niw)

niv_shell = 5000
g2_magn = fp.LocalFourPoint(matrix=g2iw_file.read_g2(channel='magn', niv=niv, niw=niw), beta=beta, wn=wn, channel='magn')
g2_dens = fp.LocalFourPoint(matrix=g2iw_file.read_g2(channel='dens', niv=niv, niw=niw), beta=beta, wn=wn, channel='dens')
niv = 60
vn = mf.vn(niv)
g2_magn.cut_iv(niv_cut=niv)
g2_dens.cut_iv(niv_cut=niv)

chi_magn = fp.chir_from_g2(g2=g2_magn, giw=giw_obj)
chi_dens = fp.chir_from_g2(g2=g2_dens, giw=giw_obj)
gchi0_inv = fp.BubbleGenerator(wn=wn, giw=giw_obj, niv=niv, is_inv=True, do_shell=False)
F_magn = fp.Fob2_from_chir(chi_magn,gchi0_inv)
F_dens = fp.Fob2_from_chir(chi_dens,gchi0_inv)
F_upup = fp.LocalFourPoint(matrix=0.5*(F_dens.mat + F_magn.mat), channel='upup', beta=F_magn.beta, wn=F_magn.wn)

gchi0 = fp.BubbleGenerator(wn=wn, giw=giw_obj, niv=niv, is_inv=False, do_shell=True)
siw_upup = fp.schwinger_dyson_F(F_upup,gchi0,giw=giw_obj.mat,u=u,totdens=0) # avoid Hartree contribution by setting totdens = 0
siw_magn = fp.schwinger_dyson_F(F_magn,gchi0,giw=giw_obj.mat,u=u,totdens=0) # avoid Hartree contribution by setting totdens = 0
siw_dens = fp.schwinger_dyson_F(F_dens,gchi0,giw=giw_obj.mat,u=u,totdens=0) # avoid Hartree contribution by setting totdens = 0

# get vrg:

lam_magn = fp.lam_from_chir(chi_magn, gchi0, u=u)
chi_phys_magn = fp.chi_phys_tilde(chir=chi_magn, gchi0=gchi0, lam=lam_magn, u=u)
lam_dens = fp.lam_from_chir(chi_dens, gchi0, u=u)
chi_phys_dens = fp.chi_phys_tilde(chir=chi_dens, gchi0=gchi0, lam=lam_dens, u=u)
chi_phys_updo = fp.subtract_chi(chi_phys_dens,chi_phys_magn)
chi_phys_updo.mat = 0.5 * chi_phys_updo.mat
chi_phys_upup = fp.add_chi(chi_phys_dens,chi_phys_magn)
chi_phys_upup.mat = 0.5 * chi_phys_upup.mat
vrg_dens = fp.vrg_from_lam(chir=chi_phys_dens,lam=lam_dens,u=u)
vrg_magn = fp.vrg_from_lam(chir=chi_phys_magn,lam=lam_magn,u=u)


vrg_upup = fp.add_vrg(vrg_dens,vrg_magn)
vrg_upup._mat = 0.5 * vrg_upup.mat
vrg_updo = fp.subtract_vrg(vrg_dens,vrg_magn)
vrg_updo._mat = 0.5 * vrg_updo.mat


# get sigma:
siw_magn_vrg = fp.schwinger_dyson_vrg(vrg=vrg_magn,chir_phys=chi_phys_magn,giw=giw_obj,u=u,do_tilde=False)
siw_dens_vrg = fp.schwinger_dyson_vrg(vrg=vrg_dens,chir_phys=chi_phys_dens,giw=giw_obj,u=u,do_tilde=False)
siw_vrg_updo_chi_updo = fp.schwinger_dyson_vrg(vrg=vrg_updo,chir_phys=chi_phys_updo,giw=giw_obj,u=u,do_tilde=False,scalfac=0,scalfac_2=0)
siw_vrg_upup_chi_updo = fp.schwinger_dyson_vrg(vrg=vrg_upup,chir_phys=chi_phys_updo,giw=giw_obj,u=u,do_tilde=False,scalfac=0,scalfac_2=0)
siw_vrg_magn_chi_upup = fp.schwinger_dyson_vrg(vrg=vrg_magn,chir_phys=chi_phys_upup,giw=giw_obj,u=u,do_tilde=False,scalfac=1)

siw_vrg_upup_chi_updo = fp.schwinger_dyson_vrg(vrg=vrg_upup,chir_phys=chi_phys_updo,giw=giw_obj,u=u,do_tilde=False,scalfac=1,scalfac_2=1,sign=+1)
siw_vrg_updo_chi_upup = fp.schwinger_dyson_vrg(vrg=vrg_updo,chir_phys=chi_phys_upup,giw=giw_obj,u=u,do_tilde=False,scalfac=0,scalfac_2=0,sign=+1)



plt.figure()
plt.plot(vn,-siw_upup.imag,'-o',color='cornflowerblue',markeredgecolor='k',label=r'$F_{uu}$')
plt.plot(vn,-2*(siw_vrg_upup_chi_updo.imag+siw_vrg_updo_chi_upup.imag),'-o',color='firebrick',markeredgecolor='k')
# plt.plot(vn,siw_vrg_updo_chi_upup.imag,'-o',color='firebrick',markeredgecolor='k')
# plt.plot(vn,siw_vrg_updo_chi_updo.imag,'-o',color='navy',markeredgecolor='k')
# plt.plot(vn,-siw_vrg_upup_chi_updo.imag,'-o',color='blue',markeredgecolor='k')
# plt.plot(vn,2*(siw_vrg_magn_chi_upup.imag+siw_vrg_updo_chi_updo.imag-siw_vrg_upup_chi_updo.imag),'-o',color='k',markeredgecolor='k',ms=10)
# plt.plot(vn,(siw_vrg_upup.imag-siw_vrg_updo.imag),'-o',color='k',markeredgecolor='k')
# plt.plot(vn,-siw_magn.imag,'-o',color='firebrick',markeredgecolor='k',label=r'$F_{m}$')
# plt.plot(vn,siw_dens.imag,'-o',color='seagreen',markeredgecolor='k',label=r'$F_{d}$')
plt.legend()
#
# plt.plot(vn,2*siw_magn_vrg.imag,'-h',color='crimson',markeredgecolor='k')
# plt.plot(vn,2*siw_dens_vrg.imag,'-h',color='forestgreen',markeredgecolor='k')
# plt.plot(vn,1/2*siw_dens_vrg.imag+3/2*siw_magn_vrg.imag,'-h',color='forestgreen',markeredgecolor='k')
plt.xlim(0,60)
plt.savefig('SDE_tests.png')
# plt.ylim(None,0)
plt.show()

#
# #%%
# plt.figure()
# plt.plot(vn,-siw_magn.imag,'-o',color='firebrick',markeredgecolor='k',ms=10)
# plt.plot(vn,0.5*(siw_dens.imag-siw_magn.imag),'-o',color='seagreen',markeredgecolor='k')
# plt.plot(vn,(siw_magn_vrg.imag*2),'-o',color='k',markeredgecolor='k')
# plt.xlim(0,60)
# plt.show()

# plt.figure()
# plt.pcolormesh(vrg_upup.mat.real,cmap='RdBu')
# plt.colorbar()
# plt.show()



# gamma_magn = fp.gamob2_from_chir(chi_magn,gchi0_inv)
# niv_array = np.array([niv,100, 200, 500, 1000,20000])
# niv_array = np.array([niv,200,500,1000,20000])
# colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(niv_array)))
# gchi0 = []
# lam_magn = []
# chi_phys_magn = []
# chi_phys_updo = []
# vrg_magn = []
# siw_magn = []
#
# lam_dens = []
# chi_phys_dens = []
# vrg_dens = []
# vrg_upup = []
# vrg_updo = []
# siw_dens = []
# siw_updo = []
#
# siw_range = []
# siw_range_v2 = []
# siw_w_asympt = []
# siw_magn_range = []
# wn_2 = mf.wn(200)
# for n in niv_array:
#     gchi0.append(fp.LocalBubble(wn=wn, giw=giw_obj, niv=niv, niv_shell=n))
#
#     # Magn:
#     # lam_magn.append(fp.lam_from_chir_FisUr(chi_magn, gchi0[-1], u=u))
#     # chi_phys_magn.append(fp.chi_phys_tilde_FisUr(chir=chi_magn, gchi0=gchi0[-1], lam=lam_magn[-1], u=u))
#     lam_magn.append(fp.lam_from_chir(chi_magn, gchi0[-1], u=u))
#     chi_phys_magn.append(fp.chi_phys_tilde(chir=chi_magn, gchi0=gchi0[-1], lam=lam_magn[-1], u=u))
#     # lam_dens.append(fp.lam_from_chir_FisUr(chi_dens, gchi0[-1], u=u))
#     # chi_phys_dens.append(fp.chi_phys_tilde_FisUr(chir=chi_dens, gchi0=gchi0[-1], lam=lam_dens[-1], u=u))
#     lam_dens.append(fp.lam_from_chir(chi_dens, gchi0[-1], u=u))
#     chi_phys_dens.append(fp.chi_phys_tilde(chir=chi_dens, gchi0=gchi0[-1], lam=lam_dens[-1], u=u))
#     chi_phys_updo.append(fp.subtract_chi(chi_phys_dens[-1],chi_phys_magn[-1]))
#     vrg_magn.append(fp.vrg_from_lam(chir=chi_phys_magn[-1],lam=lam_magn[-1],u=u))
#     siw_magn.append(fp.schwinger_dyson_vrg(vrg=vrg_magn[-1],chir_phys=chi_phys_magn[-1],giw=giw_obj,u=u,do_tilde=False))
#     vrg_dens.append(fp.vrg_from_lam(chir=chi_phys_dens[-1],lam=lam_dens[-1],u=u))
#     vrg_updo.append(fp.subtract_vrg(vrg_dens[-1], vrg_magn[-1]))
#     vrg_upup.append(fp.add_vrg(vrg_dens[-1], vrg_magn[-1]))
#     siw_dens.append(fp.schwinger_dyson_vrg(vrg=vrg_dens[-1],chir_phys=chi_phys_dens[-1],giw=giw_obj,u=u,do_tilde=False))
#     # siw_updo.append(fp.schwinger_dyson_vrg_updo(vrg=vrg_magn[-1],chir_phys=chi_phys_updo[-1],giw=giw_obj,u=u,do_tilde=False))
#     siw_updo.append(fp.schwinger_dyson_vrg_updo(vrg=vrg_upup[-1],chir_phys=chi_phys_magn[-1],giw=giw_obj,u=u,do_tilde=False))
#
#     siw_range.append(0.5*siw_dens[-1]+1.5*siw_magn[-1]+hartree)
#
# plt.figure(dpi=300)
# vn_dmft = giw_obj.wn
# plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# for i,siw in enumerate(siw_range):
#     plt.plot(vn,siw.imag, '-+', markeredgecolor='k', color = colors[i], alpha = 0.7,ms=2)
# plt.xlim(0,60)
# plt.show()
#
# plt.figure(dpi=300)
# vn_dmft = giw_obj.wn
# plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# for i,siw in enumerate(siw_magn):
#     plt.plot(vn,2*siw.imag, '-+', markeredgecolor='k', color = colors[i], alpha = 0.7,ms=2)
# plt.xlim(0,60)
# plt.show()
#
# plt.figure(dpi=300)
# vn_dmft = giw_obj.wn
# # plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# for i,siw in enumerate(siw_range):
#     plt.semilogy(vn,np.abs(mf.cut_v_1d(siw_dmft.imag,niv)-siw.imag), '-o', markeredgecolor='k', color = colors[i], alpha = 0.7)
# plt.xlim(0,60)
# plt.show()
#
# plt.figure(dpi=300)
# vn_dmft = giw_obj.wn
# # plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# for i,siw in enumerate(siw_range):
#     plt.semilogy(vn,np.abs(2*siw_magn[i].imag-siw.imag), '-o', markeredgecolor='k', color = colors[i], alpha = 0.7)
# plt.xlim(0,60)
# plt.show()
#
# ##%%
# plt.figure(dpi=300)
# vn_dmft = giw_obj.wn
# # plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# plt.plot([1/beta*np.sum(chi_phys_dens[i].mat - chi_phys_magn[i].mat) for i in range(len(niv_array))], '-o', markeredgecolor='k', color = colors[0], alpha = 0.7)
# plt.xlim(0,60)
# plt.show()
#
# #%%
# fun = lambda a,x: a/x
# x = np.linspace(0.1,30,1000)
#
# plt.figure(dpi=300)
# vn_dmft = giw_obj.wn
# # plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# for i,siw in enumerate(siw_range):
#     plt.semilogy(vn,np.abs(siw_updo[i].imag), '-o', markeredgecolor='k', color = colors[i], alpha = 0.7)
# plt.semilogy(x,fun(0.3,x), '-', markeredgecolor='k', color = 'k', alpha = 0.7)
# plt.xlim(0,60)
# plt.show()
#
# #%%
# plt.figure(dpi=300)
# vn_dmft = giw_obj.wn
# # plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# plt.semilogy(vn,2*np.abs(siw_magn[-1].imag), '-o', markeredgecolor='k', color = 'firebrick', alpha = 0.7)
# plt.semilogy(vn,np.abs(siw_range[-1].imag), '-o', markeredgecolor='k', color = 'seagreen', alpha = 0.7)
# plt.semilogy(vn,np.abs(siw_updo[-1].imag), '-o', markeredgecolor='k', color = 'goldenrod', alpha = 0.7)
# plt.semilogy(vn,np.abs(2*siw_magn[-1]-0.5*siw_updo[-1].imag), '-o', markeredgecolor='k', color = 'indigo', alpha = 0.7)
# vn_dmft = mf.vn(500)
# plt.semilogy(vn_dmft,np.abs(mf.cut_v_1d(siw_dmft.imag,500)), '-o', markeredgecolor='k', color = 'cornflowerblue', alpha = 0.7)
# asympt_prev = u**2/2*totdens*(1-totdens/2)/np.pi*beta
#
# plt.semilogy(vn,fun(asympt_prev,vn*2+1), '-', markeredgecolor='k', color = 'k', alpha = 0.7)
# plt.xlim(0,60)
# plt.show()

# #%%
# asympt_prev = u**2/2*totdens*(1-totdens/2)
# vn_dmft = mf.v(beta,1000)
#
# # plt.figure()
# # plt.plot(vn_dmft,np.abs(np.abs(mf.cut_v_1d(siw_dmft.imag,1000))), '-o', markeredgecolor='k', color = 'cornflowerblue', alpha = 0.7)
# # plt.plot(vn_dmft,np.abs(fun(asympt_prev,vn_dmft)), '-o', markeredgecolor='k', color = 'firebrick', alpha = 0.7)
# # plt.xlim(0,60)
# # plt.show()
#
# plt.figure()
# plt.semilogy(vn_dmft,np.abs(np.abs(mf.cut_v_1d(siw_dmft.imag,1000))-fun(asympt_prev,vn_dmft)), '-o', markeredgecolor='k', color = 'cornflowerblue', alpha = 0.7)
# plt.xlim(0,60)
# plt.show()
#
# #%% Old style SDE:
# import SDE as sde
#
# niv_core = niv
# niv_urange = 200
# ind_comp = np.where(niv_urange == niv_array)[0][0]
# iw = wn
# # Extract gamma:
# chi0_urange = fp.LocalBubble(giw=giw_obj, niv=niv_urange, wn=wn)
# chi0_core = fp.LocalBubble(giw=giw_obj, niv=niv_core, wn=wn)
# gamma_dens = fp.gammar_from_gchir(gchir=chi_dens, gchi0_urange=chi0_urange, u=u)
# gamma_magn = fp.gammar_from_gchir(gchir=chi_magn, gchi0_urange=chi0_urange, u=u)
#
# gamma_dens.cut_iv(niv_cut=niv_core)
# gamma_magn.cut_iv(niv_cut=niv_core)
#
# gchi_aux_dens_loc = fp.local_gchi_aux_from_gammar(gammar=gamma_dens, gchi0_core=chi0_core, u=u)
# gchi_aux_magn_loc = fp.local_gchi_aux_from_gammar(gammar=gamma_magn, gchi0_core=chi0_core, u=u)
#
# chi_aux_dens_loc = fp.local_susceptibility_from_four_point(four_point=gchi_aux_dens_loc, chi0_urange=chi0_urange)
# chi_aux_magn_loc = fp.local_susceptibility_from_four_point(four_point=gchi_aux_magn_loc, chi0_urange=chi0_urange)
#
# chi_dens_urange_loc = fp.local_chi_phys_from_chi_aux(chi_aux=chi_aux_dens_loc, chi0_urange=chi0_urange,
#                                                      chi0_core=chi0_core,
#                                                      u=u)
#
# chi_magn_urange_loc = fp.local_chi_phys_from_chi_aux(chi_aux=chi_aux_magn_loc, chi0_urange=chi0_urange,
#                                                      chi0_core=chi0_core,
#                                                      u=u)
#
# vrg_dens_loc = fp.local_fermi_bose_from_chi_aux_urange(gchi_aux=gchi_aux_dens_loc, gchi0=chi0_core,
#                                                        niv_urange=niv_urange)
#
# vrg_magn_loc = fp.local_fermi_bose_from_chi_aux_urange(gchi_aux=gchi_aux_magn_loc, gchi0=chi0_core,
#                                                        niv_urange=niv_urange)
#
# siw_dens_old = sde.local_dmft_sde(vrg=vrg_dens_loc.mat, chir=chi_dens_urange_loc, giw=giw, u=u)
# siw_magn_old = sde.local_dmft_sde(vrg=vrg_magn_loc.mat, chir=chi_magn_urange_loc, giw=giw, u=u)
# siw_old = siw_dens_old + siw_magn_old + hartree
#
#
# # plt.figure()
# # vn_dmft = giw_obj.wn
# # vn_urange = mf.vn(niv_urange)
# # plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# # plt.plot(vn_urange,siw_old.imag, '-o', markeredgecolor='k', color = colors[i], alpha = 0.7)
# # plt.xlim(0,60)
# #
# # plt.show()
#
# #%%
# chi_magn_core = chi_magn.contract_legs()
#
# plt.figure()
# plt.plot(wn,chi_phys_magn[ind_comp].mat_tilde, '-o', markeredgecolor='k', color='cornflowerblue', alpha=0.7,label='New')
# plt.plot(wn,chi_magn_urange_loc.mat, '-o', markeredgecolor='k', color='firebrick', alpha=0.7,label='Old')
# plt.legend()
# plt.show()
#
# plt.figure()
# plt.plot(wn,chi_phys_dens[ind_comp].mat_tilde, '-o', markeredgecolor='k', color='cornflowerblue', alpha=0.7,label='New')
# plt.plot(wn,chi_dens_urange_loc.mat, '-o', markeredgecolor='k', color='firebrick', alpha=0.7,label='Old')
# plt.legend()
# plt.show()
#
# # plt.figure()
# # plt.plot(wn,chi_phys_magn[0].mat_tilde, '-o', markeredgecolor='k', color='forestgreen', alpha=0.7)
# # plt.plot(wn,chi_magn_core.mat, '-o', markeredgecolor='k', color='goldenrod', alpha=0.7)
# # plt.show()
#
# #%%
#
# # fig, axes = plt.subplots(ncols=3,figsize=(10,4))
# #
# # im1 = axes[0].pcolormesh(vrg_magn[-1].vn,vrg_magn[-1].wn,mf.cut_v(arr=vrg_magn_loc.mat.real,niv_cut=vrg_magn[-1].niv,axes=(1,)), cmap='RdBu')
# # im2 = axes[1].pcolormesh(vrg_magn[-1].vn,vrg_magn_loc.wn,vrg_magn[ind_comp].mat.real, cmap='RdBu')
# # im3 = axes[2].pcolormesh(vrg_magn[-1].vn,vrg_magn_loc.wn,vrg_magn[ind_comp].mat.real-mf.cut_v(arr=vrg_magn_loc.mat.real,niv_cut=vrg_magn[-1].niv,axes=(1,)), cmap='RdBu')
# # fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location='right',
# #              pad=0.05)
# # fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location='right',
# #              pad=0.05)
# # fig.colorbar(im3, ax=(axes[2]), aspect=15, fraction=0.08, location='right',
# #              pad=0.05)
# # plt.tight_layout()
# # plt.show()
#
# #%%
#
# plt.figure()
# vn_dmft = giw_obj.wn
# vn_urange = mf.vn(niv_urange)
# plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# plt.plot(vn_urange,siw_old.imag, '-o', markeredgecolor='k', color ='firebrick', alpha = 0.7)
# plt.plot(vn,siw_range[ind_comp].imag, '-o', markeredgecolor='k', color = 'cornflowerblue', alpha = 0.7)
# plt.xlim(0,60)
#
# plt.show()
#
# plt.figure()
# vn_dmft = giw_obj.wn
# vn_urange = mf.vn(niv_urange)
# plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# plt.plot(vn_urange,siw_old.imag, '-o', markeredgecolor='k', color ='seagreen', alpha = 0.7)
# plt.plot(vn_urange,2*siw_magn_old.imag, '-o', markeredgecolor='k', color ='firebrick', alpha = 0.7)
# plt.plot(vn,2*siw_magn[ind_comp].imag, '-o', markeredgecolor='k', color = 'cornflowerblue', alpha = 0.7)
# plt.xlim(0,60)
#
# plt.show()
#
# # plt.figure()
# # vn_dmft = giw_obj.wn
# # vn_urange = mf.vn(niv_urange)
# # plt.loglog(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# # plt.plot(vn_urange,siw_range[ind_comp].imag-mf.cut_v_1d(siw_old.imag), '-o', markeredgecolor='k', color ='firebrick', alpha = 0.7)
# # plt.plot(vn,siw_range[ind_comp].imag, '-o', markeredgecolor='k', color = 'cornflowerblue', alpha = 0.7)
# # plt.xlim(0,60)
# #
# # plt.show()
#
#
#
#
#
#
#
#
#
#
#
