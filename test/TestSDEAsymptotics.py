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

giw_file = w2dyn_aux.w2dyn_file(fname=path + '1p-data.hdf5')
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
niv = niw
wn = mf.wn(niw)

niv_shell = 5000
g2_magn = fp.LocalFourPoint(matrix=g2iw_file.read_g2(channel='magn', niv=niv, niw=niw), beta=beta, wn=wn, channel='magn')
g2_dens = fp.LocalFourPoint(matrix=g2iw_file.read_g2(channel='dens', niv=niv, niw=niw), beta=beta, wn=wn, channel='dens')
niv = 50
vn = mf.vn(niv)
g2_magn.cut_iv(niv_cut=niv)
g2_dens.cut_iv(niv_cut=niv)

chi_magn = fp.chir_from_g2(g2=g2_magn, giw=giw_obj)
chi_dens = fp.chir_from_g2(g2=g2_dens, giw=giw_obj)
gchi0_inv = fp.LocalBubble(wn=wn, giw=giw_obj, niv=niv,is_inv=True,do_shell=False)
F_magn = fp.Fob2_from_chir(chi_magn,gchi0_inv)
gamma_magn = fp.gamob2_from_chir(chi_magn,gchi0_inv)
niv_array = np.array([niv,100, 200, 500, 1000,20000])
colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(niv_array)))
gchi0 = []
lam_magn = []
chi_phys_magn = []
vrg_magn = []
siw_magn = []

lam_dens = []
chi_phys_dens = []
vrg_dens = []
siw_dens = []

siw_range = []
siw_range_v2 = []
siw_w_asympt = []
siw_magn_range = []
wn_2 = mf.wn(200)
for n in niv_array:
    gchi0.append(fp.LocalBubble(wn=wn, giw=giw_obj, niv=niv, niv_shell=n))

    # Magn:
    # lam_magn.append(fp.lam_from_chir_FisUr(chi_magn, gchi0[-1], u=u))
    # chi_phys_magn.append(fp.chi_phys_tilde_FisUr(chir=chi_magn, gchi0=gchi0[-1], lam=lam_magn[-1], u=u))
    lam_magn.append(fp.lam_from_chir(chi_magn, gchi0[-1], u=u))
    chi_phys_magn.append(fp.chi_phys_tilde(chir=chi_magn, gchi0=gchi0[-1], lam=lam_magn[-1], u=u))
    # lam_dens.append(fp.lam_from_chir_FisUr(chi_dens, gchi0[-1], u=u))
    # chi_phys_dens.append(fp.chi_phys_tilde_FisUr(chir=chi_dens, gchi0=gchi0[-1], lam=lam_dens[-1], u=u))
    lam_dens.append(fp.lam_from_chir(chi_dens, gchi0[-1], u=u))
    chi_phys_dens.append(fp.chi_phys_tilde(chir=chi_dens, gchi0=gchi0[-1], lam=lam_dens[-1], u=u))

    vrg_magn.append(fp.vrg_from_lam(chir=chi_phys_magn[-1],lam=lam_magn[-1],u=u))
    siw_magn.append(fp.schwinger_dyson_vrg(vrg=vrg_magn[-1],chir_phys=chi_phys_magn[-1],giw=giw_obj,u=u))
    vrg_dens.append(fp.vrg_from_lam(chir=chi_phys_dens[-1],lam=lam_dens[-1],u=u))
    siw_dens.append(fp.schwinger_dyson_vrg(vrg=vrg_dens[-1],chir_phys=chi_phys_dens[-1],giw=giw_obj,u=u))

    siw_range.append(siw_dens[-1]+siw_magn[-1]+hartree)

    # gchi0_w_asympt = fp.LocalBubble(wn=wn_2, giw=giw_obj, niv=niv, niv_shell=n)
    # tmp = fp.schwinger_dyson_w_asympt(gchi0 = gchi0_w_asympt,giw=giw_obj, u = u, niv = niv)
    # tmp = tmp - fp.schwinger_dyson_w_asympt(gchi0 = gchi0[-1],giw=giw_obj, u = u, niv = niv)
    # siw_w_asympt.append(tmp)

#
# for i,obj in enumerate(chi_phys_dens):
#     plt.plot(wn,obj.shell, '-o', markeredgecolor='k', color = colors[i], alpha = 0.7, label=f'niv = {niv_array[i]}')
# plt.legend()
# plt.show()
#
# for i,obj in enumerate(chi_phys_dens):
#     plt.plot(wn,obj.mat_tilde, '-o', markeredgecolor='k', color = colors[i], alpha = 0.7, label=f'niv = {niv_array[i]}')
# plt.legend()
# plt.show()
#
# for i,obj in enumerate(chi_phys_magn):
#     plt.plot(wn,obj.shell, '-o', markeredgecolor='k', color = colors[i], alpha = 0.7, label=f'niv = {niv_array[i]}')
# plt.legend()
# plt.show()

plt.figure(dpi=300)
vn_dmft = giw_obj.wn
plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
for i,siw in enumerate(siw_range):
    plt.plot(vn,siw.imag, '-+', markeredgecolor='k', color = colors[i], alpha = 0.7,ms=2)
plt.xlim(0,60)
plt.show()

plt.figure(dpi=300)
vn_dmft = giw_obj.wn
plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
for i,siw in enumerate(siw_magn):
    plt.plot(vn,2*siw.imag, '-+', markeredgecolor='k', color = colors[i], alpha = 0.7,ms=2)
plt.xlim(0,60)
plt.show()

plt.figure(dpi=300)
vn_dmft = giw_obj.wn
# plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
for i,siw in enumerate(siw_range):
    plt.semilogy(vn,np.abs(mf.cut_v_1d(siw_dmft.imag,niv)-siw.imag), '-o', markeredgecolor='k', color = colors[i], alpha = 0.7)
plt.xlim(0,60)
plt.show()

# plt.figure(dpi=300)
# vn_dmft = giw_obj.wn
# for i,siw in enumerate(siw_range):
#     plt.plot(vn,siw_w_asympt[i], '-+', markeredgecolor='k', color = colors[i], alpha = 0.7,ms=2)
# plt.xlim(0,60)
# plt.show()
#%%



# plt.figure()
# vn_dmft = giw_obj.wn
# plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# for i,siw in enumerate(siw_range):
#     plt.plot(vn,siw_dens[i].imag, '-o', markeredgecolor='k', color = colors[i], alpha = 0.7,ms=2)
#     plt.plot(vn,siw_magn[i].imag, '-o', markeredgecolor='k', color = colors[i], alpha = 0.7,ms=2)
# plt.xlim(0,60)
#
# plt.show()

#%%
# for i,chi0 in enumerate(gchi0):
#     plt.plot(wn,chi0.chi0_tilde, '-o', markeredgecolor='k', color = colors[i], alpha = 0.7, label=f'niv = {niv_array[i]}')
# plt.xlim(0,60)
# plt.legend()
# plt.show()
#
# for i,chi0 in enumerate(gchi0):
#     plt.plot(wn,chi0.shell, '-o', markeredgecolor='k', color = colors[i], alpha = 0.7, label=f'niv = {niv_array[i]}')
# plt.xlim(0,60)
# plt.legend()
# plt.show()
#
# for i,chi0 in enumerate(gchi0):
#     plt.plot(niv_array[i],chi0.chi0_tilde[niw], '-o', markeredgecolor='k', color = colors[i], alpha = 0.7, label=f'niv = {niv_array[i]}')
# plt.legend()
# plt.show()
#
# for i,chi0 in enumerate(gchi0):
#     plt.semilogy(niv_array[i],gchi0[-1].chi0_tilde[niw]-chi0.chi0_tilde[niw], '-o', markeredgecolor='k', color = colors[i], alpha = 0.7, label=f'niv = {niv_array[i]}')
# plt.legend()
# plt.show()
#
# for i,chi0 in enumerate(gchi0):
#     plt.plot(niv_array[i],chi0.shell[niw], '-o', markeredgecolor='k', color = colors[i], alpha = 0.7, label=f'niv = {niv_array[i]}')
# plt.legend()
# plt.show()

#%% Old style SDE:
import SDE as sde

niv_core = niv
niv_urange = niv
ind_comp = np.where(niv_urange == niv_array)[0][0]
iw = wn
# Extract gamma:
chi0_urange = fp.LocalBubble(giw=giw_obj, niv=niv_urange, wn=wn)
chi0_core = fp.LocalBubble(giw=giw_obj, niv=niv_core, wn=wn)
gamma_dens = fp.gammar_from_gchir(gchir=chi_dens, gchi0_urange=chi0_urange, u=u)
gamma_magn = fp.gammar_from_gchir(gchir=chi_magn, gchi0_urange=chi0_urange, u=u)

gamma_dens.cut_iv(niv_cut=niv_core)
gamma_magn.cut_iv(niv_cut=niv_core)

gchi_aux_dens_loc = fp.local_gchi_aux_from_gammar(gammar=gamma_dens, gchi0_core=chi0_core, u=u)
gchi_aux_magn_loc = fp.local_gchi_aux_from_gammar(gammar=gamma_magn, gchi0_core=chi0_core, u=u)

chi_aux_dens_loc = fp.local_susceptibility_from_four_point(four_point=gchi_aux_dens_loc, chi0_urange=chi0_urange)
chi_aux_magn_loc = fp.local_susceptibility_from_four_point(four_point=gchi_aux_magn_loc, chi0_urange=chi0_urange)

chi_dens_urange_loc = fp.local_chi_phys_from_chi_aux(chi_aux=chi_aux_dens_loc, chi0_urange=chi0_urange,
                                                     chi0_core=chi0_core,
                                                     u=u)

chi_magn_urange_loc = fp.local_chi_phys_from_chi_aux(chi_aux=chi_aux_magn_loc, chi0_urange=chi0_urange,
                                                     chi0_core=chi0_core,
                                                     u=u)

vrg_dens_loc = fp.local_fermi_bose_from_chi_aux_urange(gchi_aux=gchi_aux_dens_loc, gchi0=chi0_core,
                                                       niv_urange=niv_urange)

vrg_magn_loc = fp.local_fermi_bose_from_chi_aux_urange(gchi_aux=gchi_aux_magn_loc, gchi0=chi0_core,
                                                       niv_urange=niv_urange)

siw_dens = sde.local_dmft_sde(vrg=vrg_dens_loc.mat, chir=chi_dens_urange_loc, giw=giw, u=u)
siw_magn = sde.local_dmft_sde(vrg=vrg_magn_loc.mat, chir=chi_magn_urange_loc, giw=giw, u=u)
siw_old = siw_dens + siw_magn + hartree


# plt.figure()
# vn_dmft = giw_obj.wn
# vn_urange = mf.vn(niv_urange)
# plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# plt.plot(vn_urange,siw_old.imag, '-o', markeredgecolor='k', color = colors[i], alpha = 0.7)
# plt.xlim(0,60)
#
# plt.show()

#%%
chi_magn_core = chi_magn.contract_legs()

plt.figure()
plt.plot(wn,chi_phys_magn[ind_comp].mat_tilde, '-o', markeredgecolor='k', color='cornflowerblue', alpha=0.7,label='New')
plt.plot(wn,chi_magn_urange_loc.mat, '-o', markeredgecolor='k', color='firebrick', alpha=0.7,label='Old')
plt.legend()
plt.show()

plt.figure()
plt.plot(wn,chi_phys_dens[ind_comp].mat_tilde, '-o', markeredgecolor='k', color='cornflowerblue', alpha=0.7,label='New')
plt.plot(wn,chi_dens_urange_loc.mat, '-o', markeredgecolor='k', color='firebrick', alpha=0.7,label='Old')
plt.legend()
plt.show()

# plt.figure()
# plt.plot(wn,chi_phys_magn[0].mat_tilde, '-o', markeredgecolor='k', color='forestgreen', alpha=0.7)
# plt.plot(wn,chi_magn_core.mat, '-o', markeredgecolor='k', color='goldenrod', alpha=0.7)
# plt.show()

#%%

# fig, axes = plt.subplots(ncols=3,figsize=(10,4))
#
# im1 = axes[0].pcolormesh(vrg_magn[-1].vn,vrg_magn[-1].wn,mf.cut_v(arr=vrg_magn_loc.mat.real,niv_cut=vrg_magn[-1].niv,axes=(1,)), cmap='RdBu')
# im2 = axes[1].pcolormesh(vrg_magn[-1].vn,vrg_magn_loc.wn,vrg_magn[ind_comp].mat.real, cmap='RdBu')
# im3 = axes[2].pcolormesh(vrg_magn[-1].vn,vrg_magn_loc.wn,vrg_magn[ind_comp].mat.real-mf.cut_v(arr=vrg_magn_loc.mat.real,niv_cut=vrg_magn[-1].niv,axes=(1,)), cmap='RdBu')
# fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location='right',
#              pad=0.05)
# fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location='right',
#              pad=0.05)
# fig.colorbar(im3, ax=(axes[2]), aspect=15, fraction=0.08, location='right',
#              pad=0.05)
# plt.tight_layout()
# plt.show()

#%%

plt.figure()
vn_dmft = giw_obj.wn
vn_urange = mf.vn(niv_urange)
plt.plot(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
plt.plot(vn_urange,siw_old.imag, '-o', markeredgecolor='k', color ='firebrick', alpha = 0.7)
plt.plot(vn,siw_range[ind_comp].imag, '-o', markeredgecolor='k', color = 'cornflowerblue', alpha = 0.7)
plt.xlim(0,60)
plt.savefig('./TestPlots/OldLocalEomRoutine.png')
plt.show()

# plt.figure()
# vn_dmft = giw_obj.wn
# vn_urange = mf.vn(niv_urange)
# plt.loglog(vn_dmft, siw_dmft.imag, '-o', markeredgecolor='k', color='k', alpha=0.7)
# plt.plot(vn_urange,siw_range[ind_comp].imag-mf.cut_v_1d(siw_old.imag), '-o', markeredgecolor='k', color ='firebrick', alpha = 0.7)
# plt.plot(vn,siw_range[ind_comp].imag, '-o', markeredgecolor='k', color = 'cornflowerblue', alpha = 0.7)
# plt.xlim(0,60)
#
# plt.show()











