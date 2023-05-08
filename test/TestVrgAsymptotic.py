import numpy as np
import matplotlib.pyplot as plt
import FourPoint as fp
import TwoPoint_old as tp
import w2dyn_aux_dga
import MatsubaraFrequencies as mf

path = './2DSquare_U8_tp-0.25_tpp0.12_beta25_n0.85/'

giw_file = w2dyn_aux.w2dyn_file(fname=path + '1p-data.hdf5')
beta = giw_file.get_beta()
u = giw_file.get_udd()
totdens = giw_file.get_totdens()
mu = giw_file.get_mu()
giw = giw_file.get_giw()[0, 0, :]
giw_obj = tp.LocalGreensFunction(mat=giw, beta=beta, hf_denom=mu - u * totdens / 2)
g2iw_file = w2dyn_aux.g4iw_file(fname=path + 'g4iw_sym.hdf5')
niw = g2iw_file.get_niw(channel='dens')
niv = niw + 1
wn = mf.wn(niw)
niv_shell = 5000
g2_magn = fp.LocalFourPoint(matrix=g2iw_file.read_g2(channel='magn', niv=niv, niw=niw), beta=beta, wn=wn, channel='magn')
g2_dens = fp.LocalFourPoint(matrix=g2iw_file.read_g2(channel='dens', niv=niv, niw=niw), beta=beta, wn=wn, channel='dens')
chi_magn = fp.chir_from_g2(g2=g2_magn, giw=giw_obj)
chi_dens = fp.chir_from_g2(g2=g2_dens, giw=giw_obj)
gchi0_inv = fp.LocalBubble(wn=wn, giw=giw_obj, niv=niv,is_inv=True,do_shell=False)
F_magn = fp.Fob2_from_chir(chi_magn,gchi0_inv)
gamma_magn = fp.gamob2_from_chir(chi_magn,gchi0_inv)
gamma_dens = fp.gamob2_from_chir(chi_dens,gchi0_inv)
niv_array = [100, 200, 500, 1000,2000,4000,8000]

gchi0 = []
lam_magn = []
chi_phys_magn = []
vrg_magn = []

lam_dens = []
chi_phys_dens = []
vrg_dens = []
for n in niv_array:
    gchi0.append(fp.LocalBubble(wn=wn, giw=giw_obj, niv=niv, niv_shell=n))
    lam_magn.append(fp.lam_from_chir(chi_magn, gchi0[-1], u=u))
    chi_phys_magn.append(fp.chi_phys_tilde(chir=chi_magn, gchi0=gchi0[-1], lam=lam_magn[-1], u=u))
    vrg_magn.append(fp.vrg_from_lam(chir=chi_phys_magn[-1],lam=lam_magn[-1],u=u))

    lam_dens.append(fp.lam_from_chir(chi_dens, gchi0[-1], u=u))
    chi_phys_dens.append(fp.chi_phys_tilde(chir=chi_dens, gchi0=gchi0[-1], lam=lam_dens[-1], u=u))
    vrg_dens.append(fp.vrg_from_lam(chir=chi_phys_dens[-1],lam=lam_dens[-1],u=u))

vrg_magn_1 = fp.vrg_from_gam(gam=gamma_magn,chi0_inv=gchi0_inv,u=u)
vrg_magn_2 = fp.vrg_from_lam(chir=chi_phys_magn[0],lam=lam_magn[0],u=u, do_tilde=False)

vrg_dens_1 = fp.vrg_from_gam(gam=gamma_dens,chi0_inv=gchi0_inv,u=u)
vrg_dens_2 = fp.vrg_from_lam(chir=chi_phys_dens[0],lam=lam_dens[0],u=u, do_tilde=False)

#%%
fig, axes = plt.subplots(ncols=3,figsize=(10,4))

im1 = axes[0].pcolormesh(vrg_magn_1.vn,vrg_magn_1.wn,vrg_magn_1.mat.real, cmap='RdBu')
im2 = axes[1].pcolormesh(vrg_magn_2.vn,vrg_magn_2.wn,vrg_magn_2.mat.real, cmap='RdBu')
im3 = axes[2].pcolormesh(vrg_magn_2.vn,vrg_magn_2.wn,vrg_magn_2.mat.real-vrg_magn_1.mat.real, cmap='RdBu')
fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location='right',
             pad=0.05)
fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location='right',
             pad=0.05)
fig.colorbar(im3, ax=(axes[2]), aspect=15, fraction=0.08, location='right',
             pad=0.05)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(ncols=3,figsize=(10,4))

im1 = axes[0].pcolormesh(vrg_magn_1.vn,vrg_magn_1.wn,vrg_dens_1.mat.real, cmap='RdBu')
im2 = axes[1].pcolormesh(vrg_magn_2.vn,vrg_magn_2.wn,vrg_dens_2.mat.real, cmap='RdBu')
im3 = axes[2].pcolormesh(vrg_magn_2.vn,vrg_magn_2.wn,vrg_dens_2.mat.real-vrg_dens_1.mat.real, cmap='RdBu')
fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location='right',
             pad=0.05)
fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location='right',
             pad=0.05)
fig.colorbar(im3, ax=(axes[2]), aspect=15, fraction=0.08, location='right',
             pad=0.05)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(ncols=3,figsize=(10,4))

im1 = axes[0].pcolormesh(vrg_magn_1.vn,vrg_magn_1.wn,lam_magn[0].mat.real, cmap='RdBu')
im2 = axes[1].pcolormesh(vrg_magn_2.vn,vrg_magn_2.wn,lam_magn[-1].mat.real, cmap='RdBu')
im3 = axes[2].pcolormesh(vrg_magn_2.vn,vrg_magn_2.wn,lam_magn[-1].mat_tilde.real, cmap='RdBu')
fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location='right',
             pad=0.05)
fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location='right',
             pad=0.05)
fig.colorbar(im3, ax=(axes[2]), aspect=15, fraction=0.08, location='right',
             pad=0.05)
plt.tight_layout()
plt.show()

#%%
colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(niv_array)))
# plt.figure()
#
# plt.plot(chi_phys_magn[0].wn,chi_phys_magn[0].mat_tilde,'-o', color='k', markeredgecolor='k', alpha=0.7)
# for i, chi_phys in enumerate(chi_phys_magn):
#     plt.plot(chi_phys.wn,chi_phys.mat_tilde,'-o', color=colors[i], markeredgecolor='k', alpha=0.7)
# plt.xlim(0,20)
# plt.show()

# plt.figure()
# for i, chi_phys in enumerate(chi_phys_magn):
#     plt.plot(i,chi_phys.mat_tilde[niw].real,'-o', color=colors[i], markeredgecolor='k', alpha=0.7)
# plt.show()

plt.figure()
for i, chi0 in enumerate(chi_phys_magn):
    plt.plot(i,chi0.mat_tilde[niw].real,'-o', color=colors[i], markeredgecolor='k', alpha=0.7)
    plt.plot(i,chi0.mat[niw].real,'-o', color=colors[i], markeredgecolor='k', alpha=0.7)
plt.show()

plt.figure()
for i, chi0 in enumerate(chi_phys_magn):
    plt.plot(chi0.wn,chi0.mat_tilde.real,'-o', color=colors[i], markeredgecolor='k', alpha=0.7)
plt.plot(chi0.wn,chi0.mat.real,'-o', color=colors[i], markeredgecolor='k', alpha=0.7)
plt.xlim(0,30)
plt.show()


plt.figure()
for i, chi0 in enumerate(gchi0):
    plt.plot(i,chi0.chi0_tilde[niw].real,'-o', color=colors[i], markeredgecolor='k', alpha=0.7)
    plt.plot(i,chi0.chi0[niw].real,'-o', color=colors[i], markeredgecolor='k', alpha=0.7)
plt.show()
