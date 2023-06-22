import numpy as np
import jax.numpy as jnp
import w2dyn_aux_dga as aux
import FourPoint as fp
import MatsubaraFrequencies as mf
import matplotlib.pyplot as plt

path = './2DSquare_U8_tp-0.25_tpp0.12_beta25_n0.85/'

giw_file = aux.w2dyn_file(fname=path + '1p-data.hdf5')

beta = giw_file.get_beta()
giw = giw_file.get_giw()[0, 0, :]

niw_giw = giw.size // 2

niw = 50
niv = niw
wn = mf.wn(niw)
iwn = wn[niw]

gchi0 = fp.LocalBubble(beta=beta, wn=wn, giw=giw, niv=niv, chi0_method='sum')
gchi0_inv = fp.LocalBubble(beta=beta, wn=wn, giw=giw, niv=niv, chi0_method='sum', is_inv=True)

unity = gchi0.mat * gchi0_inv.mat

plt.figure()
plt.imshow(unity.imag, cmap='RdBu')
plt.colorbar()
plt.show()

# %% Check Local four-point objects:
g2iw_file = aux.g4iw_file(fname=path + 'g4iw_sym.hdf5')
niw = g2iw_file.get_niw(channel='dens')
niv = niw + 1
wn = mf.wn(niw)
g2_dens = fp.LocalFourPoint(matrix=g2iw_file.read_g2(channel='dens', niv=niv, niw=niw), beta=beta, wn=wn, channel='dens')
g2_magn = fp.LocalFourPoint(matrix=g2iw_file.read_g2(channel='magn', niv=niv, niw=niw), beta=beta, wn=wn, channel='magn')

gchi_dens = fp.chir_from_g2(g2_dens, giw)
gchi_magn = fp.chir_from_g2(g2_magn, giw)

gchi0_inv = fp.LocalBubble(beta=beta, wn=wn, giw=giw, niv=niv, chi0_method='sum', is_inv=True)
F_magn = fp.Fob2_from_chir(gchi_magn, gchi0_inv)
F_dens = fp.Fob2_from_chir(gchi_dens, gchi0_inv)

gam_dens = fp.gamob2_from_chir(gchi_dens, gchi0_inv)
gam_dens.plot(pdir=path, name='Gamma_dens', niv=30)
gam_magn = fp.gamob2_from_chir(gchi_magn, gchi0_inv)
gam_magn.plot(pdir=path, name='Gamma_magn', niv=30)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()
axes[0].pcolormesh(gchi_dens.vn, gchi_dens.vn, gchi_dens.mat[niw].real, cmap='RdBu')
axes[1].pcolormesh(gchi_dens.vn, gchi_dens.vn, F_dens.mat[niw].real, cmap='RdBu')
axes[2].pcolormesh(gchi_dens.vn, gchi_dens.vn, gam_dens.mat[niw].real, cmap='RdBu')

axes[3].pcolormesh(gchi_dens.vn, gchi_dens.vn, gchi_magn.mat[niw].real, cmap='RdBu')
axes[4].pcolormesh(gchi_dens.vn, gchi_dens.vn, F_magn.mat[niw].real, cmap='RdBu')
axes[5].pcolormesh(gchi_dens.vn, gchi_dens.vn, gam_magn.mat[niw].real, cmap='RdBu')

# plt.tight_layout()
plt.show()

# %% Test Local Schwinger-Dyson equation:
totdens = giw_file.get_totdens()
u = giw_file.get_udd()
siw_dmft = giw_file.get_siw()[0, 0, :]
gchi0 = fp.LocalBubble(beta=beta, wn=wn, giw=giw, niv=niv, chi0_method='sum')
F_updo = fp.LocalFourPoint(matrix=0.5 * (F_dens.mat - F_magn.mat), beta=F_dens.beta, wn=F_dens.wn, channel='updo')
siw = fp.schwinger_dyson_F(F=F_updo, chi0=gchi0, giw=giw, u=u, totdens=totdens)
siw_dens = fp.schwinger_dyson_F(F=F_dens, chi0=gchi0, giw=giw, u=u, totdens=totdens)

# %% Test vrg three-leg vertex:
vrg_dens = fp.vrg_from_gam(gam=gam_dens, chi0_inv=gchi0_inv, u=u)
vrg_magn = fp.vrg_from_gam(gam=gam_magn, chi0_inv=gchi0_inv, u=u)
vrg_updo = fp.LocalThreePoint(matrix=0.5 * (vrg_dens.mat - vrg_magn.mat), channel='updo', beta=vrg_magn.beta, wn=vrg_dens.wn)
vrg_upup = fp.LocalThreePoint(matrix=0.5 * (vrg_dens.mat + vrg_magn.mat), channel='upup', beta=vrg_magn.beta, wn=vrg_dens.wn)
chi_dens = gchi_dens.contract_legs()
chi_magn = gchi_magn.contract_legs()
chi_updo = fp.LocalSusceptibility(matrix=0.5 * (chi_dens.mat - chi_magn.mat), channel='updo', beta=chi_magn.beta, wn=chi_magn.wn)
chi_upup = fp.LocalSusceptibility(matrix=0.5 * (chi_dens.mat + chi_magn.mat), channel='upup', beta=chi_magn.beta, wn=chi_magn.wn)
# siw_vrg = fp.schwinger_dyson_vrg(vrg = vrg_dens, chir_phys = chi_dens, giw=giw, u = u, totdens = totdens)
mat_grid = fp.wn_slices(giw, n_cut=vrg_dens.niv, wn=vrg_dens.wn)
siw_vrg = u * totdens / 2 + u * jnp.sum((1 / beta - (1 - u * chi_dens.mat[:, None]) * vrg_dens.mat) * mat_grid, axis=0)
siw_vrg_magn = u * totdens / 2 - u * jnp.sum((1 / beta - (1 + u * chi_magn.mat[:, None]) * vrg_magn.mat) * mat_grid, axis=0)
siw_vrg_updo_2 = 0.5 * (siw_vrg + siw_vrg_magn)
siw_vrg_updo_2 =  u * totdens / 2 + u/2 * jnp.sum((- (1 - u * chi_dens.mat[:, None]) * vrg_dens.mat + (1 + u * chi_magn.mat[:, None]) * vrg_magn.mat) * mat_grid, axis=0)
siw_vrg_updo = u * totdens / 2 + u * jnp.sum(
    ( - vrg_updo.mat + (u * chi_updo.mat[:, None]) * vrg_updo.mat+ (u * chi_upup.mat[:, None]) * vrg_upup.mat) * mat_grid, axis=0)
siw_vrg_upup = u * totdens / 2 + u * jnp.sum((1 / beta - (1 - u * chi_updo.mat[:, None]) * vrg_updo.mat) * mat_grid, axis=0)

vn_dmft = mf.vn(siw_dmft.size // 2)
plt.figure()
plt.plot(vn_dmft, siw_dmft.imag, '-o', color='firebrick', markeredgecolor='k', alpha=0.7)
# Equivalent:
plt.plot(F_updo.vn, siw.imag, '-h', color='cornflowerblue', markeredgecolor='k', alpha=0.7)
plt.plot(F_updo.vn, siw_vrg_updo_2.imag, '-h', color='forestgreen', markeredgecolor='k', alpha=0.7)
# Equivalent:
plt.plot(F_updo.vn, siw_dens.imag, '-h', color='seagreen', markeredgecolor='k', alpha=0.7)
# plt.plot(F_updo.vn, siw_vrg.imag, '-h', color='k', markeredgecolor='k', alpha=0.7)


# plt.plot(F_updo.vn, siw_vrg_magn.imag, '-h', color='crimson', markeredgecolor='k', alpha=0.7)
# plt.plot(F_updo.vn, siw_vrg_updo.imag, '-h', color='navy', markeredgecolor='k', alpha=0.7)

# plt.plot(F_updo.vn, siw_vrg_upup.imag, '-h', color='indigo', markeredgecolor='k', alpha=0.7)
# plt.plot(F_updo.vn,siw_vrg.imag, '-h', color='goldenrod', markeredgecolor='k', alpha=0.7)
plt.xlim(0, F_updo.niv)
plt.show()

# %% Compare vrg and sg:
sg_dens = fp.three_leg_from_F(F=F_dens, chi0=gchi0)


fig, axes = plt.subplots(ncols=2)
axes = axes.flatten()
im1 = axes[0].pcolormesh(vrg_dens.vn,vrg_dens.wn, vrg_dens.mat.real,cmap='RdBu')
im2 = axes[1].pcolormesh(vrg_dens.vn,vrg_dens.wn, -sg_dens.mat.real,cmap='RdBu')
fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location='right',
             pad=0.05)
fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location='right',
             pad=0.05)
plt.tight_layout()
plt.show()

# %% Check niv convergence of chi0:

niv_range = [10, 20, 40, 80, 160, 320, 640, 800]

chi0_conv_check = []
for n in niv_range:
    chi0_conv_check.append(fp.vec_get_chi0_sum(giw, beta, n, wn))

colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(niv_range)))
fig = plt.figure()
for i, chi0 in enumerate(chi0_conv_check):
    plt.plot(wn, chi0.real, '-o', color=colors[i], markeredgecolor='k', alpha=0.7)
plt.show()

fig = plt.figure()
for i, chi0 in enumerate(chi0_conv_check):
    plt.plot(i, chi0[niw].real, '-o', color=colors[i], markeredgecolor='k', alpha=0.7)
plt.show()


# %% Try fft routine:

# gchi0.plot(pdir=path,name='Bubble',iwn=iwn)

# def get_chi0_fft_wn(iwn=0):
#     gtau = jnp.fft.fft(jnp.fft.ifftshift(mf.cut_v_1d_iwn(giw,niv_cut=niv)))
#     return jnp.fft.ifft(gtau * gtau)
#

# gtau = jnp.fft.fft(mf.cut_v_1d_wn(giw, niw_cut=niv))
# return -1 / beta * jnp.fft.ifft(gtau * gtau)
def get_chi0_fft():
    giw_cut = mf.cut_v_1d(giw, niv_cut=niv)
    return 1 / beta * jnp.convolve(giw_cut, giw_cut, mode='same')


chi0_fft = get_chi0_fft()
#
fig = plt.figure()
#plt.plot(chi0_fft.real, '-o', color='cornflowerblue', markeredgecolor='k', alpha=0.7)
plt.plot(gchi0.wn,gchi0.chi0.real, '-o', color='firebrick', markeredgecolor='k', alpha=0.7)
plt.vlines(0,0,0.5)
plt.show()
