# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Performs the analytical continuation on a Lambda-Dga output
# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import sys, os

sys.path.append('../src/')
sys.path.append(os.environ['HOME'] + "/Programs/dga/LambdaDga/src")
sys.path.append('../ana_cont/')
sys.path.append(os.environ['HOME'] + "/Programs/dga/LambdaDga/ana_cont")

import numpy as np
import sys, os
import BrillouinZone as bz
import AnalyticContinuation as a_cont
import matplotlib.pyplot as plt
import PadeAux as pade_aux
from Plotting import MidpointNormalize
import Hk as hamk
import Output as output
import TwoPoint_old as twop
from mpl_toolkits.axes_grid1 import make_axes_locatable
import Plotting as plotting

# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def insert_colorbar(ax=None, im=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')


def plot_ploints_on_fs(output_path=None, gk_fs=None, k_grid=None, ind_fs=None, name='', shift_pi=True):
    kx = k_grid.kx
    ky = k_grid.ky
    gk_fs_plot = gk_fs
    nk = k_grid.nk
    if (shift_pi):
        gk_fs_plot = np.roll(gk_fs_plot, k_grid.nk[0] // 2, 0)
        gk_fs_plot = np.roll(gk_fs_plot, k_grid.nk[1] // 2, 1)
        kx = kx - np.pi
        ky = ky - np.pi

    extent = [kx[0], kx[-1], ky[0], ky[-1]]

    lw = 1.0

    def add_lines(ax):
        ax.plot(kx, 0 * kx, 'k', lw=lw)
        ax.plot(0 * ky, ky, 'k', lw=lw)
        ax.plot(-ky, ky - np.pi, '--k', lw=lw)
        ax.plot(ky, ky - np.pi, '--k', lw=lw)
        # kx.plot(-ky, ky + np.pi, '--k', lw=lw)
        ax.set_xlim(kx[0], kx[-1])
        ax.set_ylim(ky[0], ky[-1])
        # ax.plot(ky, ky + np.pi, '--k', lw=lw)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    add_lines(ax[0])
    add_lines(ax[1])
    add_lines(ax[2])
    im = ax[0].imshow(-1. / np.pi * gk_fs_plot.imag, cmap='RdBu_r', extent=extent)
    insert_colorbar(ax=ax[0], im=im)
    for i in ind_fs:
        ax[0].plot(-k_grid.kx[i[0]], -k_grid.ky[i[1]], 'o')

    gr = gk_fs_plot.real
    norm = MidpointNormalize(midpoint=0, vmin=gr.min(), vmax=gr.max())
    im = ax[1].imshow(gr, cmap='RdBu', extent=extent, norm=norm)
    insert_colorbar(ax=ax[1], im=im)
    for i in ind_fs:
        ax[1].plot(-k_grid.kx[i[0]], -k_grid.ky[i[1]], 'o')

    qpd = (1. / gk_fs_plot).real
    norm = MidpointNormalize(midpoint=0, vmin=qpd.min(), vmax=qpd.max())
    im = ax[2].imshow(qpd, cmap='RdBu', extent=extent, norm=norm)
    insert_colorbar(ax=ax[2], im=im)
    for i in ind_fs:
        ax[2].plot(-k_grid.kx[i[0]], -k_grid.ky[i[1]], 'o')

    plt.tight_layout()
    plt.savefig(output_path + '{}.png'.format(name))
    plt.show()
    plt.close('all')


def plot_aw_loc(v_real=None, gloc=None, output_path=None, name='', xlim=(None, None)):
    plt.figure()
    plt.plot(v_real, -1. / np.pi * gloc.imag)
    plt.vlines(0, 0, 10, 'k', '--', alpha=0.5)
    plt.ylim(0, np.max(-1. / np.pi * gloc.imag))
    plt.xlim(xlim)
    plt.xlabel('$\omega$')
    plt.ylabel('A($\omega$)')
    plt.title(name)
    plt.savefig(output_path + '{}.png'.format(name))
    plt.show()
    plt.close('all')


def plot_aw_ind(v_real=None, gk_cont=None, ind=None, output_path=None, name='', xlim=(None, None)):
    plt.figure()
    for i, i_arc in enumerate(ind):
        plt.plot(v_real, -1. / np.pi * gk_cont[:, i].imag, label='{}'.format(i_arc))
    plt.vlines(0, 0, 10, 'k', '--', alpha=0.5)
    plt.ylim(0, np.max(-1. / np.pi * gk_cont.imag))
    plt.xlim(xlim)
    plt.xlabel('$\omega$')
    plt.ylabel('A($\omega$)')
    plt.title(name)
    plt.savefig(output_path + '{}.png'.format(name))
    plt.show()
    plt.close('all')


# %%
# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

# Define input path:
# input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.2_tpp0.1_beta50_n0.95/LambdaDga_lc_sp_Nk19600_Nq19600_core60_invbse60_vurange150_wurange60/'
# input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/Testset1/LambdaDga_Python/LambdaDga_lc_sp_Nk6400_Nq6400_core59_invbse100_vurange100_wurange99/'
# input_path = '/mnt/c/users/pworm/Research/U2BenchmarkData/BenchmarkSchaefer_beta_15/LambdaDgaPython/LambdaDga_lc_sp_Nk4096_Nq4096_core27_invbse80_vurange80_wurange80/'
# input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.875/LambdaDga_lc_sp_Nk10000_Nq10000_core59_invbse60_vurange500_wurange59/'
input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.875/LambdaDga_lc_sp_Nk19600_Nq19600_core60_invbse60_vurange150_wurange60/'
# input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.875/LambdaDga_lc_sp_Nk19600_Nq19600_core60_invbse60_vurange150_wurange60/'
# input_path = '/mnt/d/Research/HoleDopedNickelates/LambdaDgaPython/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.925/LambdaDga_lc_sp_Nk6400_Nq6400_core59_invbse60_vurange500_wurange59/'
# input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.90/LambdaDga_lc_sp_Nk10000_Nq10000_core59_invbse60_vurange500_wurange59/'

output_path = input_path

# Load config:
config = np.load(input_path + 'config.npy', allow_pickle=True).item()
dga_sde = np.load(input_path + 'dga_sde.npy', allow_pickle=True).item()
dmft1p = config['dmft1p']
hr = config['system']['hr']
niv_urange = config['box_sizes']['niv_urange']
niv_core = config['box_sizes']['niv_core']
nk = config['box_sizes']['nk']

# Specify MaxEnt parameters:
t = 1.00
wmax = 15 * t
v_real = a_cont.v_real_tan(wmax=wmax, nw=501)
use_preblur = True
bw = 0.1
bw_dmft = 0.01
adjust_mu = True
err = 1e-2
err_dmft = 1e-2
nfit = config['box_sizes']['niv_core']

output_path = output.uniquify(output_path + 'AnaCont') + '/'
os.mkdir(output_path)

k_grid = bz.KGrid(nk=nk)
ek = hamk.ek_3d(kgrid=k_grid.grid, hr=hr)
k_grid.get_irrk_from_ek(ek=ek, dec=9)

# Continue the DMFT Green's function:
gloc_dmft_cont, gk_dmft = a_cont.max_ent_loc(v_real=v_real, sigma=dmft1p['sloc'], config=config, k_grid=k_grid,
                                             niv_cut=niv_urange, use_preblur=use_preblur, bw=bw_dmft, err=err_dmft,
                                             nfit=nfit)
plotting.plot_aw_loc(output_path=output_path, v_real=v_real, gloc=gloc_dmft_cont, name='aw-dmft')
n_int = a_cont.check_filling(v_real=v_real, gloc_cont=gloc_dmft_cont)
np.savetxt(output_path + 'n_dmft.txt', [n_int, gk_dmft['n']], delimiter=',', fmt='%.9f')
# %% Continue points along the Fermi surface:
gk_fs_dmft_cont, ind_gf0_dmft = a_cont.max_ent_on_fs(v_real=v_real, sigma=dmft1p['sloc'], config=config, k_grid=k_grid,
                                                     niv_cut=niv_urange, use_preblur=use_preblur, bw=bw_dmft,
                                                     err=err_dmft, nfit=nfit)

plotting.plot_ploints_on_fs(output_path=output_path, gk_fs=gk_dmft['gk'][:, :, 0, gk_dmft['niv']], k_grid=k_grid,
                   ind_fs=ind_gf0_dmft,
                   name='fermi_surface_dmft')

plotting.plot_aw_ind(output_path=output_path, v_real=v_real, gk_cont=gk_fs_dmft_cont, ind=ind_gf0_dmft,
            name='aw-dmft-fs-wide')
plotting.plot_aw_ind(output_path=output_path, v_real=v_real, gk_cont=gk_fs_dmft_cont, ind=ind_gf0_dmft,
            name='aw-dmft-fs-narrow', xlim=(-t, t))

# %% Perform continuation in the irreduzible BZ:
cont_dict_dmft = a_cont.max_ent_irrk(v_real=v_real, sigma=dmft1p['sloc'], config=config, k_grid=k_grid, niv_cut=niv_urange,
                                     use_preblur=use_preblur, bw=bw_dmft, err=err_dmft, nfit=nfit, scal=2)

plotting.plot_ploints_on_fs(output_path=output_path, gk_fs=gk_dmft['gk'][:, :, 0, gk_dmft['niv']], k_grid=k_grid, ind_fs=cont_dict_dmft['ind_irrk_fbz'],
                   name='fermi_surface_dmft_irrk_points', shift_pi=True)

w_int = -0.2
plotting.plot_cont_fs(output_path = output_path,name='fermi_surface_dmft_cont_wint-0.2', gk=cont_dict_dmft['gk_cont'],v_real=v_real, k_grid =cont_dict_dmft['k_grid'],w_int=w_int)

w_int = -0.1
plotting.plot_cont_fs(output_path = output_path,name='fermi_surface_dmft_cont_wint-0.1', gk=cont_dict_dmft['gk_cont'],v_real=v_real, k_grid =cont_dict_dmft['k_grid'],w_int=w_int)

# # %%
# # Perform continuation along zeros of Green's function:
# gk_fs_dmft = gk_dmft['gk'][:, :, :, niv_urange]
# ind_gf0_dmft = bz.find_qpd_zeros(qpd=gk_fs_dmft.real, kgrid=k_grid)
# ind_gf0_dmft_max_ent = ind_gf0_dmft[::4]
#
# plot_ploints_on_fs(output_path=output_path, gk_fs=gk_fs_dmft[:, :, 0], k_grid=k_grid, ind_fs=ind_gf0_dmft_max_ent,
#                    name='fermi_surface_dmft')
#
# w_int = -0.2
# # wm1 = np.argmin(np.abs(v_real_max_ent-0.2))
# ind_int = np.logical_and(v_real < 0, w_int < v_real)
# awk_fs = np.trapz(-1. / np.pi * gk_dga_max_ent_fbz[:, :, 0, ind_int].imag, v_real_max_ent[ind_int])
# awk_fs = np.squeeze(bz.shift_mat_by_pi(mat=awk_fs, nk=k_grid_small.nk))
# # awk_fs = np.squeeze(-1./np.pi * gk_dga_max_ent_fbz[:,:,0,wm1].imag)
# plt.figure()
# plt.imshow(awk_fs, cmap='terrain', extent=bz.get_extent_pi_shift(kgrid=k_grid_small), vmin=0.02)
# plt.colorbar()
# plt.savefig(output_path + 'aw_dga_fs.png')
# plt.show()
#
# # %%
# gk_dmft_max_ent = a_cont.do_max_ent_on_ind(mat=gk_dmft['gk'], ind_list=ind_gf0_dmft_max_ent, v_real=v_real_max_ent,
#                                            beta=dmft1p['beta'],
#                                            n_fit=60, err=1e-2, alpha_det_method='chi2kink')
#
# plot_aw_ind(output_path=output_path, v_real=v_real_max_ent, gk_cont=gk_dmft_max_ent, ind=ind_gf0_dmft_max_ent,
#             name='aw-dmft-fs-wide')
# plot_aw_ind(output_path=output_path, v_real=v_real_max_ent, gk_cont=gk_dmft_max_ent, ind=ind_gf0_dmft_max_ent,
#             name='aw-dmft-fs-narrow', xlim=(-t, t))
#
# # %%
# # Specify MaxEnt parameters:
# wmax = 15 * t
# v_real_max_ent = a_cont.v_real_tan(wmax=wmax, nw=1001)
# use_preblur = True
# bw = 0.1
#
# # Create DGA Green's function:
# sigma = dga_sde['sigma']
# gk_dga = twop.create_gk_dict(sigma=sigma, kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'], n=dmft1p['n'],
#                              mu0=dmft1p['mu'], adjust_mu=True, niv_cut=niv_urange)
# ind_gf0_dga = bz.find_qpd_zeros(qpd=gk_dga['gk'][:, :, :, niv_urange].real, kgrid=k_grid)
# gloc_dga = gk_dga['gk'].mean(axis=(0, 1, 2))
#
# gloc_dga_cont = a_cont.max_ent(mat=gloc_dga, v_real=v_real_max_ent, beta=dmft1p['beta'], n_fit=60,
#                                alpha_det_method='chi2kink', err=1e-2, use_preblur=True, bw=bw)
#
# # Check filling:
# ind_w = v_real_max_ent < 0
# n_int = np.trapz(-1. / np.pi * gloc_dga_cont[ind_w].imag, v_real_max_ent[ind_w]) * 2
# n_mat = gk_dga['n']
# niv0 = gk_dga['gk'].shape[-1] // 2
# gk_lutt = np.ma.masked_less((1. / gk_dga['gk'][..., niv0]).real, 0)
# gk_lutt[~gk_lutt.mask] = 1.0
# gk_lutt[gk_lutt.mask] = 0.0
# n_lutt = np.sum(gk_lutt) / k_grid.nk_tot * 2
# print(f'{n_int=}')
# print(f'{n_mat=}')
# print(f'{n_lutt=}')
#
# plot_aw_loc(output_path=output_path, v_real=v_real_max_ent, gloc=gloc_dga_cont, name='aw-dga')
#
# # %%
# # Perform continuation along zeros of Green's function:
# gk_fs_dga = gk_dga['gk'][:, :, :, niv_urange]
# ind_gf0_dga = bz.find_qpd_zeros(qpd=gk_fs_dga.real, kgrid=k_grid)
# ind_gf0_dga_max_ent = ind_gf0_dga[::4]
#
# plot_ploints_on_fs(output_path=output_path, gk_fs=gk_fs_dga[:, :, 0], k_grid=k_grid, ind_fs=ind_gf0_dga_max_ent,
#                    name='fermi_surface_dga')
#
# # %%
# gk_dga_max_ent = a_cont.do_max_ent_on_ind(mat=gk_dga['gk'], ind_list=ind_gf0_dga_max_ent, v_real=v_real_max_ent,
#                                           beta=dmft1p['beta'],
#                                           n_fit=60, err=1e-2, alpha_det_method='chi2kink')
#
# plot_aw_ind(output_path=output_path, v_real=v_real_max_ent, gk_cont=gk_dga_max_ent, ind=ind_gf0_dga_max_ent,
#             name='aw-dga-fs-wide')
# plot_aw_ind(output_path=output_path, v_real=v_real_max_ent, gk_cont=gk_dga_max_ent, ind=ind_gf0_dga_max_ent,
#             name='aw-dga-fs-narrow', xlim=(-t, t))
#
# # %% Perform continuation in the irreduzible BZ:
#
# # Create reduzed k-grid:
# scal = 5
# nk_new = (k_grid.nk[0] / scal, k_grid.nk[1] / scal, 1)
# print(nk_new)
# nk_new = tuple([int(i) for i in nk_new])
# k_grid_small = bz.KGrid(nk=nk_new)
# ek = hamk.ek_3d(kgrid=k_grid_small.grid, hr=hr)
# k_grid_small.get_irrk_from_ek(ek=ek, dec=11)
# ind_irrk = [
#     np.argmin(np.abs(np.array(k_grid.irr_kgrid) - np.atleast_2d(np.array(k_grid_small.irr_kgrid)[:, i]).T).sum(axis=0))
#     for i in k_grid_small.irrk_ind_lin]
# ind_irrk = np.squeeze(np.array(np.unravel_index(k_grid.irrk_ind[ind_irrk], shape=k_grid.nk))).T
#
# plot_ploints_on_fs(output_path=output_path, gk_fs=gk_fs_dga[:, :, 0], k_grid=k_grid, ind_fs=ind_irrk,
#                    name='fermi_surface_dga_irrk_points', shift_pi=True)
#
# ind_irrk = [tuple(ind_irrk[i, :]) for i in np.arange(ind_irrk.shape[0])]
# # %%
# gk_dga_max_ent_irrk = a_cont.do_max_ent_on_ind(mat=gk_dga['gk'], ind_list=ind_irrk, v_real=v_real_max_ent,
#                                                beta=dmft1p['beta'],
#                                                n_fit=60, err=1e-2, alpha_det_method='chi2kink')
#
# plot_aw_ind(output_path=output_path, v_real=v_real_max_ent, gk_cont=gk_dga_max_ent_irrk, ind=ind_irrk,
#             name='aw-dga-irrk-fs-wide')
# plot_aw_ind(output_path=output_path, v_real=v_real_max_ent, gk_cont=gk_dga_max_ent_irrk, ind=ind_irrk,
#             name='aw-dga-irrk-fs-narrow', xlim=(-t, t))
#
# # %%
# # Plot the Fermi-surface on the real-frequency axis:
# gk_dga_max_ent_fbz = k_grid_small.irrk2fbz(mat=gk_dga_max_ent_irrk.T)
#
# w_int = -0.2
# # wm1 = np.argmin(np.abs(v_real_max_ent-0.2))
# ind_int = np.logical_and(v_real_max_ent < 0, w_int < v_real_max_ent)
# awk_fs = np.trapz(-1. / np.pi * gk_dga_max_ent_fbz[:, :, 0, ind_int].imag, v_real_max_ent[ind_int])
# awk_fs = np.squeeze(bz.shift_mat_by_pi(mat=awk_fs, nk=k_grid_small.nk))
# # awk_fs = np.squeeze(-1./np.pi * gk_dga_max_ent_fbz[:,:,0,wm1].imag)
# plt.figure()
# plt.imshow(awk_fs, cmap='terrain', extent=bz.get_extent_pi_shift(kgrid=k_grid_small), vmin=0.02)
# plt.colorbar()
# plt.savefig(output_path + 'aw_dga_fs.png')
# plt.show()
# # %%
# # ------------------------------------------------- NO MU UPDATE --------------------------------------------------------
#
# # Specify MaxEnt parameters:
# wmax = 15 * t
# v_real_max_ent = a_cont.v_real_tan(wmax=wmax, nw=1001)
# use_preblur = True
# bw = 0.2
#
# # Create DGA Green's function:
# sigma = dga_sde['sigma']
# gk_dga_nmu = twop.create_gk_dict(sigma=sigma, kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'], n=dmft1p['n'],
#                                  mu0=dmft1p['mu'], adjust_mu=False, niv_cut=niv_urange)
# ind_gf0_dga_nmu = bz.find_qpd_zeros(qpd=gk_dga_nmu['gk'][:, :, :, niv_urange].real, kgrid=k_grid)
# gloc_dga_nmu = gk_dga_nmu['gk'].mean(axis=(0, 1, 2))
#
# gloc_dga_cont_nmu = a_cont.max_ent(mat=gloc_dga_nmu, v_real=v_real_max_ent, beta=dmft1p['beta'], n_fit=60,
#                                    alpha_det_method='chi2kink', err=1e-2, use_preblur=True, bw=bw)
#
# # Check filling:
# ind_w = v_real_max_ent < 0
# n_int = np.trapz(-1. / np.pi * gloc_dga_cont_nmu[ind_w].imag, v_real_max_ent[ind_w]) * 2
# n_mat = gk_dga['n']
# print(f'{n_int=}')
# print(f'{n_mat=}')
#
# plot_aw_loc(output_path=output_path, v_real=v_real_max_ent, gloc=gloc_dga_cont_nmu, name='aw-dga-no-mu-adjust')
#
# # %%
# # Perform continuation along zeros of Green's function:
# gk_fs_dga_nmu = gk_dga_nmu['gk'][:, :, :, niv_urange]
# ind_gf0_dga_nmu = bz.find_qpd_zeros(qpd=gk_fs_dga_nmu.real, kgrid=k_grid)
# ind_gf0_dga_max_ent_nmu = ind_gf0_dga_nmu[::4]
#
# plot_ploints_on_fs(output_path=output_path, gk_fs=gk_fs_dga_nmu[:, :, 0], k_grid=k_grid, ind_fs=ind_gf0_dga_max_ent_nmu,
#                    name='fermi_surface_dga-no-mu-adjust')
#
# # %%
# gk_dga_max_ent_nmu = a_cont.do_max_ent_on_ind(mat=gk_dga_nmu['gk'], ind_list=ind_gf0_dga_max_ent_nmu,
#                                               v_real=v_real_max_ent, beta=dmft1p['beta'],
#                                               n_fit=60, err=1e-2, alpha_det_method='chi2kink')
#
# plot_aw_ind(output_path=output_path, v_real=v_real_max_ent, gk_cont=gk_dga_max_ent_nmu, ind=ind_gf0_dga_max_ent_nmu,
#             name='aw-dga-fs-wide-no-mu-adjust')
# plot_aw_ind(output_path=output_path, v_real=v_real_max_ent, gk_cont=gk_dga_max_ent_nmu, ind=ind_gf0_dga_max_ent_nmu,
#             name='aw-dga-fs-narrow-no-mu-adjust', xlim=(-t, t))

# %%
# plt.close('all')
# # %%
# # Load Green's function:
# if (adjust_mu):
#     gk_dict = np.load(input_path + 'gk_dga.npy', allow_pickle=True).item()
# else:
#     gk_dict = np.load(input_path + 'gk_dga_mu_dmft.npy', allow_pickle=True).item()
# try:
#     gk = gk_dict['gk']
# except:
#     gk = gk_dict.gk
# qpd = np.real(1. / gk)[:, :, :, niv_urange]
# gr = gk.real[:, :, :, niv_urange]
# ak_fs = -1. / np.pi * gk.imag[:, :, :, niv_urange]
# # Find location of Fermi-surface:
#
# ind_fs = bz.find_fermi_surface_peak(ak_fs=ak_fs, kgrid=k_grid)
# ind_qdp0 = bz.find_qpd_zeros(qpd=qpd, kgrid=k_grid)
# ind_gf0 = bz.find_qpd_zeros(qpd=gr, kgrid=k_grid)
#
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
# ax[0].imshow(ak_fs, cmap='RdBu_r', extent=bz.get_extent(kgrid=k_grid))
# for i in ind_fs:
#     ax[0].plot(k_grid.kx[i[0]], k_grid.ky[i[1]], 'o')
#
# for i in ind_qdp0:
#     ax[0].plot(k_grid.kx[i[0]], k_grid.ky[i[1]], 's')
#
# for i in ind_gf0:
#     ax[0].plot(k_grid.kx[i[0]], k_grid.ky[i[1]], 'p')
#
# norm = MidpointNormalize(midpoint=0, vmin=gr.min(), vmax=gr.max())
# ax[1].imshow(gr, cmap='RdBu', extent=bz.get_extent(kgrid=k_grid))
# for i in ind_fs:
#     ax[1].plot(k_grid.kx[i[0]], k_grid.ky[i[1]], 'o')
#
# for i in ind_qdp0:
#     ax[1].plot(k_grid.kx[i[0]], k_grid.ky[i[1]], 's')
#
# for i in ind_gf0:
#     ax[1].plot(k_grid.kx[i[0]], k_grid.ky[i[1]], 'p')
#
# norm = MidpointNormalize(midpoint=0, vmin=qpd.min(), vmax=qpd.max())
# ax[2].imshow(qpd, cmap='RdBu', extent=bz.get_extent(kgrid=k_grid))
# for i in ind_fs:
#     ax[2].plot(k_grid.kx[i[0]], k_grid.ky[i[1]], 'o')
#
# for i in ind_qdp0:
#     ax[2].plot(k_grid.kx[i[0]], k_grid.ky[i[1]], 's')
#
# for i in ind_gf0:
#     ax[2].plot(k_grid.kx[i[0]], k_grid.ky[i[1]], 'p')
#
# plt.tight_layout()
# plt.savefig(output_path + 'location_of_special_points.png')
# plt.show()
#
# # # Pade approximation of the Green's function:
# nw = 1501
# wmax = 15 * t
# v_real = a_cont.v_real_tan(wmax=wmax, nw=nw)
# n_pade = 10  # np.min((niv_urange, 10))
# delta = 0.  # 1./dmft1p['beta']
# n_fit = niv_core
# gk_arc_thiele = a_cont.do_pade_on_ind(mat=gk, ind_list=ind_fs, v_real=v_real, beta=dmft1p['beta'],
#                                       method='thiele', n_fit=n_fit, n_pade=n_pade, delta=delta)
#
# gk_arc_thiele = a_cont.do_pade_on_ind(mat=gk.reshape((k_grid.nk_tot, -1)), ind_list=k_grid.irrk_ind, v_real=v_real,
#                                       beta=dmft1p['beta'],
#                                       method='thiele', n_fit=n_fit, n_pade=n_pade, delta=delta)
#
# v_real_max_ent = a_cont.v_real_tan(wmax=wmax, nw=501)
#
# n_ind = len(ind_fs)
# ind_fs_max_ent = ind_fs[n_ind // 2:]
# ind_fs_max_ent = ind_fs_max_ent[0:-1:4]
# gk_arc_max_ent = a_cont.do_max_ent_on_ind(mat=gk, ind_list=ind_fs_max_ent, v_real=v_real_max_ent, beta=dmft1p['beta'],
#                                           n_fit=60, err=1e-2)
#
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# ax[0].imshow(ak_fs, cmap='RdBu_r', extent=bz.get_extent(kgrid=k_grid))
# for i in ind_fs_max_ent:
#     ax[0].plot(k_grid.kx[i[0]], k_grid.ky[i[1]], 'o')
#
# norm = MidpointNormalize(midpoint=0, vmin=gr.min(), vmax=gr.max())
# ax[1].imshow(gr, cmap='RdBu', extent=bz.get_extent(kgrid=k_grid))
# for i in ind_fs_max_ent:
#     ax[1].plot(k_grid.kx[i[0]], k_grid.ky[i[1]], 'o')
# plt.tight_layout()
# plt.savefig(output_path + 'location_of_special_points_max_ent.png')
# plt.show()
#
# # gk_arc_max_ent_fbz = a_cont.do_max_ent_on_ind(mat=gk.reshape((k_grid.nk_tot,-1)), ind_list=k_grid.irrk_ind, v_real=v_real_max_ent, beta=dmft1p['beta'],
# #                                       n_fit=60,err=1e-2)
#
# gk_arc_thiele_fbz = k_grid.irrk2fbz(mat=gk_arc_thiele.T)
#
# plt.figure()
# for i, i_arc in enumerate(ind_fs):
#     plt.plot(v_real, -1. / np.pi * gk_arc_thiele[:, i].imag, label='{}'.format(i_arc))
# plt.legend()
# # plt.xlim(-t, t)
# # plt.ylim([0, np.max(-1. / np.pi * gk_arc_thiele[:, i].imag)])
# plt.savefig(output_path + 'aw_arc_thiele.png')
# plt.show()
#
# plt.figure()
# for i, i_arc in enumerate(ind_fs_max_ent):
#     plt.plot(v_real_max_ent, -1. / np.pi * gk_arc_max_ent[:, i].imag, label='{}'.format(i_arc))
# plt.legend()
# plt.xlim(-t, t)
# plt.ylim([0, 1])
# plt.savefig(output_path + 'aw_arc_max_ent.png')
# plt.show()
#
# plt.figure()
# for i, i_arc in enumerate(ind_fs_max_ent):
#     plt.plot(v_real_max_ent, -1. / np.pi * gk_arc_max_ent[:, i].imag, label='{}'.format(i_arc))
# plt.legend()
# plt.xlim(-2, 2)
# plt.ylim([0, 1])
# plt.savefig(output_path + 'aw_arc_max_ent_big_window.png')
# plt.show()
#
# plt.figure()
# for i, i_arc in enumerate(ind_fs):
#     plt.plot(v_real, gk_arc_thiele[:, i].real, label='{}'.format(i_arc))
# plt.legend()
# plt.xlim(-t, t)
# # plt.ylim([0, np.max(-1. / np.pi * gk_arc_thiele.imag)])
# plt.savefig(output_path + 'g_real_arc_thiele.png')
# plt.show()
#
# #
# # awk_fs = bz.shift_mat_by_pi(mat=-1./np.pi * gk_arc_thiele_fbz[:,:,0,nw//2].imag, nk=k_grid.nk)
# # plt.figure()
# # plt.imshow(awk_fs, cmap='terrain', extent=bz.get_extent_pi_shift(kgrid=k_grid))#, vmin=0.05)
# # plt.colorbar()
# # plt.savefig(output_path + 'aw_cont_fs.png')
# # plt.show()
#
# wm1 = np.argwhere(np.isclose(v_real - 0.1, 0., atol=1e-06))
# awk_fs = bz.shift_mat_by_pi(mat=-1. / np.pi * gk_arc_thiele_fbz[:, :, 0, nw // 2 - 10].imag, nk=k_grid.nk)
# plt.figure()
# plt.imshow(awk_fs, cmap='terrain', extent=bz.get_extent_pi_shift(kgrid=k_grid), vmin=0.02)
# plt.colorbar()
# plt.savefig(output_path + 'aw_cont_fs_shift.png')
# plt.show()
#
# wm1 = np.argwhere(np.isclose(v_real - 0.1, 0., atol=1e-06))
# awk_fs = bz.shift_mat_by_pi(mat=-1. / np.pi * gk_arc_thiele_fbz[:, :, 0, nw // 2].imag, nk=k_grid.nk)
# plt.figure()
# plt.imshow(awk_fs, cmap='terrain', extent=bz.get_extent_pi_shift(kgrid=k_grid), vmin=0.02)
# plt.colorbar()
# plt.savefig(output_path + 'aw_cont_fs.png')
# plt.show()
#
# plt.figure()
# plt.plot(v_real, -1. / np.pi * gk_arc_thiele_fbz.imag.mean(axis=(0, 1, 2)))
# plt.savefig(output_path + 'aw_cont_dga.png')
# plt.show()
