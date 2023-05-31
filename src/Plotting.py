# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import itertools
import matplotlib
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import FourPoint as fp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import MatsubaraFrequencies as mf
import OrnsteinZernickeFunction as ozfunc
import BrillouinZone as bz
import Config as conf
import TwoPoint_old as twop
import socket
import PlotSpecs as ps

if(socket.gethostname() != 'DESKTOP-OEHIPTV'):
    matplotlib.use('agg') # non GUI backend since VSC has no display

# -------------------------------------- DEFINE MODULE WIDE VARIABLES --------------------------------------------------

# __markers__ = itertools.cycle(('o','s','v','8','v','^','<','>','p','*','h','H','+','x','D','d','1','2','3','4'))
__markers__ = ('o', 's', 'v', '8', 'v', '^', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '1', '2', '3', '4')


# ------------------------------------------------ CLASSES -------------------------------------------------------------

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------START REIMPLEMENTATION OF THE DGA ROUTINES-----------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def plot_kx_ky(mat,kx,ky,do_save=True,pdir='./',name='',cmap='RdBu',figsize=ps.FIGSIZE,verbose=False):
    fig, axes = plt.subplots(ncols=2, figsize=figsize, dpi=500)
    axes = axes.flatten()
    im1 = axes[0].pcolormesh(kx, ky, mat.real, cmap=cmap)
    im2 = axes[1].pcolormesh(kx, ky, mat.imag, cmap=cmap)
    axes[0].set_title('$\Re$')
    axes[1].set_title('$\Im$')
    for ax in axes:
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        ax.set_aspect('equal')
    fig.suptitle(name)
    fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location='right',
                 pad=0.05)
    fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location='right',
                 pad=0.05)
    plt.tight_layout()
    if (do_save): plt.savefig(pdir + '/' + name + '.png')
    if (verbose):
        plt.show()
    else:
        plt.close()




def chi_checks(chi_dens,chi_magn,labels,green,plot_dir,verbose=False,do_plot=True,name=''):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 5), dpi=500)
    axes = axes.flatten()
    niw_chi_input = np.size(chi_dens[0])

    for i,cd in enumerate(chi_dens):
        axes[0].plot(mf.wn(len(cd) // 2), cd.real, label=labels[i])
    axes[0].set_ylabel('$\Re \chi(i\omega_n)_{dens}$')
    axes[0].legend()


    for i,cd in enumerate(chi_magn):
        axes[1].plot(mf.wn(len(cd) // 2), cd.real, label=labels[i])
    axes[1].set_ylabel('$\Re \chi(i\omega_n)_{magn}$')
    axes[1].legend()

    for i,cd in enumerate(chi_dens):
        axes[2].loglog(mf.wn(len(cd) // 2), cd.real, label=labels[i],ms=0)
    axes[2].loglog(mf.wn(niw_chi_input), np.real(1 / (mf.iw(green.beta, niw_chi_input) + 0.000001) ** 2 * green.e_kin) * 2, ls='--', label='Asympt', ms=0)
    axes[2].set_ylabel('$\Re \chi(i\omega_n)_{dens}$')
    axes[2].legend()

    for i,cd in enumerate(chi_magn):
        axes[3].loglog(mf.wn(len(cd) // 2), cd.real, label=labels[i],ms=0)
    axes[3].loglog(mf.wn(niw_chi_input), np.real(1 / (mf.iw(green.beta, niw_chi_input) + 0.000001) ** 2 * green.e_kin) * 2, '--', label='Asympt', ms=0)
    axes[3].set_ylabel('$\Re \chi(i\omega_n)_{magn}$')
    axes[3].legend()
    axes[0].set_xlim(-1, 10)
    axes[1].set_xlim(-1, 10)
    plt.tight_layout()
    if(do_plot): plt.savefig(plot_dir + f'/chi_dens_magn_'+name+'.png')
    if(verbose):
        plt.show()
    else:
        plt.close()

def local_diff_checks(arrs,labels,output_dir,verbose=False,do_plot=True,name='',xmax=None):
    ''''''
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(14, 5))
    axes = axes.flatten()

    for i,arr in enumerate(arrs):
        wn_1 = mf.wn_from_mat(arr[0])
        wn_2 = mf.wn_from_mat(arr[1])
        axes[0].plot(wn_1, arr[0].real, label=labels[i][0])
        axes[0].plot(wn_2, arr[1].real, label=labels[i][1])

        axes[1].loglog(wn_1, arr[0].real, label=labels[i][0])
        axes[1].loglog(wn_2, arr[1].real, label=labels[i][1])

        diff = np.abs(arr[0].real - arr[1].real)
        axes[2].loglog(wn_1, diff, label='diff-' + labels[i][0])

    axes[0].set_ylabel(r'$\Re '+'\\' + name + '$')
    axes[0].set_xlabel(r'$\omega_n$')

    axes[1].set_ylabel(r'$\Re '+'\\' + name + '$')
    axes[1].set_xlabel(r'$\omega_n$')

    axes[2].set_ylabel(r'$diff - \Re '+'\\' + name + '$')
    axes[2].set_xlabel(r'$\omega_n$')

    for ax in axes:
        ax.legend()
        ax.set_xlim(0, xmax)
    plt.tight_layout()
    if(do_plot): plt.savefig(output_dir  + '/' + name+'_diff_check.png')
    if(verbose):
        plt.show()
    else:
        plt.close()


def sigma_loc_checks(siw_arr,labels,beta,output_dir,verbose=False,do_plot=True,name='',xmax=None):
    if(xmax is None):
        xmax = 5+2*beta
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 5))
    axes = axes.flatten()

    for i,siw in enumerate(siw_arr):
        vn = mf.vn(np.size(siw) // 2)
        axes[0].plot(vn, siw.real, label=labels[i])
    axes[0].set_ylabel(r'$\Re \Sigma(i\nu_n)$')
    axes[0].set_xlabel(r'$\nu_n$')

    for i,siw in enumerate(siw_arr):
        vn = mf.vn(np.size(siw) // 2)
        axes[1].plot(vn, siw.imag, label=labels[i])
    axes[1].set_ylabel(r'$\Im \Sigma(i\nu_n)$')
    axes[1].set_xlabel(r'$\nu_n$')

    for i,siw in enumerate(siw_arr):
        vn = mf.vn(np.size(siw) // 2)
        axes[2].loglog(vn, siw.real, label=labels[i])
    axes[2].set_ylabel(r'$\Re \Sigma(i\nu_n)$')
    axes[2].set_xlabel(r'$\nu_n$')

    for i,siw in enumerate(siw_arr):
        vn = mf.vn(np.size(siw) // 2)
        axes[3].loglog(vn, np.abs(siw.imag), label=labels[i])
    axes[3].set_ylabel(r'$|\Im \Sigma(i\nu_n)|$')
    axes[3].set_xlabel(r'$\nu_n$')

    axes[0].set_xlim(0, xmax)
    axes[1].set_xlim(0, xmax)
    plt.legend()
    axes[1].set_ylim(None, 0)
    plt.tight_layout()
    if(do_plot): plt.savefig(output_dir + f'/sde_'+name+'_check.png')
    if(verbose):
        plt.show()
    else:
        plt.close()

def plot_fourpoint_nu_nup(mat,vn,do_save = True, pdir = './', name='NoName',cmap='RdBu',figsize=(8, 4)):
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    axes = axes.flatten()
    axes[0].pcolormesh(vn, vn, mat.real, cmap=cmap)
    axes[1].pcolormesh(vn, vn, mat.imag, cmap=cmap)
    plt.tight_layout()
    if (do_save): plt.savefig(pdir + '/' + name + '.png')
    plt.show()





def add_afzb(ax=None, kx=None, ky=None, lw=1.0, shift_pi=True):
    if (shift_pi):
        kx = kx - np.pi
        ky = ky - np.pi
        ax.plot(-ky, ky - np.pi, '--k', lw=lw)
        ax.plot(ky, ky - np.pi, '--k', lw=lw)
    else:
        ax.plot(ky, np.pi - ky, '--k', lw=lw)
        ax.plot(ky, ky - np.pi, '--k', lw=lw)
    ax.plot(kx, 0 * kx, 'k', lw=lw)
    ax.plot(0 * ky, ky, 'k', lw=lw)

    ax.set_xlim(kx[0], kx[-1])
    ax.set_ylim(ky[0], ky[-1])
    ax.set_xlabel('$k_y$')
    ax.set_ylabel('$k_x$')


def insert_colorbar(ax=None, im=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')

def plot_f_dmft(f=None,path=None,name=None,niv_cut=-1):
    niw = f.shape[0] // 2
    niv = f.shape[-1] // 2
    if(niv_cut == -1):
        niv_cut = niv
    plt.figure()
    plt.imshow(f[niw,niv-niv_cut:niv+niv_cut,niv-niv_cut:niv+niv_cut].real, cmap='RdBu', origin='lower')
    plt.colorbar()
    plt.xlabel(r'$\nu_p$')
    plt.xlabel(r'$\nu$')
    plt.title(name)
    plt.savefig(path + f'{name}.png')
    plt.show()
    plt.close()


def plot_bw_fit(bw_opt=None, bw=None, chi2=None, fit=None, output_path=None, name=None):
    plt.figure()
    plt.loglog(bw,chi2, '-o',label='$\chi^2(bw)$')
    plt.loglog(bw,np.exp(np.polyval(fit[0],np.log10(bw))), label='lin-fit(1)')
    plt.loglog(bw,np.exp(np.polyval(fit[1],np.log10(bw))), label='lin-fit(2)')
    plt.vlines(bw_opt,np.min(chi2), np.max(chi2), 'k',label='$bw_{opt}$')
    plt.legend()
    plt.xlabel('bw')
    plt.ylabel('$\chi^2$')
    plt.tight_layout()
    plt.savefig(output_path + '{}.png'.format(name))
    plt.show()
    plt.close()


def sigma_plots(dga_conf: conf.DgaConfig = None, sigma_dga=None, dmft_sde=None, sigma_loc=None, name='', sigma=None, sigma_nc = None):


    # Plot Siw-check:
    siw_list = [sigma_loc, dmft_sde['siw'], sigma.mean(axis=(0, 1, 2)), sigma_nc.mean(axis=(0, 1, 2))]
    labels = [r'$\Sigma_{DMFT}(\nu)$', r'$\Sigma_{DMFT-SDE}(\nu)$', r'$\Sigma_{DGA}(\nu)$', r'$\Sigma_{DGA-NC}(\nu)$']
    plot_siw(siw_list=siw_list, labels_list=labels, plot_dir=dga_conf.nam.output_path, niv_plot=100, name='siw_check'+name)

    # Plot non-local Siw contributions:
    siw_dga_magn_ksum = 3*sigma_dga['magn'].mean(axis=(0, 1, 2))  - 3 * dmft_sde['magn']
    siw_dga_dens_ksum = -sigma_dga['dens'].mean(axis=(0, 1, 2))  +  dmft_sde['dens']
    siw_list = [siw_dga_magn_ksum, siw_dga_dens_ksum]
    labels = [r'$\Sigma_{DGA-Magn}(\nu)$', r'$\Sigma_{DGA-Dens}(\nu)$']
    plot_siw(siw_list=siw_list, labels_list=labels, plot_dir=dga_conf.nam.output_path, niv_plot=100,name='siw_channel_contribution'+name)

    # Plot self-energy at the Fermi-surface:
    plot_siwk_fs(siwk=sigma, plot_dir=dga_conf.nam.output_path, kgrid=dga_conf.k_grid, do_shift=True,name=name)
    plot_siwk_fs(siwk=3*sigma_dga['magn']- 3 * dmft_sde['magn'], plot_dir=dga_conf.nam.output_path, kgrid=dga_conf.k_grid, do_shift=True, name='magn'+name)
    plot_siwk_fs(siwk=-sigma_dga['dens'] + dmft_sde['dens'], plot_dir=dga_conf.nam.output_path, kgrid=dga_conf.k_grid, do_shift=True, name='dens'+name)
    plot_siwk_fs(siwk=sigma_nc, plot_dir=dga_conf.nam.output_path, kgrid=dga_conf.k_grid, do_shift=True, name='nc'+name)

def giwk_plots(dga_conf: conf.DgaConfig = None, sigma=None, input_mu=None, name='', output_path = None, niv_plot=10):
    # Create the DGA Green's functions:
    gf_dict = twop.create_gk_dict(dga_conf=dga_conf, sigma=sigma, mu0=input_mu, adjust_mu=True)
    ind_gf0 = bz.find_qpd_zeros(qpd=(1. / gf_dict['gk'][:, :, :, gf_dict['niv']]).real, kgrid=dga_conf.k_grid)
    plot_giwk_fs(giwk=gf_dict['gk'], plot_dir=output_path, kgrid=dga_conf.k_grid, do_shift=True, name='dga'+name,
                          ind_fs=ind_gf0)
    plot_giwk_qpd(giwk=gf_dict['gk'], plot_dir=output_path, kgrid=dga_conf.k_grid, do_shift=True, name='dga'+name)
    plot_siw_along_fs(vn=dga_conf.box.vn_padded, siwk=sigma, ind_fs=ind_gf0, output_path=output_path, niv_plot=niv_plot,name='sigma_along_fs'+name)

def plot_siw_along_fs(siwk=None,ind_fs=None,vn=None, niv_plot = 10,niv_plot_min=-1, output_path=None, name='sigma_along_fs'):

    ind_v = np.logical_and(vn >= niv_plot_min, vn <= niv_plot)

    n_plots = len(ind_fs)
    line_colors = plt.cm.rainbow(np.linspace(0, 1, n_plots))
    lines = []

    plt.figure()
    plt.subplot(211)

    for i, ind in enumerate(ind_fs):
        tmp, = plt.plot(vn[ind_v], siwk[ind][ind_v].real,'-o', color=line_colors[i])
        lines.append(tmp)
    plt.legend(handles=[lines[0], lines[-1]], labels=['Anti-Node', 'Node'])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Re \Sigma$')
    plt.subplot(212)
    for i, ind in enumerate(ind_fs):
        plt.plot(vn[ind_v], siwk[ind][ind_v].imag,'-o', color=line_colors[i])
    plt.hlines(0,vn[ind_v][0],vn[ind_v][-1],linestyles='dashed', colors='k')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Im \Sigma$')
    plt.tight_layout()
    if (output_path is not None):
        plt.savefig(output_path + '{}.png'.format(name))
    plt.close()


def plot_vrg_dmft(vrg_dmft=None, beta=None, niv_plot=None, output_path=None):
    # Local spin-fermion vertex:
    plot_vrg_loc(vrg=vrg_dmft['magn'].mat * beta, niv_plot=niv_plot, pdir=output_path,
                 name='vrg_magn_loc')
    plot_vrg_loc(vrg=vrg_dmft['dens'].mat * beta, niv_plot=niv_plot, pdir=output_path,
                 name='vrg_dens_loc')


def plot_gamma_dmft(gamma_dmft=None, output_path=None, niw_core=None):
    plot_vertex_vvp(vertex=gamma_dmft['magn'].mat[niw_core, :, :].real, pdir=output_path,
                    name='gamma_magn')
    plot_vertex_vvp(vertex=gamma_dmft['dens'].mat[niw_core, :, :].real, pdir=output_path,
                    name='gamma_dens')


def plot_cont_edc_maps(v_real=None, gk_cont=None, k_grid=None, output_path=None, name=None, n_map=7, wplot=1):
    nk = k_grid.nk
    v0_ind = np.argmin(np.abs(v_real))
    gk_cont_shift = bz.shift_mat_by_pi(mat=gk_cont, nk=nk)
    extent = bz.get_extent_pi_shift(kgrid=k_grid)

    cuts = np.round(np.linspace(1, nk[0] // 4, n_map, endpoint=True)).astype(int)

    ind_fs = bz.find_qpd_zeros(qpd=(1. / gk_cont[:, :, :, v0_ind]).real, kgrid=k_grid)
    kx_fs = np.array([k_grid.kmesh[0][i] for i in ind_fs])
    ky_fs = np.array([k_grid.kmesh[1][i] for i in ind_fs])

    fig, axes = plt.subplots(2, 4, figsize=[13, 6])
    axes = axes.flatten()

    im = axes[0].imshow(-1. / np.pi * gk_cont_shift.imag[:, :, 0, v0_ind], extent=extent, cmap='terrain',
                        origin='lower', aspect='auto')
    axes[0].set_ylabel('$k_x$')
    axes[0].set_xlabel('$k_y$')
    axes[0].plot(-kx_fs, -ky_fs, color='k')
    insert_colorbar(ax=axes[0], im=im)
    axes[0].set_xlim([extent[0], 0])
    axes[0].set_ylim([extent[2], 0])
    for i in range(n_map):
        im = axes[i + 1].imshow(-1. / np.pi * gk_cont_shift.imag[cuts[i], :, 0, :].T, cmap='terrain', aspect='auto',
                                origin='lower', extent=extent)
        insert_colorbar(ax=axes[i + 1], im=im)
        axes[0].plot(k_grid.ky - np.pi, (k_grid.kx[cuts[i]] - np.pi) * np.ones((nk[1])), '--', color='orange')
        kx_line_ind = np.argmin(np.abs(ky_fs - k_grid.kx[cuts[i]]))
        axes[i + 1].vlines(-ky_fs[kx_line_ind], -wplot, wplot, 'k', ls='--')

    for ax in axes[1:]:
        ax.hlines(0, -np.pi, np.pi, 'k')
        ax.set_ylim([-wplot, wplot])
        ax.set_xlim([extent[2], 0])
        ax.set_ylabel('$\omega$')
        ax.set_xlabel('$k_y$')

    fig.suptitle(name)
    plt.tight_layout()
    plt.savefig(output_path + '{}.png'.format(name))
    plt.show()
    plt.close()


def plot_siwk_extrap(siwk_re_fs=None, siwk_im_fs=None, siwk_Z=None, output_path=None, name='', k_grid=None, lw=1):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    siwk_re_fs = np.squeeze(bz.shift_mat_by_pi(mat=siwk_re_fs, nk=k_grid.nk))
    siwk_im_fs = np.squeeze(bz.shift_mat_by_pi(mat=siwk_im_fs, nk=k_grid.nk))
    siwk_Z = np.squeeze(bz.shift_mat_by_pi(mat=siwk_Z, nk=k_grid.nk))
    norm = MidpointNormalize(midpoint=0, vmin=siwk_re_fs.min(), vmax=siwk_re_fs.max())
    im = ax[0].imshow(siwk_re_fs, cmap='RdBu_r', extent=bz.get_extent_pi_shift(kgrid=k_grid), norm=norm, origin='lower')
    insert_colorbar(ax=ax[0], im=im)
    ax[0].set_title(r'$\Re \Sigma(k,\nu=0)$')
    norm = MidpointNormalize(midpoint=0, vmin=siwk_im_fs.min(), vmax=siwk_im_fs.max())
    im = ax[1].imshow(siwk_im_fs, cmap='RdBu', extent=bz.get_extent_pi_shift(kgrid=k_grid), norm=norm, origin='lower')
    insert_colorbar(ax=ax[1], im=im)
    ax[1].set_title(r'$\Im \Sigma(k,\nu=0)$')
    norm = MidpointNormalize(midpoint=0, vmin=siwk_Z.min(), vmax=siwk_Z.max())
    im = ax[2].imshow(siwk_Z, cmap='RdBu', extent=bz.get_extent_pi_shift(kgrid=k_grid), norm=norm, origin='lower')
    insert_colorbar(ax=ax[2], im=im)
    ax[2].set_title('Z(k)')

    add_afzb(ax=ax[0], kx=k_grid.kx, ky=k_grid.kx, lw=lw)
    add_afzb(ax=ax[1], kx=k_grid.kx, ky=k_grid.kx, lw=lw)
    add_afzb(ax=ax[2], kx=k_grid.kx, ky=k_grid.kx, lw=lw)

    plt.tight_layout()
    fig.suptitle(name)
    plt.savefig(output_path + '{}.png'.format(name))
    plt.show()
    plt.close()


def plot_spin_fermion_w0_special_points(output_path=None, name='', vrg_w0=None, labels=None):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))

    n_loc = np.shape(vrg_w0)[0]
    niv = np.shape(vrg_w0)[-1] // 2
    v = mf.vn(n=niv)
    for i in range(n_loc):
        ax[0].plot(v, vrg_w0[i].real, '-o', label=labels[i])
        ax[1].plot(v, vrg_w0[i].imag, '-o', label=labels[i])

    ax[0].set_ylabel('$\Re \gamma$')
    ax[1].set_ylabel('$\Im \gamma$')

    ax[0].set_xlabel(r'$\nu$')
    ax[1].set_xlabel(r'$\nu$')

    ax[0].grid()
    ax[1].grid()

    ax[0].legend()
    plt.savefig(output_path + name + '.png')
    plt.close()


def plot_spin_fermion_fs(output_path=None, name='', vrg_fs=None, q_grid=None, lw=1.0):
    vrg_plot = vrg_fs  # np.squeeze(bz.shift_mat_by_pi(mat=vrg_fs, nk=q_grid.nk))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    im = ax[0].imshow(vrg_plot.real, cmap='terrain', extent=bz.get_extent(kgrid=q_grid), origin='lower')
    insert_colorbar(ax=ax[0], im=im)
    ax[0].set_title(r'$\Re \gamma(k,\omega=0, \nu=\pi/\beta)$')

    im = ax[1].imshow(vrg_plot.imag, cmap='terrain', extent=bz.get_extent(kgrid=q_grid), origin='lower')
    insert_colorbar(ax=ax[1], im=im)
    ax[1].set_title(r'$\Im \gamma(k,\omega=0, \nu=\pi/\beta)$')

    add_afzb(ax=ax[0], kx=q_grid.kx, ky=q_grid.kx, lw=lw, shift_pi=False)
    add_afzb(ax=ax[1], kx=q_grid.kx, ky=q_grid.kx, lw=lw, shift_pi=False)

    plt.tight_layout()
    fig.suptitle(name)
    plt.savefig(output_path + '{}.png'.format(name))
    plt.show()
    plt.close()


def plot_cont_fs(output_path=None, name='', gk=None, v_real=None, k_grid=None, w_int=None, w_plot=None, lw=1.0):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    if (w_int == None):
        w0_ind = np.argmin(np.abs(v_real-w_plot))
        gk_fs = gk[:, :, 0, w0_ind]
    else:
        ind_int = np.logical_and(v_real < 0, w_int < v_real)
        gk_fs = np.trapz(gk[:, :, 0, ind_int], v_real[ind_int]) / np.abs(w_int)

    awk_fs = np.squeeze(bz.shift_mat_by_pi(mat=-1. / np.pi * gk_fs.imag, nk=k_grid.nk))
    gk_real = np.squeeze(bz.shift_mat_by_pi(mat=gk_fs.real, nk=k_grid.nk))
    v0 = np.argmin(np.abs(v_real))
    qdp = np.squeeze(bz.shift_mat_by_pi(mat=(1. / gk)[:, :, 0, v0].real, nk=k_grid.nk))
    im = ax[0].imshow(awk_fs, cmap='RdBu_r', extent=bz.get_extent_pi_shift(kgrid=k_grid), origin='lower')
    insert_colorbar(ax=ax[0], im=im)
    ax[0].set_title('$A(k,\omega=0)$')
    norm = MidpointNormalize(midpoint=0, vmin=gk_real.min(), vmax=gk_real.max())
    im = ax[1].imshow(gk_real, cmap='RdBu', extent=bz.get_extent_pi_shift(kgrid=k_grid), norm=norm, origin='lower')
    insert_colorbar(ax=ax[1], im=im)
    ax[1].set_title('$\Re G(k,\omega=0)$')
    norm = MidpointNormalize(midpoint=0, vmin=qdp.min(), vmax=qdp.max())
    im = ax[2].imshow(qdp, cmap='RdBu', extent=bz.get_extent_pi_shift(kgrid=k_grid), norm=norm, origin='lower')
    insert_colorbar(ax=ax[2], im=im)
    ax[2].set_title('QPD')

    add_afzb(ax=ax[0], kx=k_grid.kx, ky=k_grid.kx, lw=lw)
    add_afzb(ax=ax[1], kx=k_grid.kx, ky=k_grid.kx, lw=lw)
    add_afzb(ax=ax[2], kx=k_grid.kx, ky=k_grid.kx, lw=lw)

    plt.tight_layout()
    fig.suptitle(name)
    plt.savefig(output_path + '{}.png'.format(name))
    plt.show()
    plt.close()


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

    gk_fs_plot = bz.shift_mat_by_pi(mat=gk_fs, nk=k_grid.nk)

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
    im = ax[1].imshow(gr, cmap='RdBu', extent=extent, norm=norm, origin='lower')
    insert_colorbar(ax=ax[1], im=im)
    for i in ind_fs:
        ax[1].plot(-k_grid.kx[i[0]], -k_grid.ky[i[1]], 'o')

    qpd = (1. / gk_fs_plot).real
    norm = MidpointNormalize(midpoint=0, vmin=qpd.min(), vmax=qpd.max())
    im = ax[2].imshow(qpd, cmap='RdBu', extent=extent, norm=norm, origin='lower')
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


def plot_fs_peaks(ax=None, k_grid=None, ind_fs=None, color='b'):
    kx = np.array([k_grid.kmesh[0][i] for i in ind_fs])
    ky = np.array([k_grid.kmesh[1][i] for i in ind_fs])
    shift_x = 0.0  # np.pi/(k_grid.nk[0])
    shift_y = 0.0  # np.pi/(k_grid.nk[1])
    ax.plot(kx - shift_x, ky - shift_y, color=color)


def plot_oz_fit(chi_w0=None, oz_coeff=None, qgrid=None, pdir=None, name=''):
    oz = ozfunc.oz_spin_w0(qgrid, oz_coeff[0], oz_coeff[1]).reshape(qgrid.nk)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
    ax[0].plot(qgrid.kx, chi_w0[:, qgrid.nk[1] // 2, 0].real, 'o', label='$\chi$')
    ax[0].plot(qgrid.kx, oz[:, qgrid.nk[1] // 2, 0].real, '-', label='oz-fit')
    ax[0].legend()
    ax[0].set_title('$q_y = np.pi$')
    ax[0].set_xlabel('$q_x$')
    ax[0].set_ylabel('$\chi$')

    mask = qgrid.kmesh[0] == qgrid.kmesh[1]
    ax[1].plot(qgrid.kx, chi_w0[mask].real, 'o', label='$\chi$')
    ax[1].plot(qgrid.kx, oz[mask].real, '-', label='oz-fit')
    ax[1].legend()
    ax[1].set_title('$q_y = q_x$')
    ax[1].set_xlabel('$q_x$')
    ax[1].set_ylabel('$\chi$')
    plt.savefig(pdir + name + '.png')
    plt.close()


def plot_vrg_loc(vrg=None, niw_plot=None, niv_plot=0, pdir=None, name='vrg_loc'):
    if (niw_plot is None):
        niw_plot = vrg.shape[0] // 2

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    niv_vrg = vrg.shape[-1] // 2
    vn = mf.vn(n=niv_plot)

    ax[0].plot(vn, vrg[niw_plot, niv_vrg - niv_plot:niv_vrg + niv_plot].real, 'o')
    ax[0].set_ylabel('$\Re \gamma$')

    ax[1].plot(vn, vrg[niw_plot, niv_vrg - niv_plot:niv_vrg + niv_plot].imag, 'o')
    ax[1].set_ylabel('$\Im \gamma$')
    ax[0].grid()
    ax[1].grid()
    plt.savefig(pdir + name + '.png')
    plt.close()


def plot_susc(susc=None):
    plt.plot(susc.iw_core, susc.mat.real, 'o')
    plt.xlim([-2, susc.beta])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'\chi')
    plt.close()


def get_extent(kgrid=None):
    return [kgrid.kx[0], kgrid.kx[-1], kgrid.ky[0], kgrid.ky[-1]]


def plot_chi_fs(chi=None, output_path=None, kgrid=None, niv_plot=None, kz=0, name=''):
    if (niv_plot is None):
        niv_plot = chi.shape[-1] // 2
    extent = get_extent(kgrid=kgrid)
    plt.figure()
    plt.imshow(chi[:, :, kz, niv_plot], cmap='RdBu', extent=extent, origin='lower')
    plt.xlabel(r'$k_y$')
    plt.ylabel(r'$k_x$')
    plt.colorbar()
    plt.savefig(output_path + 'chi_{}.png'.format(name))
    plt.close()


def plot_fp(fp: fp.LocalFourPoint = None, w_plot=0, name='', niv_plot=-1):
    if (niv_plot == -1):
        niv_plot = fp.niv
    A = fp
    plt.imshow(
        A.mat[A.wn == w_plot, A.niv - niv_plot:A.niv + niv_plot, A.niv - niv_plot:A.niv + niv_plot].squeeze().real,
        cmap='RdBu', extent=[-niv_plot, niv_plot, -niv_plot, niv_plot])
    plt.colorbar()
    plt.title(name + '-' + A.channel + '(' + r'$ \omega=$' + '{})'.format(w_plot))
    plt.xlabel(r'$\nu\prime$')
    plt.ylabel(r'$\nu$')
    plt.show()


def plot_tp(tp: fp.LocalThreePoint = None, niv_cut=-1, name=''):
    if (niv_cut == -1):
        niv_cut = tp.niv
    A = tp
    plt.imshow(A.mat.real[:, tp.niv - niv_cut:tp.niv + niv_cut], cmap='RdBu',
               extent=[-niv_cut, niv_cut, A.wn[0], A.wn[-1]])
    plt.colorbar()
    plt.title(r'$\Re$' + name + '-' + A.channel)
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$\omega$')
    plt.show()

    plt.imshow(A.mat.imag[:, tp.niv - niv_cut:tp.niv + niv_cut], cmap='RdBu',
               extent=[A.wn[0], A.wn[-1], -niv_cut, niv_cut])
    plt.colorbar()
    plt.title(r'$\Im$' + name + '-' + A.channel)
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$\omega$')
    plt.show()


def plot_chiw(wn_list=None, chiw_list=None, labels_list=None, channel=None, plot_dir=None, niw_plot=20):
    markers = __markers__
    np = len(wn_list)
    assert np < len(markers), 'More plot-lines requires, than markers available.'

    size = 2 * np + 1

    for i in range(len(wn_list)):
        plt.plot(wn_list[i], chiw_list[i].real, markers[i], ms=size - 2 * i, label=labels_list[i])
    plt.legend()
    plt.xlim(-2, niw_plot)
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\chi_{}$'.format(channel))
    if (plot_dir is not None):
        plt.savefig(plot_dir + 'chiw_{}.png'.format(channel))
        plt.close()


def plot_siw(siw_list=None, labels_list=None, plot_dir=None, niv_plot_min=-10, niv_plot=200,
             name='siw_check', ncol=1, ms=2):
    markers = __markers__
    nplot = len(siw_list)
    assert nplot < len(markers), 'More plots-lines requires, than markers avaiable.'

    plt.figure()
    plt.subplot(211)

    for i in range(nplot):
        niv = siw_list[i].shape[-1] // 2
        vn = mf.vn(n=niv)
        ind = np.logical_and(vn >= niv_plot_min, vn <= niv_plot)
        plt.plot(vn[ind], siw_list[i][ind].real, markers[i], ms=ms, label=labels_list[i])
    plt.legend()
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Re \Sigma$')
    plt.subplot(212)
    for i in range(nplot):
        niv = siw_list[i].shape[-1] // 2
        vn = mf.vn(n=niv)
        ind = np.logical_and(vn >= niv_plot_min, vn <= niv_plot)
        plt.plot(vn[ind], siw_list[i][ind].imag, markers[i], ms=ms, label=labels_list[i])
    plt.legend(ncol=ncol, loc='upper right')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Im \Sigma$')
    plt.tight_layout()
    if (plot_dir is not None):
        plt.savefig(plot_dir + '{}.png'.format(name))

    plt.close()


def plot_siwk_fs(siwk=None, plot_dir=None, kgrid=None, do_shift=False, kz=0, niv_plot=None, name=''):
    fig, ax = plot_fs(siwk=siwk, kgrid=kgrid, do_shift=do_shift, kz=kz, niv_plot=niv_plot)
    ax[0][0].set_title('$\Re \Sigma$')
    ax[0][1].set_title('$\Im \Sigma$')

    if (plot_dir is not None):
        plt.savefig(plot_dir + 'siwk_fermi_surface_{}.png'.format(name))
        plt.close()


def plot_giwk_fs(giwk=None, plot_dir=None, kgrid=None, do_shift=False, kz=0, niv_plot=None, name='', ind_fs=None):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    if (ind_fs is not None):
        plot_fs_peaks(ax=ax[0][0], k_grid=kgrid, ind_fs=ind_fs)
        plot_fs_peaks(ax=ax[0][1], k_grid=kgrid, ind_fs=ind_fs)
        plot_fs_peaks(ax=ax[1][0], k_grid=kgrid, ind_fs=ind_fs)
        plot_fs_peaks(ax=ax[1][1], k_grid=kgrid, ind_fs=ind_fs)

        n_plots = len(ind_fs)
        line_colors = plt.cm.rainbow(np.linspace(0, 1, n_plots))
        for i, ind in enumerate(ind_fs):
            ax[0][0].plot(-kgrid.kx[ind[0]], -kgrid.ky[ind[1]], 'o', color=line_colors[i])
            ax[0][1].plot(-kgrid.kx[ind[0]], -kgrid.ky[ind[1]], 'o', color=line_colors[i])
            ax[1][0].plot(-kgrid.kx[ind[0]], -kgrid.ky[ind[1]], 'o', color=line_colors[i])
            ax[1][1].plot(-kgrid.kx[ind[0]], -kgrid.ky[ind[1]], 'o', color=line_colors[i])


    plot_contour(ax=ax[0][0], siwk=giwk.real, kgrid=kgrid, do_shift=do_shift, kz=kz, niv_plot=niv_plot, cmap='RdBu',
                 midpoint_norm=True)
    plot_contour(ax=ax[0][1], siwk=giwk.imag, kgrid=kgrid, do_shift=do_shift, kz=kz, niv_plot=niv_plot, cmap='RdBu',
                 midpoint_norm=False)
    plot_contour(ax=ax[1][0], siwk=giwk.real, kgrid=kgrid, do_shift=do_shift, kz=kz, niv_plot=niv_plot,
                 cmap='terrain_r', midpoint_norm=False)
    plot_contour(ax=ax[1][1], siwk=giwk.imag, kgrid=kgrid, do_shift=do_shift, kz=kz, niv_plot=niv_plot,
                 cmap='terrain_r', midpoint_norm=False)

    ax[0][0].set_title('$\Re G$')
    ax[0][1].set_title('$\Im G$')

    plt.tight_layout()

    if (plot_dir is not None):
        plt.savefig(plot_dir + 'giwk_fermi_surface_{}.png'.format(name))
        plt.close()


def plot_giwk_qpd(giwk=None, plot_dir=None, kgrid=None, do_shift=False, kz=0, niv_plot=None, name=''):
    fig, ax = plot_qpd(siwk=(1. / giwk).real, kgrid=kgrid, do_shift=do_shift, kz=kz, niv_plot=niv_plot)

    ax[0][0].set_title('$\Re (1./G)$')
    ax[0][1].set_title('$\Im (1./G)$')

    if (plot_dir is not None):
        plt.savefig(plot_dir + 'QuasiParticleDispersion_{}.png'.format(name))
        plt.close()


def plot_fs(siwk=None, kgrid=None, do_shift=False, kz=0, niv_plot=None):
    if (niv_plot == None):
        niv_plot = np.shape(siwk)[-1] // 2

    siwk_plot = np.copy(siwk)
    kx = kgrid.kx
    ky = kgrid.ky
    if (do_shift):
        siwk_plot = np.roll(siwk_plot, kgrid.nk[0] // 2, 0)
        siwk_plot = np.roll(siwk_plot, kgrid.nk[1] // 2, 1)
        kx = kx - np.pi
        ky = ky - np.pi

    lw = 1.0

    def add_lines(ax):
        ax.plot(kx, 0 * kx, 'k', lw=lw)
        ax.plot(0 * ky, ky, 'k', lw=lw)
        ax.plot(-ky, ky - np.pi, '--k', lw=lw)
        ax.plot(ky, ky - np.pi, '--k', lw=lw)
        ax.plot(-ky, ky + np.pi, '--k', lw=lw)
        ax.plot(ky, ky + np.pi, '--k', lw=lw)

    def create_image(ax=None, contour=None, cmap='RdBu'):
        add_lines(ax)
        im = ax.imshow(contour, cmap=cmap, extent=[kx[0], kx[-1], ky[0], ky[-1]],
                       origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_xlabel(r'$k_y$')
        ax.set_ylabel(r'$k_x$')

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    create_image(ax=ax[0][0], contour=siwk_plot[:, :, kz, niv_plot].real, cmap='RdBu')
    create_image(ax=ax[0][1], contour=siwk_plot[:, :, kz, niv_plot].imag, cmap='RdBu')

    create_image(ax=ax[1][0], contour=siwk_plot[:, :, kz, niv_plot].real, cmap='terrain_r')
    create_image(ax=ax[1][1], contour=siwk_plot[:, :, kz, niv_plot].imag, cmap='terrain_r')

    plt.tight_layout()

    return fig, ax


def plot_contour(ax=None, siwk=None, kgrid=None, do_shift=False, kz=0, niv_plot=None, cmap='RdBu', midpoint_norm=False):
    if (niv_plot == None):
        niv_plot = np.shape(siwk)[-1] // 2

    siwk_plot = np.copy(siwk[:, :, kz, niv_plot])
    kx = kgrid.kx
    ky = kgrid.ky
    if (do_shift):
        siwk_plot = np.roll(siwk_plot, kgrid.nk[0] // 2, 0)
        siwk_plot = np.roll(siwk_plot, kgrid.nk[1] // 2, 1)
        kx = kx - np.pi
        ky = ky - np.pi

    siwk_plot = bz.shift_mat_by_pi(mat=siwk[:, :, kz, niv_plot], nk=kgrid.nk)
    lw = 1.0

    def add_lines(ax):
        ax.plot(kx, 0 * kx, 'k', lw=lw)
        ax.plot(0 * ky, ky, 'k', lw=lw)
        ax.plot(-ky, ky - np.pi, '--k', lw=lw)
        ax.plot(ky, ky - np.pi, '--k', lw=lw)
        ax.plot(-ky, ky + np.pi, '--k', lw=lw)
        ax.plot(ky, ky + np.pi, '--k', lw=lw)

    def create_image(ax=None, contour=None, cmap='RdBu', norm=None):
        add_lines(ax)
        im = ax.imshow(contour, cmap=cmap, extent=[kx[0], kx[-1], ky[0], ky[-1]],
                       origin='lower', norm=norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        ax.set_xlabel(r'$k_y$')
        ax.set_ylabel(r'$k_x$')

    if (midpoint_norm):
        norm = MidpointNormalize(midpoint=0, vmin=siwk_plot.min(), vmax=siwk_plot.max())
        create_image(ax=ax, contour=siwk_plot, cmap=cmap, norm=norm)
    else:
        create_image(ax=ax, contour=siwk_plot, cmap=cmap)
    return


def plot_qpd(siwk=None, kgrid=None, do_shift=False, kz=0, niv_plot=None):
    if (niv_plot == None):
        niv_plot = np.shape(siwk)[-1] // 2

    siwk_plot = np.copy(siwk[:, :, kz, niv_plot])
    kx = kgrid.kx
    ky = kgrid.ky
    if (do_shift):
        siwk_plot = np.roll(siwk_plot, kgrid.nk[0] // 2, 0)
        siwk_plot = np.roll(siwk_plot, kgrid.nk[1] // 2, 1)
        kx = kx - np.pi
        ky = ky - np.pi

    lw = 1.0

    def add_lines(ax):
        ax.plot(kx, 0 * kx, 'k', lw=lw)
        ax.plot(0 * ky, ky, 'k', lw=lw)
        ax.plot(-ky, ky - np.pi, '--k', lw=lw)
        ax.plot(ky, ky - np.pi, '--k', lw=lw)
        ax.plot(-ky, ky + np.pi, '--k', lw=lw)
        ax.plot(ky, ky + np.pi, '--k', lw=lw)

    def create_image(ax=None, contour=None, cmap='RdBu', norm=None):
        add_lines(ax)
        im = ax.imshow(contour, cmap=cmap, extent=[kx[0], kx[-1], ky[0], ky[-1]],
                       origin='lower', norm=norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax.set_xlabel(r'$k_y$')
        ax.set_ylabel(r'$k_x$')

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    norm = MidpointNormalize(midpoint=0, vmin=siwk_plot.min(), vmax=siwk_plot.max())
    create_image(ax=ax[0][0], contour=siwk_plot, cmap='RdBu', norm=norm)
    create_image(ax=ax[0][1], contour=siwk_plot, cmap='RdBu', norm=norm)

    bin_imag = np.copy(siwk_plot)
    bin_imag[bin_imag > 0] = 1
    bin_imag[bin_imag < 0] = 0
    create_image(ax=ax[1][0], contour=bin_imag, cmap='binary')
    create_image(ax=ax[1][1], contour=bin_imag, cmap='binary')

    plt.tight_layout()

    return fig, ax


def plot_gap_function(delta=None, pdir=None, name='', kgrid=None, do_shift=False):
    niv = np.shape(delta)[-1] // 2
    kx = kgrid.kx
    ky = kgrid.ky

    delta_plot = np.copy(delta)
    if (do_shift):
        delta_plot = np.roll(delta_plot, kgrid.nk[0] // 2, 0)
        delta_plot = np.roll(delta_plot, kgrid.nk[1] // 2, 1)
        kx = kx - np.pi
        ky = ky - np.pi

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # First positive Matsubara frequency:

    im = ax[0].imshow(delta_plot[:, :, 0, niv].real, cmap='RdBu', origin='lower', extent=[kx[0], kx[-1], ky[0], ky[-1]])
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # First negative Matsubara frequency:
    im = ax[1].imshow(delta_plot[:, :, 0, niv - 1].real, cmap='RdBu', origin='lower',
                      extent=[kx[0], kx[-1], ky[0], ky[-1]])
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax[0].set_xlabel(r'$k_x$')
    ax[0].set_ylabel(r'$k_y$')
    ax[0].set_title(r'$\nu_{n=0}$')

    ax[1].set_xlabel(r'$k_x$')
    ax[1].set_ylabel(r'$k_y$')
    ax[1].set_title(r'$\nu_{n=-1}$')

    plt.tight_layout()
    plt.savefig(pdir + 'GapFunction_{}.png'.format(name))
    plt.close()


def plot_vertex_vvp(vertex=None, cmap='RdBu', pdir=None, name='Vertex'):
    plt.figure()
    plt.imshow(vertex, cmap=cmap)
    plt.xlabel(r'\nu \' ')
    plt.xlabel(r'\nu')
    plt.colorbar()
    plt.savefig(pdir + '{}.png'.format(name))
    plt.close()
