# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import dga.matsubara_frequencies as mf
import dga.ornstein_zernicke_function as ozfunc
import dga.brillouin_zone as bz
import dga.config as config
import socket
from matplotlib.cm import ScalarMappable
import dga.plot_specs as ps

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

def get_zero_contour(data):
    indx = np.arange(0, np.shape(data)[1])
    indy = np.arange(0, np.shape(data)[0])
    fig1 = plt.figure()
    cs1 = plt.contour(indx, indy, data, cmap='RdBu', levels=[0, ])
    path = cs1.collections[0].get_paths()
    plt.close(fig1)
    for i in range(len(path)):
        if (i == 0):
            indices_x = np.array(np.round(path[i].vertices[:, 0], 0).astype(int))
            indices_y = np.array(np.round(path[i].vertices[:, 1], 0).astype(int))
        else:
            indices_x = np.concatenate((indices_x, np.array(np.round(path[i].vertices[:, 0], 0).astype(int))), axis=0)
            indices_y = np.concatenate((indices_y, np.array(np.round(path[i].vertices[:, 1], 0).astype(int))), axis=0)
    return indices_x, indices_y


def default_g2_plots(g2_dens, g2_magn, output_dir):
    ''' Default plots for the two-particle Green's function'''
    g2_dens.plot(0, pdir=output_dir, name='G2_dens')
    g2_magn.plot(0, pdir=output_dir, name='G2_magn')
    g2_magn.plot(10, pdir=output_dir, name='G2_magn')
    g2_magn.plot(-10, pdir=output_dir, name='G2_magn')


def default_gchi_plots(gchi_dens, gchi_magn, output_dir):
    ''' Default plots for the two-particle Green's function'''
    gchi_dens.plot(0, pdir=output_dir, name='Gchi_dens')
    gchi_magn.plot(0, pdir=output_dir, name='Gchi_magn')


def default_gamma_plots(gamma_dens, gamma_magn, output_dir, box_sizes, beta):
    ''' Default plots for Gamma. '''
    niv_core = box_sizes.niv_core
    gamma_dens.plot(0, pdir=output_dir, niv=min(niv_core, 2 * int(beta)), name='Gamma_dens')
    gamma_magn.plot(0, pdir=output_dir, niv=min(niv_core, 2 * int(beta)), name='Gamma_magn')
    gamma_magn.plot(10, pdir=output_dir, niv=min(niv_core, 2 * int(beta)), name='Gamma_magn')
    gamma_magn.plot(-10, pdir=output_dir, niv=min(niv_core, 2 * int(beta)), name='Gamma_magn')
    gamma_dens.plot(10, pdir=output_dir, niv=min(niv_core, 2 * int(beta)), name='Gamma_dens')


def plot_along_ind(mat, indizes, step_size=1, figsize=ps.FIGSIZE, cmap='rainbow', verbose=False, pdir='./', do_save=True
                   , niv_plot_min=0, niv_plot=-1, name='', ikz=0):
    '''
        Plot mat along indizes. This is primarily used to plot the self-energy (Green's function) along the Fermi surface.
    '''

    if (niv_plot == -1):
        niv_plot = np.shape(mat)[-1] // 2
    vn = mf.vn(mat)
    ind_v = np.logical_and(vn >= niv_plot_min, vn <= niv_plot)

    n_plots = len(indizes[::step_size])
    line_colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_plots))
    lines = []

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    for i, ind in enumerate(indizes[::step_size]):
        tmp, = axes[0].plot(vn[ind_v], mat[ind[0], ind[1], ikz, ind_v].real, '-o', color=line_colors[i], ms=4, alpha=0.8)
        lines.append(tmp)
    plt.legend(handles=[lines[0], lines[-1]], labels=['Anti-Node', 'Node'])
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\Re ' + '\\' + name + '$')

    for i, ind in enumerate(indizes[::step_size]):
        axes[1].plot(vn[ind_v], mat[ind[0], ind[1], ikz, ind_v].imag, '-o', color=line_colors[i], ms=4, alpha=0.8)
    plt.hlines(0, vn[ind_v][0], vn[ind_v][-1], linestyles='dashed', colors='k')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\Im ' + '\\' + name + '$')
    plt.tight_layout()
    if (do_save):
        plt.savefig(pdir + '/' + f'{name}_along_Fermi_surface.png')
    if (verbose):
        plt.show()
    else:
        plt.close()


def plot_kx_ky(mat, kx, ky, do_save=True, pdir='./', name='', cmap='RdBu', figsize=ps.FIGSIZE, verbose=False, scatter=None):
    '''
        mat: [nkx,nky]; object in the Brillouin zone.
    '''
    fig, axes = plt.subplots(ncols=2, figsize=figsize, dpi=500)
    axes = axes.flatten()
    im1 = axes[0].pcolormesh(kx, ky, mat.T.real, cmap=cmap)
    im2 = axes[1].pcolormesh(kx, ky, mat.T.imag, cmap=cmap)
    axes[0].set_title('$\Re$')
    axes[1].set_title('$\Im$')
    for ax in axes:
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        ax.set_aspect('equal')
        add_afzb(ax=ax, kx=kx, ky=ky, lw=1.0, marker='')
    fig.suptitle(name)
    fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location='right',
                 pad=0.05)
    fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location='right',
                 pad=0.05)
    if (scatter is not None):
        for ax in axes:
            colours = plt.cm.get_cmap(cmap)(np.linspace(0, 1, np.shape(scatter)[0]))
            ax.scatter(scatter[:, 0], scatter[:, 1], marker='o', c=colours)
    plt.tight_layout()
    if (do_save): plt.savefig(pdir + '/' + name + '.png')
    if (verbose):
        plt.show()
    else:
        plt.close()


def chi_checks(chi_dens, chi_magn, labels, green, plot_dir, verbose=False, do_plot=True, name=''):
    '''
        Routine plots to inspect chi_dens and chi_magn
    '''
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 5), dpi=500)
    axes = axes.flatten()
    niw_chi_input = np.size(chi_dens[0])

    for i, cd in enumerate(chi_dens):
        axes[0].plot(mf.wn(len(cd) // 2), cd.real, label=labels[i])
    axes[0].set_ylabel('$\Re \chi(i\omega_n)_{dens}$')
    axes[0].legend()

    for i, cd in enumerate(chi_magn):
        axes[1].plot(mf.wn(len(cd) // 2), cd.real, label=labels[i])
    axes[1].set_ylabel('$\Re \chi(i\omega_n)_{magn}$')
    axes[1].legend()

    for i, cd in enumerate(chi_dens):
        axes[2].loglog(mf.wn(len(cd) // 2), cd.real, label=labels[i], ms=0)
    axes[2].loglog(mf.wn(niw_chi_input), np.real(1 / (1j * mf.wn(green.beta, niw_chi_input) + 0.000001) ** 2 * green.e_kin) * 2,
                   ls='--', label='Asympt', ms=0)
    axes[2].set_ylabel('$\Re \chi(i\omega_n)_{dens}$')
    axes[2].legend()

    for i, cd in enumerate(chi_magn):
        axes[3].loglog(mf.wn(len(cd) // 2), cd.real, label=labels[i], ms=0)
    axes[3].loglog(mf.wn(niw_chi_input), np.real(1 / (1j * mf.wn(green.beta, niw_chi_input) + 0.000001) ** 2 * green.e_kin) * 2, \
                   '--',
                   label='Asympt', ms=0)
    axes[3].set_ylabel('$\Re \chi(i\omega_n)_{magn}$')
    axes[3].legend()
    axes[0].set_xlim(-1, 10)
    axes[1].set_xlim(-1, 10)
    plt.tight_layout()
    if (do_plot): plt.savefig(plot_dir + f'/chi_dens_magn_' + name + '.png')
    if (verbose):
        plt.show()
    else:
        plt.close()


def local_diff_checks(arrs, labels, output_dir, verbose=False, do_plot=True, name='', xmax=None):
    '''
        arrs: [n,nw,2]; plot two times n arrays and their respective differences.
    '''
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(14, 5))
    axes = axes.flatten()

    for i, arr in enumerate(arrs):
        wn_1 = mf.wn(arr[0])
        wn_2 = mf.wn(arr[1])
        axes[0].plot(wn_1, arr[0].real, label=labels[i][0])
        axes[0].plot(wn_2, arr[1].real, label=labels[i][1])

        axes[1].loglog(wn_1, arr[0].real, label=labels[i][0])
        axes[1].loglog(wn_2, arr[1].real, label=labels[i][1])

        diff = np.abs(arr[0].real - arr[1].real)
        axes[2].loglog(wn_1, diff, label='diff-' + labels[i][0])

    axes[0].set_ylabel(r'$\Re ' + '\\' + name + '$')
    axes[0].set_xlabel(r'$\omega_n$')

    axes[1].set_ylabel(r'$\Re ' + '\\' + name + '$')
    axes[1].set_xlabel(r'$\omega_n$')

    axes[2].set_ylabel(r'$diff - \Re ' + '\\' + name + '$')
    axes[2].set_xlabel(r'$\omega_n$')

    for ax in axes:
        ax.legend()
        ax.set_xlim(0, xmax)
    plt.tight_layout()
    if (do_plot): plt.savefig(output_dir + '/' + name + '_diff_check.png')
    if (verbose):
        plt.show()
    else:
        plt.close()


def sigma_loc_checks(siw_arr, labels, beta, output_dir, verbose=False, do_plot=True, name='', xmax=None):
    '''
        siw_arr: list of local self-energies for routine plots.
    '''
    if (xmax is None):
        xmax = 5 + 2 * beta
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 5))
    axes = axes.flatten()

    for i, siw in enumerate(siw_arr):
        vn = mf.vn(np.size(siw) // 2)
        axes[0].plot(vn, siw.real, label=labels[i])
    axes[0].set_ylabel(r'$\Re \Sigma(i\nu_n)$')
    axes[0].set_xlabel(r'$\nu_n$')

    for i, siw in enumerate(siw_arr):
        vn = mf.vn(np.size(siw) // 2)
        axes[1].plot(vn, siw.imag, label=labels[i])
    axes[1].set_ylabel(r'$\Im \Sigma(i\nu_n)$')
    axes[1].set_xlabel(r'$\nu_n$')

    for i, siw in enumerate(siw_arr):
        vn = mf.vn(np.size(siw) // 2)
        axes[2].loglog(vn, siw.real, label=labels[i])
    axes[2].set_ylabel(r'$\Re \Sigma(i\nu_n)$')
    axes[2].set_xlabel(r'$\nu_n$')

    for i, siw in enumerate(siw_arr):
        vn = mf.vn(np.size(siw) // 2)
        axes[3].loglog(vn, np.abs(siw.imag), label=labels[i])
    axes[3].set_ylabel(r'$|\Im \Sigma(i\nu_n)|$')
    axes[3].set_xlabel(r'$\nu_n$')

    axes[0].set_xlim(0, xmax)
    axes[1].set_xlim(0, xmax)
    axes[2].set_xlim(None, xmax)
    axes[3].set_xlim(None, xmax)
    plt.legend()
    axes[1].set_ylim(None, 0)
    plt.tight_layout()
    if (do_plot): plt.savefig(output_dir + f'/sde_' + name + '_check.png')
    if (verbose):
        plt.show()
    else:
        plt.close()


def plot_fourpoint_nu_nup(mat, vn, do_save=True, pdir='./', name='NoName', cmap='RdBu', figsize=(8, 4)):
    '''
        Default layout for plotting a fourpoint object in v and v'
    '''
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    axes = axes.flatten()
    axes[0].pcolormesh(vn, vn, mat.real, cmap=cmap)
    axes[1].pcolormesh(vn, vn, mat.imag, cmap=cmap)
    plt.tight_layout()
    if (do_save): plt.savefig(pdir + '/' + name + '.png')
    plt.show()


def add_afzb(ax=None, kx=None, ky=None, lw=1.0, marker=''):
    '''
        Add visual lines to mark the antiferromagnetic zone-boundary to existing axis.
    '''
    if (np.any(kx < 0)):
        ax.plot(np.linspace(-np.pi, 0, 101), np.linspace(0, np.pi, 101), '--k', lw=lw, marker=marker)
        ax.plot(np.linspace(-np.pi, 0, 101), np.linspace(0, -np.pi, 101), '--k', lw=lw, marker=marker)
        ax.plot(np.linspace(0, np.pi, 101), np.linspace(-np.pi, 0, 101), '--k', lw=lw, marker=marker)
        ax.plot(np.linspace(0, np.pi, 101), np.linspace(np.pi, 0, 101), '--k', lw=lw, marker=marker)
        ax.plot(kx, 0 * kx, '-k', lw=lw, marker=marker)
        ax.plot(0 * ky, ky, '-k', lw=lw, marker=marker)
    else:
        ax.plot(np.linspace(0, np.pi, 101), np.linspace(np.pi, 2 * np.pi, 101), '--k', lw=lw, marker=marker)
        ax.plot(np.linspace(np.pi, 0, 101), np.linspace(0, np.pi, 101), '--k', lw=lw, marker=marker)
        ax.plot(np.linspace(np.pi, 2 * np.pi, 101), np.linspace(0, np.pi, 101), '--k', lw=lw, marker=marker)
        ax.plot(np.linspace(np.pi, 2 * np.pi, 101), np.linspace(np.pi * 2, np.pi, 101), '--k', lw=lw, marker=marker)
        ax.plot(kx, np.pi * np.ones_like(kx), '-k', lw=lw, marker=marker)
        ax.plot(np.pi * np.ones_like(ky), ky, '-k', lw=lw, marker=marker)
    #
    # nump = 100
    # nump2 = nump//2
    # del_kx = kx[1] - kx[0]
    # del_ky = ky[1] - ky[0]
    # kx2 = np.linspace(kx[0],kx[-1]+del_kx,nump)
    # ky2 = np.linspace(ky[0],ky[-1]+del_ky,nump)
    # ax.plot(kx2[:nump2],ky2[nump2:], '--r', lw=lw, marker=marker)
    # ax.plot(kx2[:nump2],ky2[:nump2][::-1], '--k', lw=lw, marker=marker)
    # ax.plot(kx2[nump2:],ky2[nump2:][::-1], '--k', lw=lw, marker=marker)
    # ax.plot(kx2[nump2:],ky2[:nump2], '--k', lw=lw, marker=marker)

    ax.set_xlim(kx[0], kx[-1])
    ax.set_ylim(ky[0], ky[-1])
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')


def insert_colorbar(ax=None, im=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')


def plot_bw_fit(bw_opt=None, bw=None, chi2=None, fits=None, output_path=None, name=None):
    '''
        Plot the fit for estimating the optimal blur-width when using MaxEnt for analytic continuation
    '''
    plt.figure()
    plt.loglog(bw, chi2, '-o', label='$\chi^2(bw)$')
    for i, fit in enumerate(fits):
        plt.loglog(bw, fit, label=f'Fit-{i}')
    plt.vlines(bw_opt, np.min(chi2), np.max(chi2), 'k', label='$bw_{opt}$')
    plt.legend()
    plt.xlabel('bw')
    plt.ylabel('$\chi^2$')
    plt.tight_layout()
    plt.savefig(output_path + '{}.png'.format(name))
    plt.close()


def plot_cont_edc_maps(v_real=None, gk_cont=None, k_grid=None, output_path=None, name=None, n_map=7, wplot=1):
    nk = k_grid.nk
    v0_ind = np.argmin(np.abs(v_real))
    gk_cont_shift = bz.shift_mat_by_pi(mat=gk_cont, nk=nk)
    extent = bz.get_extent_pi_shift(kgrid=k_grid)

    cuts = np.round(np.linspace(1, nk[0] // 4, n_map, endpoint=True)).astype(int)

    ind_fs = bz.find_zeros((1. / gk_cont[:, :, 0, v0_ind]).real)
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


def plot_siwk_extrap(siwk_re_fs=None, siwk_im_fs=None, siwk_z=None, output_path='./', name='PolyFit', k_grid: bz.KGrid = None,
                     lw=1, verbose=False, do_save=True):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    siwk_re_fs = np.squeeze(siwk_re_fs)
    siwk_im_fs = np.squeeze(siwk_im_fs)
    siwk_z = np.squeeze(siwk_z)

    norm = MidpointNormalize(midpoint=0, vmin=siwk_re_fs.min(), vmax=siwk_re_fs.max())
    im = ax[0].pcolormesh(k_grid.kx_shift, k_grid.ky_shift, k_grid.shift_mat_by_pi(siwk_re_fs, axes=(0, 1)).T, cmap='RdBu_r',
                          norm=norm)
    insert_colorbar(ax=ax[0], im=im)
    ax[0].set_title(r'$\Re \Sigma(k,\nu=0)$')
    norm = MidpointNormalize(midpoint=0, vmin=siwk_im_fs.min(), vmax=siwk_im_fs.max())
    im = ax[1].pcolormesh(k_grid.kx_shift, k_grid.ky_shift, k_grid.shift_mat_by_pi(siwk_im_fs, axes=(0, 1)).T, cmap='RdBu_r',
                          norm=norm)
    insert_colorbar(ax=ax[1], im=im)
    ax[1].set_title(r'$\Im \Sigma(k,\nu=0)$')
    norm = MidpointNormalize(midpoint=0, vmin=siwk_z.min(), vmax=siwk_z.max())
    im = ax[2].pcolormesh(k_grid.kx_shift, k_grid.ky_shift, k_grid.shift_mat_by_pi(siwk_z, axes=(0, 1)).T, cmap='RdBu_r',
                          norm=norm)
    insert_colorbar(ax=ax[2], im=im)
    ax[2].set_title('Z(k)')

    for a in ax:
        add_afzb(ax=a, kx=k_grid.kx_shift, ky=k_grid.ky_shift, lw=lw)
        # a.set_aspect('equal')

    plt.tight_layout()
    fig.suptitle(name)
    if (do_save): plt.savefig(output_path + '{}.png'.format(name))
    if (verbose): plt.show()
    plt.close()


def plot_cont_fs(output_path=None, name='', mat=None, v_real=None, k_grid=None, w_int=None, w_plot=None, lw=1.0):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    if (w_int == None):
        w0_ind = np.argmin(np.abs(v_real - w_plot))
        mat_fs = mat[:, :, 0, w0_ind]
    else:
        ind_int = np.logical_and(v_real < 0, w_int < v_real)
        mat_fs = np.trapz(mat[:, :, 0, ind_int], v_real[ind_int]) / np.abs(w_int)

    awk_fs = np.squeeze(bz.shift_mat_by_pi(mat=-1. / np.pi * mat_fs.imag, nk=k_grid.nk))
    gk_real = np.squeeze(bz.shift_mat_by_pi(mat=mat_fs.real, nk=k_grid.nk))
    v0 = np.argmin(np.abs(v_real))
    qdp = np.squeeze(bz.shift_mat_by_pi(mat=(1. / mat)[:, :, 0, v0].real, nk=k_grid.nk))
    im = ax[0].pcolormesh(k_grid.kx_shift,k_grid.ky_shift,awk_fs.T, cmap='RdBu_r')
    insert_colorbar(ax=ax[0], im=im)
    ax[0].set_title('$-1/pi \Im (k,\omega={})$'.format(w_plot))
    norm = MidpointNormalize(midpoint=0, vmin=gk_real.min(), vmax=gk_real.max())
    im = ax[1].pcolormesh(k_grid.kx_shift, k_grid.ky_shift, gk_real.T, cmap='RdBu')
    insert_colorbar(ax=ax[1], im=im)
    ax[1].set_title('$\Re (k,\omega={})$'.format(w_plot))
    norm = MidpointNormalize(midpoint=0, vmin=qdp.min(), vmax=qdp.max())
    im = ax[2].pcolormesh(k_grid.kx_shift, k_grid.ky_shift, qdp.T, cmap='RdBu')
    insert_colorbar(ax=ax[2], im=im)
    ax[2].set_title('QPD $(1/(\Re (k, \omega = {})$'.format(w_plot))

    add_afzb(ax=ax[0], kx=k_grid.kx_shift, ky=k_grid.ky_shift, lw=lw)
    add_afzb(ax=ax[1], kx=k_grid.kx_shift, ky=k_grid.ky_shift, lw=lw)
    add_afzb(ax=ax[2], kx=k_grid.kx_shift, ky=k_grid.ky_shift, lw=lw)

    plt.tight_layout()
    fig.suptitle(name)
    plt.savefig(output_path + '{}_at_w{}.png'.format(name, w_plot))
    plt.close()


def plot_cont_fs_no_shift(output_path=None, name='', mat=None, v_real=None, k_grid=None, w_int=None, w_plot=None, lw=1.0):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    if (w_int == None):
        w0_ind = np.argmin(np.abs(v_real - w_plot))
        mat_fs = mat[:, :, 0, w0_ind]
    else:
        ind_int = np.logical_and(v_real < 0, w_int < v_real)
        mat_fs = np.trapz(mat[:, :, 0, ind_int], v_real[ind_int]) / np.abs(w_int)

    awk_fs = np.squeeze(-1. / np.pi * mat_fs.imag)
    gk_real = np.squeeze(mat_fs.real)
    v0 = np.argmin(np.abs(v_real))
    qdp = np.squeeze((1. / mat)[:, :, 0, v0].real)
    im = ax[0].imshow(awk_fs, cmap='RdBu_r', extent=bz.get_extent(kgrid=k_grid), origin='lower')
    insert_colorbar(ax=ax[0], im=im)
    ax[0].set_title('$-1/pi \Im (k,\omega={})$'.format(w_plot))
    norm = MidpointNormalize(midpoint=0, vmin=gk_real.min(), vmax=gk_real.max())
    im = ax[1].imshow(gk_real, cmap='RdBu', extent=bz.get_extent(kgrid=k_grid), norm=norm, origin='lower')
    insert_colorbar(ax=ax[1], im=im)
    ax[1].set_title('$\Re (k,\omega={})$'.format(w_plot))
    norm = MidpointNormalize(midpoint=0, vmin=qdp.min(), vmax=qdp.max())
    im = ax[2].imshow(qdp, cmap='RdBu', extent=bz.get_extent(kgrid=k_grid), norm=norm, origin='lower')
    insert_colorbar(ax=ax[2], im=im)
    ax[2].set_title('QPD $(1/(\Re (k, \omega = {})$'.format(w_plot))

    add_afzb(ax=ax[0], kx=k_grid.kx, ky=k_grid.kx, lw=lw)
    add_afzb(ax=ax[1], kx=k_grid.kx, ky=k_grid.kx, lw=lw)
    add_afzb(ax=ax[2], kx=k_grid.kx, ky=k_grid.kx, lw=lw)

    plt.tight_layout()
    fig.suptitle(name)
    plt.savefig(output_path + '{}_at_w{}.png'.format(name, w_plot))
    plt.close()


def plot_oz_fit(chi_w0=None, oz_coeff=None, qgrid=None, pdir=None, name=''):
    '''
        Default plot for the Ornstein-Zernike fit.
    '''
    oz = ozfunc.oz_spin_w0(qgrid, oz_coeff[0], oz_coeff[1]).reshape(qgrid.nk)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
    ax[0].plot(qgrid.kx, chi_w0[:, qgrid.nk[1] // 2, 0].real, 'o', label='$\chi$')
    ax[0].plot(qgrid.kx, oz[:, qgrid.nk[1] // 2, 0].real, '-', label='oz-fit')
    ax[0].legend()
    ax[0].set_title('$q_y = \pi$')
    ax[0].set_xlabel('$q_x$')
    ax[0].set_ylabel('$\chi$')

    ax[1].plot(qgrid.ky, chi_w0[qgrid.nk[0] // 2, :, 0].real, 'o', label='$\chi$')
    ax[1].plot(qgrid.ky, oz[qgrid.nk[0] // 2, :, 0].real, '-', label='oz-fit')
    ax[1].legend()
    ax[1].set_title('$q_x = \pi$')
    ax[1].set_xlabel('$q_y$')
    ax[1].set_ylabel('$\chi$')

    mask = qgrid.kmesh[0] == qgrid.kmesh[1]
    x_axis = np.linspace(0,2*np.pi,len(chi_w0[mask]))
    x_axis2 = np.linspace(0,2*np.pi,len(oz[mask]))
    ax[2].plot(x_axis, chi_w0[mask].real, 'o', label='$\chi$')
    ax[2].plot(x_axis2, oz[mask].real, '-', label='oz-fit')
    ax[2].legend()
    ax[2].set_title('$q_y = q_x$')
    ax[2].set_xlabel('$q$')
    ax[2].set_ylabel('$\chi$')
    plt.savefig(pdir + name + '.png')
    plt.close()


def plot_aw_loc(v_real=None, gloc=None, output_path=None, name='', xlim=(None, None)):
    ''' Plot the local continued Green's function. '''
    plt.figure()
    plt.plot(v_real, -1. / np.pi * gloc.imag)
    plt.vlines(0, 0, 10, 'k', '--', alpha=0.5)
    plt.ylim(0, np.max(-1. / np.pi * gloc.imag))
    plt.xlim(xlim)
    plt.xlabel('$\omega$')
    plt.ylabel('A($\omega$)')
    plt.title(name)
    plt.savefig(output_path + '{}.png'.format(name))
    plt.close()


def get_extent(kgrid=None):
    return [kgrid.kx[0], kgrid.kx[-1], kgrid.ky[0], kgrid.ky[-1]]


def plot_gap_function_kx_ky(mat, kx, ky, do_save=True, pdir='./', name='', cmap='RdBu', figsize=ps.FIGSIZE, verbose=False,
                      scatter=None):
    '''
        mat: [nkx,nky]; object in the Brillouin zone.
    '''
    fig, axes = plt.subplots(ncols=2, figsize=figsize, dpi=500)
    axes = axes.flatten()
    im1 = axes[0].pcolormesh(kx, ky, mat[...,0].T.real, cmap=cmap)
    im2 = axes[1].pcolormesh(kx, ky, mat[...,1].T.real, cmap=cmap)
    axes[0].set_title(r'$\nu_{n=0}$')
    axes[1].set_title(r'$\nu_{n=-1}$')
    for ax in axes:
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        ax.set_aspect('equal')
        add_afzb(ax=ax, kx=kx, ky=ky, lw=1.0, marker='')
    fig.suptitle(name)
    fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location='right',
                 pad=0.05)
    fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location='right',
                 pad=0.05)
    if (scatter is not None):
        for ax in axes:
            colours = plt.cm.get_cmap(cmap)(np.linspace(0, 1, np.shape(scatter)[0]))
            ax.scatter(scatter[:, 0], scatter[:, 1], marker='o', c=colours)
    plt.tight_layout()
    if (do_save): plt.savefig(pdir + '/GapFunction_' + name + '.png')
    if (verbose):
        plt.show()
    else:
        plt.close()

def gradient_fill(x, y, colours, ax=None):
    if ax is None:
        ax = plt.gca()
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(colours, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower')
    xy = np.column_stack([x, y])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return im
