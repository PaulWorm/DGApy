# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import itertools
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import FourPoint as fp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import MatsubaraFrequencies as mf
import OrnsteinZernickeFunction as ozfunc
import BrillouinZone as bz

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

def add_afzb(ax=None, kx=None, ky=None, lw=1.0, shift_pi=True):
    if(shift_pi):
        kx = kx - np.pi
        ky = ky - np.pi
    ax.plot(kx, 0 * kx, 'k', lw=lw)
    ax.plot(0 * ky, ky, 'k', lw=lw)
    ax.plot(-ky, ky - np.pi, '--k', lw=lw)
    ax.plot(ky, ky - np.pi, '--k', lw=lw)
    ax.set_xlim(kx[0], kx[-1])
    ax.set_ylim(ky[0], ky[-1])
    ax.set_xlabel('$k_y$')
    ax.set_ylabel('$k_x$')


def insert_colorbar(ax=None, im=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')


def plot_cont_fs(output_path=None, name='', gk=None, v_real=None, k_grid=None, w_int=-0.2, lw=1.0):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ind_int = np.logical_and(v_real < 0, w_int < v_real)
    gk_fs = np.trapz(gk[:, :, 0, ind_int], v_real[ind_int])
    awk_fs = np.squeeze(bz.shift_mat_by_pi(mat=-1. / np.pi * gk_fs.imag, nk=k_grid.nk))
    gk_real = np.squeeze(bz.shift_mat_by_pi(mat=gk_fs.real, nk=k_grid.nk))
    v0 = np.argmin(np.abs(v_real))
    qdp = np.squeeze(bz.shift_mat_by_pi(mat=(1. / gk)[:, :, 0, v0].real, nk=k_grid.nk))
    im = ax[0].imshow(awk_fs, cmap='RdBu_r', extent=bz.get_extent_pi_shift(kgrid=k_grid))
    insert_colorbar(ax=ax[0], im=im)
    ax[0].set_title('$A(k,\omega=0)$')
    norm = MidpointNormalize(midpoint=0, vmin=gk_real.min(), vmax=gk_real.max())
    im = ax[1].imshow(gk_real, cmap='RdBu', extent=bz.get_extent_pi_shift(kgrid=k_grid), norm=norm)
    insert_colorbar(ax=ax[1], im=im)
    ax[1].set_title('$\Re G(k,\omega=0)$')
    norm = MidpointNormalize(midpoint=0, vmin=qdp.min(), vmax=qdp.max())
    im = ax[2].imshow(qdp, cmap='RdBu', extent=bz.get_extent_pi_shift(kgrid=k_grid), norm=norm)
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


def plot_fs_peaks(ax=None, k_grid=None, ind_fs=None):
    kx = np.array([k_grid.kmesh[0][i] for i in ind_fs])
    ky = np.array([k_grid.kmesh[1][i] for i in ind_fs])
    shift_x = 0.0  # np.pi/(k_grid.nk[0])
    shift_y = 0.0  # np.pi/(k_grid.nk[1])
    ax.plot(kx - shift_x, ky - shift_y)


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
    plt.show()


def plot_susc(susc=None):
    plt.plot(susc.iw_core, susc.mat.real, 'o')
    plt.xlim([-2, susc.beta])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'\chi')
    plt.show()


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
        A.mat[A.iw == w_plot, A.niv - niv_plot:A.niv + niv_plot, A.niv - niv_plot:A.niv + niv_plot].squeeze().real,
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
               extent=[-niv_cut, niv_cut, A.iw[0], A.iw[-1]])
    plt.colorbar()
    plt.title(r'$\Re$' + name + '-' + A.channel)
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$\omega$')
    plt.show()

    plt.imshow(A.mat.imag[:, tp.niv - niv_cut:tp.niv + niv_cut], cmap='RdBu',
               extent=[A.iw[0], A.iw[-1], -niv_cut, niv_cut])
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
    try:
        plt.show()
    except:
        plt.close()


def plot_siw(vn_list=None, siw_list=None, labels_list=None, plot_dir=None, niv_plot_min=-10, niv_plot=200,
             name='siw_check', ncol=1, ms=2):
    markers = __markers__
    nplot = len(vn_list)
    assert nplot < len(markers), 'More plots-lines requires, than markers avaiable.'

    plt.figure()
    plt.subplot(211)

    for i in range(len(vn_list)):
        ind = np.logical_and(vn_list[i] >= niv_plot_min, vn_list[i] <= niv_plot)
        plt.plot(vn_list[i][ind], siw_list[i][ind].real, markers[i], ms=ms, label=labels_list[i])
    plt.legend()
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Re \Sigma$')
    plt.subplot(212)
    for i in range(len(vn_list)):
        ind = np.logical_and(vn_list[i] >= niv_plot_min, vn_list[i] <= niv_plot)
        plt.plot(vn_list[i][ind], siw_list[i][ind].imag, markers[i], ms=ms, label=labels_list[i])
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
    try:
        plt.show()
    except:
        plt.close()


def plot_giwk_fs(giwk=None, plot_dir=None, kgrid=None, do_shift=False, kz=0, niv_plot=None, name='', ind_fs=None):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    if (ind_fs is not None):
        plot_fs_peaks(ax=ax[0][0], k_grid=kgrid, ind_fs=ind_fs)
        plot_fs_peaks(ax=ax[0][1], k_grid=kgrid, ind_fs=ind_fs)
        plot_fs_peaks(ax=ax[1][0], k_grid=kgrid, ind_fs=ind_fs)
        plot_fs_peaks(ax=ax[1][1], k_grid=kgrid, ind_fs=ind_fs)
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
    try:
        plt.show()
    except:
        plt.close()


def plot_giwk_qpd(giwk=None, plot_dir=None, kgrid=None, do_shift=False, kz=0, niv_plot=None, name=''):
    fig, ax = plot_qpd(siwk=(1. / giwk).real, kgrid=kgrid, do_shift=do_shift, kz=kz, niv_plot=niv_plot)

    ax[0][0].set_title('$\Re (1./G)$')
    ax[0][1].set_title('$\Im (1./G)$')

    if (plot_dir is not None):
        plt.savefig(plot_dir + 'QuasiParticleDispersion_{}.png'.format(name))
    try:
        plt.show()
    except:
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
