import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import two_point as twop
from dga import wannier
from dga import analytic_continuation as ac
from dga import plotting


def plot_real_frequency_dispersion(gwk: twop.RealFrequencyGF, k_path: bz.KPath, pdir='./', name='gwk_cont', verbose=False,
                                   do_save=True, wmin=None, wmax=None):
    cm2in = 1 / 2.54
    if wmin is None: wmin = gwk.w[0]
    if wmax is None: wmax = gwk.w[-1]

    fig = plt.figure(figsize=(11 * cm2in, 6 * cm2in), dpi=251)
    gs = fig.add_gridspec(1, 2, height_ratios=[1, ], width_ratios=[4, 1],
                          left=0.12, right=0.95, bottom=0.1, top=0.95,
                          wspace=0.03, hspace=0.26)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cmap_p = 'magma'
    awk_kpath = k_path.map_to_kpath(gwk.awk())
    ax0.pcolormesh(k_path.k_axis, gwk.w, awk_kpath.T, cmap=cmap_p, vmax=1.5)
    ax0.set_ylim(wmin, wmax)
    ax0.set_xticks(k_path.x_ticks, k_path.labels)
    ax0.set_ylabel(r'$\omega$ [t]')
    ax0.hlines(0, k_path.k_axis[0], k_path.k_axis[-1], ls='--', color='grey', lw=1)

    colours = np.empty((100, 100, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb('cornflowerblue')
    colours[:, :, :3] = rgb
    colours[:, :, -1] = np.linspace(0.0, 1, 100)[None, :]
    plotting.gradient_fill(gwk.aw, gwk.w, colours, ax=ax1)

    ax1.plot(gwk.aw, gwk.w, '-', 'cornflowerblue')

    ind = np.logical_and(gwk.w < wmax, gwk.w > wmin)
    aw_max = np.max(gwk.aw[ind])
    ax1.hlines(0, 0.0, aw_max * 1.05, linestyles='dashed', color='grey', lw=1)

    ax1.set_xticks([0.1, ])
    ax1.set_xlim(0.00, 0.2)
    ax1.set_ylim(wmin, wmax)
    ax1.get_yaxis().set_visible(False)
    ax0.text(0.01, 0.95, '(a)', color='w', va='center', transform=ax0.transAxes)
    ax1.text(0.7, 0.95, '(b)', color='k', va='center', transform=ax1.transAxes)
    if do_save: plt.savefig(pdir + f'{name}.png', dpi=251)
    if verbose:
        plt.show()
    else:
        plt.close()


def plot_chi_along_kpath(chi, w, k_path: bz.KPath, pdir='./', name='chi_cont', verbose=False,
                         do_save=True):
    cm2in = 1 / 2.54
    wmin = w[np.argmin(np.abs(w)) + 1]
    fig = plt.figure(figsize=(11 * cm2in, 6 * cm2in), dpi=251)
    gs = fig.add_gridspec(1, 1, height_ratios=[1, ], width_ratios=[1, ],
                          left=0.12, right=0.98, bottom=0.1, top=0.95,
                          wspace=0.03, hspace=0.26)
    ax0 = fig.add_subplot(gs[0, 0])
    cmap_p = 'terrain'
    chi_kpath = k_path.map_to_kpath(chi)
    im1 = ax0.pcolormesh(k_path.k_axis, w, chi_kpath.real.T * w[:, None], cmap=cmap_p)
    peaks = np.argmax(chi_kpath * w[None, :], axis=1)
    ax0.plot(k_path.k_axis, w[peaks], '-h', color='cornflowerblue')
    wmax = np.max(w[peaks]) * 2
    ax0.set_ylim(wmin, wmax)
    ax0.set_xticks(k_path.x_ticks, k_path.labels)
    ax0.set_ylabel(r'$\omega$ [t]')
    ax0.hlines(0, k_path.k_axis[0], k_path.k_axis[-1], ls='--', color='grey', lw=1)
    plt.colorbar(im1)
    ax0.text(0.01, 0.95, '(a)', color='w', va='center', transform=ax0.transAxes)
    if do_save: plt.savefig(pdir + f'{name}.png', dpi=251)
    if verbose:
        plt.show()
    else:
        plt.close()

def plot_opt_cond(sigma, w, pdir='./', name='chi_cont', verbose=False,do_save=True):

    cm2in = 1 / 2.54
    fig = plt.figure(figsize=(11 * cm2in, 6 * cm2in), dpi=251)
    gs = fig.add_gridspec(1, 1, height_ratios=[1, ], width_ratios=[1, ],
                          left=0.12, right=0.98, bottom=0.1, top=0.95,
                          wspace=0.03, hspace=0.26)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(w, sigma.real, '-h', color='cornflowerblue')
    ax0.set_xlabel(r'$\omega$ [t]')
    ax0.set_ylabel(r'$\sigma(\omega)$ [t]')
    if do_save: plt.savefig(pdir + f'{name}.png', dpi=251)
    if verbose:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    pass
