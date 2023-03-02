import numpy as np
import sys, os

sys.path.append('../src')
sys.path.append('./src')
import TwoPoint as tp
import TestData as td
import LocalFourPoint as lfp
import MatsubaraFrequencies as mf
import BrillouinZone as bz
import Hk as hamk
import matplotlib.pyplot as plt
import w2dyn_aux
import matplotlib.colors as mc

PLOT_PATH = './TestPlots/'

from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_colorbar(fig, im, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


def plot_chi_convergence(wn, chi_dens_core, chi_magn_core, chi_dens_shell, chi_magn_shell, niv_core, niv_shell, count=1):
    n_niv = len(chi_dens_core)
    n_niv_shell = len(chi_dens_shell)
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), dpi=500)
    ax = ax.flatten()
    colors = plt.cm.jet(np.linspace(0.0, 1.0, n_niv))
    colors_shell = plt.cm.jet(np.linspace(0.0, 1.0, n_niv_shell))
    for i in range(n_niv):
        ax[0].plot(wn, chi_dens_core[i].real, '-o', color=colors[i], markeredgecolor='k', alpha=0.8, label=f'niv={niv_core[i]}')
        ax[1].plot(wn, chi_magn_core[i].real, '-o', color=colors[i], markeredgecolor='k', alpha=0.8, label=f'niv={niv_core[i]}')

    for i in range(n_niv_shell):
        ax[2].plot(wn, chi_dens_shell[i].real, '-o', color=colors_shell[i], markeredgecolor='k', alpha=0.8, label=f'niv-shell={niv_shell[i]}')
        ax[3].plot(wn, chi_magn_shell[i].real, '-o', color=colors_shell[i], markeredgecolor='k', alpha=0.8, label=f'niv-shell={niv_shell[i]}')
    ax[1].legend()
    ax[3].legend()
    for a in ax:
        a.set_xlim(0, None)
        a.set_xlabel('$n$')
    ax[0].set_ylabel(r'$\chi_{d}^{c}$')
    ax[1].set_ylabel(r'$\tilde{\chi}_{m}^{c}$')
    ax[2].set_ylabel(r'$\chi_{d}^{c}$')
    ax[3].set_ylabel(r'$\tilde{\chi}_{m}^{c}$')
    plt.tight_layout()
    plt.savefig(PLOT_PATH + f'chi_r_convergence_check_{count}.png')
    plt.close()


def plot_chi_convergence_wn(wn, chi_dens_core, chi_magn_core, chi_dens_shell, chi_magn_shell, niv_core, niv_for_shell, niv_shell, count=1):
    niw = np.size(wn) // 2
    n_niv = len(chi_dens_core)
    n_niv_shell = len(chi_dens_shell)
    colors_d = plt.cm.Blues(np.linspace(0.5, 1.0, n_niv))[::-1]
    colors_m = plt.cm.Blues(np.linspace(0.5, 1.0, n_niv))[::-1]

    colors_shell_d = plt.cm.Reds(np.linspace(0.5, 1.0, n_niv_shell))[::-1]
    colors_shell_m = plt.cm.Reds(np.linspace(0.5, 1.0, n_niv_shell))[::-1]

    chi_dens_core_w0 = [chi_dens_core[i][niw].real for i in range(n_niv)]
    chi_magn_core_w0 = [chi_magn_core[i][niw].real for i in range(n_niv)]
    chi_dens_shell_w0 = [chi_dens_shell[i][niw].real for i in range(n_niv_shell)]
    chi_magn_shell_w0 = [chi_magn_shell[i][niw].real for i in range(n_niv_shell)]

    n_w = niw
    chi_dens_core_w15 = [chi_dens_core[i][niw + n_w].real for i in range(n_niv)]
    chi_magn_core_w15 = [chi_magn_core[i][niw + n_w].real for i in range(n_niv)]
    chi_dens_shell_w15 = [chi_dens_shell[i][niw + n_w].real for i in range(n_niv_shell)]
    chi_magn_shell_w15 = [chi_magn_shell[i][niw + n_w].real for i in range(n_niv_shell)]

    fig, ax = plt.subplots(2, 2, figsize=(6, 5), dpi=500)
    ax = ax.flatten()

    ax[0].scatter(1 / niv_core, chi_dens_core_w0, marker='o', c=colors_d, edgecolors='k', alpha=0.8, label='core-scan')
    ax[1].scatter(1 / niv_core, chi_magn_core_w0, marker='o', c=colors_m, edgecolors='k', alpha=0.8)

    ax[2].scatter(1 / niv_core, chi_dens_core_w15, marker='o', c=colors_d, edgecolors='k', alpha=0.8, label='core-scan')
    ax[3].scatter(1 / niv_core, chi_magn_core_w15, marker='o', c=colors_m, edgecolors='k', alpha=0.8)

    ax[0].scatter(1 / (niv_for_shell + niv_shell), chi_dens_shell_w0, marker='o', c=colors_shell_d, edgecolors='k', alpha=0.8, label='shell-scan')
    ax[1].scatter(1 / (niv_for_shell + niv_shell), chi_magn_shell_w0, marker='o', c=colors_shell_m, edgecolors='k', alpha=0.8)

    ax[2].scatter(1 / (niv_for_shell + niv_shell), chi_dens_shell_w15, marker='o', c=colors_shell_d, edgecolors='k', alpha=0.8)
    ax[3].scatter(1 / (niv_for_shell + niv_shell), chi_magn_shell_w15, marker='o', c=colors_shell_m, edgecolors='k', alpha=0.8)
    ax[0].legend()
    ax[0].set_xlabel('$1/n$')
    ax[1].set_xlabel('$1/n$')
    ax[2].set_xlabel('$1/n$')
    ax[3].set_xlabel('$1/n$')
    # ax[2].set_xlabel('$n_{shell}$')
    # ax[3].set_xlabel('$n_{shell}$')
    ax[0].set_ylabel(r'$\chi_{d}(\omega=0)$')
    ax[1].set_ylabel(r'$\chi_{m}(\omega=0)$')
    ax[2].set_ylabel(r'$\chi_{m}' + f'(\omega={n_w})$')
    ax[3].set_ylabel(r'$\chi_{m}' + f'(\omega={n_w})$')
    # ax[2].set_ylabel(r'$\chi_{d}$')
    # ax[3].set_ylabel(r'$\chi_{m}$')
    plt.tight_layout()
    plt.savefig(PLOT_PATH + f'chi_r_convergence_check_w0_w{n_w}_{count}.png')
    plt.show()


def plot_chi_sum_asymptotic(chi_dens_shell, chi_magn_shell, niv_for_shell, beta, n, name='', count=1):
    niw = np.size(chi_dens_shell) // 2
    niw_sum = np.array([10, 20, 30, 40, 50, niw])
    chi_dens_sum = np.array([mf.wn_sum(chi_dens_shell, beta, n_sum) for n_sum in niw_sum])
    chi_magn_sum = np.array([mf.wn_sum(chi_magn_shell, beta, n_sum) for n_sum in niw_sum])

    fig, ax = plt.subplots(2, 2, figsize=(7, 5), dpi=500)
    ax = ax.flatten()
    data = 0.5 * (chi_dens_sum.real + chi_magn_sum.real)
    ax[0].plot(1 / niw_sum, data, '-o', color='cornflowerblue', markeredgecolor='k', alpha=0.8)
    ax[0].plot(1 / niw_sum, np.ones_like(niw_sum) * tp.get_sum_chiupup(n), '-', color='k')
    ax[0].vlines(1 / niv_for_shell, data.min(), data.max(), ls='--', colors='k')
    ax[0].set_ylabel('$\sum \chi_{uu}$')

    data = 0.5 * (chi_dens_sum.real - chi_magn_sum.real)
    ax[1].plot(1 / niw_sum, data, '-o', color='cornflowerblue', markeredgecolor='k', alpha=0.8)
    ax[1].vlines(1 / niv_for_shell, data.min(), data.max(), ls='--', colors='k')
    ax[1].set_ylabel('$\sum \chi_{ud}$')

    data = (chi_dens_sum.real)
    ax[2].plot(1 / niw_sum, data, '-o', color='cornflowerblue', markeredgecolor='k', alpha=0.8)
    ax[2].vlines(1 / niv_for_shell, data.min(), data.max(), ls='--', colors='k')
    ax[2].set_ylabel('$\sum \chi_{d}$')

    data = (chi_magn_sum.real)
    ax[3].plot(1 / niw_sum, data, '-o', color='cornflowerblue', markeredgecolor='k', alpha=0.8)
    ax[3].vlines(1 / niv_for_shell, data.min(), data.max(), ls='--', colors='k')
    ax[3].set_ylabel('$\sum \chi_{m}$')

    for a in ax:
        a.set_xlabel('1/n')

    plt.tight_layout()
    plt.savefig(PLOT_PATH + 'chi_' + name + f'_sum_convergence_{count}.png')
    plt.show()


def test_chi_asymptotic(siw_dmft, ek, mu, n, u, beta, g2_file, niv_core, niv_shell, niv_for_shell=30, count=1):
    sigma = tp.SelfEnergy(siw_dmft[None, None, None, :], beta, pos=False)
    giwk = tp.GreensFunction(sigma, ek, n=n)
    giwk.set_g_asympt(niv_asympt=np.max(niv_shell) + 1000)
    g_loc = giwk.k_mean(range='full')
    niw = g2_file.get_niw(channel='dens')
    wn = mf.wn(niw)
    g2_dens = lfp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='dens', iw=wn), beta=beta, wn=wn, channel='dens')
    g2_magn = lfp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='magn', iw=wn), beta=beta, wn=wn, channel='magn')
    gchi_dens = lfp.gchir_from_g2(g2_dens, g_loc)
    gchi_magn = lfp.gchir_from_g2(g2_magn, g_loc)
    chi0_gen = lfp.LocalBubble(wn, g_loc, beta)

    # niv_core = np.arange(10, niw)[::10][::-1]

    chi_dens_core, chi_magn_core = [], []
    vrg_dens_core, vrg_magn_core = [], []

    for niv in niv_core:
        gchi_dens.cut_iv(niv)
        gchi_magn.cut_iv(niv)
        vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, chi0_gen, u, niv_core=niv)
        vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir(gchi_magn, chi0_gen, u, niv_core=niv)
        chi_dens_core.append(chi_dens)
        chi_magn_core.append(chi_magn)

        vrg_dens_core.append(vrg_dens)
        vrg_magn_core.append(vrg_magn)

    gchi_dens = lfp.gchir_from_g2(g2_dens, g_loc)
    gchi_magn = lfp.gchir_from_g2(g2_magn, g_loc)

    chi_dens_shell, chi_magn_shell = [], []

    vrg_dens_shell, vrg_magn_shell = [], []
    gchi_dens.cut_iv(niv_for_shell)
    gchi_magn.cut_iv(niv_for_shell)

    for niv in niv_shell:
        vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, chi0_gen, u, niv_core=niv_for_shell, niv_shell=niv)
        vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir(gchi_magn, chi0_gen, u, niv_core=niv_for_shell, niv_shell=niv)

        chi_dens_shell.append(chi_dens)
        chi_magn_shell.append(chi_magn)
        vrg_dens_shell.append(vrg_dens)
        vrg_magn_shell.append(vrg_magn)
    wn = mf.wn(niw)
    plot_chi_convergence(wn, chi_dens_core, chi_magn_core, chi_dens_shell, chi_magn_shell, niv_core, niv_shell, count=count)
    plot_chi_convergence_wn(wn, chi_dens_core, chi_magn_core, chi_dens_shell, chi_magn_shell, niv_core, niv_for_shell, niv_shell, count=count)

    plot_chi_sum_asymptotic(chi_dens_shell[0], chi_magn_shell[0], niv_core[0], beta, n, name='shell', count=count)
    plot_chi_sum_asymptotic(chi_dens_core[0], chi_magn_core[0], niv_for_shell, beta, n, name='core', count=count)


def test_chi_asymptotic_1():
    ddict = td.get_data_set_1(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    niv_core = np.array([100, 95, 90, 80, 60, 50, 40, 30, 20, 10])
    niv_shell = np.array([10, 50, 100, 500, 1000, 2000])[::-1]
    test_chi_asymptotic(ddict['siw'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'], ddict['g2_file'], niv_core, niv_shell,
                        niv_for_shell=60, count=1)


def test_chi_asymptotic_2():
    ddict = td.get_data_set_2(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    niv_core = np.array([60, 50, 40, 30, 20, 10])
    niv_shell = np.array([10, 50, 100, 500, 1000, 2000, 5000, 10000])[::-1]
    test_chi_asymptotic(ddict['siw'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'], ddict['g2_file'], niv_core, niv_shell,
                        niv_for_shell=10, count=2)

def test_chi_asymptotic_22():
    ddict = td.get_data_set_2(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    niv_core = np.array([60, 50, 40, 30, 20, 10])
    niv_shell = np.array([10, 50, 100, 500, 1000, 2000, 5000, 10000])[::-1]
    test_chi_asymptotic(ddict['siw'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'], ddict['g2_file'], niv_core, niv_shell,
                        niv_for_shell=30, count=22)

def test_chi_asymptotic_222():
    ddict = td.get_data_set_2(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    niv_core = np.array([60, 50, 40, 30, 20, 10])
    niv_shell = np.array([10, 50, 100, 500, 1000, 2000, 5000, 10000])[::-1]
    test_chi_asymptotic(ddict['siw'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'], ddict['g2_file'], niv_core, niv_shell,
                        niv_for_shell=60, count=222)


def test_chi_asymptotic_3():
    ddict = td.get_data_set_3(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    niv_core = np.array([70, 60, 50, 40, 30, 20, 10])  # 100, 95, 90,
    niv_shell = np.array([10, 50, 100, 500, 1000, 2000])[::-1]
    test_chi_asymptotic(ddict['siw'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'], ddict['g2_file'], niv_core, niv_shell,
                        niv_for_shell=30, count=3)


def test_vertex_frequency_convention(ddict, ek, count=1):
    g2_file = ddict['g2_file']
    siw_dmft = ddict['siw']
    beta = ddict['beta']
    n = ddict['n']
    sigma = tp.SelfEnergy(siw_dmft[None, None, None, :], beta, pos=False)
    giwk = tp.GreensFunction(sigma, ek, n=n)
    giwk.set_g_asympt(niv_asympt=1000)
    g_loc = giwk.k_mean(range='full')
    niw = g2_file.get_niw(channel='dens')
    wn = mf.wn(niw)
    g2_dens = lfp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='dens', iw=wn), beta=beta, wn=wn, channel='dens')
    g2_magn = lfp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='magn', iw=wn), beta=beta, wn=wn, channel='magn')
    gchi_dens = lfp.gchir_from_g2(g2_dens, g_loc)
    gchi_magn = lfp.gchir_from_g2(g2_magn, g_loc)
    chi0_gen = lfp.LocalBubble(wn, g_loc, beta)
    gchi0_core = chi0_gen.get_gchi0(niv=gchi_dens.niv)
    F_dens = lfp.Fob2_from_chir(gchi_dens, gchi0_core)
    F_magn = lfp.Fob2_from_chir(gchi_magn, gchi0_core)

    gamma_dens = lfp.gamob2_from_chir(gchi_dens, gchi0_core)
    gamma_magn = lfp.gamob2_from_chir(gchi_magn, gchi0_core)

    def plot_vertex(fig, ax, vertex, niv_plot, wn):
        vn = (mf.vn(niv_plot) * 2 + 1)
        iwn = mf.wn_cen2lin(wn, niw)
        data = mf.cut_v(vertex.mat[iwn, :, :].real, niv_cut=niv_plot, axes=(-2, -1))
        if (data.min() < 0 and data.max() > 0):
            # norm = mc.TwoSlopeNorm(0, vmin=data.min(), vmax=data.max())
            im = ax.pcolormesh(vn, vn, data, cmap='RdBu')
        else:
            im = ax.pcolormesh(vn, vn, data, cmap='RdBu')

        add_colorbar(fig, im, ax)

    def plot_four_point(niv_plot, wn_plot):
        fig, ax = plt.subplots(2, 3, figsize=(10, 5), dpi=500)
        ax = ax.flatten()
        plot_vertex(fig, ax[0], gchi_dens, niv_plot, wn_plot)
        plot_vertex(fig, ax[1], F_dens, niv_plot, wn_plot)
        plot_vertex(fig, ax[2], gamma_dens, niv_plot, wn_plot)
        plot_vertex(fig, ax[3], gchi_magn, niv_plot, wn_plot)
        plot_vertex(fig, ax[4], F_magn, niv_plot, wn_plot)
        plot_vertex(fig, ax[5], gamma_magn, niv_plot, wn_plot)
        ax[0].set_title('$\chi_{d}$')
        ax[1].set_title('$F_{d}$')
        ax[2].set_title('$\Gamma_{d}$')

        ax[3].set_title('$\chi_{m}$')
        ax[4].set_title('$F_{m}$')
        ax[5].set_title('$\Gamma_{m}$')
        for a in ax:
            a.set_xlabel(r'$\nu^{\prime}_n$')
            a.set_ylabel(r'$\nu_n$')
        plt.tight_layout()
        plt.savefig(PLOT_PATH + f'vertices_w{wn_plot}_{count}.png')
        plt.show()

    plot_four_point(10, 0)
    plot_four_point(10, 5)


def test_vertex_frequency_convention_2():
    ddict = td.get_data_set_2(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    test_vertex_frequency_convention(ddict, ek, count=2)


def test_vertex_frequency_convention_3():
    ddict = td.get_data_set_2(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    test_vertex_frequency_convention(ddict, ek, count=3)


# def test_vrg_asymptotic():


if __name__ == '__main__':
    # test_chi_asymptotic_1()
    # test_chi_asymptotic_2()
    test_chi_asymptotic_22()
    # test_chi_asymptotic_222()
    test_chi_asymptotic_3()

    # test_vertex_frequency_convention_3()
    # test_vertex_frequency_convention_2()

    print('Finished!')
