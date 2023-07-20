import numpy as np
import sys, os

sys.path.append('../src')
sys.path.append('./src')
import dga.two_point as tp
import TestData as td
import dga.local_four_point as lfp
import dga.matsubara_frequencies as mf
import dga.brillouin_zone as bz
import dga.hk as hamk
import matplotlib.pyplot as plt
import dga.bubble as bub

PLOT_PATH = './TestPlots/TestChiPhys/'

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
        ax[2].plot(wn, chi_dens_shell[i].real, '-o', color=colors_shell[i], markeredgecolor='k', alpha=0.8,
                   label=f'niv-shell={niv_shell[i]}')
        ax[3].plot(wn, chi_magn_shell[i].real, '-o', color=colors_shell[i], markeredgecolor='k', alpha=0.8,
                   label=f'niv-shell={niv_shell[i]}')
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


def plot_chi_convergence_wn(wn, chi_dens_core, chi_magn_core, chi_dens_shell, chi_magn_shell, niv_core, niv_for_shell, niv_shell,
                            count=1):
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

    ax[0].scatter(1 / (niv_for_shell + niv_shell), chi_dens_shell_w0, marker='o', c=colors_shell_d, edgecolors='k', alpha=0.8,
                  label='shell-scan')
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


def test_chi_asymptotic(siw_dmft, g2_dens: lfp.LocalFourPoint, g2_magn: lfp.LocalFourPoint, ek, mu, n, u, beta, niv_core,
                        niv_shell, niv_for_shell=30, count=1):
    sigma = tp.SelfEnergy(siw_dmft[None, None, None, :], beta, pos=False)
    giwk = tp.GreensFunction(sigma, ek, n=n)
    giwk.set_g_asympt(niv_asympt=np.max(niv_shell) + 1000)
    g_loc = giwk.k_mean(iv_range='full')
    wn = g2_dens.wn
    gchi_dens = lfp.gchir_from_g2(g2_dens, g_loc)
    gchi_magn = lfp.gchir_from_g2(g2_magn, g_loc)
    chi0_gen = bub.BubbleGenerator(wn, giwk)

    # niv_core = np.arange(10, niw)[::10][::-1]

    chi_dens_core, chi_magn_core = [], []
    vrg_dens_core, vrg_magn_core = [], []

    for niv in niv_core:
        gchi_dens.cut_iv(niv)
        gchi_magn.cut_iv(niv)
        vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, chi0_gen, u, niv_core=niv, niv_shell=1000)
        vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir(gchi_magn, chi0_gen, u, niv_core=niv, niv_shell=1000)
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

    plot_chi_convergence(wn, chi_dens_core, chi_magn_core, chi_dens_shell, chi_magn_shell, niv_core, niv_shell, count=count)
    plot_chi_convergence_wn(wn, chi_dens_core, chi_magn_core, chi_dens_shell, chi_magn_shell, niv_core, niv_for_shell, niv_shell,
                            count=count)

    plot_chi_sum_asymptotic(chi_dens_shell[0], chi_magn_shell[0], niv_core[0], beta, n, name='shell', count=count)
    plot_chi_sum_asymptotic(chi_dens_core[0], chi_magn_core[0], niv_for_shell, beta, n, name='core', count=count)


def test_chi_asymptotic_1():
    ddict = td.get_data_set_1(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    niv_core = np.array([100, 95, 90, 80, 60, 50, 40, 30, 20, 10])
    niv_shell = np.array([10, 50, 100, 500, 1000, 2000])[::-1]
    test_chi_asymptotic(ddict['siw'], ddict['g2_dens'], ddict['g2_magn'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'],
                        niv_core, niv_shell,
                        niv_for_shell=60, count=1)


def test_chi_asymptotic_2():
    ddict = td.get_data_set_2(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    niv_core = np.array([60, 50, 40, 30, 20, 10])
    niv_shell = np.array([10, 50, 100, 500, 1000, 2000, 5000, 10000])[::-1]
    test_chi_asymptotic(ddict['siw'], ddict['g2_dens'], ddict['g2_magn'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'],
                        niv_core, niv_shell,
                        niv_for_shell=10, count=2)


def test_chi_asymptotic_22():
    ddict = td.get_data_set_2(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    niv_core = np.array([60, 50, 40, 30, 20, 10])
    niv_shell = np.array([10, 50, 100, 500, 1000, 2000, 5000, 10000])[::-1]
    test_chi_asymptotic(ddict['siw'], ddict['g2_dens'], ddict['g2_magn'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'],
                        ddict['g2_file'], niv_core, niv_shell,
                        niv_for_shell=30, count=22)


def test_chi_asymptotic_222():
    ddict = td.get_data_set_2(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    niv_core = np.array([60, 50, 40, 30, 20, 10])
    niv_shell = np.array([1000])[::-1]
    test_chi_asymptotic(ddict['siw'], ddict['g2_dens'], ddict['g2_magn'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'],
                        niv_core, niv_shell,
                        niv_for_shell=60, count=222)


def test_chi_asymptotic_3():
    ddict = td.get_data_set_3(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    niv_core = np.array([70, 60, 50, 40, 30, 20, 10])  # 100, 95, 90,
    niv_shell = np.array([10, 50, 100, 500, 1000, 2000])[::-1]
    test_chi_asymptotic(ddict['siw'], ddict['g2_dens'], ddict['g2_magn'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'],
                        niv_core, niv_shell,
                        niv_for_shell=30, count=3)


def test_chi_asymptotic_4():
    ddict = td.get_data_set_4(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    niv_core = np.array([50, 40, 30, 20, 10])  # 100, 95, 90,
    niv_shell = np.array([1000, 2000])[::-1]
    test_chi_asymptotic(ddict['siw'], ddict['g2_dens'], ddict['g2_magn'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'],
                        niv_core, niv_shell,
                        niv_for_shell=30, count=4)


def test_chi_asymptotic_5():
    ddict = td.get_data_set_5(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    niv_core = np.array([50, 40, 30, 20, 10])  # 100, 95, 90,
    niv_shell = np.array([1000, 2000])[::-1]
    test_chi_asymptotic(ddict['siw'], ddict['g2_dens'], ddict['g2_magn'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'],
                        niv_core, niv_shell,
                        niv_for_shell=30, count=5)


def chi_convergence_from_gammar_urange_niv_core(ddict, ek, niv_core, niv_shell, name):
    siw = ddict['siw']
    beta = ddict['beta']
    u = ddict['u']
    n = ddict['n']
    g2_dens = ddict['g2_dens']
    g2_magn = ddict['g2_magn']

    chi_dens_dmft = ddict['chi_dens']
    chi_magn_dmft = ddict['chi_magn']

    sigma = tp.SelfEnergy(siw[None, None, None, :], beta, pos=False)
    giwk = tp.GreensFunction(sigma, ek, n=n)
    giwk.set_g_asympt(niv_asympt=np.max(niv_core) + 1000)
    g_loc = giwk.k_mean(iv_range='full')

    wn = g2_dens.wn
    gchi_dens = lfp.gchir_from_g2(g2_dens, g_loc)
    gchi_magn = lfp.gchir_from_g2(g2_magn, g_loc)
    chi0_gen = bub.BubbleGenerator(wn, giwk)

    # niv_core = np.arange(10, niw)[::10][::-1]

    chi_dens_core, chi_magn_core = [], []
    chi_dens_simple, chi_magn_simple = [], []
    wn_list = []

    for niv in niv_core:
        gchi_dens.cut_iv(niv)
        gchi_magn.cut_iv(niv)
        gchi0_core = chi0_gen.get_gchi0(niv + niv_shell)
        gamma_dens = lfp.gammar_from_gchir(gchi_dens, gchi0_core, u)
        gamma_magn = lfp.gammar_from_gchir(gchi_magn, gchi0_core, u)

        _, chi_dens = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_dens, chi0_gen, u, niv_shell)
        _, chi_magn = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, chi0_gen, u, niv_shell)
        chi_dens_simple.append(gchi_dens.contract_legs())
        chi_magn_simple.append(gchi_magn.contract_legs())
        chi_dens_core.append(mf.cut_w(chi_dens, niv))
        chi_magn_core.append(mf.cut_w(chi_magn, niv))
        wn_list.append(mf.wn(niv))

    wn_list = wn_list

    n_niv = len(niv_core)
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), dpi=500)
    ax = ax.flatten()
    colors = plt.cm.jet(np.linspace(0.0, 1.0, n_niv))
    for i in range(n_niv):
        ax[0].loglog(wn_list[i], chi_dens_core[i].real, '-o', color=colors[i], markeredgecolor='k', alpha=0.8, label=f'niv'
                                                                                                                     f'={niv_core[i]}')
        ax[1].loglog(wn_list[i], chi_magn_core[i].real, '-o', color=colors[i], markeredgecolor='k', alpha=0.8, label=f'niv'
                                                                                                                     f'={niv_core[i]}')

    wn_dmft = mf.wn(len(chi_dens_dmft) // 2)
    ax[0].plot(wn_dmft, chi_dens_dmft.real, '-k', label='DMFT')
    ax[1].plot(wn_dmft, chi_magn_dmft.real, '-k', label='DMFT')

    for i in range(n_niv):
        niw_cut = min(len(wn_list[i]) // 2, len(wn_dmft) // 2)
        wn_cut = mf.wn(niw_cut)
        ax[2].loglog(wn_cut, np.abs(mf.cut_w(chi_dens_core[i].real, niw_cut) - mf.cut_w(chi_dens_dmft.real, niw_cut)), '-o',
                     color=colors[i], markeredgecolor='k', alpha=0.8, label=f'niv={niv_core[i]}')
        ax[3].loglog(wn_cut, np.abs(mf.cut_w(chi_magn_core[i].real, niw_cut) - mf.cut_w(chi_magn_dmft.real, niw_cut)), '-o',
                     color=colors[i], markeredgecolor='k', alpha=0.8, label=f'niv={niv_core[i]}')
    ax[1].legend()
    ax[3].legend()
    for a in ax:
        a.set_xlim(0, np.max(niv_core))
        a.set_ylim(0, None)
        a.set_xlabel('$n$')
    # ax[0].set_xlim(-5, np.max(niv_core))
    ax[0].set_ylim(-0.1, 2 * np.max(chi_dens_dmft))
    ax[1].set_xlim(-5, np.max(niv_core))
    ax[0].set_ylabel(r'$\chi_{d}^{c}$')
    ax[1].set_ylabel(r'$\tilde{\chi}_{m}^{c}$')
    ax[2].set_ylabel(r'$\chi_{d}^{c}$')
    ax[3].set_ylabel(r'$\tilde{\chi}_{m}^{c}$')
    plt.tight_layout()
    plt.savefig(PLOT_PATH + f'chi_r_convergence_check_{name}.png')
    plt.close()

    fig, ax = plt.subplots(2, 2, figsize=(6, 6), dpi=500)
    ax = ax.flatten()

    x = 1 / np.array(niv_core)
    y = np.array([chi[len(chi) // 2] for chi in chi_dens_core])
    ax[0].plot(x, y, '-o')
    ax[0].plot(0, chi_dens_dmft[len(chi_dens_dmft) // 2].real, 'h')
    ax[0].set_ylabel('$\chi_{d}(i\omega_n = 0)$')

    y = np.array([chi[len(chi) // 2] for chi in chi_magn_core])
    ax[1].plot(x, y, '-o')
    ax[1].plot(0, chi_magn_dmft[len(chi_magn_dmft) // 2].real, 'h')
    ax[1].set_ylabel('$\chi_{m}(i\omega_n = 0)$')

    y = np.array([1 / beta * np.sum(chi) for chi in chi_dens_core])
    ax[2].plot(x, y, '-o')
    ax[2].set_ylabel('$\sum_w \chi_{d}(i\omega_n)$')
    ax[2].plot(0, 1 / beta * np.sum(chi_dens_dmft.real), 'h')
    ax[2].hlines(1 / beta * np.sum(chi_dens_dmft.real), 0, np.max(x), ls='--')

    y = np.array([1 / beta * np.sum(chi) for chi in chi_magn_core])
    ax[3].plot(x, y, '-o')
    ax[3].plot(0, 1 / beta * np.sum(chi_magn_dmft.real), 'h')
    ax[3].hlines(1 / beta * np.sum(chi_magn_dmft.real), 0, np.max(x), ls='--')
    ax[3].set_ylabel('$\sum_w \chi_{m}(i\omega_n)$')

    for a in ax:
        a.set_xlabel('1/niv')

    plt.tight_layout()
    plt.savefig(PLOT_PATH + f'chi_r_point_convergence_{name}.png')
    plt.close()


def chi_convergence_from_gammar_urange_niv_shell(ddict, ek, niv_core, niv_shell, name):
    siw = ddict['siw']
    beta = ddict['beta']
    u = ddict['u']
    n = ddict['n']
    g2_dens = ddict['g2_dens']
    g2_magn = ddict['g2_magn']

    chi_dens_dmft = ddict['chi_dens']
    chi_magn_dmft = ddict['chi_magn']

    sigma = tp.SelfEnergy(siw[None, None, None, :], beta, pos=False)
    giwk = tp.GreensFunction(sigma, ek, n=n)
    giwk.set_g_asympt(niv_asympt=np.max(niv_shell) + 2000)
    g_loc = giwk.k_mean(iv_range='full')


    gchi_dens = lfp.gchir_from_g2(g2_dens, g_loc)
    gchi_magn = lfp.gchir_from_g2(g2_magn, g_loc)


    chi_dens_asympt, chi_magn_asympt = [], []
    chi_dens_urange, chi_magn_urange = [], []
    gchi_dens.cut_iv(niv_core)
    gchi_magn.cut_iv(niv_core)

    gchi_dens.is_full_w = True
    gchi_magn.is_full_w = True
    gchi_dens.cut_iw(niv_core)
    gchi_magn.cut_iw(niv_core)

    wn = gchi_magn.wn
    chi0_gen = bub.BubbleGenerator(wn, giwk)

    for niv in niv_shell:
        gchi0_shell = chi0_gen.get_gchi0(niv_core + niv)
        gamma_dens = lfp.gammar_from_gchir(gchi_dens, gchi0_shell, u)
        gamma_magn = lfp.gammar_from_gchir(gchi_magn, gchi0_shell, u)

        _, chi_dens = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_dens, chi0_gen, u, niv)
        _, chi_magn = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, chi0_gen, u, niv)
        chi_dens_asympt.append(mf.cut_w(chi_dens, niv_core))
        chi_magn_asympt.append(mf.cut_w(chi_magn, niv_core))

        chi0_asympt = chi0_gen.get_asymptotic_correction(niv_core+niv)
        chi_dens_urange.append(mf.cut_w(chi_dens-chi0_asympt, niv_core))
        chi_magn_urange.append(mf.cut_w(chi_magn-chi0_asympt, niv_core))

    wn = mf.wn(niv_core)

    n_niv = len(niv_shell)
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), dpi=500)
    ax = ax.flatten()
    colors = plt.cm.jet(np.linspace(0.0, 1.0, n_niv))
    for i in range(n_niv):
        ax[0].loglog(wn, chi_dens_asympt[i].real, '-o', color=colors[i], markeredgecolor='k', alpha=0.8, label=f'niv'
                                                                                                             f'={niv_shell[i]}')
        ax[1].loglog(wn, chi_magn_asympt[i].real, '-o', color=colors[i], markeredgecolor='k', alpha=0.8, label=f'niv'
                                                                                                             f'={niv_shell[i]}')

    wn_dmft = mf.wn(len(chi_dens_dmft) // 2)
    ax[0].plot(wn_dmft, chi_dens_dmft.real, '-k', label='DMFT')
    ax[1].plot(wn_dmft, chi_magn_dmft.real, '-k', label='DMFT')

    for i in range(n_niv):
        niw_cut = min(len(wn) // 2, len(wn_dmft) // 2)
        wn_cut = mf.wn(niw_cut)
        ax[2].loglog(wn_cut, np.abs(mf.cut_w(chi_dens_asympt[i].real, niw_cut) - mf.cut_w(chi_dens_dmft.real, niw_cut)), '-o',
                     color=colors[i], markeredgecolor='k', alpha=0.8, label=f'niv={niv_shell[i]}')
        ax[3].loglog(wn_cut, np.abs(mf.cut_w(chi_magn_asympt[i].real, niw_cut) - mf.cut_w(chi_magn_dmft.real, niw_cut)), '-o',
                     color=colors[i], markeredgecolor='k', alpha=0.8, label=f'niv={niv_shell[i]}')
    ax[1].legend()
    ax[3].legend()
    for a in ax:
        a.set_xlim(0, np.max(niv_shell))
        a.set_ylim(0, None)
        a.set_xlabel('$n$')
    # ax[0].set_xlim(-5, np.max(niv_core))
    ax[0].set_ylim(-0.1, 2 * np.max(chi_dens_dmft))
    ax[1].set_xlim(-5, np.max(niv_shell))
    ax[0].set_ylabel(r'$\chi_{d}^{c}$')
    ax[1].set_ylabel(r'$\tilde{\chi}_{m}^{c}$')
    ax[2].set_ylabel(r'$\chi_{d}^{c}$')
    ax[3].set_ylabel(r'$\tilde{\chi}_{m}^{c}$')
    plt.tight_layout()
    plt.savefig(PLOT_PATH + f'chi_r_convergence_niv_shell_check_{name}.png')
    plt.close()

    fig, ax = plt.subplots(2, 2, figsize=(6, 6), dpi=500)
    ax = ax.flatten()

    x = 1 / np.array(niv_shell)
    y = np.array([chi[len(chi) // 2] for chi in chi_dens_asympt])
    ax[0].plot(x, y, '-o')
    y = np.array([chi[len(chi) // 2] for chi in chi_dens_urange])
    ax[0].plot(x, y, '-p')
    ax[0].plot(0, chi_dens_dmft[len(chi_dens_dmft) // 2].real, 'h')
    ax[0].set_ylabel('$\chi_{d}(i\omega_n = 0)$')

    y = np.array([chi[len(chi) // 2] for chi in chi_magn_asympt])
    ax[1].plot(x, y, '-o')
    y = np.array([chi[len(chi) // 2] for chi in chi_magn_urange])
    ax[1].plot(x, y, '-p')
    ax[1].plot(0, chi_magn_dmft[len(chi_magn_dmft) // 2].real, 'h')
    ax[1].set_ylabel('$\chi_{m}(i\omega_n = 0)$')

    y = np.array([1 / beta * np.sum(chi) for chi in chi_dens_asympt])
    ax[2].plot(x, y, '-o')
    y = np.array([1 / beta * np.sum(chi) for chi in chi_dens_urange])
    ax[2].plot(x, y, '-p')
    ax[2].set_ylabel('$\sum_w \chi_{d}(i\omega_n)$')
    ax[2].plot(0, 1 / beta * np.sum(chi_dens_dmft.real), 'h')
    ax[2].hlines(1 / beta * np.sum(chi_dens_dmft.real), 0, np.max(x), ls='--')

    y = np.array([1 / beta * np.sum(chi) for chi in chi_magn_asympt])
    ax[3].plot(x, y, '-o')
    y = np.array([1 / beta * np.sum(chi) for chi in chi_magn_urange])
    ax[3].plot(x, y, '-p')
    ax[3].plot(0, 1 / beta * np.sum(chi_magn_dmft.real), 'h')
    ax[3].hlines(1 / beta * np.sum(chi_magn_dmft.real), 0, np.max(x), ls='--')

    ax[3].set_ylabel('$\sum_w \chi_{m}(i\omega_n)$')

    for a in ax:
        a.set_xlabel('1/niv')

    plt.tight_layout()
    plt.savefig(PLOT_PATH + f'chi_r_point_convergence_shell_{name}.png')
    plt.close()


def test_chi_convergence_9():
    ddict = td.get_data_set_9(load_g2=True, load_chi=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    # niv_core = np.array([100, 95, 90, 80, 60, 50, 40, 30, 20, 10])
    niv_core = np.array([40, 30, 20, 10])
    niv_shell = 100
    chi_convergence_from_gammar_urange_niv_core(ddict, ek, niv_core, niv_shell, 9)


def test_chi_convergence_niv_core_6():
    ddict = td.get_data_set_6(load_g2=True, load_chi=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    # niv_core = np.array([100, 95, 90, 80, 60, 50, 40, 30, 20, 10])
    niv_core = np.array([60, 40, 20, 10])
    niv_shell = 100
    chi_convergence_from_gammar_urange_niv_core(ddict, ek, niv_core, niv_shell, 'ds6_ns100')
    niv_shell = 0
    chi_convergence_from_gammar_urange_niv_core(ddict, ek, niv_core, niv_shell, 'ds6_ns0')
    niv_shell = 200
    chi_convergence_from_gammar_urange_niv_core(ddict, ek, niv_core, niv_shell, 'ds6_ns200')


def test_chi_convergence_niv_shell_6():
    ddict = td.get_data_set_6(load_g2=True, load_chi=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    # niv_core = np.array([100, 95, 90, 80, 60, 50, 40, 30, 20, 10])
    niv_shell = np.array([0, 50, 100, 200, 400])

    niv_core = 10
    chi_convergence_from_gammar_urange_niv_shell(ddict, ek, niv_core, niv_shell, 'ds6_nc10')
    niv_core = 20
    chi_convergence_from_gammar_urange_niv_shell(ddict, ek, niv_core, niv_shell, 'ds6_nc20')
    niv_core = 40
    chi_convergence_from_gammar_urange_niv_shell(ddict, ek, niv_core, niv_shell, 'ds6_nc40')
    niv_core = 60
    chi_convergence_from_gammar_urange_niv_shell(ddict, ek, niv_core, niv_shell, 'ds6_nc60')


def test_vertex_frequency_convention(ddict, ek, count=1):
    g2_file = ddict['g2_file']
    siw_dmft = ddict['siw']
    beta = ddict['beta']
    n = ddict['n']
    sigma = tp.SelfEnergy(siw_dmft[None, None, None, :], beta, pos=False)
    giwk = tp.GreensFunction(sigma, ek, n=n)
    giwk.set_g_asympt(niv_asympt=1000)
    g_loc = giwk.k_mean(iv_range='full')
    niw = g2_file.get_niw(channel='dens')
    wn = mf.wn(niw)
    g2_dens = lfp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='dens', iw=wn), beta=beta, wn=wn, channel='dens')
    g2_magn = lfp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='magn', iw=wn), beta=beta, wn=wn, channel='magn')
    gchi_dens = lfp.gchir_from_g2(g2_dens, g_loc)
    gchi_magn = lfp.gchir_from_g2(g2_magn, g_loc)
    chi0_gen = bub.BubbleGenerator(wn, giwk)
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
    test_chi_convergence_niv_core_6()
    test_chi_convergence_niv_shell_6()

    print('Finished!')
