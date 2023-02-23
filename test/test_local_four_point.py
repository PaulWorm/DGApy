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

PLOT_PATH = './TestPlots/'


def test_chi_asymptotic(siw_dmft, ek, mu, n, u, beta, g2_file, count=1):
    sigma = tp.SelfEnergy(siw_dmft[None, None, None, :], beta,pos=False)
    giwk = tp.GreensFunction(sigma, ek, n=n)
    giwk.set_g_asympt(niv_asympt=5000)
    g_loc = giwk.k_mean(range='full')
    niw = g2_file.get_niw(channel='dens')
    wn = mf.wn(niw)
    g2_dens = lfp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='dens', iw=wn), beta=beta, wn=wn, channel='dens')
    g2_magn = lfp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='magn', iw=wn), beta=beta, wn=wn, channel='magn')
    gchi_dens = lfp.gchir_from_g2(g2_dens, g_loc)
    gchi_magn = lfp.gchir_from_g2(g2_magn, g_loc)
    chi0_gen = lfp.LocalBubble(wn, g_loc, beta)

    # niv_core = np.arange(10, niw)[::10][::-1]

    niv_core = np.array([60,50, 40, 30, 20, 10])
    niv_shell = np.array([10, 50, 100, 500, 1000, 2000])[::-1]
    n_niv = len(niv_core)
    n_niv_shell = len(niv_shell)
    niv_for_shell = 30

    chi_dens_core = []
    chi_magn_core = []

    for niv in niv_core:
        gchi_dens.cut_iv(niv)
        gchi_magn.cut_iv(niv)
        _, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, chi0_gen, u, niv_core=niv)
        _, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir(gchi_magn, chi0_gen, u, niv_core=niv)
        chi_dens_core.append(chi_dens)
        chi_magn_core.append(chi_magn)

    gchi_dens = lfp.gchir_from_g2(g2_dens, g_loc)
    gchi_magn = lfp.gchir_from_g2(g2_magn, g_loc)

    chi_dens_shell = []
    chi_magn_shell = []
    gchi_dens.cut_iv(niv_for_shell)
    gchi_magn.cut_iv(niv_for_shell)
    for niv in niv_shell:
        _, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, chi0_gen, u, niv_core=niv_for_shell, niv_shell=niv)
        _, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir(gchi_magn, chi0_gen, u, niv_core=niv_for_shell, niv_shell=niv)
        chi_dens_shell.append(chi_dens)
        chi_magn_shell.append(chi_magn)

    wn = mf.wn(niw)
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), dpi=500)
    ax = ax.flatten()
    colors = plt.cm.jet(np.linspace(0.0, 1.0, n_niv))
    colors_shell = plt.cm.jet(np.linspace(0.0, 1.0, n_niv_shell))
    for i in range(n_niv):
        ax[0].plot(wn, chi_dens_core[i].real, '-o', color=colors[i], markeredgecolor='k', alpha=0.8, label=f'niv={niv_core[i]}')
        ax[1].plot(wn, chi_magn_core[i].real, '-o', color=colors[i], markeredgecolor='k', alpha=0.8)

    for i in range(n_niv_shell):
        ax[2].plot(wn, chi_dens_shell[i].real, '-o', color=colors_shell[i], markeredgecolor='k', alpha=0.8, label=f'niv-shell={niv_shell[i]}')
        ax[3].plot(wn, chi_magn_shell[i].real, '-o', color=colors_shell[i], markeredgecolor='k', alpha=0.8)
    ax[0].legend()
    ax[2].legend()
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

    colors_d = plt.cm.Blues(np.linspace(0.5, 1.0, n_niv))[::-1]
    colors_m = plt.cm.Blues(np.linspace(0.5, 1.0, n_niv))[::-1]

    colors_shell_d = plt.cm.Reds(np.linspace(0.5, 1.0, n_niv_shell))[::-1]
    colors_shell_m = plt.cm.Reds(np.linspace(0.5, 1.0, n_niv_shell))[::-1]

    chi_dens_core_w0 = [chi_dens_core[i][niw].real for i in range(n_niv)]
    chi_magn_core_w0 = [chi_magn_core[i][niw].real for i in range(n_niv)]
    chi_dens_shell_w0 = [chi_dens_shell[i][niw].real for i in range(n_niv_shell)]
    chi_magn_shell_w0  = [chi_magn_shell[i][niw].real for i in range(n_niv_shell)]

    n_w = niv_for_shell//2
    n_w = niw
    chi_dens_core_w15 = [chi_dens_core[i][niw+n_w].real for i in range(n_niv)]
    chi_magn_core_w15 = [chi_magn_core[i][niw+n_w].real for i in range(n_niv)]
    chi_dens_shell_w15 = [chi_dens_shell[i][niw+n_w].real for i in range(n_niv_shell)]
    chi_magn_shell_w15  = [chi_magn_shell[i][niw+n_w].real for i in range(n_niv_shell)]

    fig, ax = plt.subplots(2, 2, figsize=(6, 5), dpi=500)
    ax = ax.flatten()

    ax[0].scatter(1/niv_core, chi_dens_core_w0, marker='o', c=colors_d, edgecolors='k', alpha=0.8,label='core-scan')
    ax[1].scatter(1/niv_core, chi_magn_core_w0, marker='o', c=colors_m, edgecolors='k', alpha=0.8)

    ax[2].scatter(1/niv_core, chi_dens_core_w15, marker='o', c=colors_d, edgecolors='k', alpha=0.8,label='core-scan')
    ax[3].scatter(1/niv_core, chi_magn_core_w15, marker='o', c=colors_m, edgecolors='k', alpha=0.8)


    ax[0].scatter(1/(niv_for_shell+niv_shell), chi_dens_shell_w0, marker='o', c=colors_shell_d, edgecolors='k', alpha=0.8,label='shell-scan')
    ax[1].scatter(1/(niv_for_shell+niv_shell), chi_magn_shell_w0, marker='o', c=colors_shell_m, edgecolors='k', alpha=0.8)

    ax[2].scatter(1/(niv_for_shell+niv_shell), chi_dens_shell_w15, marker='o', c=colors_shell_d, edgecolors='k', alpha=0.8)
    ax[3].scatter(1/(niv_for_shell+niv_shell), chi_magn_shell_w15, marker='o', c=colors_shell_m, edgecolors='k', alpha=0.8)
    ax[0].legend()
    ax[0].set_xlabel('$1/n$')
    ax[1].set_xlabel('$1/n$')
    ax[2].set_xlabel('$1/n$')
    ax[3].set_xlabel('$1/n$')
    # ax[2].set_xlabel('$n_{shell}$')
    # ax[3].set_xlabel('$n_{shell}$')
    ax[0].set_ylabel(r'$\chi_{d}(\omega=0)$')
    ax[1].set_ylabel(r'$\chi_{m}(\omega=0)$')
    ax[2].set_ylabel(r'$\chi_{m}'+f'(\omega={n_w})$')
    ax[3].set_ylabel(r'$\chi_{m}'+f'(\omega={n_w})$')
    # ax[2].set_ylabel(r'$\chi_{d}$')
    # ax[3].set_ylabel(r'$\chi_{m}$')
    plt.tight_layout()
    plt.savefig(PLOT_PATH + f'chi_r_convergence_check_w0_w{n_w}_{count}.png')
    plt.show()

def test_chi_asymptotic_1():
    ddict = td.get_data_set_1(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    test_chi_asymptotic(ddict['siw'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'], ddict['g2_file'], count=1)


def test_chi_asymptotic_2():
    ddict = td.get_data_set_2(load_g2=True)
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])
    test_chi_asymptotic(ddict['siw'], ek, ddict['mu'], ddict['n'], ddict['u'], ddict['beta'], ddict['g2_file'], count=2)


if __name__ == '__main__':
    test_chi_asymptotic_1()
    test_chi_asymptotic_2()
    # test_chi_asymptotic_2()
    print('Finished!')
