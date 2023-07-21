import numpy as np
import sys, os

sys.path.append('../src')
sys.path.append('./src')
import dga.two_point as tp
import dga.matsubara_frequencies as mf
import matplotlib.pyplot as plt
import dga.brillouin_zone as bz
import dga.hk as hamk
import TestData as td

PLOT_PATH = './TestPlots/TestGF/'


def local_dmft_consistency(giw_dmft, sigma_dmft, mu_dmft, ek, u,n, beta, count=1):
    sigma_dmft = tp.SelfEnergy(sigma_dmft[None, None, None, :], beta,pos=False,smom0=tp.get_smom0(u,n),smom1=tp.get_smom1(u,n))
    giwk = tp.GreensFunction(sigma_dmft, ek, n=n)
    g_loc = giwk.k_mean()
    print('---------------')
    print(f'mu = {giwk.mu}')
    print(f'mu-w2dyn = {mu_dmft}')
    print('---------------')

    vn_core = mf.vn(giwk.niv_core)

    n_plot = 100
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.5),dpi=500)
    ax[0].plot(mf.cut_v_1d_pos(vn_core, n_plot), mf.cut_v_1d_pos(g_loc, n_plot).real, '-o', color='cornflowerblue')
    ax[0].plot(mf.cut_v_1d_pos(vn_core, n_plot), mf.cut_v_1d_pos(giw_dmft, n_plot).real, '-', color='k')
    ax[1].plot(mf.cut_v_1d_pos(vn_core, n_plot), mf.cut_v_1d_pos(g_loc, n_plot).imag, '-o', color='cornflowerblue')
    ax[1].plot(mf.cut_v_1d_pos(vn_core, n_plot), mf.cut_v_1d_pos(giw_dmft, n_plot).imag, '-', color='k')
    ax[0].set_xlabel(r'$\nu_n$')
    ax[1].set_xlabel(r'$\nu_n$')
    ax[0].set_ylabel(r'$\Re G$')
    ax[1].set_ylabel(r'$\Im G$')
    plt.tight_layout()
    plt.savefig(PLOT_PATH + f'GreensFunction_Test_{count}.png')
    plt.show()





def greens_function_asymptotic(giw_dmft, sigma_dmft, ek, u,n, beta, count=1):
    sigma_dmft = tp.SelfEnergy(sigma_dmft[None, None, None, :], beta, pos=False, smom0=tp.get_smom0(u, n), smom1=tp.get_smom1(u, n))
    giwk = tp.GreensFunction(sigma_dmft, ek, n=n)
    niv_asympt = 20000
    giwk.set_g_asympt(niv_asympt)
    g_loc = giwk.k_mean(iv_range='full')

    vn_asympt = mf.vn(giwk.niv_core + giwk.niv_asympt)


    n_start = giwk.niv_core-50
    n_plot = 300
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
    ax[0].plot(mf.cut_v_1d_pos(vn_asympt, n_plot, n_start), mf.cut_v_1d_pos(g_loc, n_plot, n_start).real, '-o', color='cornflowerblue')
    ax[0].plot(mf.cut_v_1d_pos(vn_asympt, n_plot, n_start), mf.cut_v_1d_pos(giw_dmft, n_plot, n_start).real, '-', color='k')
    ax[1].plot(mf.cut_v_1d_pos(vn_asympt, n_plot, n_start), mf.cut_v_1d_pos(g_loc, n_plot, n_start).imag, '-o', color='cornflowerblue')
    ax[1].plot(mf.cut_v_1d_pos(vn_asympt, n_plot, n_start), mf.cut_v_1d_pos(giw_dmft, n_plot, n_start).imag, '-', color='k')
    ax[0].set_xlabel(r'$\nu_n$')
    ax[1].set_xlabel(r'$\nu_n$')
    ax[0].set_ylabel(r'$\Re G$')
    ax[1].set_ylabel(r'$\Im G$')
    plt.tight_layout()
    plt.savefig(PLOT_PATH + f'GreensFunction_TestAsymptotic_{count}.png')
    plt.show()

    # n_start = 500
    # fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
    # ax[0].plot(mf.cut_v_1d_pos(vn_asympt, n_plot, n_start), mf.cut_v_1d_pos(g_loc, n_plot, n_start).real, '-o', color='cornflowerblue')
    # ax[0].plot(mf.cut_v_1d_pos(vn_asympt, n_plot, n_start), mf.cut_v_1d_pos(giw_dmft, n_plot, n_start).real, '-', color='k')
    # ax[1].plot(mf.cut_v_1d_pos(vn_asympt, n_plot, n_start), mf.cut_v_1d_pos(g_loc, n_plot, n_start).imag, '-o', color='cornflowerblue')
    # ax[1].plot(mf.cut_v_1d_pos(vn_asympt, n_plot, n_start), mf.cut_v_1d_pos(giw_dmft, n_plot, n_start).imag, '-', color='k')
    # ax[0].set_xlabel(r'$\nu_n$')
    # ax[1].set_xlabel(r'$\nu_n$')
    # ax[0].set_ylabel(r'$\Re G$')
    # ax[1].set_ylabel(r'$\Im G$')
    # plt.tight_layout()
    # plt.savefig(PLOT_PATH + f'GreensFunction_TestAsymptotic_core_{count}.png')
    # plt.show()


def test_greens_function_asymptotic_1():
    ddict = td.get_data_set_1()
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, ddict['hr'])
    greens_function_asymptotic(ddict['giw'], ddict['siw'], ek, ddict['u'], ddict['n'], ddict['beta'], count=1)


def test_greens_function_asymptotic_2():
    ddict = td.get_data_set_2()
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, ddict['hr'])
    greens_function_asymptotic(ddict['giw'], ddict['siw'], ek, ddict['u'], ddict['n'], ddict['beta'], count=2)

def test_greens_function_asymptotic_3():
    ddict = td.get_data_set_3()
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, ddict['hr'])
    greens_function_asymptotic(ddict['giw'], ddict['siw'], ek, ddict['u'], ddict['n'], ddict['beta'], count=3)


def test_greens_function_asymptotic_ed_1():
    ddict = td.get_data_set_4()
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, ddict['hr'])
    greens_function_asymptotic(ddict['giw'], ddict['siw'], ek, ddict['u'], ddict['n'], ddict['beta'], count=4)


def test_local_dmft_consistency_1():
    ddict = td.get_data_set_1()
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, ddict['hr'])
    local_dmft_consistency(ddict['giw'], ddict['siw'], ddict['mu'], ek, ddict['u'], ddict['n'], ddict['beta'], count=1)

def test_local_dmft_consistency_2():
    ddict = td.get_data_set_2()
    # Build Green's function:
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, ddict['hr'])
    local_dmft_consistency(ddict['giw'], ddict['siw'], ddict['mu'], ek, ddict['u'], ddict['n'], ddict['beta'], count=2)

def test_local_dmft_consistency_3():
    ddict = td.get_data_set_3()
    # Build Green's function:
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, ddict['hr'])
    local_dmft_consistency(ddict['giw'], ddict['siw'], ddict['mu'], ek, ddict['u'], ddict['n'], ddict['beta'], count=3)

def test_local_dmft_consistency_ed_1():
    ddict = td.get_data_set_4()
    # Build Green's function:
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, ddict['hr'])
    local_dmft_consistency(ddict['giw'], ddict['siw'], ddict['mu'], ek, ddict['u'], ddict['n'], ddict['beta'], count=4)


def test_local_dmft_consistency_9():
    ddict = td.get_data_set_9()
    # Build Green's function:
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, ddict['hr'])
    local_dmft_consistency(ddict['giw'], ddict['siw'], ddict['mu'], ek, ddict['u'], ddict['n'], ddict['beta'], count=9)

def test_greens_function_asymptotic_9():
    ddict = td.get_data_set_9()
    nk = (42, 42, 1)
    k_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid, ddict['hr'])
    greens_function_asymptotic(ddict['giw'], ddict['siw'], ek, ddict['u'], ddict['n'], ddict['beta'], count=9)

if __name__ == '__main__':

    # test_greens_function_asymptotic_1()
    # test_greens_function_asymptotic_2()
    # test_greens_function_asymptotic_3()
    #
    # test_greens_function_asymptotic_ed_1()
    # test_local_dmft_consistency_ed_1()
    #
    # test_local_dmft_consistency_1()
    # test_local_dmft_consistency_2()
    #
    # test_local_dmft_consistency_3()
    #
    # test_local_dmft_consistency_2()
    # test_greens_function_asymptotic_2()

    # test_local_dmft_consistency_9()
    test_greens_function_asymptotic_9()
