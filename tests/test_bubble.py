import sys, os
import numpy as np
from scipy import optimize as opt

from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import two_point as twop
from dga import bubble as bub
from test_util import test_data as td
from test_util import util_for_testing as t_util


def test_bubble_asymptotics():
    # prepare data:
    ddict, hr = td.load_minimal_dataset()
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    nk = (12, 12, 1)
    sym = bz.two_dimensional_square_symmetries()
    k_grid = bz.KGrid(nk=nk, symmetries=sym)
    ek = hr.get_ek(k_grid)
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'], niv_asympt=6000)

    # set up the bubble generator:
    niw_bub = 0
    wn_bub = mf.wn(niw_bub)
    bubble_gen = bub.BubbleGenerator(wn_bub, giwk_obj, is_full_wn=True)

    niv_core = 100
    asympt = bubble_gen.get_asymptotic_correction(niv_core=niv_core)
    q_list = k_grid.get_irrq_list()
    asympt_q = bubble_gen.get_asymptotic_correction_q(niv_core=niv_core, q_list=q_list)
    asympt_loc = k_grid.k_mean(asympt_q)

    t_util.test_array(asympt, asympt_loc, 'asymptotic_correction_q_and_loc')


def test_bubble_gen():
    # prepare data:
    ddict, hr = td.load_minimal_dataset()
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    nk = (12, 12, 1)
    sym = bz.two_dimensional_square_symmetries()
    k_grid = bz.KGrid(nk=nk, symmetries=sym)
    ek = hr.get_ek(k_grid)
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'])

    # set up the bubble generator:
    niw_bub = 30
    wn_bub = mf.wn(niw_bub)
    bubble_gen = bub.BubbleGenerator(wn_bub, giwk_obj, is_full_wn=True)

    # local part:
    niv_core = 30
    niv_shell = 0
    chi0 = bubble_gen.get_chi0(niv_core)
    chi0_asympt = bubble_gen.get_chi0(niv_core, do_asympt=True)
    t_util.test_array(chi0, np.conj(np.flip(chi0)), 'chi0_omega_sym')
    t_util.test_array(chi0_asympt, np.conj(np.flip(chi0_asympt)), 'chi0_asympt_omega_sym')

    gchi0 = bubble_gen.get_gchi0(niv_core)
    chi0_from_gchi0 = 1 / bubble_gen.beta ** 2 * np.sum(gchi0, axis=(-1,))
    t_util.test_array(chi0_from_gchi0, chi0, 'chi0_and_gchi0')

    # non-local part:
    q_list = k_grid.get_irrq_list()
    chi0_q = bubble_gen.get_chi0_q_list(niv_core, q_list)
    chi0_q_shell = bubble_gen.get_chi0q_shell(chi0_q, niv_core=niv_core, niv_shell=niv_shell, q_list=q_list)
    chi0_q_asympt = chi0_q + chi0_q_shell
    chi0_loc = k_grid.k_mean(chi0_q)
    chi0_loc_asympt = k_grid.k_mean(chi0_q_asympt)

    t_util.test_array(chi0_loc, chi0, 'chi0_from_chi0_q')
    t_util.test_array(chi0_loc_asympt, chi0_asympt, 'chi0_asympt_from_chi0_q')


def test_bubble_asymptotic_convergence():
    # prepare data:
    ddict, hr = td.load_minimal_dataset()
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    nk = (12, 12, 1)
    sym = bz.two_dimensional_square_symmetries()
    k_grid = bz.KGrid(nk=nk, symmetries=sym)
    ek = hr.get_ek(k_grid)
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'], niv_asympt=6000)

    # set up the bubble generator:
    niw_bub = 0
    wn_bub = mf.wn(niw_bub)
    bubble_gen = bub.BubbleGenerator(wn_bub, giwk_obj, is_full_wn=True)

    # local part:
    niv_core_list = np.array([100, 300, 1000, 2000, 3000, 4000, 5000][::-1])
    # niv_core_list = np.array([10, 30, 50, 100, 300, 2000, 3000][::-1])
    chi0 = np.array([bubble_gen.get_chi0(niv_core)[0] for niv_core in niv_core_list])
    chi0_asympt = np.array([bubble_gen.get_chi0(niv_core, do_asympt=True)[0] for niv_core in niv_core_list])

    assert t_util.is_decreasing(chi0.real), 'chi0 is not converging'
    assert t_util.is_monotonic(np.round(chi0_asympt.real, decimals=10)), 'chi0_asympt is not converging'

    fit_fun = lambda x, a, b: a + b * x

    n_fit = 5
    popt, pcov = opt.curve_fit(f=fit_fun, xdata=1 / niv_core_list[:n_fit], ydata=chi0[:n_fit].real)
    chi0_extrap = fit_fun(0, *popt)
    popt, pcov = opt.curve_fit(f=fit_fun, xdata=1 / niv_core_list, ydata=chi0_asympt.real)
    chi0_extrap_asympt = fit_fun(0, *popt)

    t_util.test_array([chi0_extrap, ], [chi0_extrap_asympt, ], 'chi0_extrap')

def test_is_full_w():
    # prepare data:
    ddict, hr = td.load_minimal_dataset()
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    nk = (12, 12, 1)
    sym = bz.two_dimensional_square_symmetries()
    k_grid = bz.KGrid(nk=nk, symmetries=sym)
    ek = hr.get_ek(k_grid)
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'], niv_asympt=6000)

    # set up the bubble generator:
    niw_bub = 50
    niv_sum = 100
    wn_bub = mf.wn(niw_bub)
    bubble_gen = bub.BubbleGenerator(wn_bub, giwk_obj, is_full_wn=True)
    bubble_gen2 = bub.BubbleGenerator(wn_bub, giwk_obj, is_full_wn=False)

    # test chi0:
    chi0 = bubble_gen.get_chi0(niv_sum)
    chi0_2 = bubble_gen2.get_chi0(niv_sum)

    t_util.test_array(chi0, chi0_2, 'chi0_full_w_consistency')

    # test gchi0
    gchi0 = bubble_gen.get_gchi0(niv_sum)
    gchi0_2 = bubble_gen2.get_gchi0(niv_sum)
    t_util.test_array(gchi0, gchi0_2, 'gchi0_full_w_consistency')

def test_q_sum_consistency():
    # prepare data:
    ddict, hr = td.load_minimal_dataset()
    smom0 = twop.get_smom0(ddict['u'], ddict['n'])
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    nk = (12, 12, 1)
    sym = bz.two_dimensional_square_symmetries()
    k_grid = bz.KGrid(nk=nk, symmetries=sym)
    q_list = k_grid.get_irrq_list()
    ek = hr.get_ek(k_grid)
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n = ddict['n'], niv_asympt=6000)
    t_util.test_array([giwk_obj.n,],[ddict['n'],], 'giwk_n_consistency')

    # set up the bubble generator:
    niw_bub = 50
    niv_sum = 100
    wn_bub = mf.wn(niw_bub)
    bubble_gen = bub.BubbleGenerator(wn_bub, giwk_obj, is_full_wn=True)

    # test chi0:
    chi0 = bubble_gen.get_chi0(niv_sum)
    chi0_q = bubble_gen.get_chi0_q_list(niv_sum,q_list)
    chi0_loc = k_grid.k_mean(chi0_q)
    t_util.test_array(chi0, chi0_loc, 'chi0_flat_q_consistency')


if __name__ == '__main__':
    test_bubble_asymptotics()
    test_bubble_gen()
    test_bubble_asymptotic_convergence()
    test_is_full_w()
    test_q_sum_consistency()
