import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import optimize as opt

from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import two_point as twop
from dga import bubble as bub
from dga import local_four_point as lfp
from test_util import test_data as td
from test_util import util_for_testing as t_util


def test_routine_consistency(input_type='minimal', verbose=False):
    # Load the data:
    ddict, hr = td.load_testdataset(input_type)

    # set up the single-particle quantities:
    nk = (32, 32, 1)
    sym = ddict['sym']
    k_grid = bz.KGrid(nk, sym)
    ek = hr.get_ek(k_grid)
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    t_util.test_array([siwk_obj.smom0, ], [twop.get_smom0(ddict['u'],ddict['n']), ],
                      f'{input_type}_smom0_consistency', rtol=1e-4, atol=1e-5)
    # giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'])
    giwk_obj = twop.GreensFunction(siwk_obj, ek, mu=ddict['mu_dmft'])
    t_util.test_array([ddict['mu_dmft'], ], [giwk_obj.mu, ], f'{input_type}_mu_dmft_consistency', rtol=1e-4, atol=1e-5)
    t_util.test_array([ddict['n'], ], [giwk_obj.n, ], f'{input_type}_n_dmft_consistency', rtol=1e-4, atol=1e-5)

    # set up G2:
    niw_core = mf.niw_from_mat(ddict['g4iw_dens'], axis=0)
    niv_core = mf.niw_from_mat(ddict['g4iw_dens'], axis=-1)

    wn_core = mf.wn(niw_core)
    vn_core = mf.vn(niv_core)
    g4iw_dens = lfp.get_g2_from_dmft_input(ddict, 'dens')
    g4iw_magn = lfp.get_g2_from_dmft_input(ddict, 'magn')

    # set up the bubble generator:
    bubble_gen = bub.BubbleGenerator(wn=wn_core, giwk_obj=giwk_obj, is_full_wn=True)
    gchi0_core = bubble_gen.get_gchi0(niv_core)
    chi0_core = bubble_gen.get_chi0(niv_core)

    # get gchi from g2:
    gchi_dens = lfp.gchir_from_g2(g4iw_dens, giwk_obj.g_loc)
    gchi_magn = lfp.gchir_from_g2(g4iw_magn, giwk_obj.g_loc)

    # test g2-gchi transform consistency:
    t_util.test_array(g4iw_magn.mat, lfp.g2_from_chir(gchi_magn, giwk_obj.g_loc).mat, f'{input_type}_g2_to_gchi_magn')
    t_util.test_array(g4iw_dens.mat, lfp.g2_from_chir(gchi_dens, giwk_obj.g_loc).mat, f'{input_type}_g2_to_gchi_dens')

    # get fob2 from gchi:
    fob2_dens = lfp.fob2_from_gchir(gchi_dens, gchi0_core)
    fob2_magn = lfp.fob2_from_gchir(gchi_magn, gchi0_core)

    # test gchi-fob2 transform consistency:
    t_util.test_array(gchi_magn.mat, lfp.gchir_from_fob2(fob2_magn, gchi0_core).mat, f'{input_type}_gchi_to_fob2_magn')
    t_util.test_array(gchi_dens.mat, lfp.gchir_from_fob2(fob2_dens, gchi0_core).mat, f'{input_type}_gchi_to_fob2_dens')

    # get Gamma from gchi:
    gamob2_dens = lfp.gamob2_from_gchir(gchi_dens, gchi0_core)
    gamob2_magn = lfp.gamob2_from_gchir(gchi_magn, gchi0_core)

    # test gchi-gamob2 transform consistency:
    t_util.test_array(gchi_magn.mat, lfp.gchir_from_gamob2(gamob2_magn, gchi0_core).mat, f'{input_type}_gchi_to_gamob2_magn')
    t_util.test_array(gchi_dens.mat, lfp.gchir_from_gamob2(gamob2_dens, gchi0_core).mat, f'{input_type}_gchi_to_gamob2_dens')

    # get fob2 from Gamma:
    fob2_dens_2 = lfp.fob2_from_gamob2_urange(gamob2_dens, gchi0_core)
    fob2_magn_2 = lfp.fob2_from_gamob2_urange(gamob2_magn, gchi0_core)

    t_util.test_array(fob2_dens.mat, fob2_dens_2.mat, f'{input_type}_fob2_from_gamob2_dens')
    t_util.test_array(fob2_magn.mat, fob2_magn_2.mat, f'{input_type}_fob2_from_gamob2_magn')

    # get gchi_aux:
    gchi_aux_dens = lfp.gchi_aux_core_from_gammar(gamob2_dens, gchi0_core)
    gchi_aux_magn = lfp.gchi_aux_core_from_gammar(gamob2_magn, gchi0_core)

    # get chi_aux:
    chi_aux_dens = gchi_aux_dens.contract_legs()
    chi_aux_magn = gchi_aux_magn.contract_legs()

    # get chi:
    chi_dens = lfp.chi_phys_urange(chi_aux_dens, chi0_core, chi0_core, ddict['u'], 'dens')
    chi_magn = lfp.chi_phys_urange(chi_aux_magn, chi0_core, chi0_core, ddict['u'], 'magn')

    chi_dens_raw = gchi_dens.contract_legs()
    chi_magn_raw = gchi_magn.contract_legs()

    if verbose:
        plt.figure()
        plt.plot(wn_core, chi_dens_raw.real, '-o', color='cornflowerblue')
        plt.plot(wn_core, chi_dens.real, '-o', color='firebrick')
        plt.ylabel(r'$\chi_d$')
        plt.show()

        plt.figure()
        plt.plot(wn_core, chi_magn_raw.real, '-o', color='cornflowerblue')
        plt.plot(wn_core, chi_magn.real, '-o', color='firebrick')
        plt.ylabel(r'$\chi_m$')
        plt.show()

    t_util.test_array(chi_dens, chi_dens_raw, f'{input_type}_chi_dens_consistency')
    t_util.test_array(chi_magn, chi_magn_raw, f'{input_type}_chi_magn_consistency')

    t_util.test_array(chi_dens, np.conj(np.flip(chi_dens)), f'{input_type}_chi_w_mw_consistency', rtol=1e-5, atol=1e-5)
    t_util.test_array(chi_magn, np.conj(np.flip(chi_magn)), f'{input_type}_chi_w_mw_consistency', rtol=1e-5, atol=1e-5)

    # get vrg:
    vrg_dens = lfp.vrg_from_gchi_aux_asympt(gchi_aux_dens, gchi0_core, chi_dens, chi_dens)
    vrg_magn = lfp.vrg_from_gchi_aux_asympt(gchi_aux_magn, gchi0_core, chi_magn, chi_magn)

    # Schwinger-Dyson equation:
    siw_sde = lfp.schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, giwk_obj.g_loc, ddict['n'])
    siw_sde_f_dens = lfp.schwinger_dyson_f(fob2_dens, gchi0_core, giwk_obj.g_loc)
    siw_sde_f_magn = lfp.schwinger_dyson_f(fob2_magn, gchi0_core, giwk_obj.g_loc)
    siw_sde_f = 0.5 * (siw_sde_f_dens - siw_sde_f_magn) + twop.get_smom0(ddict['u'], ddict['n'])

    if verbose:
        plt.figure()
        plt.plot(mf.vn(ddict['siw']), ddict['siw'].imag, '-o', color='cornflowerblue')
        plt.plot(vn_core, siw_sde.imag, '-o', color='firebrick')
        plt.plot(vn_core, siw_sde_f.imag, '-o', color='seagreen')
        plt.xlim(-5, vn_core[-1])
        plt.show()

        plt.figure()
        plt.plot(mf.vn(ddict['siw']), ddict['siw'].real, '-o', color='cornflowerblue')
        plt.plot(vn_core, siw_sde.real, '-o', color='firebrick')
        plt.plot(vn_core, siw_sde_f.real, '-o', color='seagreen')
        plt.xlim(-5, vn_core[-1])
        plt.show()

    t_util.test_array(siw_sde, np.conj(np.flip(siw_sde)), f'{input_type}_siw_sde_w_mw_consistency', rtol=1e-5, atol=1e-5)
    t_util.test_array(siw_sde, siw_sde_f, f'{input_type}_siw_sde_vrg_and_sde_f_consistency', rtol=1e-3, atol=1e-3)
    t_util.test_array(siw_sde, mf.cut_v(ddict['siw'], mf.niv_from_mat(siw_sde)), f'{input_type}_siw_sde_dmft_consistency',
                      rtol=0.3, atol=0.3)

    # f-bse:
    fob2_dens_3 = lfp.fob2_from_gamob2_urange(gamob2_dens, gchi0_core)
    fob2_magn_3 = lfp.fob2_from_gamob2_urange(gamob2_magn, gchi0_core)

    t_util.test_array(fob2_dens_2.mat, fob2_dens_3.mat, f'{input_type}_fob2_dens_from_gamob2_bse_consistency')
    t_util.test_array(fob2_magn_2.mat, fob2_magn_3.mat, f'{input_type}_fob2_magn_from_gamob2_bse_consistency')


def test_asymptotic_convergence(input_type='minimal', iwn=0):
    # Load the data:
    ddict, hr = td.load_testdataset()

    # set up the single-particle quantities:
    nk = (32, 32, 1)
    sym = ddict['sym']
    k_grid = bz.KGrid(nk, sym)
    ek = hr.get_ek(k_grid)
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'])

    # set up G2:
    niw_core = mf.niw_from_mat(ddict['g4iw_dens'], axis=0)
    niv_core = mf.niw_from_mat(ddict['g4iw_dens'], axis=-1)

    wn_core = mf.wn(niw_core)
    w_ind = np.argmin(np.abs(wn_core - iwn))
    wn_core = wn_core[w_ind:w_ind + 1]
    g4iw_dens = lfp.LocalFourPoint(channel='dens', mat=ddict['g4iw_dens'][w_ind:w_ind + 1], beta=ddict['beta'], u=ddict['u'],
                                   wn=wn_core)
    g4iw_magn = lfp.LocalFourPoint(channel='magn', mat=ddict['g4iw_magn'][w_ind:w_ind + 1], beta=ddict['beta'], u=ddict['u'],
                                   wn=wn_core)

    # set up the bubble generator:
    bubble_gen = bub.BubbleGenerator(wn=wn_core, giwk_obj=giwk_obj, is_full_wn=False)
    gchi0_core = bubble_gen.get_gchi0(niv_core)
    niv_urange_list = niv_core + np.array([0, 10, 30, 50, 80, 100, 200, 300, 500])
    gchi0_urange_list = [bubble_gen.get_gchi0(niv_urange) for niv_urange in niv_urange_list]

    # get gchi from g2:
    gchi_dens = lfp.gchir_from_g2(g4iw_dens, giwk_obj.g_loc)
    gchi_magn = lfp.gchir_from_g2(g4iw_magn, giwk_obj.g_loc)

    # get gamma-shell:
    gamma_dens_list = np.array([lfp.gamob2_from_gchir_urange(gchi_dens, gchi0_urange) for gchi0_urange in
                                gchi0_urange_list])
    gamma_magn_list = np.array([lfp.gamob2_from_gchir_urange(gchi_magn, gchi0_urange) for gchi0_urange in
                                gchi0_urange_list])

    # get chir_aux from gamma-shell:
    gchi_aux_dens_list = np.array([lfp.gchi_aux_core_from_gammar(gamma_dens, gchi0_core) for gamma_dens in
                                   gamma_dens_list])
    gchi_aux_magn_list = np.array([lfp.gchi_aux_core_from_gammar(gamma_magn, gchi0_core) for gamma_magn in
                                   gamma_magn_list])

    # get chir_aux from gchi_aux:
    chir_aux_dens_list = np.array([gchi_aux_dens.contract_legs() for gchi_aux_dens in gchi_aux_dens_list])
    chir_aux_magn_list = np.array([gchi_aux_magn.contract_legs() for gchi_aux_magn in gchi_aux_magn_list])

    # get chir-urange from chir_aux:
    chi0_core = bubble_gen.get_chi0(niv_core)
    chi0_urange_list = np.array([bubble_gen.get_chi0(niv_urange)[0] for niv_urange in niv_urange_list])
    chi_dens_urange_list = np.array([lfp.chi_phys_urange(chi_aux_dens, chi0_core, chi0_urange, ddict['u'], 'dens')[0]
                                     for chi_aux_dens, chi0_urange in zip(chir_aux_dens_list, chi0_urange_list)])
    chi_magn_urange_list = np.array([lfp.chi_phys_urange(chi_aux_magn, chi0_core, chi0_urange, ddict['u'], 'magn')[0]
                                     for chi_aux_magn, chi0_urange in zip(chir_aux_magn_list, chi0_urange_list)])

    # test consistence with chir-core:
    chi_dens_core = gchi_dens.contract_legs()
    chi_magn_core = gchi_magn.contract_legs()
    t_util.test_array(chi_dens_core, chi_dens_urange_list[0], f'{input_type}_chi_dens_core_urange_consistency_wn_{wn_core[0]}')
    t_util.test_array(chi_magn_core, chi_magn_urange_list[0], f'{input_type}_chi_magn_core_urange_consistency_wn_{wn_core[0]}')
    t_util.test_statement(t_util.is_monotonic(chi_dens_urange_list), f'{input_type}_chi_dens_urange_monotonic_wn_{wn_core[0]}')
    t_util.test_statement(t_util.is_monotonic(chi_magn_urange_list), f'{input_type}_chi_magn_urange_monotonic_wn_{wn_core[0]}')

    # add the asympt contribution:
    chi0_shell_list = np.array([bubble_gen.get_asymptotic_correction(niv_urange)[0] for niv_urange in niv_urange_list])
    chi0_tilde_list = np.array(chi0_urange_list) + np.array(chi0_shell_list)
    chi_dens_asympt_list = np.array([lfp.chi_phys_asympt(chi_dens_urange, chi0_urange, chi0_asympt) for
                                     chi_dens_urange, chi0_urange, chi0_asympt in zip(chi_dens_urange_list, chi0_urange_list,
                                                                                      chi0_tilde_list)])
    chi_magn_asympt_list = np.array([lfp.chi_phys_asympt(chi_magn_urange, chi0_urange, chi0_asympt) for
                                     chi_magn_urange, chi0_urange, chi0_asympt in zip(chi_magn_urange_list, chi0_urange_list,
                                                                                      chi0_tilde_list)])
    t_util.test_statement(t_util.is_monotonic(chi_dens_asympt_list), f'{input_type}_chi_dens_asympt_monotonic_wn_{wn_core[0]}')
    t_util.test_statement(t_util.is_monotonic(chi_magn_asympt_list), f'{input_type}_chi_magn_asympt_monotonic_wn_{wn_core[0]}')

    # test convergence:
    fit_fun = lambda x, a, b: a + b * x

    def get_asympt_extrapolation(data):
        xdata = 1 / niv_urange_list[-n_fit:][::-1]
        ydata = data[-n_fit:][::-1].real
        popt, _ = opt.curve_fit(f=fit_fun, xdata=xdata, ydata=ydata)
        return fit_fun(0, *popt)

    n_fit = 5

    t_util.test_array([get_asympt_extrapolation(chi0_urange_list), ],
                      [get_asympt_extrapolation(chi0_tilde_list), ], f'{input_type}_chi0_extrap_wn_{wn_core[0]}', atol=1e-4)
    t_util.test_array([get_asympt_extrapolation(chi_dens_urange_list), ],
                      [get_asympt_extrapolation(chi_dens_asympt_list), ],
                      f'{input_type}_chi_dens_extrap_wn_{wn_core[0]}', atol=1e-4)
    t_util.test_array([get_asympt_extrapolation(chi_magn_urange_list), ],
                      [get_asympt_extrapolation(chi_magn_asympt_list), ],
                      f'{input_type}_chi_magn_extrap_wn_{wn_core[0]}', atol=1e-4)

    # check convergence of vrg:
    vrg_dens_tilde_list = np.array([lfp.vrg_from_gchi_aux_asympt(gchi_dens_aux, gchi0_core, np.atleast_1d(chi_urange),
                                                                 np.atleast_1d(chi_asympt)).mat[0]
                                    for gchi_dens_aux, chi_urange, chi_asympt
                                    in zip(gchi_aux_dens_list, chi_dens_urange_list, chi_dens_asympt_list)])
    vrg_magn_tilde_list = np.array([lfp.vrg_from_gchi_aux_asympt(gchi_magn_aux, gchi0_core, np.atleast_1d(chi_urange),
                                                                 np.atleast_1d(chi_asympt)).mat[0]
                                    for gchi_magn_aux, chi_urange, chi_asympt
                                    in zip(gchi_aux_magn_list, chi_magn_urange_list, chi_magn_asympt_list)])

    t_util.test_statement(t_util.is_monotonic(vrg_dens_tilde_list[:, niv_core]),
                          f'{input_type}_vrg_dens_tilde_monotonic_wn_{wn_core[0]}_vn1')
    t_util.test_statement(t_util.is_monotonic(vrg_dens_tilde_list[:, niv_core - 1]),
                          f'{input_type}_vrg_dens_tilde_monotonic_wn_{wn_core[0]}_vn-1')

    t_util.test_statement(t_util.is_monotonic(vrg_magn_tilde_list[:, niv_core]),
                          f'{input_type}_vrg_magn_tilde_monotonic_wn_{wn_core[0]}_vn1')
    t_util.test_statement(t_util.is_monotonic(vrg_magn_tilde_list[:, niv_core - 1]),
                          f'{input_type}_vrg_magn_tilde_monotonic_wn_{wn_core[0]}_vn-1')

    # do not use the asymptotics:
    vrg_dens_list = np.array([lfp.vrg_from_gchi_aux_asympt(gchi_dens_aux, gchi0_core, np.atleast_1d(chi_urange),
                                                           np.atleast_1d(chi_urange)).mat[0]
                              for gchi_dens_aux, chi_urange, chi_urange
                              in zip(gchi_aux_dens_list, chi_dens_urange_list, chi_dens_urange_list)])
    vrg_magn_list = np.array([lfp.vrg_from_gchi_aux_asympt(gchi_magn_aux, gchi0_core, np.atleast_1d(chi_urange),
                                                           np.atleast_1d(chi_urange)).mat[0]
                              for gchi_magn_aux, chi_urange, chi_urange
                              in zip(gchi_aux_magn_list, chi_magn_urange_list, chi_magn_urange_list)])

    t_util.test_statement(t_util.is_monotonic(vrg_dens_list[:, niv_core]), f'{input_type}_vrg_dens_monotonic_wn_{wn_core[0]}_vn1')
    t_util.test_statement(t_util.is_monotonic(vrg_dens_list[:, niv_core - 1]),
                          f'{input_type}_vrg_dens_monotonic_wn_{wn_core[0]}_vn-1')

    t_util.test_statement(t_util.is_monotonic(vrg_magn_list[:, niv_core]), f'{input_type}_vrg_magn_monotonic_wn_{wn_core[0]}_vn1')
    t_util.test_statement(t_util.is_monotonic(vrg_magn_list[:, niv_core - 1]),
                          f'{input_type}_vrg_magn_monotonic_wn_{wn_core[0]}_vn-1')

    t_util.test_array([get_asympt_extrapolation(vrg_dens_list[:, niv_core]), ],
                      [get_asympt_extrapolation(vrg_dens_tilde_list[:, niv_core]), ],
                      f'{input_type}_vrg_dens_extrap_wn_{wn_core[0]}_vn1', atol=1e-3)
    t_util.test_array([get_asympt_extrapolation(vrg_dens_list[:, niv_core - 1]), ],
                      [get_asympt_extrapolation(vrg_dens_tilde_list[:, niv_core - 1]), ],
                      f'{input_type}_vrg_dens_extrap_wn_{wn_core[0]}_vn-1', atol=1e-3)
    t_util.test_array([get_asympt_extrapolation(vrg_dens_list[:, niv_core]), ],
                      [get_asympt_extrapolation(vrg_dens_tilde_list[:, niv_core]), ],
                      f'{input_type}_vrg_magn_extrap_wn_{wn_core[0]}_vn1', atol=1e-3)
    t_util.test_array([get_asympt_extrapolation(vrg_magn_list[:, niv_core - 1]), ],
                      [get_asympt_extrapolation(vrg_magn_tilde_list[:, niv_core - 1]), ],
                      f'{input_type}_vrg_magn_extrap_wn_{wn_core[0]}_vn-1', atol=1e-3)


def test_chi_sum_rule(input_type='minimal', verbose=False):
    # Load the data:
    ddict, hr = td.load_testdataset(input_type)

    # set up the single-particle quantities:
    nk = (32, 32, 1)
    sym = ddict['sym']
    k_grid = bz.KGrid(nk, sym)
    ek = hr.get_ek(k_grid)
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'])
    t_util.test_array([ddict['mu_dmft'], ], [giwk_obj.mu, ], f'{input_type}_mu_dmft_consistency', rtol=1e-4, atol=1e-4)

    # set up G2:
    g4iw_dens = lfp.get_g2_from_dmft_input(ddict, 'dens')
    g4iw_magn = lfp.get_g2_from_dmft_input(ddict, 'magn')
    # print(g4iw_magn.mat.shape)

    niw_core = 30
    niv_core = 30
    # niw_core = int(ddict['beta'] * 2 + 10)
    # niv_core = int(ddict['beta'] * 2 + 10)
    wn_core = mf.wn(niw_core)
    g4iw_dens.cut_iw(niw_core)
    g4iw_magn.cut_iw(niw_core)
    g4iw_dens.cut_iv(niv_core)
    g4iw_magn.cut_iv(niv_core)
    # print(g4iw_magn.mat.shape)

    # set up the bubble generator:
    bubble_gen = bub.BubbleGenerator(wn=wn_core, giwk_obj=giwk_obj, is_full_wn=False)
    gchi0_core = bubble_gen.get_gchi0(niv_core)
    niv_urange_list = niv_core + np.array([50, 80, 100, 200, 300])
    gchi0_urange_list = [bubble_gen.get_gchi0(niv_urange) for niv_urange in niv_urange_list]

    # get gchi from g2:
    gchi_dens = lfp.gchir_from_g2(g4iw_dens, giwk_obj.g_loc)
    gchi_magn = lfp.gchir_from_g2(g4iw_magn, giwk_obj.g_loc)

    # get gamma-shell:
    gamma_dens_list = [lfp.gamob2_from_gchir_urange(gchi_dens, gchi0_urange) for gchi0_urange in
                       gchi0_urange_list]
    gamma_magn_list = [lfp.gamob2_from_gchir_urange(gchi_magn, gchi0_urange) for gchi0_urange in
                       gchi0_urange_list]

    # get chir_aux from gamma-shell:
    gchi_aux_dens_list = np.array([lfp.gchi_aux_core_from_gammar(gamma_dens, gchi0_core) for gamma_dens in
                                   gamma_dens_list])
    gchi_aux_magn_list = np.array([lfp.gchi_aux_core_from_gammar(gamma_magn, gchi0_core) for gamma_magn in
                                   gamma_magn_list])

    # get chir_aux from gchi_aux:
    chir_aux_dens_list = np.array([gchi_aux_dens.contract_legs() for gchi_aux_dens in gchi_aux_dens_list])
    chir_aux_magn_list = np.array([gchi_aux_magn.contract_legs() for gchi_aux_magn in gchi_aux_magn_list])

    # get chir-urange from chir_aux:
    chi0_core = bubble_gen.get_chi0(niv_core)
    chi0_urange_list = np.array([bubble_gen.get_chi0(niv_urange) for niv_urange in niv_urange_list])

    chi_dens_urange_list = np.array([lfp.chi_phys_urange(chi_aux_dens, chi0_core, chi0_urange, ddict['u'], 'dens')
                                     for chi_aux_dens, chi0_urange in zip(chir_aux_dens_list, chi0_urange_list)])
    chi_magn_urange_list = np.array([lfp.chi_phys_urange(chi_aux_magn, chi0_core, chi0_urange, ddict['u'], 'magn')
                                     for chi_aux_magn, chi0_urange in zip(chir_aux_magn_list, chi0_urange_list)])

    # add the asympt contribution:
    chi0_shell_list = np.array([bubble_gen.get_asymptotic_correction(niv_urange) for niv_urange in niv_urange_list])
    chi0_tilde_list = np.array(chi0_urange_list) + np.array(chi0_shell_list)
    chi_dens_asympt_list = np.array([lfp.chi_phys_asympt(chi_dens_urange, chi0_urange, chi0_asympt) for
                                     chi_dens_urange, chi0_urange, chi0_asympt in zip(chi_dens_urange_list, chi0_urange_list,
                                                                                      chi0_tilde_list)])
    chi_magn_asympt_list = np.array([lfp.chi_phys_asympt(chi_magn_urange, chi0_urange, chi0_asympt) for
                                     chi_magn_urange, chi0_urange, chi0_asympt in zip(chi_magn_urange_list, chi0_urange_list,
                                                                                      chi0_tilde_list)])

    # sum rule:
    chi_dens_sum = 1 / ddict['beta'] * np.sum(chi_dens_urange_list, axis=-1)
    chi_magn_sum = 1 / ddict['beta'] * np.sum(chi_magn_urange_list, axis=-1)
    chi_upup_sum = 1 / 2 * (chi_dens_sum + chi_magn_sum)

    chi_dens_sum_tilde = 1 / ddict['beta'] * np.sum(chi_dens_asympt_list, axis=-1)
    chi_magn_sum_tilde = 1 / ddict['beta'] * np.sum(chi_magn_asympt_list, axis=-1)
    chi_upup_sum_tilde = 1 / 2 * (chi_dens_sum_tilde + chi_magn_sum_tilde)
    exact_sum_rule = twop.get_sum_chiupup(ddict['n'])

    if verbose:
        plt.figure()
        for chi_magn in chi_magn_asympt_list:
            plt.plot(wn_core, chi_magn.real, color='cornflowerblue')
        for chi_magn in chi_magn_urange_list:
            plt.plot(wn_core, chi_magn.real, color='seagreen')
        plt.plot(gchi_magn.wn, gchi_magn.contract_legs().real, color='firebrick')
        plt.show()

        plt.figure()
        for chi_dens in chi_dens_asympt_list:
            plt.plot(wn_core, chi_dens.real, color='cornflowerblue')
        for chi_dens in chi_dens_urange_list:
            plt.plot(wn_core, chi_dens.real, color='seagreen')
        plt.plot(gchi_magn.wn, gchi_dens.contract_legs().real, color='firebrick')
        plt.show()

        plt.figure()
        plt.semilogy(1 / np.array(niv_urange_list)[::-1], chi_upup_sum.real[::-1], color='cornflowerblue')
        # plt.plot(1/np.array(niv_urange_list)[::-1],chi_dens_sum.real[::-1], color='firebrick')
        # plt.plot(1/np.array(niv_urange_list)[::-1],chi_magn_sum.real[::-1], color='seagreen')
        plt.hlines(exact_sum_rule, 0.000001, 1 / np.array(niv_urange_list)[0], color='grey', linestyles='dashed')
        plt.xlim(0.000001, 1 / np.array(niv_urange_list)[0])
        plt.show()

    fit_fun = lambda x, a, b: a + b * x

    def get_asympt_extrapolation(data):
        xdata = 1 / niv_urange_list[-n_fit:][::-1]
        ydata = data[-n_fit:][::-1].real
        popt, _ = opt.curve_fit(f=fit_fun, xdata=xdata, ydata=ydata)
        return fit_fun(0, *popt)

    # test sum rule:

    n_fit = 3
    extrapolated_sum_rule = get_asympt_extrapolation(chi_upup_sum)
    # t_util.test_array(extrapolated_sum_rule, exact_sum_rule, 'chi_upup_sum_rule_exact')
    extrapolated_sum_rule_tilde = get_asympt_extrapolation(chi_upup_sum_tilde)
    # print(exact_sum_rule,extrapolated_sum_rule_tilde,extrapolated_sum_rule)
    t_util.test_array(extrapolated_sum_rule_tilde, extrapolated_sum_rule, f'{input_type}_chi_upup_sum_rule_tilde', atol=1e-3)
    t_util.test_array(extrapolated_sum_rule_tilde, exact_sum_rule, f'{input_type}_chi_upup_sum_rule_exact_tilde',
                      atol=1e-2, rtol=1e-1)


def test_schwinger_dyson(input_type='minimal', verbose=False):
    # Load the data:
    ddict, hr = td.load_testdataset(input_type)

    # set up the single-particle quantities:
    nk = (32, 32, 1)
    sym = ddict['sym']
    k_grid = bz.KGrid(nk, sym)
    ek = hr.get_ek(k_grid)
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'])

    t_util.test_array(ddict['siw'], np.conj(np.flip(ddict['siw'])), f'{input_type}_siw_dmft_w_mw_consistency', rtol=1e-10,
                      atol=1e-10)
    t_util.test_array(ddict['giw'], np.conj(np.flip(ddict['giw'])), f'{input_type}_giw_dmft_w_mw_consistency', rtol=1e-10,
                      atol=1e-10)

    t_util.test_array(giwk_obj.g_loc, np.flip(np.conj(giwk_obj.g_loc)), f'{input_type}_gloc_w_mw_consistency', rtol=1e-10,
                      atol=1e-10)
    t_util.test_array(np.squeeze(siwk_obj.get_siw(1000)), np.flip(np.conj(np.squeeze(siwk_obj.get_siw(1000)))),
                      f'{input_type}_sloc_w_mw_consistency', rtol=1e-10, atol=1e-10)

    # set up G2:
    g4iw_dens = lfp.get_g2_from_dmft_input(ddict, 'dens')
    g4iw_magn = lfp.get_g2_from_dmft_input(ddict, 'magn')
    # print(g4iw_magn.mat.shape)

    niw_core = 50
    niv_core = 50
    # niw_core = int(ddict['beta'] * 2 + 10)
    # niv_core = int(ddict['beta'] * 2 + 10)
    wn_core = mf.wn(niw_core)
    g4iw_dens.cut_iw(niw_core)
    g4iw_magn.cut_iw(niw_core)
    g4iw_dens.cut_iv(niv_core)
    g4iw_magn.cut_iv(niv_core)
    # print(g4iw_magn.mat.shape)

    # set up the bubble generator:
    bubble_gen = bub.BubbleGenerator(wn=wn_core, giwk_obj=giwk_obj, is_full_wn=False)
    chi0_core = bubble_gen.get_chi0(niv_core)
    gchi0_core = bubble_gen.get_gchi0(niv_core)
    niv_shell = 50
    niv_full = niv_core + niv_shell
    chi0_urange = bubble_gen.get_chi0(niv_full)
    gchi0_urange = bubble_gen.get_gchi0(niv_full)

    # get gchi from g2:
    gchi_dens = lfp.gchir_from_g2(g4iw_dens, giwk_obj.g_loc)
    gchi_magn = lfp.gchir_from_g2(g4iw_magn, giwk_obj.g_loc)

    # get gamma-shell:
    gamma_dens = lfp.gamob2_from_gchir_urange(gchi_dens, gchi0_urange)

    gamma_magn = lfp.gamob2_from_gchir_urange(gchi_magn, gchi0_urange)

    vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_dens, bubble_gen,
                                                                        niv_shell=niv_shell)
    vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, bubble_gen,
                                                                        niv_shell=niv_shell)
    siw_sde = lfp.schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, giwk_obj.g_loc, ddict['n'], niv_shell=niv_shell)

    t_util.test_array(siw_sde, mf.cut_v(ddict['siw'], mf.niv_from_mat(siw_sde)), f'{input_type}_siw_sde_shell_dmft_consistency',
                      rtol=0.3, atol=0.3)

    t_util.test_array(ddict['siw'], np.conj(np.flip(ddict['siw'])), f'{input_type}_siw_dmft_w_mw_consistency',
                      rtol=1e-5, atol=1e-5)
    t_util.test_array(mf.cut_v(siw_sde, niv_core), np.conj(np.flip(mf.cut_v(siw_sde, niv_core)))
                      , f'{input_type}_siw_sde_core_w_mw_consistency', rtol=1e-5, atol=1e-5)
    t_util.test_array(siw_sde, np.conj(np.flip(siw_sde)), f'{input_type}_siw_sde_w_mw_consistency', rtol=1e-5, atol=1e-5)

    # siw with u-range quantitites only:
    gchi_aux_dens = lfp.gchi_aux_core_from_gammar(gamma_dens, gchi0_core)
    gchi_aux_magn = lfp.gchi_aux_core_from_gammar(gamma_magn, gchi0_core)

    chi_aux_dens = gchi_aux_dens.contract_legs()
    chi_aux_magn = gchi_aux_magn.contract_legs()

    chi_dens_urange = lfp.chi_phys_urange(chi_aux_dens, chi0_core, chi0_urange, ddict['u'], 'dens')
    chi_magn_urange = lfp.chi_phys_urange(chi_aux_magn, chi0_core, chi0_urange, ddict['u'], 'magn')

    t_util.test_array(chi_dens_urange, np.conj(np.flip(chi_dens_urange)), f'{input_type}_chi_dens_urange_w_mw_consistency',
                      rtol=1e-5, atol=1e-5)
    t_util.test_array(chi_magn_urange, np.conj(np.flip(chi_magn_urange)), f'{input_type}_chi_magn_urange_w_mw_consistency',
                      rtol=1e-5, atol=1e-5)

    vrg_dens_urange = lfp.vrg_from_gchi_aux_asympt(gchi_aux_dens, gchi0_core, chi_dens_urange, chi_dens_urange)
    vrg_magn_urange = lfp.vrg_from_gchi_aux_asympt(gchi_aux_magn, gchi0_core, chi_magn_urange, chi_magn_urange)

    siw_sde_urange = lfp.schwinger_dyson_full(vrg_dens_urange, vrg_magn_urange, chi_dens_urange, chi_magn_urange, giwk_obj.g_loc,
                                              ddict['n'], niv_shell=niv_shell)

    t_util.test_array(mf.cut_v(siw_sde_urange, niv_core), mf.cut_v(np.conj(np.flip(siw_sde_urange)), niv_core),
                      f'{input_type}_siw_sde_core_urange_w_mw_consistency', rtol=1e-5, atol=1e-5)
    t_util.test_array(siw_sde_urange, np.conj(np.flip(siw_sde_urange)), f'{input_type}_siw_sde_urange_w_mw_consistency',
                      rtol=1e-5, atol=1e-5)
    t_util.test_array(mf.cut_v(siw_sde, niv_core), mf.cut_v(siw_sde_urange, niv_core),
                      f'{input_type}_siw_sde_shell_urange_consistency', rtol=1e-5, atol=1e-5)

    # get f-urange:
    fob2_dens = lfp.fob2_from_gamob2_urange(gamma_dens, gchi0_urange)
    fob2_magn = lfp.fob2_from_gamob2_urange(gamma_magn, gchi0_urange)

    # get f core from gchi:
    fob2_dens_core = lfp.fob2_from_gchir(gchi_dens, gchi0_core)
    fob2_magn_core = lfp.fob2_from_gchir(gchi_magn, gchi0_core)

    t_util.test_array(mf.cut_v(fob2_dens.mat, niv_core, axes=(-2, -1)), fob2_dens_core.mat,
                      f'{input_type}_fob2_dens_core_consistency')
    t_util.test_array(mf.cut_v(fob2_magn.mat, niv_core, axes=(-2, -1)), fob2_magn_core.mat,
                      f'{input_type}_fob2_magn_core_consistency')

    siw_sde_f_dens = lfp.schwinger_dyson_f(fob2_dens, gchi0_urange, giwk_obj.g_loc)
    siw_sde_f_magn = lfp.schwinger_dyson_f(fob2_magn, gchi0_urange, giwk_obj.g_loc)
    siw_sde_f = 0.5 * (siw_sde_f_dens - siw_sde_f_magn) + twop.get_smom0(ddict['u'], ddict['n'])

    if verbose:
        plt.figure()
        plt.plot(mf.vn(siw_sde), siw_sde.imag, '-o', color='cornflowerblue')
        plt.plot(mf.vn(siw_sde), siw_sde_urange.imag, '-o', color='navy', ms=6)
        plt.plot(mf.vn(siw_sde_f), siw_sde_f.imag, '-o', color='firebrick')
        plt.plot(mf.vn(ddict['siw']), ddict['siw'].imag, '-o', color='seagreen')
        plt.xlim(-niv_full, niv_full)
        plt.show()

        plt.figure()
        plt.plot(mf.vn(siw_sde), siw_sde.real, '-o', color='cornflowerblue')
        plt.plot(mf.vn(siw_sde), siw_sde_urange.real, '-o', color='navy', ms=6)
        plt.plot(mf.vn(siw_sde_f), siw_sde_f.real, '-o', color='firebrick')
        plt.plot(mf.vn(ddict['siw']), ddict['siw'].real, '-o', color='seagreen')
        plt.xlim(-niv_full, niv_full)
        plt.show()

    t_util.test_array(mf.cut_v(siw_sde, niv_core), mf.cut_v(siw_sde_f, niv_core),
                      f'{input_type}_siw_sde_vrg_and_sde_f_consistency', rtol=1e-2, atol=1e-2)

    t_util.test_array(siw_sde_f, np.conj(np.flip(siw_sde_f)), f'{input_type}_siw_sde_f_w_mw_consistency', rtol=1e-5, atol=1e-5)
    t_util.test_array(siw_sde_urange, siw_sde_f, f'{input_type}_siw_sde_vrg_urange_and_sde_f_consistency', rtol=1e-5, atol=1e-5)


def test_f_consistency(input_type='minimal', verbose=False):
    # Load the data:
    ddict, hr = td.load_testdataset(input_type)

    # set up the single-particle quantities:
    nk = (32, 32, 1)
    sym = ddict['sym']
    k_grid = bz.KGrid(nk, sym)
    ek = hr.get_ek(k_grid)
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'])

    # set up G2:
    g4iw_dens = lfp.get_g2_from_dmft_input(ddict, 'dens')
    g4iw_magn = lfp.get_g2_from_dmft_input(ddict, 'magn')

    niw_core = 30
    niv_core = 30
    vn_core = mf.vn(niv_core)
    wn_core = mf.wn(niw_core)

    g4iw_dens.cut_iw(niw_core)
    g4iw_magn.cut_iw(niw_core)
    g4iw_dens.cut_iv(niv_core)
    g4iw_magn.cut_iv(niv_core)

    # set up the bubble generator:
    bubble_gen = bub.BubbleGenerator(wn=wn_core, giwk_obj=giwk_obj, is_full_wn=True)
    gchi0_core = bubble_gen.get_gchi0(niv_core)
    niv_shell = 0
    niv_full = niv_core + niv_shell
    gchi0_urange = bubble_gen.get_gchi0(niv_full)

    chi0_core = bubble_gen.get_chi0(niv_core)
    chi0_urange = bubble_gen.get_chi0(niv_full)

    # get gchi from g2:
    gchi_dens = lfp.gchir_from_g2(g4iw_dens, giwk_obj.g_loc)
    gchi_magn = lfp.gchir_from_g2(g4iw_magn, giwk_obj.g_loc)

    # get gamma-shell:
    gamma_dens = lfp.gamob2_from_gchir_urange(gchi_dens, gchi0_urange)
    gamma_magn = lfp.gamob2_from_gchir_urange(gchi_magn, gchi0_urange)

    # get the exact f core by amputating legs:
    fob2_core_exact_dens = lfp.fob2_from_gchir(gchi_dens, gchi0_core)
    fob2_core_exact_magn = lfp.fob2_from_gchir(gchi_magn, gchi0_core)

    # get f core from the bse with gamma:
    fob2_urange_bse_dens = lfp.fob2_from_gamob2_urange(gamma_dens, gchi0_urange)
    fob2_urange_bse_magn = lfp.fob2_from_gamob2_urange(gamma_magn, gchi0_urange)
    fob2_urange_bse_dens.cut_iv(niv_core)
    fob2_urange_bse_magn.cut_iv(niv_core)

    t_util.test_array(fob2_core_exact_dens.mat, fob2_urange_bse_dens.mat, f'{input_type}_fob2_dens_from_gamob2_bse_consistency')
    t_util.test_array(fob2_core_exact_magn.mat, fob2_urange_bse_magn.mat, f'{input_type}_fob2_magn_from_gamob2_bse_consistency')

    gchi_dens_from_fob2 = lfp.gchir_from_fob2(fob2_urange_bse_dens, gchi0_core)
    gchi_magn_from_fob2 = lfp.gchir_from_fob2(fob2_urange_bse_magn, gchi0_core)

    t_util.test_array(gchi_dens.mat, gchi_dens_from_fob2.mat, f'{input_type}_gchi_dens_from_fob2_consistency')
    t_util.test_array(gchi_magn.mat, gchi_magn_from_fob2.mat, f'{input_type}_gchi_magn_from_fob2_consistency')

    vrg_dens_tilde, chi_dens_tilde = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_dens, bubble_gen
                                                                                    , niv_shell=niv_shell, sum_axis=-1)
    vrg_magn_tilde, chi_magn_tilde = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, bubble_gen
                                                                                    , niv_shell=niv_shell, sum_axis=-1)

    # gchi_aux from gamma:
    gchi_aux_dens = lfp.gchi_aux_core_from_gammar(gamma_dens, gchi0_core)
    gchi_aux_magn = lfp.gchi_aux_core_from_gammar(gamma_magn, gchi0_core)

    # gchi_aux from gamma-urange:
    gchi_aux_dens_urange = lfp.gchi_aux_core_from_gammar_urange(gamma_dens, gchi0_urange)
    gchi_aux_magn_urange = lfp.gchi_aux_core_from_gammar_urange(gamma_magn, gchi0_urange)

    gchi_aux_dens_urange.cut_iv(niv_core)
    gchi_aux_magn_urange.cut_iv(niv_core)

    t_util.test_array(gchi_aux_dens_urange.mat, gchi_aux_dens.mat, f'{input_type}_gchi_aux_dens_core_urange_consistency')
    t_util.test_array(gchi_aux_magn_urange.mat, gchi_aux_magn.mat, f'{input_type}_gchi_aux_magn_core_urange_consistency')

    chi_dens_urange = lfp.chi_phys_urange(gchi_aux_dens.contract_legs(), chi0_core, chi0_urange, ddict['u'], 'dens')
    chi_magn_urange = lfp.chi_phys_urange(gchi_aux_magn.contract_legs(), chi0_core, chi0_urange, ddict['u'], 'magn')

    u_d = lfp.get_ur(ddict['u'], 'dens')
    u_m = lfp.get_ur(ddict['u'], 'magn')

    u_mat = np.ones_like(gchi_aux_dens.mat[0]) * u_d / ddict['beta'] ** 2
    gchi_aux_u_gchi_aux = np.array([gchi_aux @ u_mat @ gchi_aux for gchi_aux in gchi_aux_dens.mat])

    aux_fac_1 = (1 - u_d * chi_dens_urange)
    aux_fac_2 = (1 - u_d * chi_dens_urange) ** 2 / (1 - u_d * chi_dens_tilde)

    aux_asympt = gchi_aux_u_gchi_aux * (-aux_fac_1 + aux_fac_2)[:, None, None]
    gchi_aux_dens_tilde = lfp.construct_lfp_from_lnp(gchi_aux_dens, gchi_aux_dens.mat + aux_asympt)

    u_mat = np.ones_like(gchi_aux_dens.mat[0]) * u_m / ddict['beta'] ** 2
    gchi_aux_u_gchi_aux = np.array([gchi_aux @ u_mat @ gchi_aux for gchi_aux in gchi_aux_magn.mat])

    aux_fac_1 = (1 - u_m * chi_magn_urange)
    aux_fac_2 = (1 - u_m * chi_magn_urange) ** 2 / (1 - u_m * chi_magn_tilde)

    aux_asympt = gchi_aux_u_gchi_aux * (-aux_fac_1 + aux_fac_2)[:, None, None]
    gchi_aux_magn_tilde = lfp.construct_lfp_from_lnp(gchi_aux_magn, gchi_aux_magn.mat + aux_asympt)

    vrg_dens_urange = lfp.vrg_from_gchi_aux_asympt(gchi_aux_dens, gchi0_core, chi_dens_urange, chi_dens_urange, sum_axis=-1)
    vrg_dens_urange_vp = lfp.vrg_from_gchi_aux_asympt(gchi_aux_dens, gchi0_core, chi_dens_urange, chi_dens_urange, sum_axis=-2)

    vrg_magn_urange = lfp.vrg_from_gchi_aux_asympt(gchi_aux_magn, gchi0_core, chi_magn_urange, chi_magn_urange, sum_axis=-1)
    vrg_magn_urange_vp = lfp.vrg_from_gchi_aux_asympt(gchi_aux_magn, gchi0_core, chi_magn_urange, chi_magn_urange, sum_axis=-2)

    vrg_dens_tilde_vp, _ = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_dens, bubble_gen
                                                                          , niv_shell=niv_shell, sum_axis=-2)
    fob2_core_vrg_dens = lfp.fob2_from_vrg_and_chir(gchi_aux_dens_tilde, vrg_dens_tilde, vrg_dens_tilde_vp, chi_dens_tilde,
                                                    gchi0_core)

    vrg_magn_tilde_vp, _ = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, bubble_gen
                                                                          , niv_shell=niv_shell, sum_axis=-2)
    fob2_core_vrg_magn = lfp.fob2_from_vrg_and_chir(gchi_aux_magn_tilde, vrg_magn_tilde, vrg_magn_tilde_vp, chi_magn_tilde,
                                                    gchi0_core)

    t_util.test_array(fob2_core_vrg_dens.mat, fob2_core_exact_dens.mat, f'{input_type}_fob2_dens_from_vrg_consistency')
    t_util.test_array(fob2_core_vrg_magn.mat, fob2_core_exact_magn.mat, f'{input_type}_fob2_magn_from_vrg_consistency')

    fob2_core_vrg_dens_urange = lfp.fob2_from_vrg_and_chir(gchi_aux_dens, vrg_dens_urange, vrg_dens_urange_vp, chi_dens_urange,
                                                           gchi0_core)
    fob2_core_vrg_magn_urange = lfp.fob2_from_vrg_and_chir(gchi_aux_magn, vrg_magn_urange, vrg_magn_urange_vp, chi_magn_urange,
                                                           gchi0_core)

    t_util.test_array(fob2_core_vrg_dens_urange.mat, fob2_core_exact_dens.mat,
                      f'{input_type}_fob2_dens_from_vrg_urange_consistency')
    t_util.test_array(fob2_core_vrg_magn_urange.mat, fob2_core_exact_magn.mat,
                      f'{input_type}_fob2_magn_from_vrg_urange_consistency')



def test_f_consistency_ed(verbose=False):
    # Load the data:
    ddict, hr = td.load_minimal_dataset_ed()

    # set up the single-particle quantities:
    nk = (32, 32, 1)
    sym = bz.two_dimensional_square_symmetries()
    k_grid = bz.KGrid(nk, sym)
    ek = hr.get_ek(k_grid)
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'])
    # giwk_obj = twop.GreensFunction(siwk_obj, ek, mu = ddict['mu_dmft'])

    # set up G2:
    g4iw_dens = lfp.get_g2_from_dmft_input(ddict, 'dens')
    g4iw_magn = lfp.get_g2_from_dmft_input(ddict, 'magn')
    # print(g4iw_magn.mat.shape)

    niw_core = 30
    niv_core = 30
    vn_core = mf.vn(niv_core)
    wn_core = mf.wn(niw_core)

    g4iw_dens.cut_iw(niw_core)
    g4iw_magn.cut_iw(niw_core)
    g4iw_dens.cut_iv(niv_core)
    g4iw_magn.cut_iv(niv_core)

    # set up the bubble generator:
    bubble_gen = bub.BubbleGenerator(wn=wn_core, giwk_obj=giwk_obj, is_full_wn=True)
    gchi0_core = bubble_gen.get_gchi0(niv_core)
    niv_shell = 0
    niv_full = niv_core + niv_shell
    gchi0_urange = bubble_gen.get_gchi0(niv_full)

    chi0_core = bubble_gen.get_chi0(niv_core)
    chi0_urange = bubble_gen.get_chi0(niv_full)

    # get gchi from g2:
    gchi_dens = lfp.gchir_from_g2(g4iw_dens, giwk_obj.g_loc)
    gchi_magn = lfp.gchir_from_g2(g4iw_magn, giwk_obj.g_loc)

    # get gamma-shell:
    gamma_dens = lfp.gamob2_from_gchir_urange(gchi_dens, gchi0_urange)
    gamma_magn = lfp.gamob2_from_gchir_urange(gchi_magn, gchi0_urange)

    # get the exact f core by amputating legs:
    fob2_core_exact_dens = lfp.fob2_from_gchir(gchi_dens, gchi0_core)
    fob2_core_exact_magn = lfp.fob2_from_gchir(gchi_magn, gchi0_core)

    # get f core from the bse with gamma:
    fob2_urange_bse_dens = lfp.fob2_from_gamob2_urange(gamma_dens, gchi0_urange)
    fob2_urange_bse_magn = lfp.fob2_from_gamob2_urange(gamma_magn, gchi0_urange)
    fob2_urange_bse_dens.cut_iv(niv_core)
    fob2_urange_bse_magn.cut_iv(niv_core)

    t_util.test_array(fob2_core_exact_dens.mat, fob2_urange_bse_dens.mat, 'ed_fob2_dens_from_gamob2_bse_consistency')
    t_util.test_array(fob2_core_exact_magn.mat, fob2_urange_bse_magn.mat, 'ed_fob2_magn_from_gamob2_bse_consistency')

    gchi_dens_from_fob2 = lfp.gchir_from_fob2(fob2_urange_bse_dens, gchi0_core)
    gchi_magn_from_fob2 = lfp.gchir_from_fob2(fob2_urange_bse_magn, gchi0_core)

    t_util.test_array(gchi_dens.mat, gchi_dens_from_fob2.mat, 'ed_gchi_dens_from_fob2_consistency')
    t_util.test_array(gchi_magn.mat, gchi_magn_from_fob2.mat, 'ed_gchi_magn_from_fob2_consistency')

    # gchi_aux from gamma:
    gchi_aux_dens = lfp.gchi_aux_core_from_gammar(gamma_dens, gchi0_core)
    gchi_aux_magn = lfp.gchi_aux_core_from_gammar(gamma_magn, gchi0_core)

    # gchi_aux from gamma-urange:
    gchi_aux_dens_urange = lfp.gchi_aux_core_from_gammar_urange(gamma_dens, gchi0_urange)
    gchi_aux_magn_urange = lfp.gchi_aux_core_from_gammar_urange(gamma_magn, gchi0_urange)

    gchi_aux_dens_urange.cut_iv(niv_core)
    gchi_aux_magn_urange.cut_iv(niv_core)

    t_util.test_array(gchi_aux_dens_urange.mat, gchi_aux_dens.mat, 'ed_gchi_aux_dens_core_urange_consistency')
    t_util.test_array(gchi_aux_magn_urange.mat, gchi_aux_magn.mat, 'ed_gchi_aux_magn_core_urange_consistency')

    chi_dens_urange = lfp.chi_phys_urange(gchi_aux_dens.contract_legs(), chi0_core, chi0_urange, ddict['u'], 'dens')
    chi_magn_urange = lfp.chi_phys_urange(gchi_aux_magn.contract_legs(), chi0_core, chi0_urange, ddict['u'], 'magn')

    vrg_dens_urange = lfp.vrg_from_gchi_aux_asympt(gchi_aux_dens, gchi0_core, chi_dens_urange, chi_dens_urange, sum_axis=-1)
    vrg_dens_urange_vp = lfp.vrg_from_gchi_aux_asympt(gchi_aux_dens, gchi0_core, chi_dens_urange, chi_dens_urange, sum_axis=-2)

    vrg_magn_urange = lfp.vrg_from_gchi_aux_asympt(gchi_aux_magn, gchi0_core, chi_magn_urange, chi_magn_urange, sum_axis=-1)
    vrg_magn_urange_vp = lfp.vrg_from_gchi_aux_asympt(gchi_aux_magn, gchi0_core, chi_magn_urange, chi_magn_urange, sum_axis=-2)

    fob2_core_vrg_dens_urange = lfp.fob2_from_vrg_and_chir(gchi_aux_dens, vrg_dens_urange, vrg_dens_urange, chi_dens_urange,
                                                           gchi0_core)
    fob2_core_vrg_magn_urange = lfp.fob2_from_vrg_and_chir(gchi_aux_magn, vrg_magn_urange, vrg_magn_urange, chi_magn_urange,
                                                           gchi0_core)

    if verbose:
        plt.figure()
        plt.pcolormesh(vn_core, wn_core, vrg_dens_urange.mat.real, cmap='RdBu')
        plt.xlabel('vn')
        plt.ylabel('wn')
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.pcolormesh(vn_core, wn_core, vrg_dens_urange_vp.mat.real, cmap='RdBu')
        plt.xlabel('vn')
        plt.ylabel('wn')
        plt.colorbar()
        plt.show()

    t_util.test_array(fob2_core_vrg_dens_urange.mat, fob2_core_exact_dens.mat, 'ed_fob2_dens_from_vrg_urange_consistency')
    t_util.test_array(fob2_core_vrg_magn_urange.mat, fob2_core_exact_magn.mat, 'ed_fob2_magn_from_vrg_urange_consistency')


def test_schwinger_dyson_ed(verbose=False):
    # Load the data:
    ddict, hr = td.load_minimal_dataset_ed()

    # set up the single-particle quantities:
    nk = (32, 32, 1)
    sym = bz.two_dimensional_square_symmetries()
    k_grid = bz.KGrid(nk, sym)
    ek = hr.get_ek(k_grid)
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'])

    t_util.test_array(ddict['siw'], np.conj(np.flip(ddict['siw'])), 'ed_siw_dmft_w_mw_consistency', rtol=1e-10, atol=1e-10)
    t_util.test_array(ddict['giw'], np.conj(np.flip(ddict['giw'])), 'ed_giw_dmft_w_mw_consistency', rtol=1e-10, atol=1e-10)

    t_util.test_array(giwk_obj.g_loc, np.flip(np.conj(giwk_obj.g_loc)), 'ed_gloc_w_mw_consistency', rtol=1e-10, atol=1e-10)
    t_util.test_array(np.squeeze(siwk_obj.get_siw(1000)), np.flip(np.conj(np.squeeze(siwk_obj.get_siw(1000)))),
                      'ed_sloc_w_mw_consistency', rtol=1e-10, atol=1e-10)

    # set up G2:
    g4iw_dens = lfp.get_g2_from_dmft_input(ddict, 'dens')
    g4iw_magn = lfp.get_g2_from_dmft_input(ddict, 'magn')

    niw_core = 50
    niv_core = 50
    wn_core = mf.wn(niw_core)
    g4iw_dens.cut_iw(niw_core)
    g4iw_magn.cut_iw(niw_core)
    g4iw_dens.cut_iv(niv_core)
    g4iw_magn.cut_iv(niv_core)

    # set up the bubble generator:
    bubble_gen = bub.BubbleGenerator(wn=wn_core, giwk_obj=giwk_obj, is_full_wn=False)
    chi0_core = bubble_gen.get_chi0(niv_core)
    gchi0_core = bubble_gen.get_gchi0(niv_core)
    niv_shell = 50
    niv_full = niv_core + niv_shell
    chi0_urange = bubble_gen.get_chi0(niv_full)
    gchi0_urange = bubble_gen.get_gchi0(niv_full)

    # get gchi from g2:
    gchi_dens = lfp.gchir_from_g2(g4iw_dens, giwk_obj.g_loc)
    gchi_magn = lfp.gchir_from_g2(g4iw_magn, giwk_obj.g_loc)

    # get gamma-shell:
    gamma_dens = lfp.gamob2_from_gchir_urange(gchi_dens, gchi0_urange)

    gamma_magn = lfp.gamob2_from_gchir_urange(gchi_magn, gchi0_urange)

    vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_dens, bubble_gen,
                                                                        niv_shell=niv_shell)
    vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, bubble_gen,
                                                                        niv_shell=niv_shell)
    siw_sde = lfp.schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, giwk_obj.g_loc, ddict['n'], niv_shell=niv_shell)

    t_util.test_array(siw_sde, mf.cut_v(ddict['siw'], mf.niv_from_mat(siw_sde)), 'ed_siw_sde_shell_dmft_consistency', rtol=1e-1,
                      atol=1e-1)

    t_util.test_array(ddict['siw'], np.conj(np.flip(ddict['siw'])), 'ed_siw_dmft_w_mw_consistency', rtol=1e-5, atol=1e-5)
    t_util.test_array(mf.cut_v(siw_sde, niv_core), np.conj(np.flip(mf.cut_v(siw_sde, niv_core)))
                      , 'ed_siw_sde_core_w_mw_consistency', rtol=1e-10, atol=1e-10)
    t_util.test_array(siw_sde, np.conj(np.flip(siw_sde)), 'ed_siw_sde_w_mw_consistency', rtol=1e-10, atol=1e-10)

    # siw with u-range quantitites only:
    gchi_aux_dens = lfp.gchi_aux_core_from_gammar(gamma_dens, gchi0_core)
    gchi_aux_magn = lfp.gchi_aux_core_from_gammar(gamma_magn, gchi0_core)

    chi_aux_dens = gchi_aux_dens.contract_legs()
    chi_aux_magn = gchi_aux_magn.contract_legs()

    chi_dens_urange = lfp.chi_phys_urange(chi_aux_dens, chi0_core, chi0_urange, ddict['u'], 'dens')
    chi_magn_urange = lfp.chi_phys_urange(chi_aux_magn, chi0_core, chi0_urange, ddict['u'], 'magn')

    t_util.test_array(chi_dens_urange, np.conj(np.flip(chi_dens_urange)), 'ed_chi_dens_urange_w_mw_consistency',
                      rtol=1e-10, atol=1e-10)
    t_util.test_array(chi_magn_urange, np.conj(np.flip(chi_magn_urange)), 'ed_chi_magn_urange_w_mw_consistency',
                      rtol=1e-10, atol=1e-10)

    vrg_dens_urange = lfp.vrg_from_gchi_aux_asympt(gchi_aux_dens, gchi0_core, chi_dens_urange, chi_dens_urange)
    vrg_magn_urange = lfp.vrg_from_gchi_aux_asympt(gchi_aux_magn, gchi0_core, chi_magn_urange, chi_magn_urange)

    vrg_dens_urange_vp = lfp.vrg_from_gchi_aux_asympt(gchi_aux_dens, gchi0_core, chi_dens_urange, chi_dens_urange, sum_axis=-2)
    vrg_magn_urange_vp = lfp.vrg_from_gchi_aux_asympt(gchi_aux_magn, gchi0_core, chi_magn_urange, chi_magn_urange, sum_axis=-2)

    vrg_dens_urange.mat = 0.5 * (vrg_dens_urange.mat + vrg_dens_urange_vp.mat)
    vrg_magn_urange.mat = 0.5 * (vrg_magn_urange.mat + vrg_magn_urange_vp.mat)
    siw_sde_urange = lfp.schwinger_dyson_full(vrg_dens_urange, vrg_magn_urange, chi_dens_urange, chi_magn_urange, giwk_obj.g_loc,
                                              ddict['n'], niv_shell=niv_shell)

    t_util.test_array(mf.cut_v(siw_sde_urange, niv_core), mf.cut_v(np.conj(np.flip(siw_sde_urange)), niv_core),
                      'ed_siw_sde_core_urange_w_mw_consistency', rtol=1e-5, atol=1e-5)
    t_util.test_array(siw_sde_urange, np.conj(np.flip(siw_sde_urange)), 'ed_siw_sde_urange_w_mw_consistency', rtol=1e-5,
                      atol=1e-5)
    t_util.test_array(mf.cut_v(siw_sde, niv_core), mf.cut_v(siw_sde_urange, niv_core), 'ed_siw_sde_shell_urange_consistency',
                      rtol=1e-5, atol=1e-5)

    # get f-urange:
    fob2_dens = lfp.fob2_from_gamob2_urange(gamma_dens, gchi0_urange)
    fob2_magn = lfp.fob2_from_gamob2_urange(gamma_magn, gchi0_urange)

    # get f core from gchi:
    fob2_dens_core = lfp.fob2_from_gchir(gchi_dens, gchi0_core)
    fob2_magn_core = lfp.fob2_from_gchir(gchi_magn, gchi0_core)

    t_util.test_array(mf.cut_v(fob2_dens.mat, niv_core, axes=(-2, -1)), fob2_dens_core.mat, 'ed_fob2_dens_core_consistency')
    t_util.test_array(mf.cut_v(fob2_magn.mat, niv_core, axes=(-2, -1)), fob2_magn_core.mat, 'ed_fob2_magn_core_consistency')

    siw_sde_f_dens = lfp.schwinger_dyson_f(fob2_dens, gchi0_urange, giwk_obj.g_loc)
    siw_sde_f_magn = lfp.schwinger_dyson_f(fob2_magn, gchi0_urange, giwk_obj.g_loc)
    siw_sde_f = 0.5 * (siw_sde_f_dens - siw_sde_f_magn) + twop.get_smom0(ddict['u'], ddict['n'])

    if verbose:
        plt.figure()
        plt.plot(mf.vn(siw_sde), siw_sde.imag, '-o', color='cornflowerblue')
        plt.plot(mf.vn(siw_sde), siw_sde_urange.imag, '-o', color='navy', ms=6)
        plt.plot(mf.vn(siw_sde_f), siw_sde_f.imag, '-o', color='firebrick')
        plt.plot(mf.vn(ddict['siw']), ddict['siw'].imag, '-o', color='seagreen')
        plt.xlim(-niv_full, niv_full)
        plt.show()

        plt.figure()
        plt.plot(mf.vn(siw_sde), siw_sde.real, '-o', color='cornflowerblue')
        plt.plot(mf.vn(siw_sde), siw_sde_urange.real, '-o', color='navy', ms=6)
        plt.plot(mf.vn(siw_sde_f), siw_sde_f.real, '-o', color='firebrick')
        plt.plot(mf.vn(ddict['siw']), ddict['siw'].real, '-o', color='seagreen')
        plt.xlim(-niv_full, niv_full)
        plt.show()

    t_util.test_array(mf.cut_v(siw_sde, niv_core), mf.cut_v(siw_sde_f, niv_core), 'ed_siw_sde_vrg_and_sde_f_consistency',
                      rtol=1e-2,
                      atol=1e-2)

    t_util.test_array(siw_sde_f, np.conj(np.flip(siw_sde_f)), 'ed_siw_sde_f_w_mw_consistency', rtol=1e-5, atol=1e-5)
    t_util.test_array(siw_sde_urange, siw_sde_f, 'ed_siw_sde_vrg_urange_and_sde_f_consistency', rtol=1e-5, atol=1e-5)


def test_crossing_symmetry(input_type='ed_minimal'):
    # Load the data:
    ddict, hr = td.load_testdataset(input_type)

    # set up the single-particle quantities:
    nk = (32, 32, 1)
    sym = ddict['sym']
    k_grid = bz.KGrid(nk, sym)
    ek = hr.get_ek(k_grid)
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'])

    t_util.test_array(ddict['siw'], np.conj(np.flip(ddict['siw'])), 'ed_siw_dmft_w_mw_consistency', rtol=1e-10, atol=1e-10)
    t_util.test_array(ddict['giw'], np.conj(np.flip(ddict['giw'])), 'ed_giw_dmft_w_mw_consistency', rtol=1e-10, atol=1e-10)

    t_util.test_array(giwk_obj.g_loc, np.flip(np.conj(giwk_obj.g_loc)), 'ed_gloc_w_mw_consistency', rtol=1e-10, atol=1e-10)
    t_util.test_array(np.squeeze(siwk_obj.get_siw(1000)), np.flip(np.conj(np.squeeze(siwk_obj.get_siw(1000)))),
                      'ed_sloc_w_mw_consistency', rtol=1e-10, atol=1e-10)

    # set up G2:
    g4iw_dens = lfp.get_g2_from_dmft_input(ddict, 'dens')
    g4iw_dens_sym = copy.deepcopy(g4iw_dens)
    g4iw_dens_sym.symmetrize_v_vp()

    g4iw_magn = lfp.get_g2_from_dmft_input(ddict, 'magn')
    g4iw_magn_sym = copy.deepcopy(g4iw_magn)
    g4iw_magn_sym.symmetrize_v_vp()

    t_util.test_array(g4iw_dens.mat, g4iw_dens_sym.mat, 'ed_g4iw_dens_v_vp_symmetry', rtol=1e-10, atol=1e-10)
    t_util.test_array(g4iw_magn.mat, g4iw_magn_sym.mat, 'ed_g4iw_magn_v_vp_symmetry', rtol=1e-10, atol=1e-10)


def test_epot(input_type='minimal',verbose=False):
    # Load the data:
    ddict, hr = td.load_testdataset(input_type)

    # set up the single-particle quantities:
    nk = (64, 64, 1)
    sym = ddict['sym']
    k_grid = bz.KGrid(nk, sym)
    ek = hr.get_ek(k_grid)
    siwk_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'], niv_asympt=4000)
    siwk_full = giwk_obj.get_sigma_full(giwk_obj.niv_asympt)
    giwk_test = twop.build_g(giwk_obj.v * 1j, giwk_obj.ek, giwk_obj.mu, siwk_full)
    t_util.test_array(giwk_test, giwk_obj.g_full(), f'{input_type}_gwk_build_from_siwk_full')

    # set up G2:
    g4iw_dens = lfp.get_g2_from_dmft_input(ddict, 'dens')
    g4iw_magn = lfp.get_g2_from_dmft_input(ddict, 'magn')

    # obtain chi:
    gchi_dens = lfp.gchir_from_g2(g4iw_dens, giwk_obj.g_loc)
    gchi_magn = lfp.gchir_from_g2(g4iw_magn, giwk_obj.g_loc)

    # create the bubble object:
    bubble_gen = bub.BubbleGenerator(wn=gchi_dens.wn, giwk_obj=giwk_obj, is_full_wn=True)
    niv_core = g4iw_magn.niv
    niv_urange = niv_core
    niv_full = niv_core + niv_urange
    gchi0_urange = bubble_gen.get_gchi0(niv_full)

    # obtain gamma:
    gamma_dens = lfp.gamob2_from_gchir_urange(gchi_dens, gchi0_urange)
    gamma_magn = lfp.gamob2_from_gchir_urange(gchi_magn, gchi0_urange)

    # obtain chi_aux:
    _, chi_dens_tilde = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_dens, bubble_gen
                                                                       , niv_shell=niv_urange, sum_axis=-1)
    _, chi_magn_tilde = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, bubble_gen
                                                                       , niv_shell=niv_urange, sum_axis=-1)

    if verbose:
        plt.figure()
        plt.plot(giwk_obj.v_core, np.mean((giwk_obj.core.real * siwk_obj.sigma_core.real), axis=(0, 1, 2)),
                 color='cornflowerblue')
        plt.plot(giwk_obj.v_core, np.mean((giwk_obj.core.imag * siwk_obj.sigma_core.imag), axis=(0, 1, 2)), color='firebrick')
        # plt.plot(giwk_obj.v_core, np.mean((giwk_obj.core * siwk_obj.sigma_core).imag, axis=(0, 1, 2)), color='firebrick')
        plt.show()

        plt.figure()
        plt.plot(mf.vn(ddict['beta'], ddict['siw']), ddict['siw'] - twop.get_smom0(ddict['u'], ddict['n']),
                 color='cornflowerblue')
        plt.plot(giwk_obj.v_core, np.mean((siwk_obj.sigma_core).real, axis=(0, 1, 2)), color='firebrick')
        plt.xlim(0, 20)
        plt.show()

        plt.figure()
        plt.plot(mf.vn(ddict['beta'], ddict['giw']), ddict['giw'], color='cornflowerblue')
        plt.plot(giwk_obj.v_core, np.mean((giwk_obj.core).real, axis=(0, 1, 2)), color='firebrick')
        plt.xlim(0, 20)
        plt.show()

    e_pot_g = giwk_obj.e_pot
    e_pot_chi = (ddict['u'] / (2 * ddict['beta']) * np.sum(chi_dens_tilde - chi_magn_tilde) + ddict['u'] * ddict[
        'n'] ** 2 / 4).real
    sum_rule = 1 / (2 * ddict['beta']) * np.sum(chi_dens_tilde + chi_magn_tilde).real
    sum_rule_2 = twop.get_sum_chiupup(ddict['n'])
    e_pot = 0.043563 * ddict['u']  # hardcopied double occupatio
    t_util.test_array(sum_rule, sum_rule_2, f'{input_type}_chiupup_sum_rule_consistency', atol=5e-2, rtol=5e-2)
    t_util.test_array([e_pot_g, ], e_pot, f'{input_type}_epot_g_chi_dmft_consistency', atol=5e-2, rtol=5e-2)


def main():
    input_types = ['minimal','quasi_1d']
    # input_types = ['quasi_1d']
    iwn = [0, 1, -1, 10, -10]
    for it in input_types:
        test_routine_consistency(input_type=it)
        test_chi_sum_rule(input_type=it)
        test_schwinger_dyson(input_type=it)
        test_f_consistency(input_type=it)
        for i in iwn:
            test_asymptotic_convergence(input_type=it, iwn=i)

    test_crossing_symmetry()
    test_epot(input_type='minimal')
    test_f_consistency_ed()
    test_schwinger_dyson_ed()
    test_epot(input_type='ed_minimal')

if __name__ == '__main__':
    main()
