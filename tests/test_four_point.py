import matplotlib.pyplot as plt
import numpy as np

from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import two_point as twop
from dga import bubble as bub
from dga import local_four_point as lfp
from dga import four_point as fp
from test_util import test_data as td
from test_util import util_for_testing as t_util


def test_consistency_with_local_routines(verbose=False, input_type='ed'):
    '''
        Test the consistency of the local routines with the non-local routines.
    '''
    # load the data:
    if input_type == 'ed':
        ddict, hr = td.load_minimal_dataset_ed()
    elif input_type == 'w2dyn':
        ddict, hr = td.load_minimal_dataset()
    else:
        raise ValueError(f'input_type = {input_type} not recognized.')

    # set up the single-particle quantities:
    nk = (6, 6, 1)
    sym = bz.two_dimensional_square_symmetries()
    k_grid = bz.KGrid(nk, sym)
    q_list = k_grid.get_irrq_list()
    ek = hr.get_ek(k_grid) * 0  # flat dispersion -> local green's function in each k-point
    siw_q_flat = 1j * mf.vn(ddict['beta'], mf.niv_from_mat(ddict['siw'])) + ddict['mu_dmft'] - 1 / ddict['giw']
    siwk_obj = twop.SelfEnergy(siw_q_flat[None, None, None, :], ddict['beta'])
    # the local green's function no longer coincides with that of DMFT
    giwk_obj = twop.GreensFunction(siwk_obj, ek, mu=ddict['mu_dmft'], niv_asympt=mf.niv_from_mat(ddict['siw']))
    t_util.test_array(giwk_obj.n, ddict['n'], f'{input_type}_g_loc_n_consistency', atol=3e-3)
    t_util.test_array(giwk_obj.mu, ddict['mu_dmft'], f'{input_type}_g_loc_mu_consistency', rtol=3e-3)
    t_util.test_array(mf.cut_v(giwk_obj.g_loc, giwk_obj.niv_asympt), mf.cut_v(ddict['giw'], giwk_obj.niv_asympt),
                      f'{input_type}_g_loc_dmft_consistency', atol=1e-4)

    # set up G2:
    g4iw_dens = lfp.get_g2_from_dmft_input(ddict, 'dens')
    g4iw_magn = lfp.get_g2_from_dmft_input(ddict, 'magn')

    # use v-vp symmetry:
    g4iw_dens.symmetrize_v_vp()
    g4iw_magn.symmetrize_v_vp()

    niw_core = int(ddict['beta'] * 2 + 10)
    niv_core = int(ddict['beta'] * 2 + 10)
    wn_core = mf.wn(niw_core)
    g4iw_dens.cut_iw(niw_core)
    g4iw_magn.cut_iw(niw_core)
    g4iw_dens.cut_iv(niv_core)
    g4iw_magn.cut_iv(niv_core)

    # set up the bubble generator:
    niv_shell = niv_core
    niv_urange = niv_core + niv_shell
    bubble_gen = bub.BubbleGenerator(wn=wn_core, giwk_obj=giwk_obj, is_full_wn=True)
    chi0_core = bubble_gen.get_chi0(niv=niv_core)
    chi0_urange = bubble_gen.get_chi0(niv=niv_urange)
    chi0_asympt = chi0_urange + bubble_gen.get_asymptotic_correction(niv_urange)

    gchi0_core = bubble_gen.get_gchi0(niv=niv_core)
    gchi0_urange = bubble_gen.get_gchi0(niv=niv_urange)

    gchi0_core_q = bubble_gen.get_gchi0_q_list(niv=niv_core, q_list=q_list)
    gchi0_urange_q = bubble_gen.get_gchi0_q_list(niv=niv_urange, q_list=q_list)
    chi0_core_q = bubble_gen.contract_legs(gchi0_core_q)
    chi0_urange_q = bubble_gen.contract_legs(gchi0_urange_q)
    chi0_asympt_q = chi0_urange_q + bubble_gen.get_asymptotic_correction(niv_urange)

    # get gchi from g2:
    gchi_dens = lfp.gchir_from_g2(g4iw_dens, giwk_obj.g_loc)
    gchi_magn = lfp.gchir_from_g2(g4iw_magn, giwk_obj.g_loc)

    # get gamma:
    gamma_dens = lfp.gamob2_from_gchir_urange(gchi_dens, gchi0_urange)
    gamma_magn = lfp.gamob2_from_gchir_urange(gchi_magn, gchi0_urange)

    # get gchi_q:
    gchi_dens_q = fp.get_gchir_from_gamma_loc_q(gamma_dens, gchi0_core_q)
    gchi_magn_q = fp.get_gchir_from_gamma_loc_q(gamma_magn, gchi0_core_q)

    # get gchi from gamob2:
    gchi_dens_urange = lfp.gchir_from_gamob2(gamma_dens, gchi0_core)
    gchi_magn_urange = lfp.gchir_from_gamob2(gamma_magn, gchi0_core)

    # This only works for niv_urange = 0 (i.e. no urange):
    t_util.test_array(k_grid.k_mean(gchi_dens_q), gchi_dens_urange.mat, f'{input_type}_gchi_dens_flat_q')
    t_util.test_array(k_grid.k_mean(gchi_magn_q), gchi_magn_urange.mat, f'{input_type}_gchi_magn_flat_q')

    # gchi_aux:
    gchi_aux_dens = lfp.gchi_aux_core_from_gammar(gamma_dens, gchi0_core)
    gchi_aux_magn = lfp.gchi_aux_core_from_gammar(gamma_magn, gchi0_core)
    gchi_aux_dens_q = fp.get_gchir_aux_from_gammar_q(gamma_dens, gchi0_core_q)
    gchi_aux_magn_q = fp.get_gchir_aux_from_gammar_q(gamma_magn, gchi0_core_q)

    t_util.test_array(k_grid.k_mean(gchi_aux_dens_q), gchi_aux_dens.mat, f'{input_type}_gchi_aux_dens_flat_q')
    t_util.test_array(k_grid.k_mean(gchi_aux_magn_q), gchi_aux_magn.mat, f'{input_type}_gchi_aux_magn_flat_q')

    # chi_aux:
    chi_aux_dens = gchi_aux_dens.contract_legs()
    chi_aux_magn = gchi_aux_magn.contract_legs()
    chi_aux_dens_q = 1 / ddict['beta'] ** 2 * np.sum(gchi_aux_dens_q, axis=(-1, -2))
    chi_aux_magn_q = 1 / ddict['beta'] ** 2 * np.sum(gchi_aux_magn_q, axis=(-1, -2))

    t_util.test_array(k_grid.k_mean(chi_aux_dens_q), chi_aux_dens, f'{input_type}_chi_aux_dens_flat_q')
    t_util.test_array(k_grid.k_mean(chi_aux_magn_q), chi_aux_magn, f'{input_type}_chi_aux_magn_flat_q')

    # chir_r:
    chi_dens = lfp.chi_phys_urange(chi_aux_dens, chi0_core, chi0_urange, ddict['u'], 'dens')
    chi_magn = lfp.chi_phys_urange(chi_aux_magn, chi0_core, chi0_urange, ddict['u'], 'magn')

    chi_dens_q = fp.chi_phys_from_chi_aux_q(chi_aux_dens_q, chi0_urange_q, chi0_core_q, ddict['u'], 'dens')
    chi_magn_q = fp.chi_phys_from_chi_aux_q(chi_aux_magn_q, chi0_urange_q, chi0_core_q, ddict['u'], 'magn')

    t_util.test_array(k_grid.k_mean(chi_dens_q), chi_dens, f'{input_type}_chi_dens_flat_q')
    t_util.test_array(k_grid.k_mean(chi_magn_q), chi_magn, f'{input_type}_chi_magn_flat_q')

    chi_dens_asympt = lfp.chi_phys_asympt(chi_dens, chi0_urange, chi0_asympt)
    chi_magn_asympt = lfp.chi_phys_asympt(chi_magn, chi0_urange, chi0_asympt)
    chi_dens_q_asympt = fp.chi_phys_asympt_q(chi_dens_q, chi0_urange_q, chi0_asympt_q)
    chi_magn_q_asympt = fp.chi_phys_asympt_q(chi_magn_q, chi0_urange_q, chi0_asympt_q)

    t_util.test_array(k_grid.k_mean(chi_dens_q_asympt), chi_dens_asympt, f'{input_type}_chi_dens_flat_q_asympt')
    t_util.test_array(k_grid.k_mean(chi_magn_q_asympt), chi_magn_asympt, f'{input_type}_chi_magn_flat_q_asympt')

    # vrg_r:
    vrg_dens = lfp.vrg_from_gchi_aux_asympt(gchi_aux_dens, gchi0_core, chi_dens, chi_dens_asympt)
    vrg_magn = lfp.vrg_from_gchi_aux_asympt(gchi_aux_magn, gchi0_core, chi_magn, chi_magn_asympt)
    vrg_dens_q = fp.vrg_from_gchi_aux_asympt(gchi_aux_dens_q, gchi0_core_q, chi_dens_q, chi_dens_q_asympt, ddict['u'], 'dens')
    vrg_magn_q = fp.vrg_from_gchi_aux_asympt(gchi_aux_magn_q, gchi0_core_q, chi_magn_q, chi_magn_q_asympt, ddict['u'], 'magn')

    t_util.test_array(k_grid.k_mean(vrg_dens_q), vrg_dens.mat, f'{input_type}_vrg_dens_flat_q')
    t_util.test_array(k_grid.k_mean(vrg_magn_q), vrg_magn.mat, f'{input_type}_vrg_magn_flat_q')

    # vrg_r urange:
    vrg_dens_urange = lfp.vrg_from_gchi_aux_asympt(gchi_aux_dens, gchi0_core, chi_dens, chi_dens)
    vrg_magn_urange = lfp.vrg_from_gchi_aux_asympt(gchi_aux_magn, gchi0_core, chi_magn, chi_magn)

    vrg_dens_urange_q = fp.vrg_from_gchi_aux_asympt(gchi_aux_dens_q, gchi0_core_q, chi_dens_q, chi_dens_q, ddict['u'], 'dens')
    vrg_magn_urange_q = fp.vrg_from_gchi_aux_asympt(gchi_aux_magn_q, gchi0_core_q, chi_magn_q, chi_magn_q, ddict['u'], 'magn')

    t_util.test_array(k_grid.k_mean(vrg_dens_urange_q), vrg_dens_urange.mat, f'{input_type}_vrg_dens_flat_q_urange')
    t_util.test_array(k_grid.k_mean(vrg_magn_urange_q), vrg_magn_urange.mat, f'{input_type}_vrg_magn_flat_q_urange')

    # DC kernel function:
    f_dc = lfp.fob2_from_gamob2_urange(gamma_magn, gchi0_urange)
    kernel_dc = fp.get_kernel_dc(f_dc, gchi0_urange_q)

    kernel_dc_fbz = k_grid.map_irrk2fbz(kernel_dc, shape='list')
    q_list_fbz = k_grid.get_q_list()

    hartree = twop.get_smom0(ddict['u'], ddict['n'])

    # sde with f_dc directly:
    siw_dc = -lfp.schwinger_dyson_f(f_dc, gchi0_urange, giwk_obj.g_loc) + hartree
    # sde with the non-local dc kernel:
    siw_dc_kernel = fp.schwinger_dyson_kernel_q(kernel_dc_fbz, giwk_obj.g_full(), giwk_obj.beta, q_list_fbz, wn_core,
                                                k_grid.nk_tot) + hartree
    siw_dc_kernel_loc = k_grid.k_mean(siw_dc_kernel, shape='fbz-mesh')
    # consistency of the f_dc and non-loc kernel:
    t_util.test_array(siw_dc_kernel_loc, siw_dc, f'{input_type}_siw_dc_kernel_q_flat_consistency', rtol=1e-5, atol=1e-5)

    vrg_dens_tilde, chi_dens_tilde = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_dens, bubble_gen, niv_shell=niv_shell)
    vrg_magn_tilde, chi_magn_tilde = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, bubble_gen, niv_shell=niv_shell)
    t_util.test_array(chi_dens_asympt.real, chi_dens_tilde.real, f'{input_type}_chi_dens_tilde_consistency', rtol=1e-5, atol=1e-5)
    t_util.test_array(vrg_dens_tilde.mat, vrg_dens.mat, f'{input_type}_vrg_dens_tilde_consistency', rtol=1e-5, atol=1e-5)

    t_util.test_array(chi_magn_asympt.real, chi_magn_tilde.real, f'{input_type}_chi_magn_tilde_consistency', rtol=1e-5, atol=1e-5)
    t_util.test_array(vrg_magn_tilde.mat, vrg_magn.mat, f'{input_type}_vrg_magn_tilde_consistency', rtol=1e-5, atol=1e-5)

    # compute the siw from the hedin equations in the magnetic channel only:
    siw_magn_hedin = 2 * lfp.schwinger_dyson_core_urange(vrg_magn, chi_magn_tilde, giwk_obj.g_loc,
                                                         niv_shell=niv_shell) + twop.get_smom0(ddict['u'], ddict['n'])
    # check consistency with the siw from the dc-kernel:
    t_util.test_array(mf.cut_v(siw_dc, niv_core), mf.cut_v(siw_magn_hedin, niv_core),
                      f'{input_type}_siw_dc_f_hedin_core_consistency', rtol=1e-5, atol=1e-5)

    # compute the siw from the hedin equations in the magnetic channel only and using urange quantities:
    siw_magn_hedin_urange = 2 * lfp.schwinger_dyson_core_urange(vrg_magn_urange, chi_magn, giwk_obj.g_loc,
                                                                niv_shell=niv_shell) + twop.get_smom0(ddict['u'], ddict['n'])
    # check consistency with the siw from the dc-kernel:
    t_util.test_array(mf.cut_v(siw_magn_hedin_urange, niv_urange), mf.cut_v(siw_magn_hedin_urange, niv_urange),
                      f'{input_type}_siw_dc_f_hedin_urange_consistency', rtol=1e-5, atol=1e-5)

    # compute the full siw from the hedin equations:
    siw_hedin = lfp.schwinger_dyson_full(vrg_dens_tilde, vrg_magn_tilde, chi_dens_tilde, chi_magn_tilde, giwk_obj.g_loc,
                                         ddict['n'], niv_shell=niv_shell)
    # map to fbz:
    vrg_dens_q_fbz = k_grid.map_irrk2fbz(vrg_dens_q, shape='list')
    vrg_magn_q_fbz = k_grid.map_irrk2fbz(vrg_magn_q, shape='list')
    chi_dens_q_fbz = k_grid.map_irrk2fbz(chi_dens_q_asympt, shape='list')
    chi_magn_q_fbz = k_grid.map_irrk2fbz(chi_magn_q_asympt, shape='list')

    # Schwinger-Dyson equation full-q:
    kernel_dc_fbz = mf.cut_v(kernel_dc_fbz, niv_core, axes=(-1))
    siw_sde_q = fp.schwinger_dyson_full_q(vrg_dens_q_fbz, vrg_magn_q_fbz, chi_dens_q_fbz, chi_magn_q_fbz, kernel_dc_fbz,
                                          giwk_obj.g_full(), giwk_obj.beta, ddict['u'], q_list_fbz, wn_core, k_grid.nk_tot,
                                          niv_shell=niv_shell) + twop.get_smom0(ddict['u'], ddict['n'])
    siw_sde_q_loc = k_grid.k_mean(siw_sde_q, shape='fbz-mesh')

    if verbose:
        plt.figure()
        plt.plot(mf.vn(niv_urange), siw_hedin.imag, '-o', label='siw', color='cornflowerblue', ms=6)
        plt.plot(mf.vn(niv_urange), siw_sde_q_loc.imag, '-o', label='siw_q_loc', color='firebrick')

        plt.plot(mf.vn(niv_urange), siw_dc.imag, '-h', label='siw_magn_kernel', color='tab:orange', ms=10)
        plt.plot(mf.vn(niv_urange), siw_magn_hedin.imag, '-h', label='siw_magn_hedin', color='indigo', ms=6)
        plt.plot(mf.vn(niv_urange), siw_magn_hedin_urange.imag, '-h', label='siw_magn_hedin_urange', color='goldenrod')
        plt.plot(mf.vn(mf.niv_from_mat(ddict['siw'])), ddict['siw'].imag, '-', label='siw_dmft', color='k')
        plt.xlim(0, niv_urange)
        plt.ylim(None, 0)
        plt.legend()
        plt.show()

    # check the consistency with the local and q-loc siw:
    t_util.test_array(mf.cut_v(siw_hedin, niv_urange), mf.cut_v(siw_sde_q_loc, niv_urange), f'{input_type}_siw_sde_consistency',
                      rtol=1e-5, atol=1e-5)
    # ---------------------------------- Test consistency of the four-point vertex -------------------------------------
    # Test fq:
    fob2_dens = lfp.fob2_from_gchir(gchi_dens, gchi0_core)
    fob2_magn = lfp.fob2_from_gchir(gchi_magn, gchi0_core)

    fob2_dens_from_gam_urange = lfp.fob2_from_gamob2_urange(gamma_dens, gchi0_urange)
    fob2_dens_from_gam_urange.cut_iv(niv_core)

    fob2_magn_from_gam_urange = lfp.fob2_from_gamob2_urange(gamma_magn, gchi0_urange)
    fob2_magn_from_gam_urange.cut_iv(niv_core)

    t_util.test_array(fob2_dens.mat, fob2_dens_from_gam_urange.mat, f'{input_type}_fob2_dens_from_chi_from_gam_loc_consistency'
                      , rtol=1e-5, atol=1e-5)
    t_util.test_array(fob2_magn.mat, fob2_magn_from_gam_urange.mat, f'{input_type}_fob2_magn_from_chi_from_gam_loc_consistency'
                      , rtol=1e-5, atol=1e-5)

    fq_dens1, fq_dens2 = fp.ladder_vertex_from_chi_aux_components(gchi_aux_dens_q, vrg_dens_urange_q, gchi0_core_q,
                                                                  ddict['beta'], lfp.get_ur(ddict['u'], 'dens'))

    fq_dens = fq_dens1 + (1 - ddict['u'] * chi_dens_q[..., None, None]) * fq_dens2
    fq_dens_loc = k_grid.k_mean(fq_dens)

    # test consistency of fq_dens with the local fq_dens:
    t_util.test_array(fq_dens_loc, ddict['beta'] ** 2 * fob2_dens.mat, f'{input_type}_fq_dens_consistency', rtol=1e-5, atol=1e-5)

    fq_magn1, fq_magn2 = fp.ladder_vertex_from_chi_aux_components(gchi_aux_magn_q, vrg_magn_urange_q, gchi0_core_q,
                                                                  ddict['beta'], lfp.get_ur(ddict['u'], 'magn'))

    u_m = lfp.get_ur(ddict['u'], 'magn')
    fq_magn = fq_magn1 + (1 - u_m * chi_magn_q[..., None, None]) * fq_magn2
    fq_magn_loc = k_grid.k_mean(fq_magn)

    # test consistency of fq_dens with the local fq_dens:
    t_util.test_array(fq_magn_loc, ddict['beta'] ** 2 * fob2_magn.mat, f'{input_type}_fq_magn_consistency', rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    test_consistency_with_local_routines(verbose=False, input_type='ed')
    test_consistency_with_local_routines(verbose=False, input_type='w2dyn')
