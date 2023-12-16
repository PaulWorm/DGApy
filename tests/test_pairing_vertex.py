import numpy as np
from mpi4py import MPI as mpi

from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import two_point as twop
from dga import bubble as bub
from dga import local_four_point as lfp
from dga import four_point as fp
from dga import pairing_vertex as pv
from dga import mpi_aux
from dga import plotting

from test_util import test_data as td
from test_util import util_for_testing as t_util


def test_pairing_vertex_consistency(verbose=False, input_type='w2dyn'):
    ddict, hr = td.load_testdataset(input_type)
    # set up the single-particle quantities:
    nk = (6, 6, 1)
    sym = ddict['sym']
    k_grid = bz.KGrid(nk, sym)
    q_list = k_grid.get_irrq_list()
    ek = hr.get_ek(k_grid) * 0  # flat dispersion -> local green's function in each k-point
    siw_q_flat = 1j * mf.vn(ddict['beta'], mf.niv_from_mat(ddict['siw'])) + ddict['mu_dmft'] - 1 / ddict['giw']
    siwk_obj = twop.SelfEnergy(siw_q_flat, ddict['beta'])
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'])

    # set up G2:
    g4iw_dens = lfp.get_g2_from_dmft_input(ddict, 'dens')
    g4iw_magn = lfp.get_g2_from_dmft_input(ddict, 'magn')
    if input_type != 'ed_minimal':
        g4iw_dens.symmetrize_v_vp()
        g4iw_magn.symmetrize_v_vp()

    niv_core = g4iw_dens.niv
    niv_shell = niv_core
    niv_full = niv_core + niv_shell
    niv_pp = niv_core // 2

    # Obtain the susceptibilities:
    gchi_dens = lfp.gchir_from_g2(g4iw_dens, giwk_obj.g_loc)
    gchi_magn = lfp.gchir_from_g2(g4iw_magn, giwk_obj.g_loc)

    # Set up the bubble generators:
    bubble_gen = bub.BubbleGenerator(g4iw_magn.wn, giwk_obj=giwk_obj, is_full_wn=True)
    gchi0_urange_q = bubble_gen.get_gchi0_q_list(niv=niv_full, q_list=q_list)
    gchi0_urange = bubble_gen.get_gchi0(niv=niv_full)
    gchi0_core_q = mf.cut_v(gchi0_urange_q, niv_cut=niv_core, axes=-1)
    chi0_core_q = 1 / ddict['beta'] ** 2 * np.sum(gchi0_core_q, axis=(-1,))
    chi0_urange_q = 1 / ddict['beta'] ** 2 * np.sum(gchi0_urange_q, axis=(-1,))

    # get gamma:
    gamma_dens = lfp.gamob2_from_gchir_urange(gchi_dens, gchi0_urange)
    gamma_magn = lfp.gamob2_from_gchir_urange(gchi_magn, gchi0_urange)

    # Get the axilliary susceptibilities:
    gchi_aux_dens_q = fp.get_gchir_aux_from_gammar_q(gamma_dens, gchi0_core_q)
    gchi_aux_magn_q = fp.get_gchir_aux_from_gammar_q(gamma_magn, gchi0_core_q)

    chi_aux_dens_q = 1 / ddict['beta'] ** 2 * np.sum(gchi_aux_dens_q, axis=(-1, -2))
    chi_aux_magn_q = 1 / ddict['beta'] ** 2 * np.sum(gchi_aux_magn_q, axis=(-1, -2))

    # construct the fermi-boson vertices:
    vrg_q_dens = fp.vrg_from_gchi_aux(gchi_aux_dens_q, gchi0_core_q)
    vrg_q_magn = fp.vrg_from_gchi_aux(gchi_aux_magn_q, gchi0_core_q)

    # Compute the ladder susceptibility:
    chi_dens_q = fp.chi_phys_from_chi_aux_q(chi_aux_dens_q, chi0_urange_q, chi0_core_q, ddict['u'], 'dens')
    chi_magn_q = fp.chi_phys_from_chi_aux_q(chi_aux_magn_q, chi0_urange_q, chi0_core_q, ddict['u'], 'magn')

    # Get the ladder vertex components:
    u_d = lfp.get_ur(ddict['u'], 'dens')
    u_m = lfp.get_ur(ddict['u'], 'magn')
    fq_dens1, fq_dens2 = fp.ladder_vertex_from_chi_aux_components(gchi_aux_dens_q, vrg_q_dens, gchi0_core_q,
                                                                  ddict['beta'], lfp.get_ur(ddict['u'], 'dens'))
    fq_magn1, fq_magn2 = fp.ladder_vertex_from_chi_aux_components(gchi_aux_magn_q, vrg_q_magn, gchi0_core_q,
                                                                  ddict['beta'], lfp.get_ur(ddict['u'], 'magn'))
    fq_dens = fq_dens1 + (1 - u_d * chi_dens_q[..., None, None]) * fq_dens2
    fq_magn = fq_magn1 + (1 - u_m * chi_magn_q[..., None, None]) * fq_magn2

    fq_dens_pp = mf.ph2pp_iwn_wc(fq_dens, 0)
    fq_magn_pp = mf.ph2pp_iwn_wc(fq_magn, 0)

    # Get the ladder vertices via the file reading-writing implemented in main:
    class DummyConfig():
        def __init__(self):
            pass

    d_cfg = DummyConfig()
    d_cfg.box = DummyConfig()
    d_cfg.sys = DummyConfig()
    d_cfg.box.niv_pp = niv_pp
    d_cfg.box.wn = g4iw_dens.wn
    d_cfg.sys.beta = ddict['beta']
    d_cfg.sys.u = ddict['u']
    comm = mpi.COMM_WORLD
    mpi_distributor = mpi_aux.MpiDistributor(ntasks=k_grid.nk_irr, comm=comm, output_path='./', name='Q')

    # write the pairing vertex parts:
    pv.write_pairing_vertex_components(d_cfg, mpi_distributor, gamma_dens.channel, gchi_aux_dens_q, vrg_q_dens, gchi0_core_q)
    pv.write_pairing_vertex_components(d_cfg, mpi_distributor, gamma_magn.channel, gchi_aux_magn_q, vrg_q_magn, gchi0_core_q)

    # load the pairing vertex:
    f1_magn, f2_magn, f1_dens, f2_dens = pv.load_pairing_vertex_from_rank_files(output_path='./', name='Q',
                                                                                mpi_size=comm.size,
                                                                                nq=k_grid.nk_irr,
                                                                                niv_pp=niv_pp)
    mpi_distributor.delete_file()
    # Test the parts:
    if verbose:
        f_magn1_loc = k_grid.k_mean(fq_magn1, shape='irrk')
        f_magn2_loc = k_grid.k_mean(fq_magn2, shape='irrk')
        ff_magn1_loc = k_grid.k_mean(f1_magn, shape='irrk')
        ff_magn2_loc = k_grid.k_mean(f2_magn, shape='irrk')

        lfp.plot_fourpoint_nu_nup(mf.ph2pp_iwn_wc(f_magn1_loc, 0), do_save=False, show=True, name='f_magn1_loc')
        lfp.plot_fourpoint_nu_nup(ff_magn1_loc,do_save=False, show=True, name='ff_magn1_loc')
        lfp.plot_fourpoint_nu_nup(mf.ph2pp_iwn_wc(f_magn2_loc, 0), do_save=False, show=True, name='f_magn2_loc')
        lfp.plot_fourpoint_nu_nup(ff_magn2_loc,do_save=False, show=True, name='ff_magn2_loc')

    # test array parts:
    t_util.test_array(f1_magn, mf.ph2pp_iwn_wc(fq_magn1, 0), f'{input_type}_f1_magn_consistency')
    t_util.test_array(f2_magn, mf.ph2pp_iwn_wc(fq_magn2, 0), f'{input_type}_f2_magn_consistency')
    t_util.test_array(f1_dens, mf.ph2pp_iwn_wc(fq_dens1, 0), f'{input_type}_f1_dens_consistency')
    t_util.test_array(f2_dens, mf.ph2pp_iwn_wc(fq_dens2, 0), f'{input_type}_f2_dens_consistency')

    chi_magn_q = k_grid.map_irrk2fbz(chi_magn_q, shape='mesh')
    chi_dens_q = k_grid.map_irrk2fbz(chi_dens_q, shape='mesh')
    # build the pp vertices:
    chi_magn_lambda_pp = pv.reshape_chi(chi_magn_q, d_cfg.box.niv_pp)
    f1_magn = k_grid.map_irrk2fbz(f1_magn, shape='mesh')
    f2_magn = k_grid.map_irrk2fbz(f2_magn, shape='mesh')
    f_magn = f1_magn + (1 + ddict['u'] * chi_magn_lambda_pp) * f2_magn

    chi_dens_lambda_pp = pv.reshape_chi(chi_dens_q, d_cfg.box.niv_pp)
    f1_dens = k_grid.map_irrk2fbz(f1_dens, shape='mesh')
    f2_dens = k_grid.map_irrk2fbz(f2_dens, shape='mesh')
    f_dens = f1_dens + (1 - ddict['u'] * chi_dens_lambda_pp) * f2_dens

    # Test the parts:
    fq_magn_pp = k_grid.map_irrk2fbz(fq_magn_pp, shape='mesh')
    fq_dens_pp = k_grid.map_irrk2fbz(fq_dens_pp, shape='mesh')

    if verbose:
        f_magn_pp_loc = k_grid.k_mean(fq_magn_pp, shape='mesh')
        f_dens_pp_loc = k_grid.k_mean(fq_dens_pp, shape='mesh')
        f_magn_loc = k_grid.k_mean(f_magn, shape='mesh')
        f_dens_loc = k_grid.k_mean(f_dens, shape='mesh')
        lfp.plot_fourpoint_nu_nup(f_magn_pp_loc, do_save=False, show=True, name='fq_magn_pp_loc')
        lfp.plot_fourpoint_nu_nup(f_magn_loc, do_save=False, show=True, name='f_magn_loc')
        lfp.plot_fourpoint_nu_nup(f_magn_pp_loc-f_magn_loc, do_save=False, show=True, name='diff_f_magn_loc')
        lfp.plot_fourpoint_nu_nup(f_dens_pp_loc, do_save=False, show=True, name='fq_dens_pp_loc')
        lfp.plot_fourpoint_nu_nup(f_dens_loc, do_save=False, show=True, name='f_dens_loc')
        lfp.plot_fourpoint_nu_nup(f_dens_pp_loc-f_dens_loc, do_save=False, show=True, name='diff_f_dens_loc')

    t_util.test_array(f_magn, fq_magn_pp, f'{input_type}_fq_pp_magn')
    t_util.test_array(f_dens, fq_dens_pp, f'{input_type}_fq_pp_dens')


def test_ph_to_pp(verbose=False):
    ddict, hr = td.load_minimal_dataset()

    # set up the single-particle quantities:
    nk = (6, 6, 1)
    sym = bz.two_dimensional_square_symmetries()
    k_grid = bz.KGrid(nk, sym)
    q_list = k_grid.get_irrq_list()
    ek = hr.get_ek(k_grid) * 0  # flat dispersion -> local green's function in each k-point
    siw_q_flat = 1j * mf.vn(ddict['beta'], mf.niv_from_mat(ddict['siw'])) + ddict['mu_dmft'] - 1 / ddict['giw']
    siwk_obj = twop.SelfEnergy(siw_q_flat, ddict['beta'])
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'])

    # set up G2:
    g4iw_dens = lfp.get_g2_from_dmft_input(ddict, 'dens')
    g4iw_magn = lfp.get_g2_from_dmft_input(ddict, 'magn')
    g4iw_dens.symmetrize_v_vp()
    g4iw_magn.symmetrize_v_vp()

    # set up G2:
    g4iw_dens = lfp.get_g2_from_dmft_input(ddict, 'dens')
    g4iw_magn = lfp.get_g2_from_dmft_input(ddict, 'magn')
    g4iw_sing = lfp.get_g2_from_dmft_input(ddict, 'sing')
    g4iw_trip = lfp.get_g2_from_dmft_input(ddict, 'trip')

    niw_core = mf.niw_from_mat(g4iw_dens.mat)
    niw_pp = niw_core // 2
    niv_pp = g4iw_sing.niv // 2
    g4iw_sing.cut_iv(niv_pp)
    g4iw_sing.cut_iw(niw_pp)

    g4iw_trip.cut_iv(niv_pp)
    g4iw_trip.cut_iw(niw_pp)

    dens, magn = mf.ph2pp_wc_md(g4iw_dens.mat, g4iw_magn.mat)

    g4iw_sing_test = +1 / 2 * dens - 3 / 2 * magn
    g4iw_trip_test = +1 / 2 * dens + 1 / 2 * magn

    if verbose:
        n_plot = 0
        lfp.plot_fourpoint_nu_nup(g4iw_sing_test[niw_pp+n_plot], do_save=False, show=True, name='g4iw_sing_test')
        lfp.plot_fourpoint_nu_nup(g4iw_sing.mat[niw_pp+n_plot], do_save=False, show=True, name='g4iw_sing')
        lfp.plot_fourpoint_nu_nup(g4iw_sing_test[niw_pp+n_plot]-g4iw_sing.mat[niw_pp+n_plot], do_save=False, show=True,
                                  name='g4iw_sing_diff')

        lfp.plot_fourpoint_nu_nup(g4iw_trip_test[niw_pp+n_plot], do_save=False, show=True, name='g4iw_trip_test')
        lfp.plot_fourpoint_nu_nup(g4iw_trip.mat[niw_pp+n_plot], do_save=False, show=True, name='g4iw_trip')
        lfp.plot_fourpoint_nu_nup(g4iw_trip_test[niw_pp+n_plot]-g4iw_sing.mat[niw_pp+n_plot], do_save=False, show=True,
                                  name='g4iw_trip_diff')

    t_util.test_array(g4iw_sing_test, g4iw_sing.mat,
                      'g4iw_sing', rtol=1e-3, atol=1e-3)

    t_util.test_array(g4iw_trip_test, g4iw_trip.mat,
                      'g4iw_trip', rtol=1e-3, atol=1e-3)


def main():
    input_types = ['minimal', 'quasi_1d', 'ed_minimal']

    for input_type in input_types:
        test_pairing_vertex_consistency(verbose=False, input_type=input_type)

    test_ph_to_pp()

if __name__ == '__main__':
    main()
