#!/usr/bin/env python
# ------------------------------------------------ COMMENTS ------------------------------------------------------------
'''
    This code performs a DGA calculation starting from DMFT quantities as input.
    For the original paper look at: PHYSICAL REVIEW B 75, 045118 (2007)
    For a detailed review of the procedure read my thesis:
    "Numerical analysis of many-body effects in cuprate and nickelate superconductors"
    Asymptotics were adapted from Kitatani et al. J. Phys. Mat. 10.1088/2515-7639/ac7e6d (2022)
'''
import os
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import socket

import numpy as np
import matplotlib
from mpi4py import MPI as mpi
import gc

from dga import config
from dga import analytic_continuation as a_cont
from dga import matsubara_frequencies as mf
from dga import dga_io
from dga import four_point as fp
from dga import lambda_correction as lc
from dga import two_point as twop
from dga import bubble as bub
from dga import mpi_aux
from dga import pairing_vertex as pv
from dga import eliashberg_equation as eq
from dga import high_level_routines as hlr
from dga import optics
from dga import plotting
from dga import plot_specs

# Define MPI communicator:
comm = mpi.COMM_WORLD


def main():
    # --------------------------------------------- CONFIGURATION ----------------------------------------------------------

    d_cfg, conf_file = config.parse_config_file(comm=comm)

    # Load the DMFT input:
    dmft_input = dga_io.load_1p_data(d_cfg.type, d_cfg.input_path, d_cfg.fname_1p, d_cfg.fname_2p)
    # Build the two-particle Green's function:
    g2_dens, g2_magn = dga_io.build_g2_obj(d_cfg, dmft_input)
    # Check consistency of frequency ranges set in the config file:
    d_cfg.optics.set_frequency_ranges(d_cfg)
    # set the system parameters from dmft input (n,beta,u,mu_dmft etc.):
    d_cfg.set_system_parameter(dmft_input)
    # Create the output folders:
    d_cfg.create_folders()
    # Save dmft input to output folder
    if (comm.rank == 0):
        ddict = dga_io.create_dmft_ddict(dmft_input, g2_dens, g2_magn)
        d_cfg.save_data(ddict, 'dmft_input')

    del dmft_input
    gc.collect()

    # Create the DGA d_cfg.logger:
    d_cfg.create_logger()
    if (d_cfg.do_sym_v_vp): d_cfg.logger.log_message('Symmetrizing g2 with respect to v-vp')
    d_cfg.logger.log_message(f'Running on {comm.size} threads.')
    d_cfg.logger.log_message(f'There are {d_cfg.lattice.k_grid.nk_irr} points in the irreducible BZ.')
    d_cfg.log_estimated_memory_consumption()
    d_cfg.log_sys_params()

    d_cfg.logger.log_event(message=' Config Init and folder set up done!')
    comm.Barrier()

    # Save the real-space Hamiltonian to file:
    if (comm.rank == 0): d_cfg.lattice.hr.save_hr(d_cfg.output_path)

    # Save the full config to yaml file:
    if (comm.rank == 0): config.save_config_file(conf_file, d_cfg.output_path)

    # Plot standard sanity plots for the two-particle Green's function:
    if (comm.rank == 0): plotting.default_g2_plots(g2_dens, g2_magn, d_cfg.pdir)

    d_cfg.logger.log_event(message=' Created default g2 plots!')
    # Build the DMFT Green's function:
    ek = d_cfg.lattice.hr.get_ek(d_cfg.lattice.k_grid)

    siwk_dmft = twop.SelfEnergy(sigma=d_cfg.sys.siw[None, None, None, :], beta=d_cfg.sys.beta)
    giwk_dmft = twop.GreensFunction(siwk_dmft, ek, mu=d_cfg.sys.mu_dmft, niv_asympt=d_cfg.box.niv_asympt)
    d_cfg.logger.log_memory_usage('Giwk-DMFT', giwk_dmft)
    d_cfg.check_gloc_dmft(giwk_dmft)
    d_cfg.logger.log_event(message=' Created Giwk-DMFT!')

    # Perform the poly-fits for the DMFT Green's function
    dga_io.dmft_poly_fit(giwk_dmft, d_cfg)

    # --------------------------------------------- LOCAL PART --------------------------------------------------------
    gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, siw_sde_full = hlr.local_sde_from_g2(g2_dens, g2_magn,
                                                                                                         giwk_dmft, d_cfg)
    d_cfg.logger.log_cpu_time(task=' Local SDE done. ')
    comm.Barrier()
    # ---------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- NON-LOCAL PART --------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------

    # Create the Bubble generator:
    bubble_gen = bub.BubbleGenerator(wn=d_cfg.box.wn, giwk_obj=giwk_dmft, is_full_wn=True)

    # Split the momenta between the different cores.
    mpi_distributor = mpi_aux.MpiDistributor(ntasks=d_cfg.lattice.q_grid.nk_irr, comm=comm,
                                             output_path=d_cfg.output_path,
                                             name='Q')
    comm.Barrier()
    my_q_list = d_cfg.lattice.q_grid.get_irrq_list()[mpi_distributor.my_slice]

    # Build the different non-local Bubbles:
    gchi0_q_urange = bubble_gen.get_gchi0_q_list(d_cfg.box.niv_full, my_q_list)
    chi0_q_urange = 1 / d_cfg.sys.beta ** 2 * np.sum(gchi0_q_urange, axis=-1)
    gchi0_q_core = mf.cut_v(gchi0_q_urange, niv_cut=d_cfg.box.niv_core, axes=-1)
    chi0_q_core = 1 / d_cfg.sys.beta ** 2 * np.sum(gchi0_q_core, axis=-1)

    d_cfg.logger.log_memory_usage('gchi0q', gchi0_q_urange, n_exists=1)
    # Construct the local double counting correction kernel:
    kernel_dc = hlr.construct_dc_kernel(gamma_magn, bubble_gen, gchi0_q_urange, d_cfg)
    d_cfg.logger.log_cpu_time(task=' DC kernel constructed. ')

    # Density channel:
    vrg_q_dens, chi_lad_dens = hlr.construct_vrg_and_chi_q_urange(gamma_dens, gchi0_q_core, chi0_q_urange, chi0_q_core,
                                                                  d_cfg, mpi_distributor)

    # Magnetic channel:
    vrg_q_magn, chi_lad_magn = hlr.construct_vrg_and_chi_q_urange(gamma_magn, gchi0_q_core, chi0_q_urange, chi0_q_core,
                                                                  d_cfg, mpi_distributor)

    d_cfg.logger.log_memory_usage('chi_lad', chi_lad_dens, n_exists=2)
    d_cfg.logger.log_memory_usage('vrg_q', vrg_q_dens, n_exists=2)

    if (d_cfg.save_vrg and comm.rank == 0):
        d_cfg.save_data(vrg_q_magn, 'vrg_q_magn')
        d_cfg.save_data(vrg_q_dens, 'vrg_q_dens')

    del gamma_magn, gamma_dens, vrg_magn, vrg_dens
    gc.collect()
    d_cfg.logger.log_cpu_time(task=' Vrg and Chi-ladder completed. ')

    # Collect the results from the different cores:
    chi_lad_dens = mpi_distributor.gather(rank_result=chi_lad_dens, root=0)
    chi_lad_magn = mpi_distributor.gather(rank_result=chi_lad_magn, root=0)

    # Build the full Brillouin zone, which is needed for the lambda-correction:
    if (comm.rank == 0):
        chi_lad_dens = d_cfg.lattice.q_grid.map_irrk2fbz(chi_lad_dens)
        chi_lad_magn = d_cfg.lattice.q_grid.map_irrk2fbz(chi_lad_magn)

    dga_io.chiq_checks(d_cfg, chi_dens, chi_magn, chi_lad_dens, chi_lad_magn, giwk_dmft, name='ladder')
    # Lambda-corrected susceptibility:

    if (comm.rank == 0):
        chi_lad_dens, chi_lad_magn, lambda_dens, lambda_magn = lc.lambda_correction(chi_lad_dens, chi_lad_magn,
                                                                                    d_cfg.sys.beta, chi_dens, chi_magn,
                                                                                    lambda_corr=d_cfg.lambda_corr)

    # Create checks of the lambda-corrected susceptibility:
    dga_io.chiq_checks(d_cfg, chi_dens, chi_magn, chi_lad_dens, chi_lad_magn, giwk_dmft, name='lam')
    if (comm.rank == 0):
        d_cfg.save_data(chi_lad_dens, 'chi_dens_lam')
        d_cfg.save_data(chi_lad_magn, 'chi_magn_lam')
        np.savetxt(d_cfg.output_path + '/lambda.txt', [[lambda_dens, lambda_magn]], header='dens magn', fmt='%.6f')
        chi0q_asympt_correction = bubble_gen.get_asymptotic_correction_q(d_cfg.box.niv_full, d_cfg.lattice.q_grid.get_irrq_list())
        chi0q_asympt_correction = d_cfg.lattice.q_grid.map_irrk2fbz(chi0q_asympt_correction)
        chi_dens_uniform_static = chi_lad_dens[0, 0, 0, d_cfg.box.niw_core].real
        chi_dens_uniform_static += chi0q_asympt_correction[0, 0, 0, d_cfg.box.niw_core].real
        chi_magn_uniform_static = chi_lad_magn[0, 0, 0, d_cfg.box.niw_core].real
        chi_magn_uniform_static += chi0q_asympt_correction[0, 0, 0, d_cfg.box.niw_core].real
        np.savetxt(d_cfg.output_path + '/UniformStaticSusceptibility.txt',
                   [[chi_dens_uniform_static, chi_magn_uniform_static]], header='dens magn', fmt='%.6f')

    del chi_dens, chi_magn
    gc.collect()
    d_cfg.logger.log_cpu_time(task=' Lambda-correction done. ')

    # Build the pairing vertex
    comm.Barrier()
    if (comm.rank == 0 and d_cfg.eliash.do_pairing_vertex):
        pv.build_pairing_vertex(d_cfg, comm, chi_lad_magn, chi_lad_dens)
        d_cfg.logger.log_cpu_time(task=' Pairing vertex saved to file. ')

    # Build the vertex for optical conductivity:
    if (comm.rank == 0 and d_cfg.optics.do_vertex):
        optics.build_vertex_for_optical_conductivity(d_cfg, comm, chi_lad_magn, chi_lad_dens)
        d_cfg.logger.log_cpu_time(task=' Optical conductivity vertex saved to file. ')

    # Build the momentum-dependent ladder vertex:
    if (comm.rank == 0 and d_cfg.save_fq):
        fp.build_vertex_fq(d_cfg, comm, chi_lad_magn, chi_lad_dens)
        d_cfg.logger.log_cpu_time(task=' Vertex saved to file. ')

    # Perform Ornstein-zernicke fitting of the susceptibility:
    if (comm.rank == 0): dga_io.fit_and_plot_oz(d_cfg.output_path, d_cfg.lattice.q_grid)

    # Non-local Schwinger-Dyson equation:
    # collect vrg vertices:
    vrg_q_dens = mpi_distributor.gather(rank_result=vrg_q_dens)
    vrg_q_magn = mpi_distributor.gather(rank_result=vrg_q_magn)

    d_cfg.logger.log_memory_usage('vrg-q', vrg_q_dens, n_exists=2)

    if (comm.rank == 0):
        vrg_q_dens = d_cfg.lattice.q_grid.map_irrk2fbz(vrg_q_dens, shape='list')
        vrg_q_magn = d_cfg.lattice.q_grid.map_irrk2fbz(vrg_q_magn, shape='list')

    # Distribute the full fbz q-points to the different cores:
    full_q_list = d_cfg.lattice.q_grid.get_q_list()

    # Create new mpi-distributor for the full fbz:
    mpi_dist_fbz = mpi_aux.MpiDistributor(ntasks=d_cfg.lattice.q_grid.nk_tot, comm=comm,
                                          name='FBZ')
    comm.Barrier()
    my_full_q_list = full_q_list[mpi_dist_fbz.my_slice]

    if (comm.rank == 0):
        # Map object to core q-list:
        chi_lad_dens = d_cfg.lattice.q_grid.map_fbz_mesh2list(chi_lad_dens)
        chi_lad_magn = d_cfg.lattice.q_grid.map_fbz_mesh2list(chi_lad_magn)

    chi_lad_dens = mpi_dist_fbz.scatter(chi_lad_dens)
    chi_lad_magn = mpi_dist_fbz.scatter(chi_lad_magn)

    vrg_q_dens = mpi_dist_fbz.scatter(vrg_q_dens)
    vrg_q_magn = mpi_dist_fbz.scatter(vrg_q_magn)

    kernel_dc = mpi_distributor.gather(kernel_dc)
    if (comm.rank == 0): kernel_dc = d_cfg.lattice.q_grid.map_irrk2fbz(kernel_dc, 'list')
    kernel_dc = mpi_dist_fbz.scatter(kernel_dc)

    siwk_dga = fp.schwinger_dyson_full_q(vrg_q_dens, vrg_q_magn, chi_lad_dens, chi_lad_magn, kernel_dc,
                                         giwk_dmft.g_full(), d_cfg.sys.beta, d_cfg.sys.u, my_full_q_list,
                                         d_cfg.box.wn, d_cfg.lattice.q_grid.nk_tot, 0, d_cfg.logger)

    del vrg_q_dens, vrg_q_magn, chi_lad_dens, chi_lad_magn, kernel_dc
    gc.collect()

    d_cfg.logger.log_cpu_time(task=' Non-local SDE done. ')

    # print(get_largest_vars())

    # Collect from the different cores:
    siwk_dga = mpi_dist_fbz.allreduce(siwk_dga)

    siwk_dga += mf.cut_v(d_cfg.sys.siw, d_cfg.box.niv_core) - mf.cut_v(siw_sde_full, d_cfg.box.niv_core)
    hartree = twop.get_smom0(d_cfg.sys.u, d_cfg.sys.n)
    siwk_dga += hartree

    # Build the DGA Self-energy with Sigma_DMFT as asymptotic
    sigma_dga = twop.create_dga_siwk_with_dmft_as_asympt(siwk_dga, siwk_dmft, d_cfg.box.niv_shell)

    dga_io.default_siwk_checks(d_cfg, sigma_dga, siw_sde_full, siwk_dmft)

    d_cfg.logger.log_cpu_time(task=' Siwk collected and plotted. ')

    # Build the DGA Green's function:
    giwk_dga = twop.GreensFunction(sigma_dga, ek, n=d_cfg.sys.n, niv_asympt=d_cfg.box.niv_asympt)

    dga_io.default_giwk_checks(d_cfg, giwk_dga, sigma_dga)

    d_cfg.logger.log_cpu_time(task=' Giwk build and plotted. ')
    if not d_cfg.debug.keep_rank_files: mpi_distributor.delete_file()

    # --------------------------------------------- POSTPROCESSING ----------------------------------------------------------

    # ---------------------------------------------------- ENERGY -----------------------------------------------------------
    # if comm.rank == 0: # not yet finished
    #     e_kin_dga = giwk_dga.e_kin
    #     e_kin_dmft = giwk_dmft.e_kin

    # --------------------------------------------- POLY-FIT ----------------------------------------------------------------
    dga_io.dga_poly_fit(d_cfg, sigma_dga, giwk_dga)

    # --------------------------------------------- ELIASHBERG ROUTINES ----------------------------------------------------------
    if comm.rank == 0 and d_cfg.eliash.do_eliash:
        d_cfg.logger.log_cpu_time(task=' Starting Eliashberg ')
        gk_dga = mf.cut_v(giwk_dga.core, d_cfg.box.niv_pp, (-1,))
        norm = np.prod(d_cfg.lattice.q_grid.nk_tot) * d_cfg.sys.beta
        gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=d_cfg.eliash.gap0_sing['k'],
                                v_type=d_cfg.eliash.gap0_sing['v'],
                                k_grid=d_cfg.lattice.q_grid.grid)

        gamma_sing = -d_cfg.eliash.load_data('F_sing_pp')
        # if d_cfg.eliash.sym_sing: gamma_sing = eq.symmetrize_gamma(gamma_sing, 'sing')
        powiter_sing = eq.EliashberPowerIteration(gamma=gamma_sing, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
                                                  n_eig=d_cfg.eliash.n_eig)

        gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=d_cfg.eliash.gap0_trip['k'],
                                v_type=d_cfg.eliash.gap0_trip['v'],
                                k_grid=d_cfg.lattice.q_grid.grid)

        gamma_trip = -d_cfg.eliash.load_data('F_trip_pp')
        # if (d_cfg.eliash.sym_trip): gamma_trip = eq.symmetrize_gamma(gamma_sing, 'trip')
        powiter_trip = eq.EliashberPowerIteration(gamma=gamma_trip, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
                                                  n_eig=d_cfg.eliash.n_eig)

        eliashberg = {
            'lambda_sing': powiter_sing.lam,
            'lambda_trip': powiter_trip.lam,
            'delta_sing': powiter_sing.gap,
            'delta_trip': powiter_trip.gap,
        }
        d_cfg.eliash.save_data(eliashberg, 'eliashberg')
        np.savetxt(d_cfg.eliash.output_path + 'eigenvalues.txt', [powiter_sing.lam.real, powiter_trip.lam.real],
                   delimiter=',',
                   fmt='%.9f')
        np.savetxt(d_cfg.eliash.output_path + 'eigenvalues_s.txt', [powiter_sing.lam_s.real, powiter_trip.lam_s.real],
                   delimiter=',', fmt='%.9f')

        for i in range(len(powiter_sing.gap)):
            kx, ky = d_cfg.lattice.k_grid.kx_shift, d_cfg.lattice.k_grid.ky_shift
            niv_pp = d_cfg.box.niv_pp
            data = d_cfg.lattice.k_grid.shift_mat_by_pi(powiter_sing.gap[i].real, axes=(0, 1))[:, :, 0,
                   niv_pp - 1:niv_pp + 1]
            plotting.plot_gap_function_kx_ky(data, kx, ky, name='sing_{}'.format(i), pdir=d_cfg.eliash.output_path)
            data = d_cfg.lattice.k_grid.shift_mat_by_pi(powiter_trip.gap[i].real, axes=(0, 1))[:, :, 0,
                   niv_pp - 1:niv_pp + 1]
            plotting.plot_gap_function_kx_ky(data, kx, ky, name='trip_{}'.format(i), pdir=d_cfg.eliash.output_path)
            d_cfg.logger.log_event('Eliashberg completed!')
            if not d_cfg.keep_pairing_vertex:
                d_cfg.eliash.clean_data('F_sing_pp')
                d_cfg.eliash.clean_data('F_trip_pp')

    # --------------------------------------------- OPTICAL CONDUCTIVITY------------------------------------------------------
    if comm.rank == 0 and 'optics' in conf_file:
        chijj_bubble = None
        chijj_vert = None

    if comm.rank == 0 and d_cfg.optics.do_bubble:
        d_cfg.logger.log_cpu_time(task=' Starting optics bubble')
        chijj_bubble = optics.vec_get_chijj_bubble(giwk_dga, d_cfg.lattice.hr, d_cfg.lattice.k_grid,
                                                   d_cfg.optics.wn_bubble(pos=True),
                                                   niv_sum=d_cfg.optics.niv_bubble, der_a=d_cfg.optics.der_a,
                                                   der_b=d_cfg.optics.der_b)
        chijj_bubble = mf.bosonic_full_nu_range(chijj_bubble)
        d_cfg.optics.save_data(chijj_bubble, 'chijj_bubble')
        d_cfg.logger.log_event('Optics bubble completed!')

    if d_cfg.optics.do_vertex:
        d_cfg.logger.log_cpu_time(task=' Starting optics vertex')
        if d_cfg.comm.rank == 0:
            f_cond = d_cfg.optics.load_data('F_cond')
            f_cond = d_cfg.lattice.q_grid.map_fbz_mesh2list(f_cond)
            d_cfg.logger.log_memory_usage('F_cond', f_cond, n_exists=1)
        else:
            f_cond = None
        d_cfg.logger.log_cpu_time(task=' Vertex loaded')
        f_cond = mpi_dist_fbz.scatter(f_cond, root=0)
        d_cfg.logger.log_memory_usage('f_cond', f_cond, n_exists=1)
        d_cfg.logger.log_cpu_time(task=' Vertex distributed among ranks.')
        chijj_vert = optics.vec_get_chijj_vert(f_cond, giwk_dga, d_cfg.lattice.hr, d_cfg.lattice.k_grid,
                                               wn_vert=d_cfg.optics.wn_vert(), wn_cond=d_cfg.optics.wn_cond(pos=True),
                                               q_list=my_full_q_list, der_a=d_cfg.optics.der_a, der_b=d_cfg.optics.der_b)
        chijj_vert = mpi_dist_fbz.allreduce(chijj_vert)  # collect results from ranks
        chijj_vert = mf.bosonic_full_nu_range(chijj_vert)
        if comm.rank == 0:
            d_cfg.optics.save_data(chijj_vert, 'chijj_vert')
        d_cfg.logger.log_event('Optics vertex completed!')

    if comm.rank == 0 and 'optics' in conf_file:
        # Plot the optical conductivity on matsubara frequencies:
        if chijj_vert is None: chijj_vert = np.zeros_like(chijj_bubble)

        optics.plot_opt_cond_matsubara(chijj_bubble, chijj_vert, do_save=True, pdir=d_cfg.optics.output_path)

        # Perform the analytic continuation:
        if ('max_ent' in conf_file['optics'] and comm.rank == 0):
            d_cfg.logger.log_event('MaxEnt for optical conductivity started.')
            conf_dict = conf_file['optics']['max_ent']
            if ('n_fit' in conf_file['optics']['max_ent']):
                if (conf_file['optics']['max_ent']['n_fit'] == -1):
                    conf_dict['n_fit'] = d_cfg.optics.niw_cond
            max_ent = a_cont.MaxEnt(d_cfg.sys.beta, 'freq_bosonic', me_config=conf_dict)

            sigma_bubble_cont = max_ent.cont_single_ind(chijj_bubble)
            chijj_full = mf.add_bosonic(chijj_bubble, chijj_vert)
            sigma_full_cont = max_ent.cont_single_ind(chijj_full)
            # Plot the optical conductivity on real frequencies:
            optics.plot_opt_cond_realf(max_ent.w, sigma_bubble_cont, sigma_full_cont, do_save=True,
                                       pdir=d_cfg.optics.output_path)
            d_cfg.optics.save_data(sigma_bubble_cont, 'sigma_bubble_cont')
            d_cfg.optics.save_data(sigma_full_cont, 'sigma_full_cont')
            d_cfg.optics.save_data(max_ent.w, 'w')
            d_cfg.logger.log_event('MaxEnt for optical conductivity finished.')

            if not d_cfg.keep_fq_optics: d_cfg.optics.clean_data('F_cond')

    # ------------------------------------------------- MAXENT ------------------------------------------------------------
    if ('max_ent' in conf_file):
        import dga.dga_max_ent as dme
        d_cfg.logger.log_cpu_time(task=' Starting Max-Ent ')
        d_cfg.logger.log_event('Path is:' + d_cfg.output_path)
        dme.main(path=d_cfg.output_path, comm=comm)
        d_cfg.logger.log_event('MaxEnt completed')

    # End program:
    d_cfg.logger.log_event('Completed DGA run!')
    comm.Barrier()
    mpi.Finalize()


if __name__ == '__main__':
    main()
