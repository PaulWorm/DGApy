import numpy as np
import gc

import dga.local_four_point as lfp
import dga.plotting as plotting
import dga.config as config
import dga.loggers as loggers
import dga.two_point as twop
import dga.bubble as bub



# ======================================================================================================================
# ---------------------------------------------- HIGH-LEVEL ROUTINES  --------------------------------------------------
# ======================================================================================================================

def local_sde_from_g2(g2_dens: lfp.LocalFourPoint, g2_magn: lfp.LocalFourPoint, giwk_dmft: twop.GreensFunction, dga_config,
                      dmft_input, comm,logger=None, write_output=True):
    '''
        Perform the local Schwinger-Dyson equation starting with g2 as input
    '''
    gchi_dens = lfp.gchir_from_g2(g2_dens, giwk_dmft.g_loc)
    gchi_magn = lfp.gchir_from_g2(g2_magn, giwk_dmft.g_loc)

    if(logger is not None): logger.log_cpu_time(task=' Data loading completed. ')
    if(logger is not None): logger.log_memory_usage()

    # ------------------------------------------- Extract the irreducible Vertex -----------------------------------------------------
    # Create Bubble generator:
    bubble_gen = bub.BubbleGenerator(wn=dga_config.box_sizes.wn, giw=giwk_dmft)
    gchi0_urange = bubble_gen.get_gchi0(dga_config.box_sizes.niv_full)
    gamma_dens = lfp.gammar_from_gchir(gchi_dens, gchi0_urange, dmft_input['u'])
    gamma_magn = lfp.gammar_from_gchir(gchi_magn, gchi0_urange, dmft_input['u'])
    del gchi_magn, gchi_dens
    gc.collect()

    if(logger is not None): logger.log_cpu_time(task=' Gamma-loc finished. ')
    if(logger is not None): logger.log_memory_usage()
    if (comm.rank == 0 and write_output):
        plotting.default_gamma_plots(gamma_dens, gamma_magn, dga_config.output_path, dga_config.box_sizes, dmft_input['beta'])
        dga_config.save_data(gamma_dens.mat, 'gamma_dens')
        dga_config.save_data(gamma_magn.mat, 'gamma_magn')

    if(logger is not None): logger.log_cpu_time(task=' Gamma-loc plotting finished. ')

    # ----------------------------------- Compute the Susceptibility and Threeleg Vertex --------------------------------------------------------
    vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_dens, bubble_gen, dmft_input['u'],
                                                                        niv_shell=dga_config.box_sizes.niv_shell)
    vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, bubble_gen, dmft_input['u'],
                                                                        niv_shell=dga_config.box_sizes.niv_shell)

    # Create checks of the susceptibility:
    if (comm.rank == 0 and write_output): plotting.chi_checks([chi_dens, ], [chi_magn, ], ['Loc-tilde', ], giwk_dmft, dga_config.output_path,
                                             verbose=False,
                                             do_plot=True, name='loc')

    if(logger is not None): logger.log_cpu_time(task=' Vrg and Chi-phys loc done. ')
    if(logger is not None): logger.log_memory_usage()
    # --------------------------------------- Local Schwinger-Dyson equation --------------------------------------------------------

    # Perform the local SDE for box-size checks:
    siw_sde_full = lfp.schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, giwk_dmft.g_loc,
                                            dmft_input['u'],
                                            dmft_input['n'],
                                            niv_shell=dga_config.box_sizes.niv_shell)

    # Create checks of the self-energy:
    if (comm.rank == 0 and write_output):
        plotting.sigma_loc_checks([siw_sde_full, dmft_input['siw']], ['SDE', 'Input'], dmft_input['beta'], dga_config.output_path,
                                  verbose=False,
                                  do_plot=True, xmax=dga_config.box_sizes.niv_full)
        dga_config.save_data(siw_sde_full, 'siw_sde_full')
        dga_config.save_data(chi_dens, 'chi_dens_loc')
        dga_config.save_data(chi_magn, 'chi_magn_loc')

    if(logger is not None): logger.log_cpu_time(task=' Local SDE finished. ')
    if(logger is not None): logger.log_memory_usage()

    return gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, siw_sde_full