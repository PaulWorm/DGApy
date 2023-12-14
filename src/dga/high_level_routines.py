import numpy as np
import gc

from dga import plotting
from dga import config
from dga import loggers
from dga import matsubara_frequencies as mf
from dga import two_point as twop
from dga import bubble as bub
from dga import local_four_point as lfp
from dga import four_point as fp
from dga import mpi_aux
from dga import pairing_vertex as pv
from dga import optics


# ======================================================================================================================
# ---------------------------------------------- HIGH-LEVEL ROUTINES  --------------------------------------------------
# ======================================================================================================================

def local_sde_from_g2(g2_dens: lfp.LocalFourPoint, g2_magn: lfp.LocalFourPoint, giwk_dmft: twop.GreensFunction,
                      d_cfg: config.DgaConfig):
    '''
        Perform the local Schwinger-Dyson equation starting with g2 as input
    '''
    logger = d_cfg.logger
    comm = d_cfg.comm
    verbosity = d_cfg.verbosity

    gchi_dens = lfp.gchir_from_g2(g2_dens, giwk_dmft.g_loc)
    gchi_magn = lfp.gchir_from_g2(g2_magn, giwk_dmft.g_loc)
    if (comm.rank == 0 and verbosity > 0):
        plotting.default_gchi_plots(gchi_dens, gchi_magn, d_cfg.pdir)
    # ------------------------------------------- Extract the irreducible Vertex -------------------------------------------------
    # Create Bubble generator:
    bubble_gen = bub.BubbleGenerator(wn=d_cfg.box.wn, giwk_obj=giwk_dmft, is_full_wn=True)
    gchi0_urange = bubble_gen.get_gchi0(d_cfg.box.niv_full)
    gamma_dens = lfp.gamob2_from_gchir_urange(gchi_dens, gchi0_urange)
    gamma_magn = lfp.gamob2_from_gchir_urange(gchi_magn, gchi0_urange)
    logger.log_memory_usage('Gamma-dens', gamma_dens, n_exists=6)
    del gchi_magn, gchi_dens
    gc.collect()

    logger.log_cpu_time(task=' Gamma-loc finished. ')
    if (comm.rank == 0 and verbosity > 0):
        plotting.default_gamma_plots(gamma_dens, gamma_magn, d_cfg.pdir, d_cfg.box, d_cfg.sys.beta)
        d_cfg.save_data(gamma_dens.mat, 'gamma_dens')
        d_cfg.save_data(gamma_magn.mat, 'gamma_magn')

    logger.log_cpu_time(task=' Gamma-loc plotting finished. ')

    # ----------------------------------- Compute the Susceptibility and Threeleg Vertex -----------------------------------------
    vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_gammar_urange(gamma_dens, bubble_gen,
                                                                       niv_shell=d_cfg.box.niv_shell)
    vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_gammar_urange(gamma_magn, bubble_gen,
                                                                       niv_shell=d_cfg.box.niv_shell)

    # Create checks of the susceptibility:
    if (comm.rank == 0 and verbosity > 0): plotting.chi_checks([chi_dens, ], [chi_magn, ], ['Loc-tilde', ], giwk_dmft,
                                                               d_cfg.pdir, verbose=False, do_plot=True, name='loc')

    logger.log_cpu_time(task=' Vrg and Chi-phys loc done. ')
    # --------------------------------------- Local Schwinger-Dyson equation --------------------------------------------------------

    # Perform the local SDE for box-size checks:
    siw_sde_full = lfp.schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, giwk_dmft.g_loc, d_cfg.sys.n,
                                            niv_shell=d_cfg.box.niv_shell)

    # Create checks of the self-energy:
    if (comm.rank == 0 and verbosity > 0):
        plotting.sigma_loc_checks([siw_sde_full, d_cfg.sys.siw], ['SDE', 'Input'], d_cfg.sys.beta, d_cfg.pdir,
                                  verbose=False,
                                  do_plot=True, xmax=d_cfg.box.niv_full)
        d_cfg.save_data(siw_sde_full, 'siw_sde_full')
        d_cfg.save_data(chi_dens, 'chi_dens_loc')
        d_cfg.save_data(chi_magn, 'chi_magn_loc')

    logger.log_cpu_time(task=' Local SDE finished. ')

    return gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, siw_sde_full

def construct_dc_kernel(gamma_magn: lfp.LocalFourPoint, bubble_gen: bub.BubbleGenerator, gchi0_q_urange, d_cfg: config.DgaConfig):
    ''' Construct the double-counting kernel '''

    # Build the local magnetic vertex including the shell asymptotics:
    gchi0_urange = bubble_gen.get_gchi0(d_cfg.box.niv_full)
    f_dc = lfp.fob2_from_gamob2_urange(gamma_magn, gchi0_urange)

    # double-counting kernel:
    if(d_cfg.comm.rank == 0):
        f_dc.plot(pdir=d_cfg.pdir + '/', name='F_dc')

    kernel_dc = mf.cut_v(fp.get_kernel_dc(f_dc, gchi0_q_urange), d_cfg.box.niv_core, axes=(-1,))
    d_cfg.logger.log_memory_usage('kernel-dc', kernel_dc, n_exists=1)
    return kernel_dc


def construct_vrg_and_chi_q_urange(gamma_r: lfp.LocalFourPoint, gchi0_q_core, chi0_q_urange, chi0_q_core
                                   , d_cfg: config.DgaConfig, mpi_dist: mpi_aux.MpiDistributor):
    gchiq_aux = fp.get_gchir_aux_from_gammar_q(gamma_r, gchi0_q_core)
    chiq_aux = 1 / d_cfg.sys.beta ** 2 * np.sum(gchiq_aux, axis=(-1, -2))

    chi_lad_urange = fp.chi_phys_from_chi_aux_q(chiq_aux, chi0_q_urange, chi0_q_core, d_cfg.sys.u, gamma_r.channel)
    vrg_q_dens = fp.vrg_from_gchi_aux(gchiq_aux, gchi0_q_core)

    if (d_cfg.eliash.do_pairing_vertex):
        pv.write_pairing_vertex_components(d_cfg, mpi_dist, gamma_r.channel, gchiq_aux, vrg_q_dens, gchi0_q_core)

    if (d_cfg.optics.do_vertex):
        optics.write_optics_vertex_components(d_cfg, mpi_dist, gamma_r.channel, gchiq_aux, vrg_q_dens, gchi0_q_core)

    if (d_cfg.save_fq):
        fp.write_vertex_components(d_cfg, mpi_dist, gamma_r.channel, gchiq_aux, vrg_q_dens, gchi0_q_core, name='fq_')

    return vrg_q_dens, chi_lad_urange