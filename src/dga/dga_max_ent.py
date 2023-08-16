#!/usr/bin/env python
# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Main file for analytic continuation of the output from a DGA calcuation
# ----------------------------------------------------------------------------------------------------------------------
import sys, os
from importlib.resources import path

import numpy as np
from mpi4py import MPI as mpi
from ruamel.yaml import YAML

import dga.config as config
import dga.loggers as loggers
import dga.dga_io as dga_io
import dga.two_point as twop
import dga.hk as hamk
import dga.analytic_continuation as a_cont

import dga.plot_specs


def main(path='./', comm=None):
    # Define MPI communicator:
    if (comm is None): comm = mpi.COMM_WORLD

    if (comm.rank == 0):
        conf_file = YAML().load(open(path + 'dga_config.yaml'))
    else:
        conf_file = None
    conf_file = comm.bcast(conf_file, root=0)
    dga_config = config.DgaConfig(conf_file, comm=comm)
    logger = loggers.MpiLogger(logfile='max_ent.log', comm=comm, output_path=path)

    dmft_input = np.load(path + 'dmft_input.npy', allow_pickle=True).item()
    ek = hamk.ek_3d(dga_config.lattice.k_grid.grid, dga_config.lattice.hr)
    # Broadcast bw_opt_dga
    max_ent_dir = path + '/MaxEnt/'
    dga_config.output_path = path
    if (comm.rank == 0 and not os.path.exists(max_ent_dir)): os.mkdir(max_ent_dir)

    me_conf = config.MaxEntConfig(t=dga_config.lattice.hr[0, 0], beta=dmft_input['beta'], config_dict=conf_file['max_ent'],
                                  output_path_loc=max_ent_dir)

    if comm.rank == 0 and me_conf.cont_g_loc:
        siwk_dmft = twop.SelfEnergy(sigma=dmft_input['siw'][None, None, None, :], beta=dmft_input['beta'])
        giwk_dmft = twop.GreensFunction(siwk_dmft, ek, mu=dmft_input['mu_dmft'], niv_asympt=dga_config.box_sizes.niv_asympt)
        g_loc_dmft = giwk_dmft.g_loc
        a_cont.max_ent_loc_bw_range(g_loc_dmft, me_conf, name='dmft')
    # Broadcast bw_opt_dga

    sigma_dga = np.load(path + 'siwk_dga.npy', allow_pickle=True)
    sigma_dga = twop.SelfEnergy(sigma=sigma_dga, beta=dmft_input['beta'])

    giwk_dga = twop.GreensFunction(sigma_dga, ek, n=dmft_input['n'], niv_asympt=dga_config.box_sizes.niv_full)

    if comm.rank == 0 and me_conf.cont_g_loc:
        g_loc_dga = giwk_dga.g_loc
        bw_opt_dga = a_cont.max_ent_loc_bw_range(g_loc_dga, me_conf, name='dga')

        logger.log_cpu_time(task=' MaxEnt local ')
        logger.log_memory_usage()
    else:
        bw_opt_dga = None

    bw_opt_dga = comm.bcast(bw_opt_dga, root=0)
    # me_conf.bw_dga.append(bw_opt_dga)

    if me_conf.cont_s_nl:
        me_conf.output_path_nl_s = dga_config.output_path + '/MaxEntSiwk/'
        if (comm.rank == 0 and not os.path.exists(me_conf.output_path_nl_s)): os.mkdir(me_conf.output_path_nl_s)

        for bw in me_conf.bw_dga:
            a_cont.max_ent_irrk_bw_range_sigma(sigma_dga, dga_config.lattice.k_grid, me_conf, comm, bw, logger=logger,
                                               name='siwk_dga')
        logger.log_cpu_time(task=' MaxEnt Siwk ')
        logger.log_memory_usage()

    if me_conf.cont_g_nl:
        me_conf.output_path_nl_g = dga_config.output_path + '/MaxEntGiwk/'
        if (comm.rank == 0 and not os.path.exists(me_conf.output_path_nl_g)): os.mkdir(me_conf.output_path_nl_g)

        for bw in me_conf.bw_dga:
            a_cont.max_ent_irrk_bw_range_green(giwk_dga, dga_config.lattice.k_grid, me_conf, comm, bw, logger=logger,
                                               name='Giwk_dga')
        logger.log_cpu_time(task=' MaxEnt Giwk ')
        logger.log_memory_usage()

    # Continue the susceptibility:

    if me_conf.cont_chi_d:
        chi = np.load(path + 'chi_dens_lam.npy', allow_pickle=True)
        output_path = dga_config.output_path + '/MaxEntChiDens/'
        if (comm.rank == 0 and not os.path.exists(output_path)): os.mkdir(output_path)

        for bw in me_conf.bw_dga:
            a_cont.max_ent_irrk_bw_range_chi(chi, dga_config.lattice.k_grid, me_conf, comm, bw, logger=logger, name='dens',
                                             out_dir=output_path)
        logger.log_cpu_time(task=' MaxEnt Chi-dens ')
        logger.log_memory_usage()

    if me_conf.cont_chi_m:
        chi = np.load(path + 'chi_magn_lam.npy', allow_pickle=True)
        output_path = dga_config.output_path + '/MaxEntChiMagn/'
        if (comm.rank == 0 and not os.path.exists(output_path)): os.mkdir(output_path)

        for bw in me_conf.bw_chi:
            a_cont.max_ent_irrk_bw_range_chi(chi, dga_config.lattice.k_grid, me_conf, comm, bw, logger=logger, name='magn',
                                             out_dir=output_path)
        logger.log_cpu_time(task=' MaxEnt Chi-magn ')
        logger.log_memory_usage()


if __name__ == '__main__':
    main(path='./', comm=None)
