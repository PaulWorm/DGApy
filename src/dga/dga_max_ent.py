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
import dga.mpi_aux as mpi_aux
import dga.plotting as plotting
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
    logger = loggers.MpiLogger(logfile=path + 'max_ent.log', comm=comm, output_path=path)

    dmft_input = np.load(path + 'dmft_input.npy', allow_pickle=True).item()
    ek = hamk.ek_3d(dga_config.lattice.k_grid.grid, dga_config.lattice.hr)
    dga_config.output_path = path

    max_ent_config = conf_file['max_ent']
    me_conf = config.MaxEntConfig(1,dmft_input['beta'],max_ent_config)
    if comm.rank == 0 and 'loc' in max_ent_config:
        loc_dir = path + 'MaxEnt/'
        loc_dir = dga_io.set_output_path(loc_dir,comm=comm)
        siwk_dmft = twop.SelfEnergy(sigma=dmft_input['siw'][None, None, None, :], beta=dmft_input['beta'])
        giwk_dmft = twop.GreensFunction(siwk_dmft, ek, mu=dmft_input['mu_dmft'], niv_asympt=dga_config.box_sizes.niv_asympt)
        g_loc_dmft = giwk_dmft.g_loc
        a_cont.max_ent_loc_bw_range(g_loc_dmft, me_conf, name='dmft', out_dir=loc_dir)


    if(comm.rank == 0):
        sigma_dga = np.load(path + 'siwk_dga.npy', allow_pickle=True)
        sigma_dga = twop.SelfEnergy(sigma=sigma_dga, beta=dmft_input['beta'])

        giwk_dga = twop.GreensFunction(sigma_dga, ek, n=dmft_input['n'], niv_asympt=dga_config.box_sizes.niv_full)
    comm.barrier()

    # Perform analytic continuation of local quantities:
    if comm.rank == 0 and 'loc' in max_ent_config:
        g_loc_dga = giwk_dga.g_loc
        bw_opt_dga = a_cont.max_ent_loc_bw_range(g_loc_dga, me_conf, name='dga', out_dir=loc_dir)

        logger.log_cpu_time(task=' MaxEnt local ')
        logger.log_memory_usage()
    else:
        bw_opt_dga = None

    # Broadcast bw_opt_dga
    bw_opt_dga = comm.bcast(bw_opt_dga, root=0)
    # me_conf.bw_dga.append(bw_opt_dga)


    # Create a mpi distributor:
    mpi_dist = mpi_aux.MpiDistributor(ntasks=dga_config.lattice.q_grid.nk_irr, comm=comm,
                                          output_path=dga_config.output_path + '/',
                                          name='MaxEntIrrk')



    # Continuation of the k-dependent self-energy:
    if 's_dga' in max_ent_config:
        siw_cont_path = path + 'MaxEntSiwk'
        siw_cont_path = dga_io.set_output_path(siw_cont_path,comm=comm)
        me_controller = a_cont.MaxEnt(dmft_input['beta'],'freq_fermionic',comm=comm,config=max_ent_config['s_dga'])
        if(me_controller.do_cont):
            logger.log_cpu_time(task=' MaxEnt controller for Siwk created ')
            if(comm.rank == 0):
                sigma_mat = sigma_dga.get_siw(me_controller.n_fit)-sigma_dga.smom0 # subtract Hartree
                sigma_mat = dga_config.lattice.k_grid.map_fbz2irrk(sigma_mat)
            else:
                sigma_mat = None

            siw_cont = a_cont.mpi_ana_cont(sigma_mat, me_controller, mpi_dist, 'siwk_dga', logger = logger)
            if(comm.rank == 0):
                siw_cont = dga_config.lattice.k_grid.map_irrk2fbz(siw_cont) + sigma_dga.smom0  # add Hartree
                a_cont.save_and_plot_cont_fermionic(siw_cont, me_controller.w, dga_config.lattice.k_grid, 'swk_dga', siw_cont_path)
            logger.log_cpu_time(task=' MaxEnt Siwk ')
            logger.log_memory_usage()

    # Continuation of the k-dependent Green's function
    if 'g_dga' in max_ent_config:
        cont_path = path + 'MaxEntGiwk'
        cont_path = dga_io.set_output_path(cont_path,comm=comm)
        me_controller = a_cont.MaxEnt(dmft_input['beta'],'freq_fermionic',comm=comm,config=max_ent_config['g_dga'])
        if (me_controller.do_cont):
            logger.log_cpu_time(task=' MaxEnt controller for Giwk created ')
            if(comm.rank == 0):
                mat = giwk_dga.g_full()
                mat = dga_config.lattice.k_grid.map_fbz2irrk(mat)
            else:
                mat = None

            cont = a_cont.mpi_ana_cont(mat, me_controller, mpi_dist, 'giwk_dga', logger = logger)
            if(comm.rank == 0):
                cont = dga_config.lattice.k_grid.map_irrk2fbz(cont)
                a_cont.save_and_plot_cont_fermionic(cont, me_controller.w, dga_config.lattice.k_grid, 'gwk_dga', cont_path)
            logger.log_cpu_time(task=' MaxEnt Giwk ')
            logger.log_memory_usage()

    # Continuation of the q-dependent density lambda-susceptbility
    if 'chi_d' in max_ent_config:
        cont_path = path + 'MaxEntChiDens'
        cont_path = dga_io.set_output_path(cont_path,comm=comm)
        me_controller = a_cont.MaxEnt(dmft_input['beta'],'freq_bosonic',comm=comm,config=max_ent_config['chi_d'])
        if (me_controller.do_cont):
            logger.log_cpu_time(task=' MaxEnt controller for Chi-dens created ')
            if(comm.rank == 0):
                mat = np.load(path + 'chi_dens_lam.npy', allow_pickle=True)
                mat = dga_config.lattice.q_grid.map_fbz2irrk(mat)
            else:
                mat = None

            cont = a_cont.mpi_ana_cont(mat, me_controller, mpi_dist, 'chi_dens', logger = logger)
            if(comm.rank == 0):
                cont = dga_config.lattice.q_grid.map_irrk2fbz(cont)
                a_cont.save_and_plot_cont_bosonic(cont, me_controller.w, dga_config.lattice.q_grid, 'chi_dens', cont_path)
            logger.log_cpu_time(task=' MaxEnt Chi-dens ')
            logger.log_memory_usage()

    # Continuation of the q-dependent magnetic lambda-susceptbility
    if 'chi_m' in max_ent_config:
        cont_path = path + 'MaxEntChiMagn'
        cont_path = dga_io.set_output_path(cont_path,comm=comm)
        me_controller = a_cont.MaxEnt(dmft_input['beta'],'freq_bosonic',comm=comm,config=max_ent_config['chi_m'])
        if (me_controller.do_cont):
            logger.log_cpu_time(task=' MaxEnt controller for Chi-dens created ')
            if(comm.rank == 0):
                mat = np.load(path + 'chi_magn_lam.npy', allow_pickle=True)
                mat = dga_config.lattice.q_grid.map_fbz2irrk(mat)
            else:
                mat = None

            cont = a_cont.mpi_ana_cont(mat, me_controller, mpi_dist, 'chi_magn', logger = logger)
            if(comm.rank == 0):
                cont = dga_config.lattice.q_grid.map_irrk2fbz(cont)
                a_cont.save_and_plot_cont_bosonic(cont, me_controller.w, dga_config.lattice.q_grid, 'chi_magn', cont_path)
            logger.log_cpu_time(task=' MaxEnt Chi-magn ')
            logger.log_memory_usage()

    mpi_dist.delete_file()



if __name__ == '__main__':
    main(path='./', comm=None)
