#!/usr/bin/env python
# ------------------------------------------------ COMMENTS ------------------------------------------------------------
'''
    Main file for analytic continuation of the output from a DGA calcuation
'''
# ----------------------------------------------------------------------------------------------------------------------
import os
import socket

import numpy as np
from mpi4py import MPI as mpi
from ruamel.yaml import YAML
import matplotlib

from dga import config
from dga import loggers
from dga import dga_io
from dga import two_point as twop
from dga import analytic_continuation as a_cont
from dga import brillouin_zone as bz
from dga import mpi_aux
from dga import high_level_plotting as hlp
from dga import optics
import dga.plot_specs

if (socket.gethostname() != 'DESKTOP-OEHIPTV'):
    matplotlib.use('agg')  # non GUI backend since VSC has no display


def main(path='./', comm=None):
    print('Starting MaxEnt at:' + path)
    # Define MPI communicator:
    if (comm is None): comm = mpi.COMM_WORLD

    if (comm.rank == 0):
        conf_file = YAML().load(open(path + 'dga_config.yaml'))
    else:
        conf_file = None
    conf_file = comm.bcast(conf_file, root=0)
    d_cfg = config.DgaConfig(conf_file, comm=comm)
    logger = loggers.MpiLogger(logfile=path + 'max_ent.log', comm=comm, output_path=path)
    logger.log_event(f'Path is: {path}')

    fname_dmft = path + 'dmft_input.npy'
    if(os.path.isfile(fname_dmft)):
        dmft_input = np.load(path + 'dmft_input.npy', allow_pickle=True).item()
    else:
        raise ValueError(f'File: {fname_dmft} does not exist.')

    ek = d_cfg.lattice.hr.get_ek(d_cfg.lattice.k_grid)
    d_cfg.output_path = path
    d_cfg.set_system_parameter(dmft_input)

    # define k-path objects that might be used later:
    bz_kpath_string_list = ['Gamma-X-M2-Gamma','Gamma-Y-M2-Gamma','Gamma-X-M-Gamma','Gamma-Y-M-Gamma','Gamma-X-M-Y-Gamma']
    k_path_list = [bz.KPath(d_cfg.lattice.k_grid.nk, bz_kpath_string) for bz_kpath_string in bz_kpath_string_list]

    max_ent_config = conf_file['max_ent']
    me_conf = config.MaxEntConfig(1,dmft_input['beta'],max_ent_config)
    if comm.rank == 0 and 'loc' in max_ent_config:
        loc_dir = path + 'MaxEnt/'
        loc_dir = dga_io.set_output_path(loc_dir,comm=comm)
        siwk_dmft = twop.SelfEnergy(sigma=dmft_input['siw'][None, None, None, :], beta=dmft_input['beta'])
        giwk_dmft = twop.GreensFunction(siwk_dmft, ek, mu=dmft_input['mu_dmft'], niv_asympt=d_cfg.box.niv_asympt)
        g_loc_dmft = giwk_dmft.g_loc
        a_cont.max_ent_loc_bw_range(g_loc_dmft, me_conf, name='dmft', out_dir=loc_dir)


    if(comm.rank == 0):
        sigma_dga = np.load(path + 'siwk_dga.npy', allow_pickle=True)
        sigma_dga = twop.SelfEnergy(sigma=sigma_dga, beta=dmft_input['beta'])

        giwk_dga = twop.GreensFunction(sigma_dga, ek, n=dmft_input['n'], niv_asympt=d_cfg.box.niv_full)
    comm.barrier()

    # Perform analytic continuation of local quantities:
    if comm.rank == 0 and 'loc' in max_ent_config:
        g_loc_dga = giwk_dga.g_loc
        _ = a_cont.max_ent_loc_bw_range(g_loc_dga, me_conf, name='dga', out_dir=loc_dir)

        logger.log_cpu_time(task=' MaxEnt local ')

    # Create a mpi distributor:
    mpi_dist = mpi_aux.MpiDistributor(ntasks=d_cfg.lattice.q_grid.nk_irr, comm=comm,
                                          name='MaxEntIrrk')

    # Continuation of the k-dependent self-energy:
    if 's_dga' in max_ent_config:
        siw_cont_path = path + 'MaxEntSiwk'
        siw_cont_path = dga_io.set_output_path(siw_cont_path,comm=comm)
        me_controller = a_cont.MaxEnt(dmft_input['beta'], 'freq_fermionic', comm=comm, me_config=max_ent_config, **max_ent_config[
            's_dga'])
        if(me_controller.do_cont):
            logger.log_cpu_time(task=' MaxEnt controller for Siwk created ')
            if(comm.rank == 0):
                sigma_mat = sigma_dga.get_siw(me_controller.n_fit)-sigma_dga.smom0 # subtract Hartree
                sigma_mat = d_cfg.lattice.k_grid.map_fbz2irrk(sigma_mat)
            else:
                sigma_mat = None

            siw_cont = a_cont.mpi_ana_cont(sigma_mat, me_controller, mpi_dist, 'siwk_dga', logger = logger)
            if comm.rank == 0:
                siw_cont = d_cfg.lattice.k_grid.map_irrk2fbz(siw_cont) + sigma_dga.smom0  # add Hartree
                a_cont.save_and_plot_cont_fermionic(siw_cont, me_controller.w, d_cfg.lattice.k_grid, 'swk_dga', siw_cont_path)
                # build the Green's function from the self-energy:
                gwk_cont = twop.RealFrequencyGF(me_controller.w,siw_cont,ek,n=dmft_input['n'], deltino=0.04)
                a_cont.save_and_plot_cont_fermionic(gwk_cont.gwk(), gwk_cont.w, d_cfg.lattice.k_grid, 'gwk_dga',
                                                    siw_cont_path)
                t_unit = 1 # update this later
                t_unit = np.abs(t_unit)
                for k_path in k_path_list:
                    hlp.plot_real_frequency_dispersion(gwk_cont, k_path, pdir=siw_cont_path, name=f'gwk_dga_{k_path.path}',
                                                       wmin = -2*t_unit, wmax = 2*t_unit)
                    hlp.plot_real_frequency_dispersion(gwk_cont, k_path, pdir=siw_cont_path, name=f'gwk_dga_wfull_{k_path.path}')

                # create bubble optical conductivity:
                w_max_bub = me_controller.w[-1]/3
                sigma_bub, w_bub = optics.vec_get_sigma_bub_realf(gwk_cont,d_cfg.lattice.hr,d_cfg.lattice.k_grid,
                                                           d_cfg.sys.beta,w_max_bub)
                np.save(siw_cont_path + 'sigma_bub.npy', sigma_bub)
                np.save(siw_cont_path + 'w_bub.npy', w_bub)
                hlp.plot_opt_cond(sigma_bub,w_bub,pdir=siw_cont_path,name='sigma_bub')

            logger.log_cpu_time(task=' MaxEnt Siwk ')
            logger.log_memory_usage('siw_cont',siw_cont, n_exists=1)

    # Continuation of the k-dependent Green's function
    if 'g_dga' in max_ent_config:
        cont_path = path + 'MaxEntGiwk'
        cont_path = dga_io.set_output_path(cont_path,comm=comm)
        me_controller = a_cont.MaxEnt(dmft_input['beta'], 'freq_fermionic', comm=comm, me_config=max_ent_config,
                                      **max_ent_config['g_dga'])
        if (me_controller.do_cont):
            logger.log_cpu_time(task=' MaxEnt controller for Giwk created ')
            if(comm.rank == 0):
                mat = giwk_dga.g_full()
                mat = d_cfg.lattice.k_grid.map_fbz2irrk(mat)
            else:
                mat = None

            cont = a_cont.mpi_ana_cont(mat, me_controller, mpi_dist, 'giwk_dga', logger = logger)
            if(comm.rank == 0):
                cont = d_cfg.lattice.k_grid.map_irrk2fbz(cont)
                a_cont.save_and_plot_cont_fermionic(cont, me_controller.w, d_cfg.lattice.k_grid, f'gwk_dga', cont_path)
            logger.log_cpu_time(task=' MaxEnt Giwk ')
            logger.log_memory_usage('giwk_cont', cont, n_exists=1)

    # Continuation of the q-dependent density lambda-susceptbility
    if 'chi_d' in max_ent_config:
        cont_path = path + 'MaxEntChiDens'
        cont_path = dga_io.set_output_path(cont_path,comm=comm)
        me_controller = a_cont.MaxEnt(dmft_input['beta'], 'freq_bosonic', comm=comm, me_config=max_ent_config, **max_ent_config[
            'chi_d'])
        if (me_controller.do_cont):
            logger.log_cpu_time(task=' MaxEnt controller for Chi-dens created ')
            if(comm.rank == 0):
                mat = np.load(path + 'chi_dens_lam.npy', allow_pickle=True)
                mat = d_cfg.lattice.q_grid.map_fbz2irrk(mat)
            else:
                mat = None

            cont = a_cont.mpi_ana_cont(mat, me_controller, mpi_dist, 'chi_dens', logger = logger)
            if(comm.rank == 0):
                cont = d_cfg.lattice.q_grid.map_irrk2fbz(cont)
                a_cont.save_and_plot_cont_bosonic(cont, me_controller.w, d_cfg.lattice.q_grid, 'chi_dens', cont_path)
                for k_path in k_path_list:
                    hlp.plot_chi_along_kpath(cont, me_controller.w, k_path,
                                                       pdir=cont_path, name=f'chi_dens_{k_path.path}')
            logger.log_cpu_time(task=' MaxEnt Chi-dens ')

    # Continuation of the q-dependent magnetic lambda-susceptbility
    if 'chi_m' in max_ent_config:
        cont_path = path + 'MaxEntChiMagn'
        cont_path = dga_io.set_output_path(cont_path,comm=comm)
        me_controller = a_cont.MaxEnt(dmft_input['beta'], 'freq_bosonic', comm=comm, me_config=max_ent_config, **max_ent_config[
            'chi_m'])
        if (me_controller.do_cont):
            logger.log_cpu_time(task=' MaxEnt controller for Chi-magn created ')
            if(comm.rank == 0):
                mat = np.load(path + 'chi_magn_lam.npy', allow_pickle=True)
                mat = d_cfg.lattice.q_grid.map_fbz2irrk(mat)
            else:
                mat = None

            cont = a_cont.mpi_ana_cont(mat, me_controller, mpi_dist, 'chi_magn', logger = logger)
            if(comm.rank == 0):
                cont = d_cfg.lattice.q_grid.map_irrk2fbz(cont)
                a_cont.save_and_plot_cont_bosonic(cont, me_controller.w, d_cfg.lattice.q_grid, 'chi_magn', cont_path)
                for k_path in k_path_list:
                    hlp.plot_chi_along_kpath(cont, me_controller.w, k_path,
                                             pdir=cont_path, name=f'chi_dens_{k_path.path}')
            logger.log_cpu_time(task=' MaxEnt Chi-magn ')

    mpi_dist.delete_file()



if __name__ == '__main__':
    main(path='./', comm=None)
