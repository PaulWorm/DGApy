#!/usr/bin/env python
# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# This code performs a DGA calculation starting from DMFT quantities as io.
# For the original paper look at: PHYSICAL REVIEW B 75, 045118 (2007)
# For a detailed review of the procedure read my thesis: "Numerical analysis of many-body effects in cuprate and nickelate superconductors"
# Asymptotics were adapted from Kitatani et al. J. Phys. Mat. 10.1088/2515-7639/ac7e6d (2022)
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI as mpi
from ruamel.yaml import YAML
import gc

import dga.config as config
import dga.matsubara_frequencies as mf
import dga.brillouin_zone as bz
import dga.loggers as loggers
import dga.dga_io as dga_io
import dga.four_point as fp
import dga.local_four_point as lfp
import dga.plotting as plotting
import dga.lambda_correction as lc
import dga.two_point as twop
import dga.bubble as bub
import dga.hk as hamk
import dga.mpi_aux as mpi_aux
import dga.pairing_vertex as pv
import dga.eliashberg_equation as eq
import dga.util as util

import dga.plot_specs

# Define MPI communicator:
comm = mpi.COMM_WORLD

# --------------------------------------------- CONFIGURATION ----------------------------------------------------------
# Parse config line arguments:
parser = config.create_dga_argparser()
if (comm.rank == 0):
    args = parser.parse_args()
    # args = parser.parse_args(args=['--config','dga_config.yaml'])
    assert hasattr(args, 'config'), 'Config file location must be provided.'
    conf_file = YAML().load(open(args.path + args.config))
else:
    conf_file = None

conf_file = comm.bcast(conf_file, root=0)
lambda_correction_type = conf_file['dga']['lambda_corr']

dga_config = config.DgaConfig(conf_file)

# %%
# --------------------------------------------------- LOAD THE INPUT -------------------------------------------------------------
dmft_input = dga_io.load_1p_data(dga_config.input_type, dga_config.input_path, dga_config.fname_1p, dga_config.fname_2p)
g2_dens, g2_magn = dga_io.load_g2(dga_config.box_sizes, dmft_input)



# ------------------------------------------- DEFINE THE OUTPUT DIRECTORY --------------------------------------------------------
# Define output directory:
output_dir = dga_config.input_path + '/LambdaDga_lc_{}_Nk{}_Nq{}_wcore{}_vcore{}_vshell{}'.format(lambda_correction_type,
                                                                                                  dga_config.lattice.nk_tot,
                                                                                                  dga_config.lattice.nq_tot,
                                                                                                  dga_config.box_sizes.niw_core,
                                                                                                  dga_config.box_sizes.niv_core,
                                                                                                  dga_config.box_sizes.niv_shell)

# Create output directory:
comm.barrier()
dga_config.set_output_path(output_dir,comm)
comm.barrier()

if ('pairing' in conf_file):
    pairing_config = config.EliashbergConfig(conf_file['pairing'])
    pairing_config.set_output_path(dga_config.output_path + 'Eliashberg/',comm)
else:
    pairing_config = config.EliashbergConfig()
# %%
hr = dga_config.lattice.set_hr()


logger = loggers.MpiLogger(logfile=dga_config.output_path + '/dga.log', comm=comm, output_path=dga_config.output_path)
logger.log_message(f'Running on {comm.size} threads.')
logger.log_memory_usage()
logger.log_event(message=' Config Init and folder set up done!')

# Save dmft input to output folder
if(comm.rank == 0): dga_config.save_data(dmft_input,'dmft_input')

# Save the full config to yaml file:
if (comm.rank == 0):
    with open(dga_config.output_path + "/config.yaml", "w+") as file:
        yaml = YAML()
        yaml.dump(dga_config.as_dict(), file)
        file.close()
    with open(dga_config.output_path + "/raw_config_file.yaml", "w+") as file:
        yaml = YAML()
        yaml.dump(conf_file, file)
        file.close()

if (comm.rank == 0): plotting.default_g2_plots(g2_dens, g2_magn, dga_config.output_path)

# Build Green's function and susceptibility:
ek = hamk.ek_3d(dga_config.lattice._k_grid.grid, hr)

siwk_dmft = twop.SelfEnergy(sigma=dmft_input['siw'][None, None, None, :], beta=dmft_input['beta'])
giwk_dmft = twop.GreensFunction(siwk_dmft, ek, mu=dmft_input['mu_dmft'], niv_asympt=dga_config.box_sizes.niv_asympt)

if (dga_config.do_poly_fitting):
    poly_fit_dir = dga_config.output_path + '/PolyFits/'
    if (comm.rank == 0): os.mkdir(poly_fit_dir)
    if comm.rank == 0:
        dga_io.poly_fit(giwk_dmft.g_full(pi_shift=True), dmft_input['beta'], dga_config.lattice._k_grid, dga_config.n_fit,
                        dga_config.o_fit,
                        name='Giwk_dmft_poly_cont_',
                        output_path=poly_fit_dir)

# Broadcast bw_opt_dga
if ('max_ent' in conf_file):
    max_ent_dir = dga_config.output_path + '/MaxEnt/'
    if (comm.rank == 0 and not os.path.exists(max_ent_dir)): os.mkdir(max_ent_dir)
    me_conf = config.MaxEntConfig(t=hr[0, 0], beta=dmft_input['beta'], config_dict=conf_file['max_ent'],
                                  output_path_loc=max_ent_dir)

    if comm.rank == 0 and me_conf.cont_g_loc:
        g_loc_dmft = giwk_dmft.g_loc
        dga_io.max_ent_loc_bw_range(g_loc_dmft, me_conf, name='dmft')

gchi_dens = lfp.gchir_from_g2(g2_dens, giwk_dmft.g_loc)
gchi_magn = lfp.gchir_from_g2(g2_magn, giwk_dmft.g_loc)

del g2_dens, g2_magn
gc.collect()

logger.log_cpu_time(task=' Data loading completed. ')
logger.log_memory_usage()

# ------------------------------------------- Extract the irreducible Vertex -----------------------------------------------------
# Create Bubble generator:
bubble_gen = bub.LocalBubble(wn=dga_config.box_sizes.wn, giw=giwk_dmft)
gchi0_core = bubble_gen.get_gchi0(dga_config.box_sizes.niv_core)
gchi0_urange = bubble_gen.get_gchi0(dga_config.box_sizes.niv_full)
gamma_dens = lfp.gammar_from_gchir(gchi_dens, gchi0_urange, dmft_input['u'])
gamma_magn = lfp.gammar_from_gchir(gchi_magn, gchi0_urange, dmft_input['u'])
del gchi_magn, gchi_dens
gc.collect()
logger.log_cpu_time(task=' Gamma-loc finished. ')
logger.log_memory_usage()
if (comm.rank == 0):
    plotting.default_gamma_plots(gamma_dens, gamma_magn, dga_config.output_path, dga_config.box_sizes, dmft_input['beta'])
    dga_config.save_data(gamma_dens.mat,'gamma_dens')
    dga_config.save_data(gamma_magn.mat,'gamma_magn')

logger.log_cpu_time(task=' Gamma-loc plotting finished. ')

# ----------------------------------- Compute the Susceptibility and Threeleg Vertex --------------------------------------------------------
vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_dens, bubble_gen, dmft_input['u'],
                                                                    niv_shell=dga_config.box_sizes.niv_shell)
vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, bubble_gen, dmft_input['u'],
                                                                    niv_shell=dga_config.box_sizes.niv_shell)

# Create checks of the susceptibility:
if (comm.rank == 0): plotting.chi_checks([chi_dens, ], [chi_magn, ], ['Loc-tilde', ], giwk_dmft, dga_config.output_path, verbose=False,
                                         do_plot=True, name='loc')

logger.log_cpu_time(task=' Vrg and Chi-phys loc done. ')
logger.log_memory_usage()
# --------------------------------------- Local Schwinger-Dyson equation --------------------------------------------------------

# Perform the local SDE for box-size checks:
siw_sde_full = lfp.schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, giwk_dmft.g_loc,
                                        dmft_input['u'],
                                        dmft_input['n'],
                                        niv_shell=dga_config.box_sizes.niv_shell)

# Create checks of the self-energy:
if (comm.rank == 0):
    plotting.sigma_loc_checks([siw_sde_full, dmft_input['siw']], ['SDE', 'Input'], dmft_input['beta'], dga_config.output_path, verbose=False,
                              do_plot=True, xmax=dga_config.box_sizes.niv_full)
    dga_config.save_data(siw_sde_full,'siw_sde_full')
    dga_config.save_data(chi_dens,'chi_dens_loc')
    dga_config.save_data(chi_magn,'chi_magn_loc')


logger.log_cpu_time(task=' Local SDE finished. ')
logger.log_memory_usage()
# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- NON-LOCAL PART --------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# Split the momenta between the different cores.
mpi_distributor = mpi_aux.MpiDistributor(ntasks=dga_config.lattice._q_grid.nk_irr, comm=comm, output_path=dga_config.output_path,
                                         name='Q')
my_q_list = dga_config.lattice._q_grid.irrk_mesh_ind.T[mpi_distributor.my_slice]
# %%
F_dc = lfp.Fob2_from_gamob2_urange(gamma_magn, gchi0_urange, dmft_input['u'])
# #
#     Compute the fermi-bose vertex and susceptibility using the asymptotics proposed in
#     Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005
# #
niv_full = dga_config.box_sizes.niv_full
beta = dmft_input['beta']
u = dmft_input['u']
niv_core = dga_config.box_sizes.niv_core
# Build the different non-local Bubbles:
gchi0_q_urange = bubble_gen.get_gchi0_q_list(niv_full, my_q_list)
chi0_q_urange = 1 / beta ** 2 * np.sum(gchi0_q_urange, axis=-1)
gchi0_q_core = mf.cut_v(gchi0_q_urange, niv_cut=niv_core, axes=-1)
chi0_q_core = 1 / beta ** 2 * np.sum(gchi0_q_core, axis=-1)
chi0q_shell = bubble_gen.get_asymptotic_correction_q(niv_full, my_q_list)
chi0q_shell_dc = bubble_gen.get_asymptotic_correction_q(niv_full, my_q_list)
if (logger is not None): logger.log_cpu_time(task=' Bubbles constructed. ')

# double-counting kernel:
if (logger is not None):
    if (logger.is_root):
        F_dc.plot(pdir=logger.out_dir + '/', name='F_dc')

kernel_dc = mf.cut_v(fp.get_kernel_dc(F_dc.mat, gchi0_q_urange, u, 'magn'), niv_core, axes=(-1,))
if (logger is not None): logger.log_cpu_time(task=' DC kernel constructed. ')

# Density channel:
gchiq_aux = fp.get_gchir_aux_from_gammar_q(gamma_dens, gchi0_q_core, u)
chiq_aux = 1 / beta ** 2 * np.sum(gchiq_aux, axis=(-1, -2))
chi_lad_urange = fp.chi_phys_from_chi_aux_q(chiq_aux, chi0_q_urange, chi0_q_core, u, gamma_dens.channel)
chi_lad_dens = fp.chi_phys_asympt_q(chi_lad_urange, chi0_q_urange, chi0_q_urange + chi0q_shell)

vrg_q_dens = fp.vrg_from_gchi_aux(gchiq_aux, gchi0_q_core, chi_lad_urange, chi_lad_dens, u, gamma_dens.channel)

if('pairing' in conf_file):
    niv_pp = dga_config.box_sizes.niv_pp
    omega = pv.get_omega_condition(niv_pp=niv_pp)
    channel = 'dens'
    mpi_distributor.open_file()
    for i,iq in enumerate(mpi_distributor.my_tasks):
        for j,iw in enumerate(dga_config.box_sizes.wn):
            if np.abs(iw) < 2 * niv_pp:
                condition = omega == iw

                f1_slice, f2_slice = pv.ladder_vertex_from_chi_aux_components(gchi_aux=gchiq_aux[i,j],
                                                                              vrg=vrg_q_dens[i,j],
                                                                              gchi0=gchi0_q_core[i,j],
                                                                              beta=beta,
                                                                              u_r=fp.get_ur(u,channel))
                group = '/irrq{:03d}wn{:04d}/'.format(iq,iw)

                mpi_distributor.file[group + f'f1_{channel}/'] = pv.get_pp_slice_4pt(mat=f1_slice, condition=condition,
                                                                     niv_pp=niv_pp)
                mpi_distributor.file[group + f'f2_{channel}/'] = pv.get_pp_slice_4pt(mat=f2_slice, condition=condition,
                                                                     niv_pp=niv_pp)

                mpi_distributor.file[group + 'condition/'] = condition
    mpi_distributor.close_file()



# Magnetic channel:
gchiq_aux = fp.get_gchir_aux_from_gammar_q(gamma_magn, gchi0_q_core, u)
chiq_aux = 1 / beta ** 2 * np.sum(gchiq_aux, axis=(-1, -2))
chi_lad_urange = fp.chi_phys_from_chi_aux_q(chiq_aux, chi0_q_urange, chi0_q_core, u, gamma_magn.channel)
chi_lad_magn = fp.chi_phys_asympt_q(chi_lad_urange, chi0_q_urange, chi0_q_urange + chi0q_shell)

u_r = fp.get_ur(u, gamma_magn.channel)
# 1/beta**2 since we want F/beta**2
kernel_dc += u_r / gamma_magn.beta * (1 - u_r * chi_magn[None, :, None]) * vrg_magn.mat[None, :, :] * chi0q_shell_dc[
                                                                                                              :, :, None]
vrg_q_magn = fp.vrg_from_gchi_aux(gchiq_aux, gchi0_q_core, chi_lad_urange, chi_lad_magn, u, gamma_magn.channel)

if('pairing' in conf_file):
    niv_pp = dga_config.box_sizes.niv_pp
    omega = pv.get_omega_condition(niv_pp=niv_pp)
    channel = 'magn'
    mpi_distributor.open_file()
    for i,iq in enumerate(mpi_distributor.my_tasks):
        for j,iw in enumerate(dga_config.box_sizes.wn):
            if np.abs(iw) < 2 * niv_pp:
                condition = omega == iw

                f1_slice, f2_slice = pv.ladder_vertex_from_chi_aux_components(gchi_aux=gchiq_aux[i,j],
                                                                              vrg=vrg_q_magn[i,j],
                                                                              gchi0=gchi0_q_core[i,j],
                                                                              beta=beta,
                                                                              u_r=fp.get_ur(u,channel))
                group = '/irrq{:03d}wn{:04d}/'.format(iq,iw)

                mpi_distributor.file[group + f'f1_{channel}/'] = pv.get_pp_slice_4pt(mat=f1_slice, condition=condition,
                                                                     niv_pp=niv_pp)
                mpi_distributor.file[group + f'f2_{channel}/'] = pv.get_pp_slice_4pt(mat=f2_slice, condition=condition,
                                                                     niv_pp=niv_pp)

    mpi_distributor.close_file()


del gamma_magn, gamma_dens, F_dc, vrg_magn, vrg_dens, gchiq_aux, chiq_aux
gc.collect()
logger.log_cpu_time(task=' Vrg and Chi-ladder completed. ')
logger.log_memory_usage()



# %% Collect the results from the different cores:
chi_lad_dens = mpi_distributor.gather(rank_result=chi_lad_dens, root=0)
chi_lad_magn = mpi_distributor.gather(rank_result=chi_lad_magn, root=0)

# Build the full Brillouin zone, which is needed for the lambda-correction:
if (comm.rank == 0):
    chi_lad_dens = dga_config.lattice._q_grid.map_irrk2fbz(chi_lad_dens)
    chi_lad_magn = dga_config.lattice._q_grid.map_irrk2fbz(chi_lad_magn)
# %%

if (comm.rank == 0):
    chi_lad_dens_loc = dga_config.lattice._q_grid.k_mean(chi_lad_dens, type='fbz-mesh')
    chi_lad_magn_loc = dga_config.lattice._q_grid.k_mean(chi_lad_magn, type='fbz-mesh')
    plotting.chi_checks([chi_dens, chi_lad_dens_loc], [chi_magn, chi_lad_magn_loc], ['loc', 'ladder-sum'], giwk_dmft, dga_config.output_path,
                        verbose=False,
                        do_plot=True, name='lad_q_tilde')

    plotting.plot_kx_ky(chi_lad_dens[..., 0, dga_config.box_sizes.niw_core], dga_config.lattice._q_grid.kx,
                        dga_config.lattice._q_grid.ky, pdir=dga_config.output_path, name='Chi_ladder_dens_kz0')
    plotting.plot_kx_ky(chi_lad_magn[..., 0, dga_config.box_sizes.niw_core], dga_config.lattice._q_grid.kx,
                        dga_config.lattice._q_grid.ky, pdir=dga_config.output_path, name='Chi_ladder_magn_kz0')

# %% Lambda-corrected susceptibility:

if (comm.rank == 0):
    chi_lad_dens, chi_lad_magn, lambda_dens, lambda_magn = lc.lambda_correction(chi_lad_dens, chi_lad_magn, dmft_input['beta'],
                                                                                chi_dens, chi_magn,
                                                                                type=dga_config.lambda_corr)

# %%
if (comm.rank == 0):
    chi_lad_dens_loc = dga_config.lattice._q_grid.k_mean(chi_lad_dens, type='fbz-mesh')
    chi_lad_magn_loc = dga_config.lattice._q_grid.k_mean(chi_lad_magn, type='fbz-mesh')
    plotting.chi_checks([chi_lad_dens_loc, chi_dens], [chi_lad_magn_loc, chi_magn], ['lam-loc', 'loc'], giwk_dmft, dga_config.output_path,
                        verbose=False,
                        do_plot=True, name='q_lam')

    plotting.plot_kx_ky(chi_lad_dens[..., 0, dga_config.box_sizes.niw_core], dga_config.lattice._q_grid.kx,
                        dga_config.lattice._q_grid.ky, pdir=dga_config.output_path, name='Chi_lam_dens_kz0')
    plotting.plot_kx_ky(chi_lad_magn[..., 0, dga_config.box_sizes.niw_core], dga_config.lattice._q_grid.kx,
                        dga_config.lattice._q_grid.ky, pdir=dga_config.output_path, name='Chi_lam_magn_kz0')

    dga_config.save_data(chi_lad_dens,'chi_dens_lam')
    dga_config.save_data(chi_lad_magn,'chi_magn_lam')
    np.savetxt(dga_config.output_path + '/lambda.txt', [[lambda_dens, lambda_magn]], header='dens magn', fmt='%.6f')

del chi_dens, chi_magn
gc.collect()
logger.log_cpu_time(task=' Lambda-correction done. ')
logger.log_memory_usage()


#%% Build the pairing vertex
comm.Barrier()
if(comm.rank == 0 and 'pairing' in conf_file):
    f1_magn, f2_magn, f1_dens, f2_dens = pv.load_pairing_vertex_from_rank_files(output_path=dga_config.output_path, name='Q',
                                                                                mpi_size=comm.size,
                                                                                nq=dga_config.lattice._q_grid.nk_irr,
                                                                                niv_pp=dga_config.box_sizes.niv_pp)
    chi_magn_lambda_pp = pv.reshape_chi(chi_lad_magn,dga_config.box_sizes.niv_pp)
    f1_magn = dga_config.lattice._q_grid.map_irrk2fbz(f1_magn,shape='mesh')
    f2_magn = dga_config.lattice._q_grid.map_irrk2fbz(f2_magn,shape='mesh')
    f_magn = f1_magn + (1 + dmft_input['u'] * chi_magn_lambda_pp) * f2_magn

    chi_dens_lambda_pp = pv.reshape_chi(chi_lad_dens,dga_config.box_sizes.niv_pp)
    f1_dens = dga_config.lattice._q_grid.map_irrk2fbz(f1_dens,shape='mesh')
    f2_dens = dga_config.lattice._q_grid.map_irrk2fbz(f2_dens,shape='mesh')
    f_dens = f1_dens + (1 - dmft_input['u'] * chi_dens_lambda_pp) * f2_dens

    # Build singlet and triplet vertex:
    f_sing = -1.5 * f_magn + 0.5 * f_dens
    f_trip = -0.5 * f_magn - 0.5 * f_dens


    f_sing_loc = dga_config.lattice._q_grid.k_mean(f_sing,'fbz-mesh')
    f_trip_loc = dga_config.lattice._q_grid.k_mean(f_trip,'fbz-mesh')

    lfp.plot_fourpoint_nu_nup(f_sing_loc,mf.vn(dga_config.box_sizes.niv_pp),pdir=dga_config.output_path,name='F_sing_pp')
    lfp.plot_fourpoint_nu_nup(f_trip_loc,mf.vn(dga_config.box_sizes.niv_pp),pdir=dga_config.output_path,name='F_trip_pp')

    pairing_config.save_data(f_sing,'F_sing_pp')
    pairing_config.save_data(f_trip,'F_trip_pp')
    del f_sing, f_trip, f_dens, f_magn
    gc.collect()

logger.log_cpu_time(task=' Pairing vertex saved to file. ')
# %% Perform Ornstein-zernicke fitting of the susceptibility:
if (comm.rank == 0): dga_io.fit_and_plot_oz(dga_config.output_path, dga_config.lattice._q_grid)

# %% Non-local Schwinger-Dyson equation:
# collect vrg vertices:
vrg_q_dens = mpi_distributor.gather(rank_result=vrg_q_dens)
vrg_q_magn = mpi_distributor.gather(rank_result=vrg_q_magn)

if(comm.rank == 0):
    vrg_q_dens = dga_config.lattice._q_grid.map_irrk2fbz(vrg_q_dens, shape='list')
    vrg_q_magn = dga_config.lattice._q_grid.map_irrk2fbz(vrg_q_magn, shape='list')

# %%
# Distribute the full fbz q-points to the different cores:
full_q_list = np.array([dga_config.lattice._q_grid.kmesh_ind[i].flatten() for i in range(3)]).T
q_dupl = np.ones((np.shape(full_q_list)[0]))

# Create new mpi-distributor for the full fbz:
mpi_dist_fbz = mpi_aux.MpiDistributor(ntasks=dga_config.lattice._q_grid.nk_tot, comm=comm, output_path=dga_config.output_path + '/',
                                      name='FBZ')
my_full_q_list = full_q_list[mpi_dist_fbz.my_slice]

if(comm.rank == 0):
    # Map object to core q-list:
    chi_lad_dens = dga_config.lattice._q_grid.map_fbz_mesh2list(chi_lad_dens)
    chi_lad_magn = dga_config.lattice._q_grid.map_fbz_mesh2list(chi_lad_magn)

chi_lad_dens = mpi_dist_fbz.scatter(chi_lad_dens)
chi_lad_magn = mpi_dist_fbz.scatter(chi_lad_magn)

vrg_q_dens = mpi_dist_fbz.scatter(vrg_q_dens)
vrg_q_magn = mpi_dist_fbz.scatter(vrg_q_magn)


#%%
kernel_dc = mpi_distributor.gather(kernel_dc)
if(comm.rank == 0):
    kernel_dc = dga_config.lattice._q_grid.map_irrk2fbz(kernel_dc, 'list')
    kernel_dc_mesh = dga_config.lattice._q_grid.map_fbz_list2mesh(kernel_dc)
    plotting.plot_kx_ky(kernel_dc_mesh[:, :, 0, dga_config.box_sizes.niw_core, dga_config.box_sizes.niv_core],
                        dga_config.lattice._q_grid.kx, dga_config.lattice._q_grid.ky,
                        pdir=dga_config.output_path, name='dc_kernel')

kernel_dc = mpi_dist_fbz.scatter(kernel_dc)
siwk_dga = fp.schwinger_dyson_full_q(vrg_q_dens, vrg_q_magn, chi_lad_dens, chi_lad_magn, kernel_dc,
                                     giwk_dmft.g_full(), dmft_input['beta'], dmft_input['u'], my_full_q_list,
                                     dga_config.box_sizes.wn,
                                     np.prod(dga_config.lattice._q_grid.nk), 0, logger)

del vrg_q_dens, vrg_q_magn, chi_lad_dens, chi_lad_magn, kernel_dc
gc.collect()

logger.log_cpu_time(task=' Non-local SDE done. ')
logger.log_memory_usage()

# %%
# Collect from the different cores:
siwk_dga = mpi_dist_fbz.allreduce(siwk_dga)

siwk_dga += mf.cut_v(dmft_input['siw'], dga_config.box_sizes.niv_core) - mf.cut_v(siw_sde_full, dga_config.box_sizes.niv_core)
hartree = twop.get_smom0(dmft_input['u'], dmft_input['n'])
siwk_dga += hartree
# %%
# Build the DGA Self-energy with Sigma_DMFT as asymptotic
sigma_dga = twop.create_dga_siwk_with_dmft_as_asympt(siwk_dga,siwk_dmft, dga_config.box_sizes.niv_shell)
if (comm.rank == 0):
    siwk_dga_shift = sigma_dga.get_siw(dga_config.box_sizes.niv_full, pi_shift=True)
    kx_shift = np.linspace(-np.pi, np.pi, dga_config.lattice.nk[0], endpoint=False)
    ky_shift = np.linspace(-np.pi, np.pi, dga_config.lattice.nk[1], endpoint=False)

    plotting.plot_kx_ky(siwk_dga_shift[..., 0, dga_config.box_sizes.niv_full], kx_shift, ky_shift, pdir=dga_config.output_path,
                        name='Siwk_dga_kz0')
    siw_dga_loc = np.mean(siwk_dga_shift, axis=(0, 1, 2))
    plotting.sigma_loc_checks([siw_sde_full, siw_dga_loc],
                              ['SDE-loc', 'DGA-loc', 'DC-loc', 'Dens-loc', 'Magn-loc'], dmft_input['beta'],
                              dga_config.output_path, verbose=False, do_plot=True, name='dga_loc', xmax=dga_config.box_sizes.niv_full)

    plotting.sigma_loc_checks([siw_sde_full, siw_dga_loc],
                              ['SDE-loc', 'DGA-loc', 'DC-loc', 'Dens-loc', 'Magn-loc'], dmft_input['beta'],
                              dga_config.output_path, verbose=False, do_plot=True, name='dga_loc_core', xmax=dga_config.box_sizes.niv_core)

    dga_config.save_data(sigma_dga.get_siw(dga_config.box_sizes.niv_full,pi_shift=False),'siwk_dga')

logger.log_cpu_time(task=' Siwk collected and plotted. ')
logger.log_memory_usage()

# %% Build the DGA Green's function:
giwk_dga = twop.GreensFunction(sigma_dga, ek, n=dmft_input['n'], niv_asympt=dga_config.box_sizes.niv_full)

if (comm.rank == 0):
    giwk_shift = giwk_dga.g_full(pi_shift=False)[..., 0, giwk_dga.niv_full][:dga_config.lattice.nk[0] // 2,
                 :dga_config.lattice.nk[1] // 2]
    fs_ind = bz.find_zeros(giwk_shift)
    n_fs = np.shape(fs_ind)[0]
    fs_ind = fs_ind[:n_fs // 2]

    fs_points = np.stack((dga_config.lattice._k_grid.kx[fs_ind[:, 0]], dga_config.lattice._k_grid.ky[fs_ind[:, 1]]), axis=1)
    plotting.plot_kx_ky(giwk_dga.g_full(pi_shift=True)[..., 0, giwk_dga.niv_full], kx_shift, ky_shift,
                        pdir=dga_config.output_path, name='Giwk_dga_kz0', scatter=fs_points)

    dga_config.save_data(giwk_dga.g_full(),'giwk_dga.npy')
    np.savetxt(dga_config.output_path + '/mu.txt', [[giwk_dga.mu, dmft_input['mu_dmft']]], header='mu_dga, mu_dmft', fmt='%1.6f')

    # Plots along Fermi-Luttinger surface:
    plotting.plot_along_ind(sigma_dga.get_siw(dga_config.box_sizes.niv_full), fs_ind, pdir=dga_config.output_path, niv_plot_min=0,
                            niv_plot=20, name='Sigma_{dga}')

logger.log_cpu_time(task=' Giwk build and plotted. ')
logger.log_memory_usage()
mpi_distributor.delete_file()
mpi_dist_fbz.delete_file()
logger.log_memory_usage()
# --------------------------------------------- POSTPROCESSING ----------------------------------------------------------

# %%
if (dga_config.do_poly_fitting):
    # ------------------------------------------------- POLYFIT ------------------------------------------------------------
    # Extrapolate the self-energy to the Fermi-level via polynomial fit:
    if comm.rank == 0: dga_io.poly_fit(sigma_dga.get_siw(dga_config.box_sizes.niv_full, pi_shift=True), dmft_input['beta'],
                                       dga_config.lattice._k_grid, dga_config.n_fit,
                                       dga_config.o_fit, name='Siwk_poly_cont_', output_path=poly_fit_dir)

    if comm.rank == 0:
        dga_io.poly_fit(giwk_dga.g_full(pi_shift=True), dmft_input['beta'], dga_config.lattice._k_grid, dga_config.n_fit,
                        dga_config.o_fit,
                        name='Giwk_dga_poly_cont_',
                        output_path=poly_fit_dir)

    logger.log_cpu_time(task=' Poly-fits ')

# --------------------------------------------- ANALYTIC CONTINUATION --------------------------------------------------
# %%
# Broadcast bw_opt_dga
if ('max_ent' in conf_file):
    if comm.rank == 0 and me_conf.cont_g_loc:
        g_loc_dga = giwk_dga.g_loc
        bw_opt_dga = dga_io.max_ent_loc_bw_range(g_loc_dga, me_conf, name='dga')

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
            dga_io.max_ent_irrk_bw_range_sigma(sigma_dga, dga_config.lattice._k_grid, me_conf, comm, bw, logger=logger,
                                               name='siwk_dga')
        logger.log_cpu_time(task=' MaxEnt Siwk ')
        logger.log_memory_usage()

    if me_conf.cont_g_nl:
        me_conf.output_path_nl_g = dga_config.output_path + '/MaxEntGiwk/'
        if (comm.rank == 0 and not os.path.exists(me_conf.output_path_nl_g)): os.mkdir(me_conf.output_path_nl_g)

        for bw in me_conf.bw_dga:
            dga_io.max_ent_irrk_bw_range_green(giwk_dga, dga_config.lattice._k_grid, me_conf, comm, bw, logger=logger,
                                               name='Giwk_dga')
        logger.log_cpu_time(task=' MaxEnt Giwk ')
        logger.log_memory_usage()

#%%Perform the Eliashberg Routine:
if(comm.rank == 0 and 'pairing' in conf_file):
    gk_dga = mf.cut_v(giwk_dga.core,dga_config.box_sizes.niv_pp,(-1,))
    norm = np.prod(dga_config.lattice._q_grid.nk_tot) * dmft_input['beta']
    gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=pairing_config.gap0_sing['k'], v_type=pairing_config.gap0_sing['v'],
                            k_grid=dga_config.lattice._q_grid.grid)

    gamma_sing = -pairing_config.load_data('F_sing_pp')
    if(pairing_config.sym_sing): eq.symmetrize_gamma(gamma_sing,'sing')
    powiter_sing = eq.EliashberPowerIteration(gamma=gamma_sing, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
                                              n_eig=pairing_config.n_eig)

    gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=pairing_config.gap0_trip['k'], v_type=pairing_config.gap0_trip['v'],
                            k_grid=dga_config.lattice._q_grid.grid)

    gamma_trip = -pairing_config.load_data('F_trip_pp')
    if (pairing_config.sym_trip): eq.symmetrize_gamma(gamma_sing, 'trip')
    powiter_trip = eq.EliashberPowerIteration(gamma=gamma_trip, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
                                              n_eig=pairing_config.n_eig)

    eliashberg = {
        'lambda_sing': powiter_sing.lam,
        'lambda_trip': powiter_trip.lam,
        'delta_sing': powiter_sing.gap,
        'delta_trip': powiter_trip.gap,
    }
    pairing_config.save_data(eliashberg,'eliashberg')
    np.savetxt(pairing_config.output_path + 'eigenvalues.txt', [powiter_sing.lam.real, powiter_trip.lam.real], delimiter=',', fmt='%.9f')
    np.savetxt(pairing_config.output_path + 'eigenvalues_s.txt', [powiter_sing.lam_s.real, powiter_trip.lam_s.real],
               delimiter=',', fmt='%.9f')

    for i in range(len(powiter_sing.gap)):
        plotting.plot_gap_function(delta=powiter_sing.gap[i].real, pdir=pairing_config.output_path, name='sing_{}'.format(i),
                                   kgrid=dga_config.lattice._q_grid,do_shift=True)
        plotting.plot_gap_function(delta=powiter_sing.gap[i].real, pdir=pairing_config.output_path, name='trip_{}'.format(i),
                                   kgrid=dga_config.lattice._q_grid,do_shift=True)

    logger.log_event('Eliashberg completed!')

# End program:
logger.log_event('Completed DGA run!')
comm.Barrier()
mpi.Finalize()
