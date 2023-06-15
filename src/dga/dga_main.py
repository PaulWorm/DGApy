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

import dga.config as config
import dga.matsubara_frequencies as mf
import dga.brillouin_zone as bz
import dga.loggers as loggers
import dga.io as io
import dga.four_point as fp
import dga.local_four_point as lfp
import dga.plotting as plotting
import dga.lambda_correction as lc
import dga.two_point as twop
import dga.bubble as bub
import dga.hk as hamk
import dga.mpi_aux as mpi_aux

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
dmft_input = io.load_1p_data(dga_config.input_type, dga_config.input_path, dga_config.fname_1p, dga_config.fname_2p)
g2_dens, g2_magn = io.load_g2(dga_config.box_sizes,dmft_input)

# ------------------------------------------- DEFINE THE OUTPUT DIRECTORY --------------------------------------------------------
# Define output directory:
output_dir = dga_config.input_path + '/LambdaDga_lc_{}_Nk{}_Nq{}_wcore{}_vcore{}_vshell{}'.format(lambda_correction_type,
                                                                                                 dga_config.lattice.nk_tot,
                                                                                                 dga_config.lattice.nq_tot,
                                                                                                 dga_config.box_sizes.niw_core,
                                                                                                 dga_config.box_sizes.niv_core,
                                                                                                 dga_config.box_sizes.niv_shell)

# %%
hr = dga_config.lattice.set_hr()

# Create output directory:
output_dir = io.uniquify(output_dir)
dga_config.output_path = output_dir
comm.barrier()
if (comm.rank == 0): os.mkdir(output_dir)
comm.barrier()
logger = loggers.MpiLogger(logfile=output_dir + '/dga.log', comm=comm, output_path=output_dir)
logger.log_message(f'Running on {comm.size} threads.')
logger.log_event(message=' Config Init and folder set up done!')
# Save the full config to yaml file:
if(comm.rank == 0):
    with open(dga_config.output_path + "/config.yaml", "w+") as file:
        yaml = YAML()
        yaml.dump(dga_config.as_dict(), file)
        file.close()


if (comm.rank == 0): plotting.default_g2_plots(g2_dens,g2_magn,output_dir)

# Build Green's function and susceptibility:
ek = hamk.ek_3d(dga_config.lattice._k_grid.grid, hr)

siwk_dmft = twop.SelfEnergy(sigma=dmft_input['siw'][None, None, None, :], beta=dmft_input['beta'])
giwk_dmft = twop.GreensFunction(siwk_dmft, ek, mu=dmft_input['mu_dmft'], niv_asympt=dga_config.box_sizes.niv_asympt)

gchi_dens = lfp.gchir_from_g2(g2_dens, giwk_dmft.g_loc)
gchi_magn = lfp.gchir_from_g2(g2_magn, giwk_dmft.g_loc)

logger.log_cpu_time(task=' Data loading completed. ')

# ------------------------------------------- Extract the irreducible Vertex --------------------------------------------------------
# Create Bubble generator:
bubble_gen = bub.LocalBubble(wn=g2_dens.wn, giw=giwk_dmft)
gchi0_core = bubble_gen.get_gchi0(dga_config.box_sizes.niv_core)
gchi0_urange = bubble_gen.get_gchi0(dga_config.box_sizes.niv_full)
gamma_dens = lfp.gammar_from_gchir(gchi_dens, gchi0_urange, dmft_input['u'])
gamma_magn = lfp.gammar_from_gchir(gchi_magn, gchi0_urange, dmft_input['u'])
logger.log_cpu_time(task=' Gamma-loc finished. ')

if (comm.rank == 0):
    plotting.default_gamma_plots(gamma_dens,gamma_magn,output_dir,dga_config.box_sizes,dmft_input['beta'])
    np.save(output_dir + '/gamma_dens.npy', gamma_dens.mat)
    np.save(output_dir + '/gamma_magn.npy', gamma_magn.mat)

logger.log_cpu_time(task=' Gamma-loc plotting finished. ')

# ----------------------------------- Compute the Susceptibility and Threeleg Vertex --------------------------------------------------------
vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_dens, bubble_gen, dmft_input['u'],
                                                                    niv_shell=dga_config.box_sizes.niv_shell)
vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, bubble_gen, dmft_input['u'],
                                                                    niv_shell=dga_config.box_sizes.niv_shell)

# Create checks of the susceptibility:
if (comm.rank == 0): plotting.chi_checks([chi_dens, ], [chi_magn, ], ['Loc-tilde', ], giwk_dmft, output_dir, verbose=False,
                                         do_plot=True, name='loc')

logger.log_cpu_time(task=' Vrg and Chi-phys loc done. ')
# --------------------------------------- Local Schwinger-Dyson equation --------------------------------------------------------

# Perform the local SDE for box-size checks:
siw_sde_full = lfp.schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, giwk_dmft.g_loc,
                                        dmft_input['u'],
                                        dmft_input['n'],
                                        niv_shell=dga_config.box_sizes.niv_shell)

# Create checks of the self-energy:
if (comm.rank == 0):
    plotting.sigma_loc_checks([siw_sde_full, dmft_input['siw']], ['SDE', 'Input'], dmft_input['beta'], output_dir, verbose=False,
                              do_plot=True, xmax=dga_config.box_sizes.niv_full)
    np.save(output_dir + '/siw_sde_loc.npy', siw_sde_full)
    np.save(output_dir + '/chi_dens_loc.npy', chi_dens)
    np.save(output_dir + '/chi_magn_loc.npy', chi_magn)

logger.log_cpu_time(task=' Local SDE finished. ')

# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- NON-LOCAL PART --------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# Split the momenta between the different cores.
mpi_distributor = mpi_aux.MpiDistributor(ntasks=dga_config.lattice._q_grid.nk_irr, comm=comm, output_path=dga_config.output_path,
                                         name='Q')
my_q_list = dga_config.lattice._q_grid.irrk_mesh_ind.T[mpi_distributor.my_slice]
# %%
# F_dc = lfp.Fob2_from_chir(gchi_magn,gchi0_core)
F_dc = lfp.Fob2_from_gamob2_urange(gamma_magn, gchi0_urange, dmft_input['u'])
vrg_q_dens, vrg_q_magn, chi_lad_dens, chi_lad_magn, kernel_dc = fp.get_vrg_and_chir_lad_from_gammar_uasympt_q(gamma_dens,
                                                                                                              gamma_magn,
                                                                                                              F_dc,
                                                                                                              vrg_magn,
                                                                                                              chi_magn,
                                                                                                              bubble_gen,
                                                                                                              dmft_input['u'],
                                                                                                              my_q_list,
                                                                                                              niv_shell=dga_config.box_sizes.niv_shell,
                                                                                                              logger=logger)

logger.log_cpu_time(task=' Vrg and Chi-ladder completed. ')
# %% Collect the results from the different cores:
chi_lad_dens = mpi_distributor.allgather(rank_result=chi_lad_dens)
chi_lad_magn = mpi_distributor.allgather(rank_result=chi_lad_magn)

# Build the full Brillouin zone, which is needed for the lambda-correction:
chi_lad_dens = dga_config.lattice._q_grid.map_irrk2fbz(chi_lad_dens)
chi_lad_magn = dga_config.lattice._q_grid.map_irrk2fbz(chi_lad_magn)
# %%

if (comm.rank == 0):
    chi_lad_dens_loc = dga_config.lattice._q_grid.k_mean(chi_lad_dens, type='fbz-mesh')
    chi_lad_magn_loc = dga_config.lattice._q_grid.k_mean(chi_lad_magn, type='fbz-mesh')
    plotting.chi_checks([chi_dens, chi_lad_dens_loc], [chi_magn, chi_lad_magn_loc], ['loc', 'ladder-sum'], giwk_dmft, output_dir,
                        verbose=False,
                        do_plot=True, name='lad_q_tilde')

    plotting.plot_kx_ky(chi_lad_dens[..., 0, dga_config.box_sizes.niw_core], dga_config.lattice._q_grid.kx,
                        dga_config.lattice._q_grid.ky, pdir=output_dir, name='Chi_ladder_dens_kz0')
    plotting.plot_kx_ky(chi_lad_magn[..., 0, dga_config.box_sizes.niw_core], dga_config.lattice._q_grid.kx,
                        dga_config.lattice._q_grid.ky, pdir=output_dir, name='Chi_ladder_magn_kz0')

# %% Lambda-corrected susceptibility:

chi_lad_dens, chi_lad_magn, lambda_dens, lambda_magn = lc.lambda_correction(chi_lad_dens, chi_lad_magn, dmft_input['beta'],
                                                                            chi_dens, chi_magn,
                                                                            type=dga_config.lambda_corr)

# %%
if (comm.rank == 0):
    chi_lad_dens_loc = dga_config.lattice._q_grid.k_mean(chi_lad_dens, type='fbz-mesh')
    chi_lad_magn_loc = dga_config.lattice._q_grid.k_mean(chi_lad_magn, type='fbz-mesh')
    plotting.chi_checks([chi_lad_dens_loc, chi_dens], [chi_lad_magn_loc, chi_magn], ['lam-loc', 'loc'], giwk_dmft, output_dir,
                        verbose=False,
                        do_plot=True, name='q_lam')

    plotting.plot_kx_ky(chi_lad_dens[..., 0, dga_config.box_sizes.niw_core], dga_config.lattice._q_grid.kx,
                        dga_config.lattice._q_grid.ky, pdir=output_dir, name='Chi_lam_dens_kz0')
    plotting.plot_kx_ky(chi_lad_magn[..., 0, dga_config.box_sizes.niw_core], dga_config.lattice._q_grid.kx,
                        dga_config.lattice._q_grid.ky, pdir=output_dir, name='Chi_lam_magn_kz0')

    np.save(output_dir + '/chi_dens_lam.npy', chi_lad_dens)
    np.save(output_dir + '/chi_magn_lam.npy', chi_lad_magn)
    np.savetxt(output_dir + '/lambda.txt', [[lambda_dens, lambda_magn]], header='dens magn', fmt='%.6f')

logger.log_cpu_time(task=' Lambda-correction done. ')

# %% Perform Ornstein-zernicke fitting of the susceptibility:
if (comm.rank == 0): io.fit_and_plot_oz(output_dir, dga_config.lattice._q_grid)

# %% Non-local Schwinger-Dyson equation:
# collect vrg vertices:
vrg_q_dens = mpi_distributor.allgather(rank_result=vrg_q_dens)
vrg_q_magn = mpi_distributor.allgather(rank_result=vrg_q_magn)

vrg_q_dens = dga_config.lattice._q_grid.map_irrk2fbz(vrg_q_dens, shape='list')
vrg_q_magn = dga_config.lattice._q_grid.map_irrk2fbz(vrg_q_magn, shape='list')

# %%
if (comm.rank == 0):
    plotting.default_vrg_plots(vrg_q_dens,vrg_q_magn,vrg_dens, vrg_magn, dga_config)

# %%
# Distribute the full fbz q-points to the different cores:
full_q_list = np.array([dga_config.lattice._q_grid.kmesh_ind[i].flatten() for i in range(3)]).T
q_dupl = np.ones((np.shape(full_q_list)[0]))

# Map object to core q-list:
chi_lad_dens = dga_config.lattice._q_grid.map_fbz_mesh2list(chi_lad_dens)
chi_lad_magn = dga_config.lattice._q_grid.map_fbz_mesh2list(chi_lad_magn)

# Create new mpi-distributor for the full fbz:
mpi_dist_fbz = mpi_aux.MpiDistributor(ntasks=dga_config.lattice._q_grid.nk_tot, comm=comm, output_path=output_dir + '/', name='FBZ')
my_full_q_list = full_q_list[mpi_dist_fbz.my_slice]
chi_lad_dens = chi_lad_dens[mpi_dist_fbz.my_slice, ...]
chi_lad_magn = chi_lad_magn[mpi_dist_fbz.my_slice, ...]

vrg_q_dens = vrg_q_dens[mpi_dist_fbz.my_slice, ...]
vrg_q_magn = vrg_q_magn[mpi_dist_fbz.my_slice, ...]

# %%
kernel_dc = mpi_distributor.allgather(kernel_dc)
kernel_dc = dga_config.lattice._q_grid.map_irrk2fbz(kernel_dc, 'list')
if(comm.rank == 0):
    kernel_dc_mesh = dga_config.lattice._q_grid.map_fbz_list2mesh(kernel_dc)
    plotting.plot_kx_ky(kernel_dc_mesh[:,:,0,dga_config.box_sizes.niw_core,dga_config.box_sizes.niv_core],
                        dga_config.lattice._q_grid.kx,dga_config.lattice._q_grid.ky,
                        pdir=dga_config.output_path, name='dc_kernel')

kernel_dc = kernel_dc[mpi_dist_fbz.my_slice, ...]
print(f'Rank {comm.rank} is doing {mpi_dist_fbz.my_slice}')
siwk_dga = fp.schwinger_dyson_full_q(vrg_q_dens, vrg_q_magn, chi_lad_dens, chi_lad_magn, kernel_dc,
                                     giwk_dmft.g_full(), dmft_input['beta'], dmft_input['u'], my_full_q_list, g2_magn.wn,
                                     np.prod(dga_config.lattice._q_grid.nk), 0, logger)

logger.log_cpu_time(task=' Non-local SDE done. ')

# siw_dga = dga_config.lattice._q_grid.k_mean(siwk_dga,type='fbz-mesh')

# plotting.sigma_loc_checks([siw_dga],
#                           ['DGA-loc',], dmft_input['beta'],
#                           output_dir, verbose=False, do_plot=True, name=f'dga_loc_rank_{comm.rank}',
#                           xmax=dga_config.box_sizes.niv_core)
# %%
# Collect from the different cores:
siwk_dga = mpi_dist_fbz.allreduce(siwk_dga)

siwk_dga += mf.cut_v(dmft_input['siw'], dga_config.box_sizes.niv_core) - mf.cut_v(siw_sde_full, dga_config.box_sizes.niv_core)
hartree = twop.get_smom0(dmft_input['u'], dmft_input['n'])
siwk_dga += hartree
# %%

siwk_shell = siwk_dmft.get_siw(dga_config.box_sizes.niv_full)
siwk_shell = np.ones(dga_config.lattice.nk)[:, :, :, None] * \
             mf.inv_cut_v(siwk_shell, niv_core=dga_config.box_sizes.niv_core, niv_shell=dga_config.box_sizes.niv_shell,
                          axes=-1)[0, 0, 0, :][None, None, None, :]
siwk_dga = mf.concatenate_core_asmypt(siwk_dga, siwk_shell)
siwk_dga_shift = twop.SelfEnergy(siwk_dga, dmft_input['beta']).get_siw(dga_config.box_sizes.niv_full, pi_shift=True)

if (comm.rank == 0):
    kx_shift = np.linspace(-np.pi, np.pi, dga_config.lattice.nk[0], endpoint=False)
    ky_shift = np.linspace(-np.pi, np.pi, dga_config.lattice.nk[1], endpoint=False)

    plotting.plot_kx_ky(siwk_dga_shift[..., 0, dga_config.box_sizes.niv_full], kx_shift, ky_shift, pdir=output_dir,
                        name='Siwk_dga_kz0')
    siw_dga_loc = np.mean(siwk_dga_shift, axis=(0, 1, 2))
    plotting.sigma_loc_checks([siw_sde_full, siw_dga_loc],
                              ['SDE-loc', 'DGA-loc', 'DC-loc', 'Dens-loc', 'Magn-loc'], dmft_input['beta'],
                              output_dir, verbose=False, do_plot=True, name='dga_loc', xmax=dga_config.box_sizes.niv_full)

    plotting.sigma_loc_checks([siw_sde_full, siw_dga_loc],
                              ['SDE-loc', 'DGA-loc', 'DC-loc', 'Dens-loc', 'Magn-loc'], dmft_input['beta'],
                              output_dir, verbose=False, do_plot=True, name='dga_loc_core', xmax=dga_config.box_sizes.niv_core)

    np.save(output_dir + '/siwk_dga.npy', siwk_dga)

logger.log_cpu_time(task=' Siwk collected and plotted. ')
# %% Build the DGA Green's function:
sigma_dga = twop.SelfEnergy(siwk_dga, dmft_input['beta'])
giwk_dga = twop.GreensFunction(sigma_dga, ek, n=dmft_input['n'], niv_asympt=dga_config.box_sizes.niv_full * 2)
if (comm.rank == 0):
    giwk_shift = giwk_dga.g_full(pi_shift=False)[..., 0, giwk_dga.niv_full][:dga_config.lattice.nk[0] // 2, :dga_config.lattice.nk[1] // 2]
    fs_ind = bz.find_zeros(giwk_shift)
    n_fs = np.shape(fs_ind)[0]
    fs_ind = fs_ind[:n_fs // 2]

    fs_points = np.stack((dga_config.lattice._k_grid.kx[fs_ind[:, 0]], dga_config.lattice._k_grid.ky[fs_ind[:, 1]]), axis=1)
    plotting.plot_kx_ky(giwk_dga.g_full(pi_shift=True)[..., 0, giwk_dga.niv_full], kx_shift, ky_shift,
                        pdir=output_dir, name='Giwk_dga_kz0', scatter=fs_points)

    np.save(output_dir + '/giwk_dga.npy', giwk_dga.g_full())
    np.savetxt(output_dir + '/mu.txt', [[giwk_dga.mu, dmft_input['mu_dmft']]], header='mu_dga, mu_dmft', fmt='%1.6f')

    # Plots along Fermi-Luttinger surface:
    siwk_dga = sigma_dga.get_siw(dga_config.box_sizes.niv_full)
    plotting.plot_along_ind(siwk_dga, fs_ind, pdir=output_dir, niv_plot_min=0, niv_plot=20, name='Sigma_{dga}')

logger.log_cpu_time(task=' Giwk build and plotted. ')
mpi_distributor.delete_file()
mpi_dist_fbz.delete_file()
# --------------------------------------------- POSTPROCESSING ----------------------------------------------------------

# %%
if (dga_config.do_poly_fitting):
    poly_fit_dir = output_dir + '/PolyFits/'
    if (comm.rank == 0): os.mkdir(poly_fit_dir)
    # ------------------------------------------------- POLYFIT ------------------------------------------------------------
    # Extrapolate the self-energy to the Fermi-level via polynomial fit:
    if comm.rank == 0: io.poly_fit(sigma_dga.get_siw(dga_config.box_sizes.niv_full, pi_shift=True), dmft_input['beta'],
                                   dga_config.lattice._k_grid, dga_config.n_fit,
                                   dga_config.o_fit, name='Siwk_poly_cont_', output_path=poly_fit_dir)

    if comm.rank == 0:
        io.poly_fit(giwk_dga.g_full(pi_shift=True), dmft_input['beta'], dga_config.lattice._k_grid, dga_config.n_fit, dga_config.o_fit,
                    name='Giwk_dga_poly_cont_',
                    output_path=poly_fit_dir)

    if comm.rank == 0:
        io.poly_fit(giwk_dmft.g_full(pi_shift=True), dmft_input['beta'], dga_config.lattice._k_grid, dga_config.n_fit, dga_config.o_fit,
                    name='Giwk_dmft_poly_cont_',
                    output_path=poly_fit_dir)

    logger.log_cpu_time(task=' Poly-fits ')

# --------------------------------------------- ANALYTIC CONTINUATION --------------------------------------------------
# %%
# Broadcast bw_opt_dga
comm.Barrier()
if ('max_ent' in conf_file):
    max_ent_dir = output_dir + '/MaxEnt/'
    if (comm.rank == 0 and not os.path.exists(max_ent_dir)): os.mkdir(max_ent_dir)
    me_conf = config.MaxEntConfig(t=hr[0,0],beta=dmft_input['beta'],config_dict=conf_file['max_ent'],
                                  output_path_loc=max_ent_dir)

    if comm.rank == 0 and me_conf.cont_g_loc:
        g_loc_dmft = giwk_dmft.g_loc
        io.max_ent_loc_bw_range(g_loc_dmft,me_conf, name='dmft')

        g_loc_dga  = giwk_dga.g_loc
        me_conf.bw_dga.append(io.max_ent_loc_bw_range(g_loc_dga,me_conf, name='dga'))
        logger.log_cpu_time(task=' MaxEnt local ')
#%%
if ('max_ent' in conf_file):
    comm.barrier()
    if me_conf.cont_s_nl:
        me_conf.output_path_nl_s = output_dir + '/MaxEntSiwk/'
        if (comm.rank == 0 and not os.path.exists(me_conf.output_path_nl_s)): os.mkdir(me_conf.output_path_nl_s)

        for bw in me_conf.bw_dga:
            io.max_ent_irrk_bw_range_sigma(sigma_dga, dga_config.lattice._k_grid, me_conf, comm, bw, logger=logger,
                                           name='siwk_dga')
#%%
if ('max_ent' in conf_file):
    comm.barrier()
    if me_conf.cont_g_nl:
        me_conf.output_path_nl_g = output_dir + '/MaxEntGiwk/'
        if (comm.rank == 0 and not os.path.exists(me_conf.output_path_nl_g)): os.mkdir(me_conf.output_path_nl_g)

        for bw in me_conf.bw_dga:
            io.max_ent_irrk_bw_range_green(giwk_dga, dga_config.lattice._k_grid, me_conf, comm, bw, logger=logger,
                                           name='Giwk_dga')


# End program:
logger.log_event('Completed DGA run!')
comm.Barrier()
mpi.Finalize()
