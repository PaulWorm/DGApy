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
import dga.analytic_continuation as a_cont
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
import dga.wannier as wannier
import dga.mpi_aux as mpi_aux
import dga.pairing_vertex as pv
import dga.eliashberg_equation as eq
import dga.util as util
import dga.high_level_routines as hlr

import dga.plot_specs

# Define MPI communicator:
comm = mpi.COMM_WORLD

def get_largest_vars(num=5):
    mem = []
    names = []
    vs = globals()
    for k, v in vs.items():
        # mem.append(v.size * v.itemsize * 1e-6)
        mem.append(sys.getsizeof(v)  * 1e-6)
        names.append(k)
    mem = np.array(mem)
    names = np.array(names)
    sort_ind = np.argsort(mem, axis=0)[::-1]
    mem[:] = mem[sort_ind]
    names[:] = names[sort_ind]
    n = min(len(mem), num)
    string =[f'{i}: {names[i]} uses {mem[i]} MB |' for i in range(n)]
    string = f'Total consumption is : {np.sum(mem)} MB |' + ' '.join(string)
    return string

# --------------------------------------------- CONFIGURATION ----------------------------------------------------------

dga_config, conf_file = config.parse_config_file(comm=comm)

if(comm.size > dga_config.lattice.k_grid.nk_irr):
    raise ValueError('Number of processes may not be larger than points in the irreducible BZ for distribution.')

# %%
# --------------------------------------------------- LOAD THE INPUT -------------------------------------------------------------
dmft_input = dga_io.load_1p_data(dga_config.input_type, dga_config.input_path, dga_config.fname_1p, dga_config.fname_2p)
g2_dens, g2_magn = dga_io.load_g2(dga_config.box_sizes, dmft_input)

# ------------------------------------------- DEFINE THE OUTPUT DIRECTORY --------------------------------------------------------
comm.barrier()
dga_config.create_dga_ouput_folder(comm=comm)
dga_config.eliash.set_output_path(dga_config.output_path + 'Eliashberg/', comm)
comm.barrier()

# Create the DGA logger:
logger = loggers.MpiLogger(logfile=dga_config.output_path + '/dga.log', comm=comm, output_path=dga_config.output_path)
logger.log_message(f'Running on {comm.size} threads.')
logger.log_memory_usage()
#print(get_largest_vars())
logger.log_event(message=' Config Init and folder set up done!')
comm.Barrier()

# Save dmft input to output folder
if (comm.rank == 0): dga_config.save_data(dmft_input, 'dmft_input')

if (comm.rank == 0): dga_config.lattice.hr.save_hr(dga_config.output_path)

# Save the full config to yaml file:
if (comm.rank == 0): config.save_config_file(conf_file, dga_config.output_path)

# Plot standard sanity plots for the two-particle Green's function:
if (comm.rank == 0): plotting.default_g2_plots(g2_dens, g2_magn, dga_config.output_path)

# Build the DMFT Green's function:
ek = dga_config.lattice.hr.get_ek_one_band(dga_config.lattice.k_grid)

siwk_dmft = twop.SelfEnergy(sigma=dmft_input['siw'][None, None, None, :], beta=dmft_input['beta'])
giwk_dmft = twop.GreensFunction(siwk_dmft, ek, mu=dmft_input['mu_dmft'], niv_asympt=dga_config.box_sizes.niv_asympt)

# Perform the poly-fits for the DMFT Green's function
dga_config.create_poly_fit_folder(comm=comm)

if comm.rank == 0 and dga_config.do_poly_fitting:
    dga_io.poly_fit(giwk_dmft.g_full(pi_shift=True), dmft_input['beta'], dga_config.lattice.k_grid, dga_config.n_fit,
                    dga_config.o_fit,
                    name='Giwk_dmft_poly_cont_',
                    output_path=dga_config.poly_fit_dir)
    gamma_dmft, bandshift_dmft, Z_dmft = a_cont.get_gamma_bandshift_Z(mf.vn(dmft_input['beta'], siwk_dmft.sigma[0, 0, 0, :],
                                                                            pos=True),
                                                                      siwk_dmft.sigma[0, 0, 0, :],
                                                                      order=dga_config.o_fit, N=dga_config.n_fit)
    np.savetxt(dga_config.poly_fit_dir + 'Siw_DMFT_polyfit.txt', np.array([[bandshift_dmft, ], [gamma_dmft, ], [Z_dmft, ]]).T,
               header='bandshift gamma Z', fmt='%.6f')

# --------------------------------------------- LOCAL PART --------------------------------------------------------
gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, siw_sde_full = hlr.local_sde_from_g2(g2_dens, g2_magn, giwk_dmft,
                                                                                                     dga_config, dmft_input, comm,
                                                                                                     logger=logger)
logger.log_memory_usage()
#print(get_largest_vars())
comm.Barrier()
# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- NON-LOCAL PART --------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# Create the Bubble generator:
bubble_gen = bub.BubbleGenerator(wn=dga_config.box_sizes.wn, giw=giwk_dmft)
gchi0_urange = bubble_gen.get_gchi0(dga_config.box_sizes.niv_full)

# Split the momenta between the different cores.
if (dga_config.eliash.do_pairing_vertex):
    mpi_distributor = mpi_aux.MpiDistributor(ntasks=dga_config.lattice.q_grid.nk_irr, comm=comm,
                                             output_path=dga_config.output_path,
                                             name='Q')
else:
    mpi_distributor = mpi_aux.MpiDistributor(ntasks=dga_config.lattice.q_grid.nk_irr, comm=comm,
                                             name='Q')
comm.Barrier()
my_q_list = dga_config.lattice.q_grid.irrk_mesh_ind.T[mpi_distributor.my_slice]
# %%
F_dc = lfp.Fob2_from_gamob2_urange(gamma_magn, gchi0_urange, dmft_input['u'])
del gchi0_urange
gc.collect()
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
if (logger is not None): logger.log_cpu_time(task=' Bubbles constructed. ')

# double-counting kernel:
if (logger is not None):
    if (logger.is_root):
        F_dc.plot(pdir=logger.out_dir + '/', name='F_dc')

kernel_dc = mf.cut_v(fp.get_kernel_dc(F_dc.mat, gchi0_q_urange, u, 'magn'), niv_core, axes=(-1,))
print(get_largest_vars(5))
del gchi0_q_urange
gc.collect()
if (logger is not None): logger.log_cpu_time(task=' DC kernel constructed. ')

# Density channel:
gchiq_aux = fp.get_gchir_aux_from_gammar_q(gamma_dens, gchi0_q_core, u)
chiq_aux = 1 / beta ** 2 * np.sum(gchiq_aux, axis=(-1, -2))
chi_lad_urange = fp.chi_phys_from_chi_aux_q(chiq_aux, chi0_q_urange, chi0_q_core, u, gamma_dens.channel)
chi_lad_dens = fp.chi_phys_asympt_q(chi_lad_urange, chi0_q_urange, chi0_q_urange + chi0q_shell)

vrg_q_dens = fp.vrg_from_gchi_aux(gchiq_aux, gchi0_q_core, chi_lad_urange, chi_lad_dens, u, gamma_dens.channel)

if (dga_config.eliash.do_pairing_vertex):
    niv_pp = dga_config.box_sizes.niv_pp
    omega = pv.get_omega_condition(niv_pp=niv_pp)
    channel = 'dens'
    mpi_distributor.open_file()
    for i, iq in enumerate(mpi_distributor.my_tasks):
        for j, iw in enumerate(dga_config.box_sizes.wn):
            if np.abs(iw) < 2 * niv_pp:
                condition = omega == iw

                f1_slice, f2_slice = pv.ladder_vertex_from_chi_aux_components(gchi_aux=gchiq_aux[i, j],
                                                                              vrg=vrg_q_dens[i, j],
                                                                              gchi0=gchi0_q_core[i, j],
                                                                              beta=beta,
                                                                              u_r=fp.get_ur(u, channel))
                group = '/irrq{:03d}wn{:04d}/'.format(iq, iw)

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
kernel_dc += u_r / gamma_magn.beta * (1 - u_r * chi_magn[None, :, None]) * vrg_magn.mat[None, :, :] * chi0q_shell[
                                                                                                      :, :, None]
vrg_q_magn = fp.vrg_from_gchi_aux(gchiq_aux, gchi0_q_core, chi_lad_urange, chi_lad_magn, u, gamma_magn.channel)

if (dga_config.eliash.do_pairing_vertex):
    niv_pp = dga_config.box_sizes.niv_pp
    omega = pv.get_omega_condition(niv_pp=niv_pp)
    channel = 'magn'
    mpi_distributor.open_file()
    for i, iq in enumerate(mpi_distributor.my_tasks):
        for j, iw in enumerate(dga_config.box_sizes.wn):
            if np.abs(iw) < 2 * niv_pp:
                condition = omega == iw

                f1_slice, f2_slice = pv.ladder_vertex_from_chi_aux_components(gchi_aux=gchiq_aux[i, j],
                                                                              vrg=vrg_q_magn[i, j],
                                                                              gchi0=gchi0_q_core[i, j],
                                                                              beta=beta,
                                                                              u_r=fp.get_ur(u, channel))
                group = '/irrq{:03d}wn{:04d}/'.format(iq, iw)

                mpi_distributor.file[group + f'f1_{channel}/'] = pv.get_pp_slice_4pt(mat=f1_slice, condition=condition,
                                                                                     niv_pp=niv_pp)
                mpi_distributor.file[group + f'f2_{channel}/'] = pv.get_pp_slice_4pt(mat=f2_slice, condition=condition,
                                                                                     niv_pp=niv_pp)

    mpi_distributor.close_file()

del gamma_magn, gamma_dens, F_dc, vrg_magn, vrg_dens, gchiq_aux, chiq_aux
gc.collect()
logger.log_cpu_time(task=' Vrg and Chi-ladder completed. ')
logger.log_memory_usage()
#print(get_largest_vars())

# %% Collect the results from the different cores:
chi_lad_dens = mpi_distributor.gather(rank_result=chi_lad_dens, root=0)
chi_lad_magn = mpi_distributor.gather(rank_result=chi_lad_magn, root=0)

# Build the full Brillouin zone, which is needed for the lambda-correction:
if (comm.rank == 0):
    chi_lad_dens = dga_config.lattice.q_grid.map_irrk2fbz(chi_lad_dens)
    chi_lad_magn = dga_config.lattice.q_grid.map_irrk2fbz(chi_lad_magn)
# %%

if (comm.rank == 0):
    chi_lad_dens_loc = dga_config.lattice.q_grid.k_mean(chi_lad_dens, type='fbz-mesh')
    chi_lad_magn_loc = dga_config.lattice.q_grid.k_mean(chi_lad_magn, type='fbz-mesh')
    plotting.chi_checks([chi_dens, chi_lad_dens_loc], [chi_magn, chi_lad_magn_loc], ['loc', 'ladder-sum'], giwk_dmft,
                        dga_config.output_path,
                        verbose=False,
                        do_plot=True, name='lad_q_tilde')

    plotting.plot_kx_ky(chi_lad_dens[..., 0, dga_config.box_sizes.niw_core], dga_config.lattice.q_grid.kx,
                        dga_config.lattice.q_grid.ky, pdir=dga_config.output_path, name='Chi_ladder_dens_kz0')
    plotting.plot_kx_ky(chi_lad_magn[..., 0, dga_config.box_sizes.niw_core], dga_config.lattice.q_grid.kx,
                        dga_config.lattice.q_grid.ky, pdir=dga_config.output_path, name='Chi_ladder_magn_kz0')

# %% Lambda-corrected susceptibility:

if (comm.rank == 0):
    chi_lad_dens, chi_lad_magn, lambda_dens, lambda_magn = lc.lambda_correction(chi_lad_dens, chi_lad_magn, dmft_input['beta'],
                                                                                chi_dens, chi_magn,
                                                                                type=dga_config.lambda_corr)

# %%
if (comm.rank == 0):
    chi_lad_dens_loc = dga_config.lattice.q_grid.k_mean(chi_lad_dens, type='fbz-mesh')
    chi_lad_magn_loc = dga_config.lattice.q_grid.k_mean(chi_lad_magn, type='fbz-mesh')
    plotting.chi_checks([chi_lad_dens_loc, chi_dens], [chi_lad_magn_loc, chi_magn], ['lam-loc', 'loc'], giwk_dmft,
                        dga_config.output_path,
                        verbose=False,
                        do_plot=True, name='q_lam')

    plotting.plot_kx_ky(chi_lad_dens[..., 0, dga_config.box_sizes.niw_core], dga_config.lattice.q_grid.kx,
                        dga_config.lattice.q_grid.ky, pdir=dga_config.output_path, name='Chi_lam_dens_kz0')
    plotting.plot_kx_ky(chi_lad_magn[..., 0, dga_config.box_sizes.niw_core], dga_config.lattice.q_grid.kx,
                        dga_config.lattice.q_grid.ky, pdir=dga_config.output_path, name='Chi_lam_magn_kz0')

    dga_config.save_data(chi_lad_dens, 'chi_dens_lam')
    dga_config.save_data(chi_lad_magn, 'chi_magn_lam')
    np.savetxt(dga_config.output_path + '/lambda.txt', [[lambda_dens, lambda_magn]], header='dens magn', fmt='%.6f')

del chi_dens, chi_magn
gc.collect()
logger.log_cpu_time(task=' Lambda-correction done. ')
logger.log_memory_usage()
#print(get_largest_vars())

# %% Build the pairing vertex
comm.Barrier()
if (comm.rank == 0 and dga_config.eliash.do_pairing_vertex):
    f1_magn, f2_magn, f1_dens, f2_dens = pv.load_pairing_vertex_from_rank_files(output_path=dga_config.output_path, name='Q',
                                                                                mpi_size=comm.size,
                                                                                nq=dga_config.lattice.q_grid.nk_irr,
                                                                                niv_pp=dga_config.box_sizes.niv_pp)
    chi_magn_lambda_pp = pv.reshape_chi(chi_lad_magn, dga_config.box_sizes.niv_pp)
    f1_magn = dga_config.lattice.q_grid.map_irrk2fbz(f1_magn, shape='mesh')
    f2_magn = dga_config.lattice.q_grid.map_irrk2fbz(f2_magn, shape='mesh')
    f_magn = f1_magn + (1 + dmft_input['u'] * chi_magn_lambda_pp) * f2_magn

    chi_dens_lambda_pp = pv.reshape_chi(chi_lad_dens, dga_config.box_sizes.niv_pp)
    f1_dens = dga_config.lattice.q_grid.map_irrk2fbz(f1_dens, shape='mesh')
    f2_dens = dga_config.lattice.q_grid.map_irrk2fbz(f2_dens, shape='mesh')
    f_dens = f1_dens + (1 - dmft_input['u'] * chi_dens_lambda_pp) * f2_dens

    # Build singlet and triplet vertex:
    f_sing = -1.5 * f_magn + 0.5 * f_dens
    f_trip = -0.5 * f_magn - 0.5 * f_dens

    plotting.plot_kx_ky(f_sing[:, :, 0, dga_config.box_sizes.niv_pp, dga_config.box_sizes.niv_pp], dga_config.lattice.k_grid.kx,
                        dga_config.lattice.k_grid.kx, pdir=dga_config.eliash.output_path, name='F_sing_pp')

    plotting.plot_kx_ky(f_trip[:, :, 0, dga_config.box_sizes.niv_pp, dga_config.box_sizes.niv_pp], dga_config.lattice.k_grid.kx,
                        dga_config.lattice.k_grid.kx, pdir=dga_config.eliash.output_path, name='F_trip_pp')

    f_sing_loc = dga_config.lattice.q_grid.k_mean(f_sing, 'fbz-mesh')
    f_trip_loc = dga_config.lattice.q_grid.k_mean(f_trip, 'fbz-mesh')

    lfp.plot_fourpoint_nu_nup(f_sing_loc, mf.vn(dga_config.box_sizes.niv_pp), pdir=dga_config.eliash.output_path,
                              name='F_sing_pp_loc')
    lfp.plot_fourpoint_nu_nup(f_trip_loc, mf.vn(dga_config.box_sizes.niv_pp), pdir=dga_config.eliash.output_path,
                              name='F_trip_pp_loc')

    dga_config.eliash.save_data(f_sing, 'F_sing_pp')
    dga_config.eliash.save_data(f_trip, 'F_trip_pp')
    del f_sing, f_trip, f_dens, f_magn
    gc.collect()

logger.log_cpu_time(task=' Pairing vertex saved to file. ')
# %% Perform Ornstein-zernicke fitting of the susceptibility:
if (comm.rank == 0): dga_io.fit_and_plot_oz(dga_config.output_path, dga_config.lattice.q_grid)

# %% Non-local Schwinger-Dyson equation:
# collect vrg vertices:
vrg_q_dens = mpi_distributor.gather(rank_result=vrg_q_dens)
vrg_q_magn = mpi_distributor.gather(rank_result=vrg_q_magn)

if (comm.rank == 0):
    vrg_q_dens = dga_config.lattice.q_grid.map_irrk2fbz(vrg_q_dens, shape='list')
    vrg_q_magn = dga_config.lattice.q_grid.map_irrk2fbz(vrg_q_magn, shape='list')

# %%
# Distribute the full fbz q-points to the different cores:
full_q_list = np.array([dga_config.lattice.q_grid.kmesh_ind[i].flatten() for i in range(3)]).T
q_dupl = np.ones((np.shape(full_q_list)[0]))

# Create new mpi-distributor for the full fbz:
mpi_dist_fbz = mpi_aux.MpiDistributor(ntasks=dga_config.lattice.q_grid.nk_tot, comm=comm,
                                      name='FBZ')
comm.Barrier()
my_full_q_list = full_q_list[mpi_dist_fbz.my_slice]

if (comm.rank == 0):
    # Map object to core q-list:
    chi_lad_dens = dga_config.lattice.q_grid.map_fbz_mesh2list(chi_lad_dens)
    chi_lad_magn = dga_config.lattice.q_grid.map_fbz_mesh2list(chi_lad_magn)

chi_lad_dens = mpi_dist_fbz.scatter(chi_lad_dens)
chi_lad_magn = mpi_dist_fbz.scatter(chi_lad_magn)

vrg_q_dens = mpi_dist_fbz.scatter(vrg_q_dens)
vrg_q_magn = mpi_dist_fbz.scatter(vrg_q_magn)

# %%
kernel_dc = mpi_distributor.gather(kernel_dc)
if (comm.rank == 0):
    kernel_dc = dga_config.lattice.q_grid.map_irrk2fbz(kernel_dc, 'list')
    kernel_dc_mesh = dga_config.lattice.q_grid.map_fbz_list2mesh(kernel_dc)
    plotting.plot_kx_ky(kernel_dc_mesh[:, :, 0, dga_config.box_sizes.niw_core, dga_config.box_sizes.niv_core],
                        dga_config.lattice.q_grid.kx, dga_config.lattice.q_grid.ky,
                        pdir=dga_config.output_path, name='dc_kernel')

kernel_dc = mpi_dist_fbz.scatter(kernel_dc)
siwk_dga = fp.schwinger_dyson_full_q(vrg_q_dens, vrg_q_magn, chi_lad_dens, chi_lad_magn, kernel_dc,
                                     giwk_dmft.g_full(), dmft_input['beta'], dmft_input['u'], my_full_q_list,
                                     dga_config.box_sizes.wn,
                                     np.prod(dga_config.lattice.q_grid.nk), 0, logger)

del vrg_q_dens, vrg_q_magn, chi_lad_dens, chi_lad_magn, kernel_dc
gc.collect()

logger.log_cpu_time(task=' Non-local SDE done. ')
logger.log_memory_usage()
#print(get_largest_vars())

# %%
# Collect from the different cores:
siwk_dga = mpi_dist_fbz.allreduce(siwk_dga)

siwk_dga += mf.cut_v(dmft_input['siw'], dga_config.box_sizes.niv_core) - mf.cut_v(siw_sde_full, dga_config.box_sizes.niv_core)
hartree = twop.get_smom0(dmft_input['u'], dmft_input['n'])
siwk_dga += hartree
# %%
# Build the DGA Self-energy with Sigma_DMFT as asymptotic
sigma_dga = twop.create_dga_siwk_with_dmft_as_asympt(siwk_dga, siwk_dmft, dga_config.box_sizes.niv_shell)
if (comm.rank == 0):
    siwk_dga_shift = sigma_dga.get_siw(dga_config.box_sizes.niv_full, pi_shift=True)
    kx_shift = np.linspace(-np.pi, np.pi, dga_config.lattice.nk[0], endpoint=False)
    ky_shift = np.linspace(-np.pi, np.pi, dga_config.lattice.nk[1], endpoint=False)

    plotting.plot_kx_ky(siwk_dga_shift[..., 0, dga_config.box_sizes.niv_full], kx_shift, ky_shift, pdir=dga_config.output_path,
                        name='Siwk_dga_kz0')
    siw_dga_loc = np.mean(siwk_dga_shift, axis=(0, 1, 2))
    plotting.sigma_loc_checks([siw_sde_full, siw_dga_loc],
                              ['SDE-loc', 'DGA-loc', 'DC-loc', 'Dens-loc', 'Magn-loc'], dmft_input['beta'],
                              dga_config.output_path, verbose=False, do_plot=True, name='dga_loc',
                              xmax=dga_config.box_sizes.niv_full)

    plotting.sigma_loc_checks([siw_sde_full, siw_dga_loc],
                              ['SDE-loc', 'DGA-loc', 'DC-loc', 'Dens-loc', 'Magn-loc'], dmft_input['beta'],
                              dga_config.output_path, verbose=False, do_plot=True, name='dga_loc_core',
                              xmax=dga_config.box_sizes.niv_core)

    dga_config.save_data(sigma_dga.get_siw(dga_config.box_sizes.niv_full, pi_shift=False), 'siwk_dga')

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

    fs_points = np.stack((dga_config.lattice.k_grid.kx[fs_ind[:, 0]], dga_config.lattice.k_grid.ky[fs_ind[:, 1]]), axis=1)
    plotting.plot_kx_ky(giwk_dga.g_full(pi_shift=True)[..., 0, giwk_dga.niv_full], kx_shift, ky_shift,
                        pdir=dga_config.output_path, name='Giwk_dga_kz0', scatter=fs_points)

    dga_config.save_data(giwk_dga.g_full(), 'giwk_dga')
    np.savetxt(dga_config.output_path + '/mu.txt', [[giwk_dga.mu, dmft_input['mu_dmft']]], header='mu_dga, mu_dmft', fmt='%1.6f')

    # Plots along Fermi-Luttinger surface:
    plotting.plot_along_ind(sigma_dga.get_siw(dga_config.box_sizes.niv_full), fs_ind, pdir=dga_config.output_path, niv_plot_min=0,
                            niv_plot=20, name='Sigma_{dga}')

logger.log_cpu_time(task=' Giwk build and plotted. ')
mpi_distributor.delete_file()
mpi_dist_fbz.delete_file()
logger.log_memory_usage()
#print(get_largest_vars())
# --------------------------------------------- POSTPROCESSING ----------------------------------------------------------

# %%
if (dga_config.do_poly_fitting):
    # ------------------------------------------------- POLYFIT ------------------------------------------------------------
    # Extrapolate the self-energy to the Fermi-level via polynomial fit:
    if comm.rank == 0: dga_io.poly_fit(sigma_dga.get_siw(dga_config.box_sizes.niv_full, pi_shift=True), dmft_input['beta'],
                                       dga_config.lattice.k_grid, dga_config.n_fit,
                                       dga_config.o_fit, name='Siwk_poly_cont_', output_path=dga_config.poly_fit_dir)

    if comm.rank == 0:
        dga_io.poly_fit(giwk_dga.g_full(pi_shift=True), dmft_input['beta'], dga_config.lattice.k_grid, dga_config.n_fit,
                        dga_config.o_fit,
                        name='Giwk_dga_poly_cont_',
                        output_path=dga_config.poly_fit_dir)

    logger.log_cpu_time(task=' Poly-fits ')

# %%Perform the Eliashberg Routine:
if (comm.rank == 0 and dga_config.eliash.do_eliash):
    logger.log_cpu_time(task=' Staring Eliashberg ')
    gk_dga = mf.cut_v(giwk_dga.core, dga_config.box_sizes.niv_pp, (-1,))
    norm = np.prod(dga_config.lattice.q_grid.nk_tot) * dmft_input['beta']
    gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=dga_config.eliash.gap0_sing['k'],
                            v_type=dga_config.eliash.gap0_sing['v'],
                            k_grid=dga_config.lattice.q_grid.grid)

    gamma_sing = -dga_config.eliash.load_data('F_sing_pp')
    if (dga_config.eliash.sym_sing): eq.symmetrize_gamma(gamma_sing, 'sing')
    powiter_sing = eq.EliashberPowerIteration(gamma=gamma_sing, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
                                              n_eig=dga_config.eliash.n_eig)

    gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=dga_config.eliash.gap0_trip['k'],
                            v_type=dga_config.eliash.gap0_trip['v'],
                            k_grid=dga_config.lattice.q_grid.grid)

    gamma_trip = -dga_config.eliash.load_data('F_trip_pp')
    if (dga_config.eliash.sym_trip): eq.symmetrize_gamma(gamma_sing, 'trip')
    powiter_trip = eq.EliashberPowerIteration(gamma=gamma_trip, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
                                              n_eig=dga_config.eliash.n_eig)

    eliashberg = {
        'lambda_sing': powiter_sing.lam,
        'lambda_trip': powiter_trip.lam,
        'delta_sing': powiter_sing.gap,
        'delta_trip': powiter_trip.gap,
    }
    dga_config.eliash.save_data(eliashberg, 'eliashberg')
    np.savetxt(dga_config.eliash.output_path + 'eigenvalues.txt', [powiter_sing.lam.real, powiter_trip.lam.real], delimiter=',',
               fmt='%.9f')
    np.savetxt(dga_config.eliash.output_path + 'eigenvalues_s.txt', [powiter_sing.lam_s.real, powiter_trip.lam_s.real],
               delimiter=',', fmt='%.9f')

    for i in range(len(powiter_sing.gap)):
        plotting.plot_gap_function(delta=powiter_sing.gap[i].real, pdir=dga_config.eliash.output_path, name='sing_{}'.format(i),
                                   kgrid=dga_config.lattice.q_grid, do_shift=True)
        plotting.plot_gap_function(delta=powiter_trip.gap[i].real, pdir=dga_config.eliash.output_path, name='trip_{}'.format(i),
                                   kgrid=dga_config.lattice.q_grid, do_shift=True)

    logger.log_event('Eliashberg completed!')

# ------------------------------------------------- MAXENT ------------------------------------------------------------
if ('max_ent' in conf_file):
    import dga.dga_max_ent as dme

    logger.log_cpu_time(task=' Starting Max-Ent ')
    logger.log_event('Path is:' + dga_config.output_path)
    dme.main(path=dga_config.output_path, comm=comm)
    logger.log_event('MaxEnt completed')

# End program:
logger.log_event('Completed DGA run!')
comm.Barrier()
mpi.Finalize()
