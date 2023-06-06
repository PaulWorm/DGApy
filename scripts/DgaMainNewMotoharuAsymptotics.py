# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# This code performs a DGA calculation starting from DMFT quantities as input.
# For the original paper look at: PHYSICAL REVIEW B 75, 045118 (2007)
# For a detailed review of the procedure read my thesis: "Numerical analysis of many-body effects in cuprate and nickelate superconductors"
# Asymptotics were adapted from Phys. Rev. B 106, 205101 (2022)
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import sys, os

sys.path.insert(0, '../src')
import h5py
import numpy as np
import Output as out
from mpi4py import MPI as mpi
import MpiAux as mpi_aux
import numpy as np
import Input as input
import LocalFourPoint as lfp
import FourPoint as fp
import Hr as hamr
import Hk as hamk
import BrillouinZone as bz
import TwoPoint as twop
import Bubble as bub
import Plotting as plotting
import LambdaCorrection as lc
import matplotlib.pyplot as plt
import MatsubaraFrequencies as mf
import Loggers as loggers
import Output as output
import PlotSpecs

# Define MPI communicator:
comm = mpi.COMM_WORLD

# --------------------------------------------- CONFIGURATION ----------------------------------------------------------

# Momentum and frequency grids:
niw_core = 30
niv_core = 30
niv_shell = 60
niv_full = niv_core + niv_shell
lambda_correction_type = 'spch'  # lambda-correction options not yet implemented
nk = (16, 16, 1)
nq = nk  # Currently nk and nq have to be equal. For inequal grids the q-list would have to be adjusted.
symmetries = bz.two_dimensional_square_symmetries()
hr = hamr.one_band_2d_t_tp_tpp(1.0, -0.2, 0.1)

# Parameters for Poly fitting:
n_fit = 4
order = 3

# Input and output directories:
# input_type = 'EDFermion'  # 'w2dyn'
input_type = 'w2dyn'  # 'w2dyn'
input_dir = '../test/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
output_dir = input_dir + 'LambdaDga_lc_{}_Nk{}_Nq{}_wcore{}_vcore{}_vshell{}'.format(lambda_correction_type, np.prod(nk),
                                                                                     np.prod(nq),
                                                                                     niw_core, niv_core, niv_shell)

# Create output directory:
output_dir = out.uniquify(output_dir)
poly_fit_dir = output_dir + '/PolyFits/'
comm.barrier()
if (comm.rank == 0): os.mkdir(output_dir)
if (comm.rank == 0): os.mkdir(poly_fit_dir)
comm.barrier()

# Initialize Logger:
logger = loggers.MpiLogger(logfile=output_dir + '/dga.log', comm=comm, output_path=output_dir)

logger.log_event(message=' Config Init and folder set up done!')
# %%
# ------------------------------------------- LOAD THE INPUT --------------------------------------------------------
dmft_input = input.load_1p_data(input_type, input_dir)

# cut the two-particle Green's functions:
g2_dens = lfp.LocalFourPoint(channel='dens', matrix=dmft_input['g4iw_dens'], beta=dmft_input['beta'])
g2_magn = lfp.LocalFourPoint(channel='magn', matrix=dmft_input['g4iw_magn'], beta=dmft_input['beta'])

# Cut frequency ranges:
g2_dens.cut_iv(niv_core)
g2_dens.cut_iw(niw_core)

g2_magn.cut_iv(niv_core)
g2_magn.cut_iw(niw_core)

if (comm.rank == 0):
    g2_dens.plot(0, pdir=output_dir, name='G2_dens')
    g2_magn.plot(0, pdir=output_dir, name='G2_magn')
    g2_magn.plot(10, pdir=output_dir, name='G2_magn')
    g2_magn.plot(-10, pdir=output_dir, name='G2_magn')

# Build Green's function and susceptibility:
k_grid = bz.KGrid(nk=nk, symmetries=symmetries)
ek = hamk.ek_3d(k_grid.grid, hr)

siwk_dmft = twop.SelfEnergy(sigma=dmft_input['siw'][None, None, None, :], beta=dmft_input['beta'])
giwk_dmft = twop.GreensFunction(siwk_dmft, ek, mu=dmft_input['mu_dmft'], niv_asympt=niv_full * 2 + niw_core * 2)

gchi_dens = lfp.gchir_from_g2(g2_dens, giwk_dmft.g_loc)
gchi_magn = lfp.gchir_from_g2(g2_magn, giwk_dmft.g_loc)

logger.log_cpu_time(task=' Data loading completed. ')
# ------------------------------------------- Extract the irreducible Vertex --------------------------------------------------------
# Create Bubble generator:
bubble_gen = bub.LocalBubble(wn=g2_dens.wn, giw=giwk_dmft)
gchi0_core = bubble_gen.get_gchi0(niv_core)
gchi0_urange = bubble_gen.get_gchi0(niv_full)
gamma_dens = lfp.gammar_from_gchir(gchi_dens, gchi0_urange, dmft_input['u'])
gamma_magn = lfp.gammar_from_gchir(gchi_magn, gchi0_urange, dmft_input['u'])

if (comm.rank == 0):
    gamma_dens.plot(0, pdir=output_dir, niv=min(niv_core, 2 * int(dmft_input['beta'])), name='Gamma_dens')
    gamma_magn.plot(0, pdir=output_dir, niv=min(niv_core, 2 * int(dmft_input['beta'])), name='Gamma_magn')
    gamma_magn.plot(10, pdir=output_dir, niv=min(niv_core, 2 * int(dmft_input['beta'])), name='Gamma_magn')
    gamma_magn.plot(-10, pdir=output_dir, niv=min(niv_core, 2 * int(dmft_input['beta'])), name='Gamma_magn')
    gamma_dens.plot(10, pdir=output_dir, niv=min(niv_core, 2 * int(dmft_input['beta'])), name='Gamma_dens')

    np.save(output_dir + '/gamma_dens.npy', gamma_dens.mat)
    np.save(output_dir + '/gamma_magn.npy', gamma_magn.mat)

logger.log_cpu_time(task=' Gamma-loc finished. ')
# ----------------------------------- Compute the Susceptibility and Threeleg Vertex --------------------------------------------------------


vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_dens, bubble_gen, dmft_input['u'],
                                                                      niv_shell=niv_shell)
vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, bubble_gen, dmft_input['u'],
                                                                      niv_shell=niv_shell)

# if( comm.rank == 0):
#     F_dens.plot(pdir=output_dir,name='F_dens_dc')
#     F_magn.plot(pdir=output_dir,name='F_magn_dc')

# Create checks of the susceptibility:
if (comm.rank == 0): plotting.chi_checks([chi_dens, ], [chi_magn, ], ['Loc-tilde', ], giwk_dmft, output_dir, verbose=False,
                                         do_plot=True, name='loc')

logger.log_cpu_time(task=' Vrg and Chi-phys loc done. ')
# --------------------------------------- Local Schwinger-Dyson equation --------------------------------------------------------

# Perform the local SDE for box-size checks:
siw_sde_full = lfp.schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, giwk_dmft.g_loc, dmft_input['u'], dmft_input['n'],
                                        niv_shell=niv_shell)

# Create checks of the self-energy:
if (comm.rank == 0):
    plotting.sigma_loc_checks([siw_sde_full, dmft_input['siw']], ['SDE', 'Input'], dmft_input['beta'], output_dir, verbose=False,
                              do_plot=True, xmax=niv_full)
    np.save(output_dir + '/siw_sde_loc.npy', siw_sde_full)
    np.save(output_dir + '/chi_dens_loc.npy', chi_dens)
    np.save(output_dir + '/chi_magn_loc.npy', chi_magn)

logger.log_cpu_time(task=' Local SDE finished. ')

# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- NON-LOCAL PART --------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# Split the momenta between the different cores.
q_grid = bz.KGrid(nk=nq, symmetries=symmetries)
mpi_distributor = mpi_aux.MpiDistributor(ntasks=q_grid.nk_irr, comm=comm, output_path=output_dir + '/', name='Q')
my_q_list = q_grid.irrk_mesh_ind.T[mpi_distributor.my_slice]
# %%
# F_dc = lfp.Fob2_from_chir(gchi_magn,gchi0_core)
F_dc = lfp.Fob2_from_gamob2_urange(gamma_magn,gchi0_urange,dmft_input['u'])
vrg_q_dens, vrg_q_magn, chi_lad_dens, chi_lad_magn, kernel_dc = fp.get_vrg_and_chir_lad_from_gammar_uasympt_q(gamma_dens,
                                                                                                              gamma_magn,
                                                                                                              F_dc,
                                                                                                              vrg_magn,
                                                                                                              chi_magn,
                                                                                                              bubble_gen,
                                                                                                              dmft_input['u'],
                                                                                                              my_q_list,
                                                                                                              niv_shell=niv_shell,
                                                                                                              logger=logger)

logger.log_cpu_time(task=' Vrg and Chi-ladder completed. ')
# %% Collect the results from the different cores:

chi_lad_dens = mpi_distributor.allgather(rank_result=chi_lad_dens)
chi_lad_magn = mpi_distributor.allgather(rank_result=chi_lad_magn)

# Build the full Brillouin zone, which is needed for the lambda-correction:
chi_lad_dens = q_grid.map_irrk2fbz(chi_lad_dens)
chi_lad_magn = q_grid.map_irrk2fbz(chi_lad_magn)
# %%

if (comm.rank == 0):
    chi_lad_dens_loc = q_grid.k_mean(chi_lad_dens, type='fbz-mesh')
    chi_lad_magn_loc = q_grid.k_mean(chi_lad_magn, type='fbz-mesh')
    plotting.chi_checks([chi_dens, chi_lad_dens_loc], [chi_magn, chi_lad_magn_loc], ['loc', 'ladder-sum'], giwk_dmft, output_dir,
                        verbose=False,
                        do_plot=True, name='lad_q_tilde')

    plotting.plot_kx_ky(chi_lad_dens[..., 0, niw_core], q_grid.kx, q_grid.ky, pdir=output_dir, name='Chi_ladder_dens_kz0')
    plotting.plot_kx_ky(chi_lad_magn[..., 0, niw_core], q_grid.kx, q_grid.ky, pdir=output_dir, name='Chi_ladder_magn_kz0')

# %% Lambda-corrected susceptibility:

chi_lad_dens, chi_lad_magn, lambda_dens, lambda_magn = lc.lambda_correction(chi_lad_dens, chi_lad_magn, dmft_input['beta'],
                                                                            chi_dens, chi_magn,
                                                                            type=lambda_correction_type)

# %%
if (comm.rank == 0):
    chi_lad_dens_loc = q_grid.k_mean(chi_lad_dens, type='fbz-mesh')
    chi_lad_magn_loc = q_grid.k_mean(chi_lad_magn, type='fbz-mesh')
    plotting.chi_checks([chi_lad_dens_loc, chi_dens], [chi_lad_magn_loc, chi_magn], ['lam-loc', 'loc'], giwk_dmft, output_dir,
                        verbose=False,
                        do_plot=True, name='q_lam')

    plotting.plot_kx_ky(chi_lad_dens[..., 0, niw_core], q_grid.kx, q_grid.ky, pdir=output_dir, name='Chi_lam_dens_kz0')
    plotting.plot_kx_ky(chi_lad_magn[..., 0, niw_core], q_grid.kx, q_grid.ky, pdir=output_dir, name='Chi_lam_magn_kz0')

    np.save(output_dir + '/chi_dens_lam.npy', chi_lad_dens)
    np.save(output_dir + '/chi_magn_lam.npy', chi_lad_magn)
    np.savetxt(output_dir + '/lambda.txt', [[lambda_dens, lambda_magn]], header='dens magn', fmt='%.6f')

logger.log_cpu_time(task=' Lambda-correction done. ')

# %% Perform Ornstein-zernicke fitting of the susceptibility:
if (comm.rank == 0): output.fit_and_plot_oz(output_dir, q_grid)

# %% Non-local Schwinger-Dyson equation:
# collect vrg vertices:
vrg_q_dens = mpi_distributor.allgather(rank_result=vrg_q_dens)
vrg_q_magn = mpi_distributor.allgather(rank_result=vrg_q_magn)

vrg_q_dens = q_grid.map_irrk2fbz(vrg_q_dens, shape='list')
vrg_q_magn = q_grid.map_irrk2fbz(vrg_q_magn, shape='list')

# %%
if (comm.rank == 0):
    vrg_q_dens_sum = np.mean(vrg_q_dens, axis=0)
    vrg_q_magn_sum = np.mean(vrg_q_magn, axis=0)

    plotting.plot_kx_ky(vrg_q_dens.reshape(q_grid.nk + vrg_q_dens.shape[1:])[:, :, 0, niw_core, niv_core], q_grid.kx, q_grid.ky,
                        pdir=output_dir, name='Vrg_dens_w0')
    plotting.plot_kx_ky(vrg_q_magn.reshape(q_grid.nk + vrg_q_magn.shape[1:])[:, :, 0, niw_core, niv_core], q_grid.kx, q_grid.ky,
                        pdir=output_dir, name='Vrg_magn_w0')

    iwn_plot = niv_core
    vn_core = mf.vn(niv_core)
    fig, axes = plt.subplots(2, 2, dpi=500, figsize=(8, 8))
    axes = axes.flatten()
    axes[0].plot(vn_core, vrg_q_dens_sum[iwn_plot].real)
    axes[0].plot(vn_core, vrg_dens.mat[iwn_plot].real)

    axes[1].plot(vn_core, vrg_q_dens_sum[iwn_plot].imag)
    axes[1].plot(vn_core, vrg_dens.mat[iwn_plot].imag)

    axes[2].plot(vn_core, vrg_q_magn_sum[iwn_plot].real)
    axes[2].plot(vn_core, vrg_magn.mat[iwn_plot].real)

    axes[3].plot(vn_core, vrg_q_magn_sum[iwn_plot].imag)
    axes[3].plot(vn_core, vrg_magn.mat[iwn_plot].imag)
    for ax in axes:
        ax.set_xlim(0, None)
    plt.legend()
    plt.savefig(output_dir + f'/TestVrg_loc_wn{iwn_plot}.png')
    plt.close()

# %%
# Distribute the full fbz q-points to the different cores:
full_q_list = np.array([q_grid.kmesh_ind[i].flatten() for i in range(3)]).T
q_dupl = np.ones((np.shape(full_q_list)[0]))

# Map object to core q-list:
chi_lad_dens = q_grid.map_fbz_mesh2list(chi_lad_dens)
chi_lad_magn = q_grid.map_fbz_mesh2list(chi_lad_magn)

# Create new mpi-distributor for the full fbz:
mpi_dist_fbz = mpi_aux.MpiDistributor(ntasks=q_grid.nk_tot, comm=comm, output_path=output_dir + '/', name='FBZ')
my_full_q_list = full_q_list[mpi_dist_fbz.my_slice]
chi_lad_dens = chi_lad_dens[mpi_dist_fbz.my_slice, ...]
chi_lad_magn = chi_lad_magn[mpi_dist_fbz.my_slice, ...]

vrg_q_dens = vrg_q_dens[mpi_dist_fbz.my_slice, ...]
vrg_q_magn = vrg_q_magn[mpi_dist_fbz.my_slice, ...]

# %%
kernel_dc = mpi_distributor.allgather(kernel_dc)
kernel_dc = q_grid.map_irrk2fbz(kernel_dc, 'list')[mpi_dist_fbz.my_slice, ...]


siwk_dga = fp.schwinger_dyson_full_q(vrg_q_dens, vrg_q_magn, chi_lad_dens, chi_lad_magn, kernel_dc,
                                     giwk_dmft.g_full(), dmft_input['beta'], dmft_input['u'], my_full_q_list, g2_magn.wn,
                                     np.prod(q_grid.nk), 0, logger)

logger.log_cpu_time(task=' Non-local SDE done. ')

# %%
hartree = twop.get_smom0(dmft_input['u'], dmft_input['n'])
siwk_dga += mf.cut_v(dmft_input['siw'], niv_core) - mf.cut_v(siw_sde_full, niv_core)

# Collect from the different cores:
siwk_dga = mpi_dist_fbz.allreduce(siwk_dga)

siwk_dga += hartree
# %%
siwk_shell = siwk_dmft.get_siw(niv_full)
siwk_shell = np.ones(nk)[:,:,:,None] * \
             mf.inv_cut_v(siwk_shell,niv_core=niv_core,niv_shell=niv_shell,axes=-1)[0,0,0,:][None,None,None,:]
siwk_dga = mf.concatenate_core_asmypt(siwk_dga,siwk_shell)
siwk_dga_shift = twop.SelfEnergy(siwk_dga, dmft_input['beta']).get_siw(niv_full,pi_shift=True)

if (comm.rank == 0):
    kx_shift = np.linspace(-np.pi, np.pi, nk[0], endpoint=False)
    ky_shift = np.linspace(-np.pi, np.pi, nk[1], endpoint=False)

    plotting.plot_kx_ky(siwk_dga_shift[..., 0, niv_full], kx_shift, ky_shift, pdir=output_dir, name='Siwk_dga_kz0')
    siw_dga_loc = np.mean(siwk_dga_shift, axis=(0, 1, 2))
    plotting.sigma_loc_checks([siw_sde_full, siw_dga_loc],
                              ['SDE-loc', 'DGA-loc', 'DC-loc', 'Dens-loc', 'Magn-loc'], dmft_input['beta'],
                              output_dir, verbose=False, do_plot=True, name='dga_loc', xmax=niv_full)

    plotting.sigma_loc_checks([siw_sde_full, siw_dga_loc],
                              ['SDE-loc', 'DGA-loc', 'DC-loc', 'Dens-loc', 'Magn-loc'], dmft_input['beta'],
                              output_dir, verbose=False, do_plot=True, name='dga_loc_core', xmax=niv_core)

    np.save(output_dir + '/siwk_dga.npy', siwk_dga)

logger.log_cpu_time(task=' Siwk collected and plotted. ')
# %% Build the DGA Green's function:

if (comm.rank == 0):
    sigma_dga = twop.SelfEnergy(siwk_dga, dmft_input['beta'])
    giwk_dga = twop.GreensFunction(sigma_dga, ek, n=dmft_input['n'], niv_asympt=niv_full * 2)



    giwk_shift = giwk_dga.g_full(pi_shift=False)[..., 0, giwk_dga.niv_full][:nk[0] // 2, :nk[1] // 2]
    fs_ind = bz.find_zeros(giwk_shift)
    n_fs = np.shape(fs_ind)[0]
    fs_ind = fs_ind[:n_fs // 2]

    fs_points = np.stack((k_grid.kx[fs_ind[:, 0]], k_grid.ky[fs_ind[:, 1]]), axis=1)
    plotting.plot_kx_ky(giwk_dga.g_full(pi_shift=True)[..., 0, giwk_dga.niv_full], kx_shift, ky_shift,
                        pdir=output_dir, name='Giwk_dga_kz0', scatter=fs_points)

    np.save(output_dir + '/giwk_dga.npy', giwk_dga.g_full())
    np.savetxt(output_dir + '/mu.txt', [[giwk_dga.mu, dmft_input['mu_dmft']]], header='mu_dga, mu_dmft', fmt='%1.6f')

    # Plots along Fermi-Luttinger surface:
    siwk_dga = sigma_dga.get_siw(niv_full)
    plotting.plot_along_fs(siwk_dga, fs_ind, pdir=output_dir, niv_plot_min=0, niv_plot=20, name='Sigma_{dga}')

logger.log_cpu_time(task=' Giwk build and plotted. ')
mpi_distributor.delete_file()
mpi_dist_fbz.delete_file()
# --------------------------------------------- POSTPROCESSING ----------------------------------------------------------

# %%
# ------------------------------------------------- POLYFIT ------------------------------------------------------------
# Extrapolate the self-energy to the Fermi-level via polynomial fit:
if comm.rank == 0: output.poly_fit(sigma_dga.get_siw(niv_full, pi_shift=True), dmft_input['beta'], k_grid, n_fit, order,
                                   name='Siwk_poly_cont_',
                                   output_path=poly_fit_dir)

if comm.rank == 0:
    output.poly_fit(giwk_dga.g_full(pi_shift=True), dmft_input['beta'], k_grid, n_fit, order, name='Giwk_dga_poly_cont_',
                    output_path=poly_fit_dir)

if comm.rank == 0:
    output.poly_fit(giwk_dmft.g_full(pi_shift=True), dmft_input['beta'], k_grid, n_fit, order, name='Giwk_dmft_poly_cont_',
                    output_path=poly_fit_dir)

logger.log_cpu_time(task=' Poly-fits ')

# %%
