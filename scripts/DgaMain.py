# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Warning: many of the files I prepared from Motoharu do not have smom stored.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import sys, os

sys.path.append('../src/')
sys.path.append(os.environ['HOME'] + "/Programs/dga/src")
import numpy as np
import Hr as hr_mod
import Hk as hamk
import Indizes as ind
import Output as output
import Input as input
import TwoPoint as twop
import Plotting as plotting
from mpi4py import MPI as mpi
import MpiAux as mpiaux
import Config as conf
import LocalRoutines as lr
import SDE as sde
import FourPoint as fp
import LambdaCorrection as lc
import Loggers as loggers
import LambdaDga as lambda_dga

# ----------------------------------------------- PARAMETERS -----------------------------------------------------------

# Define MPI communicator:
comm = mpi.COMM_WORLD

# Define Config objects:
names = conf.Names()
options = conf.Options()
sys_param = conf.SystemParamter()
box_sizes = conf.BoxSizes()

# Define paths of datasets:
names.input_path = './'
#names.input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta6.6666_n0.99/'
# names.input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta30_n0.90/'
names.input_path = '/mnt/d/Research/U2BenchmarkData/BenchmarkSchaefer_beta_15/LambdaDgaPython/'
# names.input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta15_n0.975/'
# names.input_path = '/mnt/c/Users/pworm/Research/Ba2CuO4/Plane1/U3.0eV_n0.93_b040/'
# names.input_path = '/mnt/d/Research/BenchmarkEliashberg/'
# names.input_path = '/mnt/d/Research/HoleDopedNickelates/2DSquare_U8_tp-0.25_tpp0.12_beta50_n0.875/LambdaDgaPython/'
names.output_path = names.input_path

# Define names of input/output files:
names.fname_dmft = '1p-data.hdf5'
names.fname_g2 = 'g4iw_sym.hdf5'  # 'Vertex_sym.hdf5' #'g4iw_sym.hdf5'
names.fname_ladder_vertex = 'LadderVertex.hdf5'

# Define options:
options.do_max_ent_loc = False  # Perform analytic continuation using MaxEnt from Josef Kaufmann's ana_cont package.
options.do_max_ent_irrk = False  # Perform analytic continuation using MaxEnt from Josef Kaufmann's ana_cont package.
options.do_pairing_vertex = False
options.keep_ladder_vertex = False
options.lambda_correction_type = 'sp'  # Available: ['spch','sp','none','sp_only']
options.use_urange_for_lc = False  # Use with care. This is not really tested and at least low k-grid samples don't look too good.
options.lc_use_only_positive = True  # Use only frequency box where susceptibility is positive for lambda correction.
options.analyse_spin_fermion_contributions = True  # Analyse the contributions of the Re/Im part of the spin-fermion vertex seperately
options.analyse_w0_contribution = False  # Analyse the w0 contribution to the self-energy
options.use_fbz = True  # Perform the calculation in the full BZ
options.use_gloc_dmft = True # Use DMFT output g_loc

# Analytic continuation flags:
dmft_fs_cont = False
dmft_fbz_cont = False
no_mu_adjust_fs_cont = False
no_mu_adjust_fbz_cont = False

# Create the real-space Hamiltonian:
t = 0.25
hr = hr_mod.one_band_2d_t_tp_tpp(t=t, tp=-0.0 * t, tpp=0.0 * t)
# hr = hr_mod.Ba2CuO4_plane()
sys_param.hr = hr
# Eliashberg config object:
el_conf = conf.EliashbergConfig(k_sym='d-wave')

# Anacont parameters:
nwr = 401

# Pairing vertex symmetries:
n_eig = 2
sym_sing = True
sym_trip = True

# Define frequency box-sizes:
box_sizes.niw_core = 30
box_sizes.niw_urange = 30  # This seems not save enough to be used.
box_sizes.niv_core = 30
box_sizes.niv_invbse = 30
box_sizes.niv_urange = 30  # Must be larger than niv_invbse

# Box size for saving the spin-fermion vertex:
box_sizes.niw_vrg_save = 5
box_sizes.niv_vrg_save = 5

# Define k-ranges:
nkx = 16
nky = nkx
nqx = 16
nqy = nkx

box_sizes.nk = (nkx, nky, 1)
box_sizes.nq = (nqx, nqy, 1)

output_folder = 'LambdaDga_lc_{}_Nk{}_Nq{}_core{}_invbse{}_vurange{}_wurange{}'.format(options.lambda_correction_type,
                                                                                       np.prod(box_sizes.nk),
                                                                                       np.prod(box_sizes.nq),
                                                                                       box_sizes.niw_core,
                                                                                       box_sizes.niv_invbse,
                                                                                       box_sizes.niv_urange,
                                                                                       box_sizes.niw_urange)
names.output_path = output.uniquify(names.output_path + output_folder) + '/'
fname_ladder_vertex = names.output_path + names.fname_ladder_vertex

names.output_path_sp = output.uniquify(names.output_path + 'SpinFermion') + '/'
names.output_path_pf = output.uniquify(names.output_path + 'PolyFit') + '/'
names.output_path_ac = output.uniquify(names.output_path + 'AnaCont') + '/'
names.output_path_el = output.uniquify(names.output_path + 'Eliashberg') + '/'

# Create the DGA Config object:
dga_conf = conf.DgaConfig(BoxSizes=box_sizes, Options=options, SystemParameter=sys_param, Names=names,
                          ek_funk=hamk.ek_3d)

# Parameter for the polynomial extrapolation to the Fermi-level:
dga_conf.npf = 4
dga_conf.opf = 3

# --------------------------------------------- CREATE THE OUTPUT PATHS -------------------------------------------------
comm.Barrier()
if (comm.rank == 0): os.mkdir(dga_conf.nam.output_path)
if (comm.rank == 0): os.mkdir(dga_conf.nam.output_path_sp)
if (comm.rank == 0): os.mkdir(dga_conf.nam.output_path_pf)
if (comm.rank == 0): os.mkdir(dga_conf.nam.output_path_ac)
if (comm.rank == 0): os.mkdir(dga_conf.nam.output_path_el)
comm.Barrier()

# Initialize Logger:
logger = loggers.MpiLogger(logfile=dga_conf.nam.output_path + 'dga.log', comm=comm)

logger.log_event(message=' Config Init and folder set up done!')
# ------------------------------------------------ DMFT 1P INPUT -------------------------------------------------------
dmft1p = input.load_1p_data(dga_conf=dga_conf)
if comm.rank == 0: np.save(dga_conf.nam.output_path + 'dmft1p.npy', dmft1p, allow_pickle=True)
box_sizes.niv_dmft = dmft1p['niv']
sys_param.beta = dmft1p['beta']
sys_param.u = dmft1p['u']
sys_param.n = dmft1p['n']
sys_param.mu_dmft = dmft1p['mu']

if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'config.npy', dga_conf)

# ------------------------------------------ ANALYTIC CONTINUATION SET-UP ----------------------------------------------
me_conf = conf.MaxEntConfig(t=t, beta=dga_conf.sys.beta, mesh_type='lorentzian')
me_conf.nwr = nwr
n_fit = me_conf.get_n_fit_opt(n_fit_min=dga_conf.box.niv_core, n_fit_max=dga_conf.box.niv_urange)
bw = me_conf.get_bw_opt()

logger.log_event(message=' DMFT one-particle input read!')
# ----------------------------------------------- LOAD GAMMA -----------------------------------------------------------
gamma_dmft = input.get_gamma_loc(dga_conf=dga_conf, giw_dmft=dmft1p['gloc'])

if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'gamma_dmft.npy', gamma_dmft, allow_pickle=True)
if (comm.rank == 0): plotting.plot_gamma_dmft(gamma_dmft=gamma_dmft, output_path=dga_conf.nam.output_path,
                                              niw_core=dga_conf.box.niw_core)

logger.log_cpu_time(task=' DMFT gamma extraction ')
# -------------------------------------- LOCAL SCHWINGER DYSON EQUATION ------------------------------------------------

ldga = lambda_dga.LambdaDga(config=dga_conf, comm=comm, sigma_dmft=dmft1p['sloc'],gloc_dmft=dmft1p['gloc'], sigma_start=dmft1p['sloc'],
                            gamma_magn=gamma_dmft['magn'], gamma_dens=gamma_dmft['dens'], adjust_mu=False)
#ldga.g_loc = dmft1p['gloc']
ldga.local_sde(safe_output=True, interactive=True)

# ----------------------------------------- NON-LOCAL RPA SUCEPTIBILITY  -----------------------------------------------
ldga.rpa_susceptibility()
logger.log_cpu_time(task=' RPA susceptibility ')
# ----------------------------------------- NON-LOCAL LADDER SUCEPTIBILITY  --------------------------------------------
ldga.dga_bse_susceptibility(save_vrg=True)
logger.log_cpu_time(task=' ladder susceptibility ')
# ----------------------------------------------- LAMBDA-CORRECTION ----------------------------------------------------
ldga.lambda_correction(save_output=True)
logger.log_cpu_time(task=' lambda correction ')
# ------------------------------------------- DGA SCHWINGER-DYSON EQUATION ---------------------------------------------
ldga.dga_sde(interactive=True)
logger.log_cpu_time(task=' DGA SDE ')

ldga.build_dga_sigma(interactive=True)
sigma = ldga.sigma
sigma_nc = ldga.sigma_nc

if comm.rank == 0: output.prepare_and_plot_vrg_dga(dga_conf=dga_conf, distributor=ldga.qiw_distributor)

logger.log_cpu_time(task=' Plotting ')

if comm.rank == 0: plotting.plot_f_dmft(f=ldga.f_loc['dens'], path=dga_conf.nam.output_path, name='f_dmft_dens')
if comm.rank == 0: plotting.plot_f_dmft(f=ldga.f_loc['magn'], path=dga_conf.nam.output_path, name='f_dmft_magn')

# ------------------------------------------------- OZ-FIT -------------------------------------------------------------
if comm.rank == 0: output.fit_and_plot_oz(dga_conf=dga_conf)

logger.log_cpu_time(task=' OZ-fit ')
# ------------------------------------------------- POLYFIT ------------------------------------------------------------
# Extrapolate the self-energy to the Fermi-level via polynomial fit:
if comm.rank == 0: output.poly_fit(dga_conf=dga_conf, mat_data=sigma, name='sigma_extrap')
if comm.rank == 0: output.poly_fit(dga_conf=dga_conf, mat_data=sigma_nc, name='sigma_nc_extrap')

if comm.rank == 0:
    gk = twop.create_gk_dict(dga_conf=dga_conf, sigma=dmft1p['sloc'], mu0=dmft1p['mu'], adjust_mu=True)
    output.poly_fit(dga_conf=dga_conf, mat_data=gk['gk'], name='giwk_dmft')

if comm.rank == 0:
    gk = twop.create_gk_dict(dga_conf=dga_conf, sigma=sigma, mu0=dmft1p['mu'], adjust_mu=True)
    output.poly_fit(dga_conf=dga_conf, mat_data=gk['gk'], name='giwk')

if comm.rank == 0:
    gk = twop.create_gk_dict(dga_conf=dga_conf, sigma=sigma_nc, mu0=dmft1p['mu'], adjust_mu=True)
    output.poly_fit(dga_conf=dga_conf, mat_data=gk['gk'], name='giwk_nc')

logger.log_cpu_time(task=' Poly-fits ')

# --------------------------------------------- ANALYTIC CONTINUATION --------------------------------------------------
# %%
# Broadcast bw_opt_dga
comm.Barrier()
if comm.rank == 0:
    bw_opt_dga = np.empty((1,))
    bw_opt_vrg_re = np.empty((1,))
else:
    bw_opt_dga = np.empty((1,))
    bw_opt_vrg_re = np.empty((1,))

if dga_conf.opt.do_max_ent_loc:
    if comm.rank == 0:
        bw_range_loc = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]) * bw
        output.max_ent_loc_bw_range(dga_conf=dga_conf, me_conf=me_conf, bw_range=bw_range_loc, sigma=dmft1p['sloc'],
                                    n_fit=n_fit, adjust_mu=True, name='dmft')
        bw_opt_dga[0] = output.max_ent_loc_bw_range(dga_conf=dga_conf, me_conf=me_conf, bw_range=bw_range_loc,
                                                    sigma=sigma,
                                                    n_fit=n_fit, adjust_mu=True, name='dga')
        output.max_ent_loc_bw_range(dga_conf=dga_conf, me_conf=me_conf, bw_range=bw_range_loc,
                                    sigma=sigma_nc, n_fit=n_fit, adjust_mu=True, name='dga_nc')

    logger.log_cpu_time(task=' MaxEnt local ')

    if dga_conf.opt.analyse_spin_fermion_contributions:
        comm.Bcast(bw_opt_vrg_re, root=0)
    comm.Bcast(bw_opt_dga, root=0)

# %%
if dga_conf.opt.do_max_ent_irrk:
    # Do analytic continuation within the irreducible Brillouin Zone:
    output.max_ent_irrk_bw_range(comm=comm, dga_conf=dga_conf, me_conf=me_conf, bw_range=bw_opt_dga,
                                 sigma=sigma, n_fit=n_fit, name='dga')
    logger.log_cpu_time(task=' MaxEnt irrk ')

# ------------------------------------------------ PAIRING VERTEX ------------------------------------------------------
if (dga_conf.opt.do_pairing_vertex and comm.rank == 0):
    output.load_and_construct_pairing_vertex(dga_conf=dga_conf, comm=comm)

    logger.log_cpu_time(task=' Load and construct pairing vertex ')

# ----------------------------------------------- Eliashberg Equation --------------------------------------------------
if (dga_conf.opt.do_pairing_vertex and comm.rank == 0):
    output.perform_eliashberg_routine(dga_conf=dga_conf, sigma=sigma, el_conf=el_conf)

    logger.log_cpu_time(task=' Performed eliashberg equation ')
#
#
# # ---------------------------------------------- REMOVE THE RANK FILES -------------------------------------------------
comm.Barrier()
qiw = mpiaux.MpiDistributor(ntasks=dga_conf.k_grid.nk_irr, comm=comm,
                            output_path=dga_conf.nam.output_path,
                            name='Qiw')
qiw.delete_file()
