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
names.input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.85/KonvergenceAnalysis/'
#names.input_path = '/mnt/d/Research/U2BenchmarkData/BenchmarkSchaefer_beta_15/LambdaDgaPython/'
# names.input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta15_n0.975/'
#names.input_path = '/mnt/c/Users/pworm/Research/Ba2CuO4/Plane1/U3.0eV_n0.93_b040/'
#names.input_path = '/mnt/d/Research/BenchmarkEliashberg/'
#names.input_path = '/mnt/d/Research/HoleDopedNickelates/2DSquare_U8_tp-0.25_tpp0.12_beta50_n0.875/LambdaDgaPython/'
names.output_path = names.input_path

# Define names of input/output files:
names.fname_dmft = '1p-data.hdf5'
names.fname_g2 = 'g4iw_sym.hdf5'  # 'Vertex_sym.hdf5' #'g4iw_sym.hdf5'
names.fname_ladder_vertex = 'LadderVertex.hdf5'

# Define options:
options.do_max_ent_loc = True # Perform analytic continuation using MaxEnt from Josef Kaufmann's ana_cont package.
options.do_max_ent_irrk = True  # Perform analytic continuation using MaxEnt from Josef Kaufmann's ana_cont package.
options.do_pairing_vertex = True
options.keep_ladder_vertex = False
options.lambda_correction_type = 'spch'  # Available: ['spch','sp','none','sp_only']
options.use_urange_for_lc = True  # Use with care. This is not really tested and at least low k-grid samples don't look too good.
options.lc_use_only_positive = True  # Use only frequency box where susceptibility is positive for lambda correction.
options.analyse_spin_fermion_contributions = False  # Analyse the contributions of the Re/Im part of the spin-fermion vertex seperately
options.analyse_w0_contribution = False  # Analyse the w0 contribution to the self-energy
options.use_fbz = False  # Perform the calculation in the full BZ

# Analytic continuation flags:
dmft_fs_cont = False
dmft_fbz_cont = False
no_mu_adjust_fs_cont = False
no_mu_adjust_fbz_cont = False

# Create the real-space Hamiltonian:
t = 1.00
hr = hr_mod.one_band_2d_t_tp_tpp(t=t, tp=-0.2 * t, tpp=0.10 * t)
#hr = hr_mod.Ba2CuO4_plane()
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
box_sizes.niw_core = 10
box_sizes.niw_urange = 200  # This seems not to be save enough to be used.
box_sizes.niv_core = 10
box_sizes.niv_invbse = 10
box_sizes.niv_urange = 200  # Must be larger than niv_invbse

# Box size for saving the spin-fermion vertex:
box_sizes.niw_vrg_save = 5
box_sizes.niv_vrg_save = 5

# Define k-ranges:
nkx = 24
nky = nkx
nqx = 24
nqy = nkx

box_sizes.nk = (nkx, nky, 1)
box_sizes.nq = (nqx, nqy, 1)

output_folder = 'LambdaDga_lc_{}_local_core{}_invbse{}_vurange{}_wurange{}'.format(options.lambda_correction_type,
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
# ----------------------------------------------- Chi0 URANGE ----------------------------------------------------------
chi0_urange = fp.LocalBubble(giw=dmft1p['gloc'], beta=dga_conf.sys.beta, niv_sum=dga_conf.box.niv_urange, iw=dga_conf.box.wn_core)
if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'chi0_urange.npy', chi0_urange, allow_pickle=True)
# ----------------------------------------------- LOAD GAMMA -----------------------------------------------------------
gamma_dmft = input.get_gamma_loc(dga_conf=dga_conf, giw_dmft=dmft1p['gloc'])

if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'gamma_dmft.npy', gamma_dmft, allow_pickle=True)
if (comm.rank == 0): plotting.plot_gamma_dmft(gamma_dmft=gamma_dmft, output_path=dga_conf.nam.output_path,
                                              niw_core=dga_conf.box.niw_core)

logger.log_cpu_time(task=' DMFT gamma extraction ')
# -------------------------------------- LOCAL SCHWINGER DYSON EQUATION ------------------------------------------------
dmft_sde, chi_dmft, vrg_dmft, sigma_com_loc = lr.local_dmft_sde_from_gamma(dga_conf=dga_conf, giw=dmft1p['gloc'],
                                                                           gamma_dmft=gamma_dmft, ana_w0=False)
rpa_sde_loc, chi_rpa_loc = sde.local_rpa_sde_correction(dmft_input=dmft1p, box_sizes=dga_conf.box,
                                                        iw=dga_conf.box.wn_rpa)
dmft_sde = lr.add_rpa_correction(dmft_sde=dmft_sde, rpa_sde_loc=rpa_sde_loc, wn_rpa=dga_conf.box.wn_rpa,
                                 sigma_comp=sigma_com_loc)

if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'dmft_sde.npy', dmft_sde, allow_pickle=True)
if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'rpa_sde_loc.npy', rpa_sde_loc, allow_pickle=True)
if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'chi_dmft.npy', chi_dmft, allow_pickle=True)
if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'chi_rpa_loc.npy', chi_rpa_loc, allow_pickle=True)
if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'vrg_dmft.npy', vrg_dmft, allow_pickle=True)
if (comm.rank == 0): plotting.plot_vrg_dmft(vrg_dmft=vrg_dmft, beta=dga_conf.sys.beta, niv_plot=dga_conf.box.niv_urange,
                                            output_path=dga_conf.nam.output_path)



logger.log_cpu_time(task=' DMFT SDE ')
# ------------------------------------------- ANALYZE OMEGA=0 CONTRIBUTION ---------------------------------------------
if (dga_conf.opt.analyse_w0_contribution):
    dmft_sde_w0, _, _, sigma_com_loc_w0 = lr.local_dmft_sde_from_gamma(dga_conf=dga_conf, giw=dmft1p['gloc'],
                                                                       gamma_dmft=gamma_dmft, ana_w0=True)

# Plot Chi DMFT:
wn_list = [dga_conf.box.wn_core, dga_conf.box.wn_core]
chiw_list = [chi_dmft['dens'].mat, chi_dmft['magn'].mat]
labels = [r'$\chi_{dens}$', r'$\chi_{magn}$']
import matplotlib.pyplot as plt
plt.figure()
plotting.plot_chiw(wn_list=wn_list, chiw_list=chiw_list, labels_list=labels, channel='both', plot_dir=dga_conf.nam.output_path, niw_plot=20)

# Plot Siw-check:
vn_list = [dga_conf.box.vn_dmft, dga_conf.box.vn_urange]
siw_list = [dmft1p['sloc'], dmft_sde['siw']]
labels = [r'$\Sigma_{DMFT}(\nu)$', r'$\Sigma_{DMFT-SDE}(\nu)$']
plotting.plot_siw(vn_list=vn_list, siw_list=siw_list, labels_list=labels, plot_dir=dga_conf.nam.output_path, niv_plot=100, name='siw_check')
