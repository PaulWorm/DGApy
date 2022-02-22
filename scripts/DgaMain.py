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
#names.input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.85/KonvergenceAnalysis/'
#names.input_path = '/mnt/d/Research/U2BenchmarkData/BenchmarkSchaefer_beta_15/LambdaDgaPython/'
# names.input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta15_n0.975/'
#names.input_path = '/mnt/c/Users/pworm/Research/Ba2CuO4/Plane1/U3.0eV_n0.93_b040/'
#names.input_path = '/mnt/d/Research/BenchmarkEliashberg/'
names.input_path = '/mnt/d/Research/HoleDopedNickelates/2DSquare_U8_tp-0.25_tpp0.12_beta50_n0.875/LambdaDgaPython/'
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
t = 0.25
hr = hr_mod.one_band_2d_t_tp_tpp(t=t, tp=-0.25 * t, tpp=0.12 * t)
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
box_sizes.niw_core = 20
box_sizes.niw_urange = 20  # This seems not to save enough to be used.
box_sizes.niv_core = 20
box_sizes.niv_invbse = 20
box_sizes.niv_urange = 80  # Must be larger than niv_invbse
box_sizes.niv_asympt = 0  # Don't use this for now.

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

# ----------------------------------------- CREATE MPI DISTRIBUTORS ----------------------------------------------------
qiw_distributor = mpiaux.MpiDistributor(ntasks=dga_conf.box.wn_core_plus.size * dga_conf.q_grid.nk_irr, comm=comm,
                                        output_path=dga_conf.nam.output_path,
                                        name='Qiw')
index_grid_keys = ('irrq', 'iw')
qiw_grid = ind.IndexGrids(grid_arrays=(dga_conf.q_grid.irrk_ind_lin,) + (dga_conf.box.wn_core_plus,),
                          keys=index_grid_keys,
                          my_slice=qiw_distributor.my_slice)

index_grid_keys_fbz = ('qx', 'qy', 'qz', 'iw')
qiw_grid_fbz = ind.IndexGrids(grid_arrays=dga_conf.q_grid.grid + (dga_conf.box.wn_core,), keys=index_grid_keys_fbz)

qiw_distributor_rpa = mpiaux.MpiDistributor(ntasks=dga_conf.box.wn_rpa_plus.size * dga_conf.q_grid.nk_irr, comm=comm)
qiw_grid_rpa = ind.IndexGrids(grid_arrays=(dga_conf.q_grid.irrk_ind_lin,) + (dga_conf.box.wn_rpa_plus,),
                              keys=index_grid_keys,
                              my_slice=qiw_distributor_rpa.my_slice)

# ----------------------------------------- NON-LOCAL RPA SUCEPTIBILITY  -----------------------------------------------
chi_rpa = fp.rpa_susceptibility(dmft_input=dmft1p, dga_conf=dga_conf, qiw_indizes=qiw_grid_rpa.my_mesh,
                                sigma=dmft1p['sloc'])

logger.log_cpu_time(task=' RPA susceptibility ')
# ----------------------------------------- NON-LOCAL LADDER SUCEPTIBILITY  --------------------------------------------
qiw_distributor.open_file()
chi_dga, vrg_dga = fp.dga_susceptibility(dga_conf=dga_conf, dmft_input=dmft1p, qiw_grid=qiw_grid.my_mesh,
                                         file=qiw_distributor.file, gamma_dmft=gamma_dmft, k_grid=dga_conf.k_grid,
                                         q_grid=dga_conf.q_grid, hr=dga_conf.sys.hr, sigma=dmft1p['sloc'])
qiw_distributor.close_file()


logger.log_cpu_time(task=' ladder susceptibility ')
# ----------------------------------------------- LAMBDA-CORRECTION ----------------------------------------------------


chi_dens_rpa = fp.ladder_susc_allgather_qiw_and_build_fbziw(dga_conf=dga_conf, distributor=qiw_distributor_rpa,
                                                            mat=chi_rpa['dens'].mat, qiw_grid=qiw_grid_rpa,
                                                            qiw_grid_fbz=qiw_grid_fbz, channel='dens')
chi_magn_rpa = fp.ladder_susc_allgather_qiw_and_build_fbziw(dga_conf=dga_conf, distributor=qiw_distributor_rpa,
                                                            mat=chi_rpa['magn'].mat, qiw_grid=qiw_grid_rpa,
                                                            qiw_grid_fbz=qiw_grid_fbz, channel='magn')

chi_rpa = {
    'dens': chi_dens_rpa,
    'magn': chi_magn_rpa,
}

chi_dens_ladder = fp.ladder_susc_allgather_qiw_and_build_fbziw(dga_conf=dga_conf, distributor=qiw_distributor,
                                                               mat=chi_dga['dens'].mat, qiw_grid=qiw_grid,
                                                               qiw_grid_fbz=qiw_grid_fbz, channel='dens')
chi_magn_ladder = fp.ladder_susc_allgather_qiw_and_build_fbziw(dga_conf=dga_conf, distributor=qiw_distributor,
                                                               mat=chi_dga['magn'].mat, qiw_grid=qiw_grid,
                                                               qiw_grid_fbz=qiw_grid_fbz, channel='magn')
chi_ladder = {
    'dens': chi_dens_ladder,
    'magn': chi_magn_ladder,
}

if (comm.rank == 0):
    np.save(dga_conf.nam.output_path + 'chi_ladder.npy', chi_ladder, allow_pickle=True)
    plotting.plot_chi_fs(chi=chi_ladder['magn'].mat.real, output_path=dga_conf.nam.output_path,
                         kgrid=dga_conf.q_grid,
                         name='magn_ladder_w0')

lambda_, n_lambda = lc.lambda_correction(dga_conf=dga_conf, chi_ladder=chi_ladder, chi_rpa=chi_rpa, chi_dmft=chi_dmft,
                                         chi_rpa_loc=chi_rpa_loc)

lc.build_chi_lambda(dga_conf=dga_conf, chi_ladder=chi_dga, chi_rpa=chi_rpa, lambda_=lambda_)

if (comm.rank == 0):
    string_temp = 'Number of positive frequencies used for {}: {}'
    np.savetxt(dga_conf.nam.output_path + 'n_lambda_correction.txt',
               [string_temp.format('magn', n_lambda['magn']), string_temp.format('dens', n_lambda['dens'])],
               delimiter=' ', fmt='%s')
    string_temp = 'Lambda for {}: {}'
    np.savetxt(dga_conf.nam.output_path + 'lambda.txt',
               [string_temp.format('magn', lambda_['magn']), string_temp.format('dens', lambda_['dens'])],
               delimiter=' ', fmt='%s')


chi_dens_lambda = fp.ladder_susc_allgather_qiw_and_build_fbziw(dga_conf=dga_conf, distributor=qiw_distributor,
                                                               mat=chi_dga['dens'].mat, qiw_grid=qiw_grid,
                                                               qiw_grid_fbz=qiw_grid_fbz, channel='dens')
chi_magn_lambda = fp.ladder_susc_allgather_qiw_and_build_fbziw(dga_conf=dga_conf, distributor=qiw_distributor,
                                                               mat=chi_dga['magn'].mat, qiw_grid=qiw_grid,
                                                               qiw_grid_fbz=qiw_grid_fbz, channel='magn')
chi_lambda = {
    'dens': chi_dens_lambda,
    'magn': chi_magn_lambda,
}

if (comm.rank == 0):
    fp.save_and_plot_chi_lambda(dga_conf=dga_conf, chi_lambda=chi_lambda, distributor=qiw_distributor,
                                                 qiw_grid=qiw_grid, qiw_grid_fbz=qiw_grid_fbz)

logger.log_cpu_time(task=' lambda correction plots ')
# ------------------------------------------- DGA SCHWINGER-DYSON EQUATION ---------------------------------------------

sigma_dga, sigma_dga_components = sde.sde_dga_wrapper(dga_conf=dga_conf, vrg=vrg_dga, chi=chi_dga,
                                                      qiw_mesh=qiw_grid.my_mesh,
                                                      dmft_input=dmft1p, distributor=qiw_distributor)

sigma_rpa = sde.rpa_sde_wrapper(dga_conf=dga_conf, dmft_input=dmft1p, chi=chi_rpa, qiw_grid=qiw_grid_rpa,
                                distributor=qiw_distributor_rpa)
if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'sigma_rpa.npy', sigma_rpa, allow_pickle=True)

sigma, sigma_nc = sde.build_dga_sigma(dga_conf=dga_conf, sigma_dga=sigma_dga, sigma_rpa=sigma_rpa, dmft_sde=dmft_sde,
                                dmft1p=dmft1p)
logger.log_cpu_time(task=' DGA SDE ')
# ------------------------------------------- ANALYZE OMEGA=0 CONTRIBUTION ---------------------------------------------

if (options.analyse_w0_contribution):
    ind_w0 = qiw_grid.my_mesh[:, 1] == 0
    sigma_dga_w0, sigma_dga_components_w0 = sde.sde_dga_wrapper(dga_conf=dga_conf, vrg=vrg_dga, chi=chi_dga,
                                                                qiw_mesh=qiw_grid.my_mesh[ind_w0, :],
                                                                dmft_input=dmft1p, distributor=qiw_distributor)
    sigma_w0, sigma_nc_w0 = sde.build_dga_sigma(dga_conf=dga_conf, sigma_dga=sigma_dga_w0, sigma_rpa=sigma_rpa,
                                       dmft_sde=dmft_sde_w0,
                                       dmft1p=dmft1p)
    if comm.rank == 0: np.save(dga_conf.nam.output_path + 'sigma_dga_w0.npy', sigma_dga_w0, allow_pickle=True)


logger.log_cpu_time(task=' DGA SDE (w=0) ')
# --------------------------------------------------- PLOTTING ---------------------------------------------------------
if comm.rank == 0: np.save(dga_conf.nam.output_path + 'sigma_dga.npy', sigma_dga, allow_pickle=True)
if comm.rank == 0: np.save(dga_conf.nam.output_path + 'sigma.npy', sigma, allow_pickle=True)
if comm.rank == 0: np.save(dga_conf.nam.output_path + 'sigma_nc.npy', sigma, allow_pickle=True)
if comm.rank == 0: plotting.sigma_plots(dga_conf=dga_conf, sigma_dga=sigma_dga, dmft_sde=dmft_sde, dmft1p=dmft1p, sigma=sigma, sigma_nc=sigma_nc)
if (dga_conf.opt.analyse_w0_contribution):
    if comm.rank == 0: plotting.sigma_plots(dga_conf=dga_conf, sigma_dga=sigma_dga_w0, dmft_sde=dmft_sde_w0,
                                            dmft1p=dmft1p, name='_w0', sigma=sigma_w0, sigma_nc=sigma_nc_w0)
if comm.rank == 0: plotting.giwk_plots(dga_conf=dga_conf, sigma=sigma, dmft1p=dmft1p,
                                       output_path=dga_conf.nam.output_path)
if comm.rank == 0: plotting.giwk_plots(dga_conf=dga_conf, sigma=sigma_nc, dmft1p=dmft1p, name='_nc',
                                       output_path=dga_conf.nam.output_path)

# Plot the contribution of the real/imaginary part of the spin-fermion vertex.
if dga_conf.opt.analyse_spin_fermion_contributions:
    if comm.rank == 0: output.spin_fermion_contributions_output(dga_conf=dga_conf,
                                                                sigma_dga_contributions=sigma_dga_components)
    sigma_vrg_re = sde.buid_dga_sigma_vrg_re(dga_conf=dga_conf, sigma_dga_components=sigma_dga_components,
                                             sigma_rpa=sigma_rpa, dmft_sde_comp=sigma_com_loc,
                                             dmft1p=dmft1p)
    if comm.rank == 0: plotting.giwk_plots(dga_conf=dga_conf, sigma=sigma_vrg_re, dmft1p=dmft1p, name='_vrg_re',
                                           output_path=dga_conf.nam.output_path_sp)

if comm.rank == 0: output.prepare_and_plot_vrg_dga(dga_conf=dga_conf, distributor=qiw_distributor)

logger.log_cpu_time(task=' Plotting ')
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

if comm.rank == 0 and dga_conf.opt.analyse_spin_fermion_contributions:
    gk = twop.create_gk_dict(dga_conf=dga_conf, sigma=sigma_vrg_re, mu0=dmft1p['mu'], adjust_mu=True)
    output.poly_fit(dga_conf=dga_conf, mat_data=gk['gk'], name='giwk_vrg_re')

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
        if dga_conf.opt.analyse_spin_fermion_contributions:
            bw_opt_vrg_re[0] = output.max_ent_loc_bw_range(dga_conf=dga_conf, me_conf=me_conf, bw_range=bw_range_loc,
                                                           sigma=sigma_vrg_re,
                                                           n_fit=n_fit, adjust_mu=True, name='dga_vrg_re')

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

    if dga_conf.opt.analyse_spin_fermion_contributions:
        # Do analytic continuation within the irreducible Brillouin Zone:
        output.max_ent_irrk_bw_range(comm=comm, dga_conf=dga_conf, me_conf=me_conf, bw_range=bw_opt_vrg_re,
                                     sigma=sigma_vrg_re, n_fit=n_fit, name='dga_vrg_re')
        logger.log_cpu_time(task=' MaxEnt irrk vrg_re ')

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
