# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Warning: many of the files I prepared from Motoharu do not have smom stored.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import sys, os

sys.path.append('../src/')
sys.path.append(os.environ['HOME'] + "/Programs/dga/LambdaDga/src")
sys.path.append('../ana_cont/')
sys.path.append(os.environ['HOME'] + "/Programs/dga/LambdaDga/ana_cont")
import numpy as np
import Hr as hr_mod
import Hk as hamk
import Indizes as ind
import w2dyn_aux
import MatsubaraFrequencies as mf
import BrillouinZone as bz
import LambdaDga as ldga
import time
import Output as output
import Input as input
import ChemicalPotential as chempot
import TwoPoint as twop
import OrnsteinZernickeFunction as ozfunc
import RealTime as rt
import gc
import Plotting as plotting
from mpi4py import MPI as mpi
import MpiAux as mpiaux
import Config as conf
import LocalRoutines as lr
import SDE as sde
import FourPoint as fp
import LambdaCorrection as lc

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
names.output_path = names.input_path

# Define names of input/output files:
names.fname_dmft = '1p-data.hdf5'
names.fname_g2 = 'g4iw_sym.hdf5'  # 'Vertex_sym.hdf5' #'g4iw_sym.hdf5'
names.fname_ladder_vertex = 'LadderVertex.hdf5'

# Define options:
options.do_analytic_continuation = True  # Perform analytic continuation using MaxEnt from Josef Kaufmann's ana_cont package.
options.do_pairing_vertex = True
options.keep_ladder_vertex = False
options.lambda_correction_type = 'sp'  # Available: ['spch','sp','none','sp_only']
options.use_urange_for_lc = False  # Use with care. This is not really tested and at least low k-grid samples don't look too good.
options.lc_use_only_positive = True  # Use only frequency box where susceptibility is positive for lambda correction.
options.analyse_spin_fermion_contributions = True  # Analyse the contributions of the Re/Im part of the spin-fermion vertex seperately
options.use_fbz = False  # Perform the calculation in the full BZ

# Analytic continuation flags:
dmft_fs_cont = False
dmft_fbz_cont = False
no_mu_adjust_fs_cont = False
no_mu_adjust_fbz_cont = False

# Create the real-space Hamiltonian:
t = 1.0
sys_param.hr = hr_mod.one_band_2d_t_tp_tpp(t=t, tp=-0.2 * t, tpp=0.1 * t)

gap0_sing = {
    'k': 'd-wave',
    'v': 'even'
}

gap0_trip = {
    'k': 'd-wave',
    'v': 'odd'
}

# Pairing vertex symmetries:
n_eig = 2
sym_sing = True
sym_trip = True

# Define frequency box-sizes:
box_sizes.niw_core = 10
box_sizes.niw_urange = 10  # This seems not to save enough to be used.
box_sizes.niv_core = 10
box_sizes.niv_invbse = 10
box_sizes.niv_urange = 20  # Must be larger than niv_invbse
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

# analytic continuation specifications:
nwr = 1001
wmax = 15 * t
use_preblur = True
bw_dmft = 0.01
err = 1e-2

# Parameter for the polynomial extrapolation to the Fermi-level:
n_extrap = 4
order_extrap = 3

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


# Create the DGA Config object:
dga_conf = conf.DgaConfig(BoxSizes=box_sizes, Options=options, SystemParameter=sys_param, Names=names,
                          ek_funk=hamk.ek_3d)

# --------------------------------------------- CREATE THE OUTPUT PATHS -------------------------------------------------
if (comm.rank == 0): os.mkdir(dga_conf.nam.output_path)
if (comm.rank == 0): os.mkdir(dga_conf.nam.output_path_sp)

# ------------------------------------------------ DMFT 1P INPUT -------------------------------------------------------
dmft1p = input.load_1p_data(dga_conf=dga_conf)
box_sizes.niv_dmft = dmft1p['niv']
sys_param.beta = dmft1p['beta']
sys_param.u = dmft1p['u']
sys_param.n = dmft1p['n']

if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'config.npy', dga_conf)

# ----------------------------------------------- LOAD GAMMA -----------------------------------------------------------
gamma_dmft = input.get_gamma_loc(dga_conf=dga_conf, giw_dmft=dmft1p['gloc'])

if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'gamma_dmft.npy', gamma_dmft, allow_pickle=True)
if (comm.rank == 0): plotting.plot_gamma_dmft(gamma_dmft=gamma_dmft, output_path=dga_conf.nam.output_path,
                                              niw_core=dga_conf.box.niw_core)

# -------------------------------------- LOCAL SCHWINGER DYSON EQUATION ------------------------------------------------
dmft_sde, chi_dmft, vrg_dmft = lr.local_dmft_sde_from_gamma(dga_conf=dga_conf, giw=dmft1p['gloc'],
                                                            gamma_dmft=gamma_dmft)
rpa_sde_loc, chi_rpa_loc = sde.local_rpa_sde_correction(dmft_input=dmft1p, box_sizes=dga_conf.box,
                                                        iw=dga_conf.box.wn_rpa)
dmft_sde = lr.add_rpa_correction(dmft_sde=dmft_sde, rpa_sde_loc=rpa_sde_loc, wn_rpa=dga_conf.box.wn_rpa)

if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'dmft_sde.npy', dmft_sde, allow_pickle=True)
if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'rpa_sde_loc.npy', rpa_sde_loc, allow_pickle=True)
if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'chi_dmft.npy', chi_dmft, allow_pickle=True)
if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'chi_rpa_loc.npy', chi_rpa_loc, allow_pickle=True)
if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'vrg_dmft.npy', vrg_dmft, allow_pickle=True)
if (comm.rank == 0): plotting.plot_vrg_dmft(vrg_dmft=vrg_dmft, beta=dga_conf.sys.beta, niv_plot=dga_conf.box.niv_urange,
                                            output_path=dga_conf.nam.output_path)

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
chi_rpa = fp.rpa_susceptibility(dmft_input=dmft1p, dga_conf=dga_conf, qiw_indizes=qiw_grid_rpa.my_mesh)

# ----------------------------------------- NON-LOCAL LADDER SUCEPTIBILITY  --------------------------------------------
qiw_distributor.open_file()
chi_dga, vrg_dga = fp.dga_susceptibility(dga_conf=dga_conf, dmft_input=dmft1p, qiw_grid=qiw_grid.my_mesh,
                                         file=qiw_distributor.file, gamma_dmft=gamma_dmft)
qiw_distributor.close_file()

# ----------------------------------------------- LAMBDA-CORRECTION ----------------------------------------------------


chi_dens_rpa = fp.ladder_susc_gather_qiw_and_build_fbziw(distributor=qiw_distributor_rpa, mat=chi_rpa['dens'].mat,
                                             qiw_grid=qiw_grid_rpa, qiw_grid_fbz=qiw_grid_fbz, dga_conf=dga_conf,
                                             channel='dens')
chi_magn_rpa = fp.ladder_susc_gather_qiw_and_build_fbziw(distributor=qiw_distributor_rpa, mat=chi_rpa['magn'].mat,
                                             qiw_grid=qiw_grid_rpa, qiw_grid_fbz=qiw_grid_fbz, dga_conf=dga_conf,
                                             channel='magn')

chi_rpa = {
    'dens': chi_dens_rpa,
    'magn': chi_magn_rpa,
}

chi_dens_ladder = fp.ladder_susc_gather_qiw_and_build_fbziw(distributor=qiw_distributor, mat=chi_dga['dens'].mat, qiw_grid=qiw_grid,
                                                qiw_grid_fbz=qiw_grid_fbz, dga_conf=dga_conf, channel='dens')
chi_magn_ladder = fp.ladder_susc_gather_qiw_and_build_fbziw(distributor=qiw_distributor, mat=chi_dga['magn'].mat, qiw_grid=qiw_grid,
                                                qiw_grid_fbz=qiw_grid_fbz, dga_conf=dga_conf, channel='magn')
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

if (comm.rank == 0):
    string_temp = 'Number of positive frequencies used for {}: {}'
    np.savetxt(dga_conf.nam.output_path + 'n_lambda_correction.txt',
               [string_temp.format('magn', n_lambda['magn']), string_temp.format('dens', n_lambda['dens'])],
               delimiter=' ', fmt='%s')
    string_temp = 'Lambda for {}: {}'
    np.savetxt(dga_conf.nam.output_path + 'lambda.txt',
               [string_temp.format('magn', lambda_['magn']), string_temp.format('dens', lambda_['dens'])],
               delimiter=' ', fmt='%s')

lc.build_chi_lambda(dga_conf=dga_conf, chi_ladder=chi_dga, chi_rpa=chi_rpa, lambda_=lambda_)

if (comm.rank == 0): fp.gather_save_and_plot_chi_lambda(dga_conf=dga_conf, chi_dga=chi_dga, distributor=qiw_distributor,
                                                        qiw_grid=qiw_grid, qiw_grid_fbz=qiw_grid_fbz)

# ------------------------------------------- DGA SCHWINGER-DYSON EQUATION ---------------------------------------------

sigma_dga, sigma_dga_components = sde.sde_dga_wrapper(dga_conf=dga_conf, vrg=vrg_dga, chi=chi_dga, qiw_grid=qiw_grid,
                                              dmft_input=dmft1p, distributor=qiw_distributor)

sigma_rpa = sde.rpa_sde_wrapper(dga_conf = dga_conf, dmft_input=dmft1p, chi=chi_rpa, qiw_grid=qiw_grid_rpa, distributor=qiw_distributor_rpa)
if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'sigma_rpa.npy', sigma_rpa, allow_pickle=True)

sigma_dga = sde.build_dga_sigma(dga_conf = dga_conf, sigma_dga = sigma_dga, sigma_rpa = sigma_rpa, dmft_sde = dmft_sde, dmft1p=dmft1p)

# --------------------------------------------------- PLOTTING ---------------------------------------------------------
if (comm.rank == 0): np.save(dga_conf.nam.output_path + 'sigma_dga.npy', sigma_dga, allow_pickle=True)
if (comm.rank == 0): plotting.sigma_plots(dga_conf = dga_conf, sigma_dga=sigma_dga, dmft_sde=dmft_sde, dmft1p=dmft1p)
if (comm.rank == 0): plotting.giwk_plots(dga_conf = dga_conf, sigma=sigma_dga['sigma'],  dmft1p=dmft1p, output_path=dga_conf.nam.output_path)
if (comm.rank == 0): plotting.giwk_plots(dga_conf = dga_conf, sigma=sigma_dga['sigma_nc'],  dmft1p=dmft1p, name='_nc', output_path=dga_conf.nam.output_path)



# Plot the contribution of the real/imaginary part of the spin-fermion vertex.
if(dga_conf.opt.analyse_spin_fermion_contributions and comm.rank == 0):
    output.spin_fermion_contributions_output(dga_conf=dga_conf, sigma_dga_contributions=sigma_dga_components)
    sigma_vrg_re = sde.buid_dga_sigma_vrg_re(dga_conf = dga_conf, sigma_dga_components=sigma_dga_components, sigma_rpa=sigma_rpa, dmft_sde=dmft_sde,
                          dmft1p=dmft1p)
    plotting.giwk_plots(dga_conf=dga_conf, sigma=sigma_vrg_re, dmft1p=dmft1p, name='_vrg_re',
                        output_path=dga_conf.nam.output_path_sp)

if (comm.rank == 0): output.prepare_and_plot_vrg_dga(dga_conf = dga_conf, distributor=qiw_distributor)

# ---------------------------------------------- SPIN FERMION VERTEX ---------------------------------------------------
# # %%



#
# # %%
# # ------------------------------------------------ MAIN ----------------------------------------------------------------
# if (comm.rank == 0):
#     log = lambda s, *a: sys.stderr.write(str(s) % a + "\n")
#     rerr = sys.stderr
# else:
#     log = lambda s, *a: None
#     rerr = open(os.devnull, "w")
#
# log("Running on %d core%s", comm.size, " s"[comm.size > 1])
# log("Calculation started %s", time.strftime("%c"))
#
# comm.Barrier()
#
# if (comm.rank == 0):
#     os.mkdir(output_path)
#     np.save(output_path + 'config.npy', config_dump)
#
# comm.Barrier()
#

#
#
#
#
# #%% OZ-fit to the lambda-correction susceptibility:
# if(comm.rank == 0):
#     chi_lambda = np.load(output_path + 'chi_lambda.npy', allow_pickle=True).item()
#     np.savetxt(output_path + 'lambda_values.txt', [chi_lambda['lambda_dens'], chi_lambda['lambda_magn']], delimiter=',',
#                fmt='%.9f')
#
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.plot(chi_lambda['chi_magn_lambda'].mat.real[ind_node])
#     plt.plot(chi_lambda['chi_magn_lambda'].mat.real[ind_anti_node])
#     plt.savefig(output_path + 'chi_magn_lambda_node_anti_node.png')
#     plt.show()
#
#     plt.figure()
#     plt.plot(chi_lambda['chi_magn_lambda'].mat.imag[ind_node])
#     plt.plot(chi_lambda['chi_magn_lambda'].mat.imag[ind_anti_node])
#     plt.savefig(output_path + 'chi_magn_lambda_node_anti_node_imag.png')
#     plt.show()
#
#     try:
#         oz_coeff, _ = ozfunc.fit_oz_spin(q_grid, chi_lambda['chi_magn_lambda'].mat[:, :, :, niw_core].real.flatten())
#     except:
#         oz_coeff = [-1,-1]
#
#     np.savetxt(output_path + 'oz_coeff.txt', oz_coeff, delimiter=',', fmt='%.9f')
#     plotting.plot_oz_fit(chi_w0=chi_lambda['chi_magn_lambda'].mat[:, :, :, niw_core], oz_coeff=oz_coeff, qgrid=q_grid,
#                          pdir=output_path, name='oz_fit')
#
#     plotting.plot_chi_fs(chi=chi_lambda['chi_magn_lambda'].mat.real, output_path=output_path, kgrid=q_grid,
#                          name='magn_w0')
#     plotting.plot_chi_fs(chi=chi_lambda['chi_dens_lambda'].mat.real, output_path=output_path, kgrid=q_grid,
#                          name='dens_w0')
#

# #%%
# # Extrapolate the self-energy to the Fermi-level via polynomial fit:
# if(comm.rank == 0):
#     import AnalyticContinuation as a_cont
#     v = mf.v_plus(beta=dmft1p['beta'], n=niv_urange)
#
#     siwk_re_fs, siwk_im_fs, siwk_Z = a_cont.extract_coeff_on_ind(siwk=dga_sde['sigma'].reshape(-1,dga_sde['sigma'].shape[-1])[:,niv_urange:],indizes=k_grid.irrk_ind, v=v, N=n_extrap, order=order_extrap)
#     siwk_re_fs = k_grid.irrk2fbz(mat=siwk_re_fs)
#     siwk_im_fs = k_grid.irrk2fbz(mat=siwk_im_fs)
#     siwk_Z = k_grid.irrk2fbz(mat=siwk_Z)
#
#     siwk_extrap = {
#         'siwk_re_fs': siwk_re_fs,
#         'siwk_im_fs': siwk_im_fs,
#         'siwk_Z': siwk_Z
#     }
#
#     np.save(output_path + 'siwk_extrap.npy', siwk_extrap, allow_pickle=True)
#
#     plotting.plot_siwk_extrap(siwk_re_fs=siwk_re_fs, siwk_im_fs=siwk_im_fs, siwk_Z=siwk_Z, output_path=output_path, name='siwk_fs_extrap', k_grid=k_grid)
#
#     # Do the same for the nc self-energy
#     siwk_re_fs, siwk_im_fs, siwk_Z = a_cont.extract_coeff_on_ind(siwk=dga_sde['sigma_nc'].reshape(-1,dga_sde['sigma_nc'].shape[-1])[:,niv_urange:],indizes=k_grid.irrk_ind, v=v, N=n_extrap, order=order_extrap)
#     siwk_re_fs = k_grid.irrk2fbz(mat=siwk_re_fs)
#     siwk_im_fs = k_grid.irrk2fbz(mat=siwk_im_fs)
#     siwk_Z = k_grid.irrk2fbz(mat=siwk_Z)
#
#     siwk_extrap_nc = {
#         'siwk_re_fs': siwk_re_fs,
#         'siwk_im_fs': siwk_im_fs,
#         'siwk_Z': siwk_Z
#     }
#
#     np.save(output_path + 'siwk_extrap_nc.npy', siwk_extrap_nc, allow_pickle=True)
#
#     plotting.plot_siwk_extrap(siwk_re_fs=siwk_re_fs, siwk_im_fs=siwk_im_fs, siwk_Z=siwk_Z, output_path=output_path, name='siwk_fs_extrap_nc', k_grid=k_grid)
#
#
#     # Do the same, but for the Green's function:
#     giwk_re_fs, giwk_im_fs, giwk_Z = a_cont.extract_coeff_on_ind(siwk=gf_dict['gk'].reshape(-1,dga_sde['sigma'].shape[-1])[:,niv_urange:],indizes=k_grid.irrk_ind, v=v, N=n_extrap, order=order_extrap)
#     giwk_re_fs = k_grid.irrk2fbz(mat=giwk_re_fs)
#     giwk_im_fs = k_grid.irrk2fbz(mat=giwk_im_fs)
#     giwk_Z = k_grid.irrk2fbz(mat=giwk_Z)
#
#     giwk_extrap = {
#         'giwk_re_fs': giwk_re_fs,
#         'giwk_im_fs': giwk_im_fs,
#         'giwk_Z': giwk_Z
#     }
#
#     np.save(output_path + 'giwk_extrap.npy', giwk_extrap, allow_pickle=True)
#
#     plotting.plot_siwk_extrap(siwk_re_fs=giwk_re_fs, siwk_im_fs=giwk_im_fs, siwk_Z=giwk_Z, output_path=output_path, name='giwk_fs_extrap', k_grid=k_grid)
#
#     # Do the same, but for the Green's function:
#     giwk_re_fs, giwk_im_fs, giwk_Z = a_cont.extract_coeff_on_ind(siwk=gf_dict_mu_dmft['gk'].reshape(-1,dga_sde['sigma'].shape[-1])[:,niv_urange:],indizes=k_grid.irrk_ind, v=v, N=n_extrap, order=order_extrap)
#     giwk_re_fs = k_grid.irrk2fbz(mat=giwk_re_fs)
#     giwk_im_fs = k_grid.irrk2fbz(mat=giwk_im_fs)
#     giwk_Z = k_grid.irrk2fbz(mat=giwk_Z)
#
#     giwk_extrap = {
#         'giwk_re_fs': giwk_re_fs,
#         'giwk_im_fs': giwk_im_fs,
#         'giwk_Z': giwk_Z
#     }
#
#     np.save(output_path + 'giwk_extrap_mu_dmft.npy', giwk_extrap, allow_pickle=True)
#
#     plotting.plot_siwk_extrap(siwk_re_fs=giwk_re_fs, siwk_im_fs=giwk_im_fs, siwk_Z=giwk_Z, output_path=output_path, name='giwk_fs_extrap_mu_dmft', k_grid=k_grid)
#
#     del gf_dict, gf_dict_mu_dmft
#     gc.collect()

#
#
# # --------------------------------------------- ANALYTIC CONTINUATION --------------------------------------------------
# # %%
#
# if(do_analytic_continuation):
#     import RealTime as rt
#     import AnalyticContinuation as a_cont
#
#     nfit = np.min((np.max((niv_core, int(dmft1p['beta'] * 4))),niv_urange))
#     v_real = a_cont.v_real_tan(wmax=wmax, nw=nwr)
#     #Use a range of blur widths to have an automatic check on the stability and quality of continuation results.
#     bw_range_loc = [3./dmft1p['beta'] * t*0.5, 3./dmft1p['beta'] * t, 3./dmft1p['beta'] * t*2, 0.0]
#     bw_range_arc = [3./dmft1p['beta'] * t]
#     # For irrk use only continuation, as this will otherwise take too long and generate too many files.
#     bw_irrk_range = [3./dmft1p['beta'] * t*2, 3./dmft1p['beta'] * t] # I use 3 instead of pi here.
#     output_path_ana_cont = output.uniquify(output_path + 'AnaCont') + '/'
#
#
#     if(comm.rank==0):
#         os.mkdir(output_path_ana_cont)
#         realt = rt.real_time()
#         realt.create_file(fname=output_path_ana_cont+'cpu_time_ana_cont.txt')
#         text_file = open(output_path_ana_cont + 'cont_settings.txt', 'w')
#         text_file.write(f'nfit={nfit} \n')
#         text_file.write(f'err={err} \n')
#         text_file.write(f'wmax={wmax} \n')
#         text_file.write(f'nwr={nwr} \n')
#         text_file.close()
#
#
# # Do analytic continuation of local part:
# if(do_analytic_continuation and comm.rank == 0):
#
#     # DMFT Green's function:
#     gloc_dmft_cont, gk_dmft = a_cont.max_ent_loc(v_real=v_real, sigma=dmft1p['sloc'], config=config, k_grid=k_grid,
#                                                  niv_cut=niv_urange, use_preblur=use_preblur, bw=bw_dmft, err=err,
#                                                  nfit=nfit, adjust_mu=True)
#     plotting.plot_aw_loc(output_path=output_path_ana_cont, v_real=v_real, gloc=gloc_dmft_cont, name='aw-dmft')
#     n_int = a_cont.check_filling(v_real=v_real, gloc_cont=gloc_dmft_cont)
#     np.savetxt(output_path_ana_cont + 'n_dmft.txt', [n_int, gk_dmft['n']], delimiter=',', fmt='%.9f')
#     np.save(output_path_ana_cont + 'gloc_cont_dmft.npy',gloc_dmft_cont, allow_pickle=True)
#
#     for bw in bw_range_loc:
#         # mu-adjusted DGA Green's function:
#         gloc_dga_cont, gk_dga = a_cont.max_ent_loc(v_real=v_real, sigma=dga_sde['sigma'], config=config, k_grid=k_grid,
#                                                      niv_cut=niv_urange, use_preblur=use_preblur, bw=bw, err=err,
#                                                      nfit=nfit, adjust_mu=True)
#         plotting.plot_aw_loc(output_path=output_path_ana_cont, v_real=v_real, gloc=gloc_dga_cont, name='aw-dga-bw{}'.format(bw))
#         n_int = a_cont.check_filling(v_real=v_real, gloc_cont=gloc_dga_cont)
#         np.savetxt(output_path_ana_cont + 'n_dga_bw{}.txt'.format(bw), [n_int, gk_dga['n']], delimiter=',', fmt='%.9f')
#         np.save(output_path_ana_cont + 'gloc_cont_dga_bw{}.npy'.format(bw),gloc_dga_cont, allow_pickle=True)
#
#         # not mu-adjusted DGA Green's function:
#         gloc_dga_cont_nma, gk_dga_nma = a_cont.max_ent_loc(v_real=v_real, sigma=dga_sde['sigma'], config=config, k_grid=k_grid,
#                                                      niv_cut=niv_urange, use_preblur=use_preblur, bw=bw, err=err,
#                                                      nfit=nfit, adjust_mu=False)
#         plotting.plot_aw_loc(output_path=output_path_ana_cont, v_real=v_real, gloc=gloc_dga_cont_nma, name='aw-dga-no-mu-adjust-bw{}'.format(bw))
#         n_int = a_cont.check_filling(v_real=v_real, gloc_cont=gloc_dga_cont_nma)
#         np.savetxt(output_path_ana_cont + 'n_dga_no_mu_adjust_bw{}.txt'.format(bw), [n_int, gk_dga_nma['n']], delimiter=',', fmt='%.9f')
#         np.save(output_path_ana_cont + 'gloc_cont_dga_no_mu_adjust_bw{}.npy'.format(bw),gloc_dga_cont_nma, allow_pickle=True)
#
#     if(comm.rank==0): realt.write_time_to_file(string='Local spectral function continuation:',rank=comm.rank)
#
#
#
# #%%
# # Do analytic continuation along Fermi-surface:
# if(do_analytic_continuation and comm.rank == 0):
#
#     if(dmft_fs_cont):
#         # DMFT Green's function:
#         gk_fs_dmft_cont, ind_gf0_dmft, gk_dmft = a_cont.max_ent_on_fs(v_real=v_real, sigma=dmft1p['sloc'], config=config,
#                                                              k_grid=k_grid,
#                                                              niv_cut=niv_urange, use_preblur=use_preblur, bw=bw_dmft,
#                                                              err=err, nfit=nfit)
#
#         plotting.plot_ploints_on_fs(output_path=output_path_ana_cont, gk_fs=gk_dmft['gk'][:, :, 0, gk_dmft['niv']], k_grid=k_grid,
#                                     ind_fs=ind_gf0_dmft,
#                                     name='fermi_surface_dmft')
#
#         plotting.plot_aw_ind(output_path=output_path_ana_cont, v_real=v_real, gk_cont=gk_fs_dmft_cont, ind=ind_gf0_dmft,
#                              name='aw-dmft-fs-wide')
#         plotting.plot_aw_ind(output_path=output_path_ana_cont, v_real=v_real, gk_cont=gk_fs_dmft_cont, ind=ind_gf0_dmft,
#                              name='aw-dmft-fs-narrow', xlim=(-t, t))
#
#         np.save(output_path_ana_cont + 'gk_fs_dmft_cont.npy',gk_fs_dmft_cont, allow_pickle=True)
#
#     for bw in bw_range_arc:
#         # mu-adjusted DGA Green's function:
#         gk_fs_dga_cont, ind_gf0_dga, gk_dga = a_cont.max_ent_on_fs(v_real=v_real, sigma=dga_sde['sigma'], config=config,
#                                                              k_grid=k_grid,
#                                                              niv_cut=niv_urange, use_preblur=use_preblur, bw=bw,
#                                                              err=err, nfit=nfit)
#
#         plotting.plot_ploints_on_fs(output_path=output_path_ana_cont, gk_fs=gk_dga['gk'][:, :, 0, gk_dga['niv']], k_grid=k_grid,
#                                     ind_fs=ind_gf0_dga,
#                                     name='fermi_surface_dga'.format(bw))
#
#         plotting.plot_aw_ind(output_path=output_path_ana_cont, v_real=v_real, gk_cont=gk_fs_dga_cont, ind=ind_gf0_dga,
#                              name='aw-dga-fs-wide-bw{}'.format(bw))
#         plotting.plot_aw_ind(output_path=output_path_ana_cont, v_real=v_real, gk_cont=gk_fs_dga_cont, ind=ind_gf0_dga,
#                              name='aw-dga-fs-narrow-bw{}'.format(bw), xlim=(-t, t))
#         np.save(output_path_ana_cont + 'gk_fs_dga_cont_bw{}.npy'.format(bw), gk_fs_dga_cont, allow_pickle=True)
#
#         # not mu-adjusted DGA Green's function:
#         if(no_mu_adjust_fs_cont):
#             gk_fs_dga_cont_nma, ind_gf0_dgat_nma, gk_dgat_nma = a_cont.max_ent_on_fs(v_real=v_real, sigma=dga_sde['sigma'], config=config,
#                                                                  k_grid=k_grid,
#                                                                  niv_cut=niv_urange, use_preblur=use_preblur, bw=bw,
#                                                                  err=err, nfit=nfit, adjust_mu=False)
#
#             plotting.plot_ploints_on_fs(output_path=output_path_ana_cont, gk_fs=gk_dgat_nma['gk'][:, :, 0, gk_dgat_nma['niv']], k_grid=k_grid,
#                                         ind_fs=ind_gf0_dgat_nma,
#                                         name='fermi_surface_dga-no-mu-adjust'.format(bw))
#
#             plotting.plot_aw_ind(output_path=output_path_ana_cont, v_real=v_real, gk_cont=gk_fs_dga_cont_nma, ind=ind_gf0_dgat_nma,
#                                  name='aw-dga-fs-wide-no-mu-adjust-bw{}'.format(bw))
#             plotting.plot_aw_ind(output_path=output_path_ana_cont, v_real=v_real, gk_cont=gk_fs_dga_cont_nma, ind=ind_gf0_dgat_nma,
#                                  name='aw-dga-fs-narrow-no-mu-adjust-bw{}'.format(bw), xlim=(-t, t))
#             np.save(output_path_ana_cont + 'gk_fs_dga_cont_no_mu_adjust_bw{}.npy'.format(bw), gk_fs_dga_cont_nma, allow_pickle=True)
#
#     realt.write_time_to_file(string='Continuation of spectral function along Fermi-arc:',rank=comm.rank)
#
# #%%
# # Do analytic continuation within the irreducible Brillouin Zone:
# gc.collect() # garbage collection
# if(do_analytic_continuation):
#
#     irrk_distributor = mpiaux.MpiDistributor(ntasks=k_grid.nk_irr, comm=comm)
#
#     index_grid_keys = ('irrk',)
#     irrk_grid = ind.IndexGrids(grid_arrays=(k_grid.irrk_ind_lin,), keys=index_grid_keys,
#                               my_slice=irrk_distributor.my_slice)
#     ind_irrk = np.squeeze(np.array(np.unravel_index(k_grid.irrk_ind[irrk_grid.my_indizes], shape=k_grid.nk))).T
#     if(np.size(ind_irrk.shape) > 1):
#         ind_irrk = [tuple(ind_irrk[i, :]) for i in np.arange(ind_irrk.shape[0])]
#     else:
#         ind_irrk = tuple(ind_irrk)
#
#     if(dmft_fbz_cont):
#         # DMFT Green's function:
#         gk = twop.create_gk_dict(sigma=dmft1p['sloc'], kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'], n=dmft1p['n'],
#                                       mu0=dmft1p['mu'], adjust_mu=True, niv_cut=niv_urange)
#
#
#
#         gk_my_cont = a_cont.do_max_ent_on_ind_T(mat=gk['gk'], ind_list=ind_irrk, v_real=v_real,
#                                                        beta=dmft1p['beta'],
#                                                        n_fit=nfit, err=err, alpha_det_method='chi2kink', use_preblur = use_preblur, bw=bw_dmft)
#
#         gk_cont = irrk_distributor.allgather(rank_result=gk_my_cont)
#         if(comm.rank == 0):
#             gk_cont_fbz = k_grid.irrk2fbz(mat=gk_cont)
#             plotting.plot_cont_fs(output_path=output_path_ana_cont, name='fermi_surface_dmft_cont',
#                                   gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=None)
#             np.save(output_path_ana_cont + 'gk_dmft_cont_fbz.npy', gk_cont_fbz, allow_pickle=True)
#             plotting.plot_cont_edc_maps(v_real=v_real, gk_cont=gk_cont_fbz, k_grid=k_grid, output_path=output_path_ana_cont, name='fermi_surface_dmft_cont_edc_maps')
#
#     for bw in bw_irrk_range:
#         # mu-adjust DGA Green's function:
#         gk = twop.create_gk_dict(sigma=dga_sde['sigma'], kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'], n=dmft1p['n'],
#                                       mu0=dmft1p['mu'], adjust_mu=True, niv_cut=niv_urange)
#         gk_my_cont = a_cont.do_max_ent_on_ind_T(mat=gk['gk'], ind_list=ind_irrk, v_real=v_real,
#                                                        beta=dmft1p['beta'],
#                                                        n_fit=nfit, err=err, alpha_det_method='chi2kink', use_preblur = use_preblur, bw=bw)
#
#         comm.Barrier()
#         gk_cont = irrk_distributor.allgather(rank_result=gk_my_cont)
#         comm.Barrier()
#         if(comm.rank == 0):
#             gk_cont_fbz = k_grid.irrk2fbz(mat=gk_cont)
#
#             plotting.plot_cont_fs(output_path=output_path_ana_cont,
#                                   name='fermi_surface_dga_cont_wint-0.1-bw{}'.format(bw),
#                                   gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=-0.1)
#             plotting.plot_cont_fs(output_path=output_path_ana_cont, name='fermi_surface_dga_cont_wint-0.05-bw{}'.format(bw),
#                                   gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=-0.05)
#             plotting.plot_cont_fs(output_path=output_path_ana_cont, name='fermi_surface_dga_cont_w0-bw{}'.format(bw),
#                                   gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=None)
#             np.save(output_path_ana_cont + 'gk_dga_cont_fbz_bw{}.npy'.format(bw), gk_cont_fbz, allow_pickle=True)
#             plotting.plot_cont_edc_maps(v_real=v_real, gk_cont=gk_cont_fbz, k_grid=k_grid,
#                                         output_path=output_path_ana_cont, name='fermi_surface_dga_cont_edc_maps_bw{}'.format(bw))
#
#         # not mu-adjust DGA Green's function:
#         if(no_mu_adjust_fbz_cont):
#             gk = twop.create_gk_dict(sigma=dga_sde['sigma'], kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'], n=dmft1p['n'],
#                                      mu0=dmft1p['mu'], adjust_mu=False, niv_cut=niv_urange)
#             gk_my_cont = a_cont.do_max_ent_on_ind_T(mat=gk['gk'], ind_list=ind_irrk, v_real=v_real,
#                                                   beta=dmft1p['beta'],
#                                                   n_fit=nfit, err=err, alpha_det_method='chi2kink', use_preblur=use_preblur,
#                                                   bw=bw)
#
#             comm.Barrier()
#             gk_cont = irrk_distributor.allgather(rank_result=gk_my_cont)
#             comm.Barrier()
#             if (comm.rank == 0):
#                 gk_cont_fbz = k_grid.irrk2fbz(mat=gk_cont)
#                 plotting.plot_cont_fs(output_path=output_path_ana_cont,
#                                       name='fermi_surface_dga_cont_wint-0.1_no_mu_adjust_bw{}'.format(bw),
#                                       gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=-0.1)
#                 plotting.plot_cont_fs(output_path=output_path_ana_cont, name='fermi_surface_dga_cont_wint-0.05_no_mu_adjust_bw{}'.format(bw),
#                                       gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=-0.05)
#                 plotting.plot_cont_fs(output_path=output_path_ana_cont, name='fermi_surface_dga_cont_w0_no_mu_adjust_bw{}'.format(bw),
#                                       gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=None)
#
#                 np.save(output_path_ana_cont + 'gk_dga_cont_fbz_no_mu_adjust_bw{}.npy'.format(bw), gk_cont_fbz, allow_pickle=True)
#                 plotting.plot_cont_edc_maps(v_real=v_real, gk_cont=gk_cont_fbz, k_grid=k_grid,
#                                             output_path=output_path_ana_cont, name='fermi_surface_dga_no_mu_adjust_cont_edc_maps_bw{}'.format(bw))
#
#         gc.collect() # Garbage collection
#
#     if(comm.rank==0):  realt.write_time_to_file(string='Continuation of spectral function within the irreduzible Brillouin Zone:',rank=comm.rank)
#
# # ------------------------------------------------ PAIRING VERTEX ------------------------------------------------------
# # %%
# if (do_pairing_vertex and comm.rank == 0):
#     import RealTime as rt
#     import PairingVertex as pv
#
#
#     realt = rt.real_time()
#     realt.create_file(fname=output_path+'cpu_time_eliashberg.txt')
#
#     realt.write_time_to_file(string='Start pairing vertex:', rank=comm.rank)
#
#     qiw_distributor = mpiaux.MpiDistributor(ntasks=box_sizes['niw_core'] * q_grid.nk_irr, comm=comm,
#                                             output_path=output_path,
#                                             name='Qiw')
#
#     qiw_grid = ind.IndexGrids(grid_arrays=q_grid.grid + (grids['wn_core'],),
#                               keys=('qx', 'qy', 'qz', 'iw'),
#                               my_slice=None)
#
#     f1_magn, f2_magn, f1_dens, f2_dens = pv.load_pairing_vertex_from_rank_files(output_path=output_path, name='Qiw',
#                                                                                 mpi_size=comm.size, nq=q_grid.nk_irr,
#                                                                                 niv_pp=niv_pp)
#
#     # Create f_magn:
#     f1_magn = q_grid.irrk2fbz(mat=f1_magn)
#     f2_magn = q_grid.irrk2fbz(mat=f2_magn)
#     chi_magn_lambda_pp = pv.reshape_chi(chi=chi_lambda['chi_magn_lambda'].mat, niv_pp=niv_pp)
#     f_magn = f1_magn + (1 + dmft1p['u'] * chi_magn_lambda_pp) * f2_magn
#     del f1_magn, f2_magn, chi_magn_lambda_pp
#     gc.collect()
#     plotting.plot_vertex_vvp(vertex=f_magn.mean(axis=(0, 1, 2)).real, pdir=output_path, name='f_magn_loc')
#
#     # Create f_dens:
#     f1_dens = q_grid.irrk2fbz(mat=f1_dens)
#     f2_dens = q_grid.irrk2fbz(mat=f2_dens)
#     chi_dens_lambda_pp = pv.reshape_chi(chi=chi_lambda['chi_dens_lambda'].mat, niv_pp=niv_pp)
#     f_dens = f1_dens + (1 - dmft1p['u'] * chi_dens_lambda_pp) * f2_dens
#     del f1_dens, f2_dens, chi_dens_lambda_pp, chi_lambda
#     gc.collect()
#     plotting.plot_vertex_vvp(vertex=f_dens.mean(axis=(0, 1, 2)).real, pdir=output_path, name='f_dens_loc')
#
#     # Create f_sing and f_trip:
#     f_sing = -1.5 * f_magn + 0.5 * f_dens
#     f_trip = -0.5 * f_magn - 0.5 * f_dens
#     plotting.plot_vertex_vvp(vertex=f_sing.mean(axis=(0, 1, 2)).real, pdir=output_path, name='f_sing_loc')
#     plotting.plot_vertex_vvp(vertex=f_trip.mean(axis=(0, 1, 2)).real, pdir=output_path, name='f_trip_loc')
#
#     pairing_vertices = {
#         'f_sing': f_sing,
#         'f_trip': f_trip
#     }
#
#     np.save(output_path + 'pairing_vertices.npy', pairing_vertices)
#     realt.write_time_to_file(string='End pairing vertex:',rank=comm.rank)
# #
# # ----------------------------------------------- Eliashberg Equation --------------------------------------------------
# # %%
#
# if (do_pairing_vertex and comm.rank == 0):
#     import TwoPoint as twop
#     import EliashbergEquation as eq
#
#     realt.write_time_to_file(string='Start Eliashberg:', rank=comm.rank)
#     output_path_el = output.uniquify(output_path + 'Eliashberg') + '/'
#     os.mkdir(output_path_el)
#     gamma_sing = -f_sing
#     gamma_trip = -f_trip
#     #
#     if (sym_sing):
#         gamma_sing = 0.5 * (gamma_sing + np.flip(gamma_sing, axis=(-1)))
#
#     if (sym_trip):
#         gamma_trip = 0.5 * (gamma_trip - np.flip(gamma_trip, axis=(-1)))
#
#     plotting.plot_vertex_vvp(vertex=gamma_trip.mean(axis=(0, 1, 2)).real, pdir=output_path, name='gamma_trip_loc')
#
#     g_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=q_grid.grid, hr=hr,
#                                                sigma=dga_sde['sigma'])
#     mu_dga = g_generator.adjust_mu(n=dmft1p['n'], mu0=dmft1p['mu'])
#     gk_dga = g_generator.generate_gk(mu=mu_dga, qiw=[0, 0, 0, 0], niv=niv_pp).gk
#
#     gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=gap0_sing['k'], v_type=gap0_sing['v'],
#                             k_grid=q_grid.grid)
#     norm = np.prod(nq) * dmft1p['beta']
#     powiter_sing = eq.EliashberPowerIteration(gamma=gamma_sing, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
#                                               n_eig=n_eig)
#
#     gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=gap0_trip['k'], v_type=gap0_trip['v'],
#                             k_grid=q_grid.grid)
#     powiter_trip = eq.EliashberPowerIteration(gamma=gamma_trip, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
#                                               n_eig=n_eig)
#
#     eliashberg = {
#         'lambda_sing': powiter_sing.lam,
#         'lambda_trip': powiter_trip.lam,
#         'delta_sing': powiter_sing.gap,
#         'delta_trip': powiter_trip.gap,
#     }
#     np.save(output_path_el + 'eliashberg.npy', eliashberg)
#     np.savetxt(output_path_el + 'eigenvalues.txt', [powiter_sing.lam.real, powiter_trip.lam.real], delimiter=',',
#                fmt='%.9f')
#     np.savetxt(output_path_el + 'eigenvalues_s.txt', [powiter_sing.lam_s.real, powiter_trip.lam_s.real], delimiter=',',
#                fmt='%.9f')
#
#     for i in range(len(powiter_sing.gap)):
#         plotting.plot_gap_function(delta=powiter_sing.gap[i].real, pdir=output_path_el, name='sing_{}'.format(i),
#                                    kgrid=q_grid,
#                                    do_shift=True)
#         plotting.plot_gap_function(delta=powiter_sing.gap[i].real, pdir=output_path_el, name='trip_{}'.format(i),
#                                    kgrid=q_grid,
#                                    do_shift=True)
#     realt.write_time_to_file(string='End Eliashberg:', rank=comm.rank)
#
#
# # ---------------------------------------------- REMOVE THE RANK FILES -------------------------------------------------
# comm.Barrier()
# qiw = mpiaux.MpiDistributor(ntasks=k_grid.nk_irr, comm=comm,
#                                             output_path=output_path,
#                                             name='Qiw')
# qiw.delete_file()
