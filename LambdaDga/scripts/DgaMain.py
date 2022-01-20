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
import ChemicalPotential as chempot
import TwoPoint as twop
import OrnsteinZernickeFunction as ozfunc

import Plotting as plotting
from mpi4py import MPI as mpi
import MpiAux as mpiaux
# ----------------------------------------------- PARAMETERS -----------------------------------------------------------

# Define MPI communicator:
comm = mpi.COMM_WORLD

# Define paths of datasets:
input_path = './'
# input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
# input_path = '/mnt/c/users/pworm/Research/BEPS_Project/TriangularLattice/DGA/TriangularLattice_U8.0_tp1.0_tpp0.0_beta10_n1.0/'
# input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U8_b010_tp0_tpp0_n0.85/LambdaDgaPython/'
# input_path = '/mnt/c/users/pworm/Research/BEPS_Project/TriangularLattice/DGA/TriangularLattice_U9.5_tp1.0_tpp0.0_beta10_n1.0/'
# input_path = '/mnt/d/Research/BEPS_Project/TriangularLattice/TriangularLattice_U9.0_tp1.0_tpp0.0_beta10_n1.0/'
# input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/KonvergenceAnalysis/'
# input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
# input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta80_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
# input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/NdNiO2_U8_n0.85_b75/'
# input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.2_tpp0.1_beta30_n0.90/'
# input_path = '/mnt/c/users/pworm/Research/U2BenchmarkData/2DSquare_U2_tp-0.0_tpp0.0_beta20_mu1/'
# input_path = '/mnt/c/users/pworm/Research/U2BenchmarkData/BenchmarkSchaefer_beta_15/LambdaDgaPython/'
# input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/Testset1/LambdaDga_Python/'
# input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.25_tpp0.12_beta12.5_n0.85/'
# input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.2_tpp0.1_beta70_n0.75/'
# input_path = '/mnt/c/users/pworm/Research/BEPS_Project/ElectronDoping/2DSquare_U8_tp-0.2_tpp0.1_beta25_n1.02/'
# input_path = '/mnt/c/users/pworm/Research/Ba2CuO4/Plane1/U3.0eV_n0.93_b040/'
#input_path = '/mnt/d/Research/HoleDopedNickelates/2DSquare_U8_tp-0.25_tpp0.12_beta25_n0.95/LambdaDgaPython/'
input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.875/'
output_path = input_path

fname_dmft = '1p-data.hdf5'
fname_g2 = 'g4iw_sym.hdf5'  # 'Vertex_sym.hdf5' #'g4iw_sym.hdf5'
fname_ladder_vertex = 'LadderVertex.hdf5'

# Define options:
do_analytic_continuation = True # Perform analytic continuation using MaxEnt from Josef Kaufmann's ana_cont package.
do_pairing_vertex = True
keep_ladder_vertex = False
lambda_correction_type = 'sp'  # Available: ['spch','sp','none','sp_only']
use_urange_for_lc = True  # Use with care. This is not really tested and at least low k-grid samples don't look too good.
verbose = True
lc_use_only_positive = True  # Use only frequency box where susceptibility is positive for lambda correction.

# Create the real-space Hamiltonian:
t = 1.0
hr = hr_mod.one_band_2d_t_tp_tpp(t=t,tp=0,tpp=0)
#hr = hr_mod.standard_cuprates(t=t)
# hr = hr_mod.unfrustrated_square(t=t)
#hr = hr_mod.motoharu_nickelates(t=t)
# hr = hr_mod.motoharu_nickelates_2(t=t)
# hr = hr_mod.Ba2CuO4_plane()


gap0_sing = {
    'k': 'd-wave',
    'v': 'even'
}

gap0_trip = {
    'k': 'd-wave',
    'v': 'odd'
}

# Pairing vertex symmetries:

sym_sing = True
sym_trip = True

# Define frequency box-sizes:
niw_core = 10
niw_urange = 10  # This seems not to save enough to be used.
niv_core = 10
niv_invbse = 10
niv_urange = 50  # Must be larger than niv_invbse
niv_asympt = 0  # Don't use this for now.

niv_pp = np.min((niw_core // 2, niv_core // 2))

# Define k-ranges:
nkx = 16
nky = nkx
nqx = 16
nqy = nqx

nk = (nkx, nky, 1)
nq = (nqx, nqy, 1)

# analytic continuation specifications:
nwr = 1001
wmax = 15*t
use_preblur = True
bw = 0.2
bw_dmft = 0.01
err = 1e-2
nfit = niv_core


output_folder = 'LambdaDga_lc_{}_Nk{}_Nq{}_core{}_invbse{}_vurange{}_wurange{}'.format(lambda_correction_type,
                                                                                       np.prod(nk), np.prod(nq),
                                                                                       niw_core, niv_invbse, niv_urange,
                                                                                       niw_urange)
output_path = output.uniquify(output_path + output_folder) + '/'
fname_ladder_vertex = output_path + fname_ladder_vertex

# Generate k-meshes:
k_grid = bz.KGrid(nk=nk)
q_grid = bz.KGrid(nk=nq)
ek = hamk.ek_3d(kgrid=k_grid.grid, hr=hr)
eq = hamk.ek_3d(kgrid=q_grid.grid, hr=hr)

k_grid.get_irrk_from_ek(ek=ek, dec=11)
q_grid.get_irrk_from_ek(ek=eq, dec=11)
print(f'{q_grid.nk_irr=}')

# load contents from w2dynamics DMFT file:
f1p = w2dyn_aux.w2dyn_file(fname=input_path + fname_dmft)
dmft1p = f1p.load_dmft1p_w2dyn()
f1p.close()
niv_dmft = dmft1p['niv']
if (dmft1p['n'] == 0.0): dmft1p['n'] = 1.0

# Define system paramters, like interaction or inverse temperature.
# Note: I want this to be decoupled from dmft1p, because of later RPA/FLEX stuff.

options = {
    'do_pairing_vertex': do_pairing_vertex,
    'lambda_correction_type': lambda_correction_type,
    'lc_use_only_positive': lc_use_only_positive,
    'use_urange_for_lc': use_urange_for_lc,
    'sym_sing': sym_sing,
    'sym_trip': sym_trip
}

system = {
    'u': dmft1p['u'],
    'beta': dmft1p['beta'],
    'n': dmft1p['n'],
    'hr': hr
}

names = {
    'input_path': input_path,
    'output_path': output_path,
    'fname_g2': fname_g2,
    'fname_ladder_vertex': fname_ladder_vertex
}

box_sizes = {
    "niv_dmft": niv_dmft,
    "niw_core": niw_core,
    "niw_urange": niw_urange,
    "niv_core": niv_core,
    "niv_invbse": niv_invbse,
    "niv_urange": niv_urange,
    "niv_asympt": niv_asympt,
    "niv_pp": niv_pp,
    "nk": nk,
    "nq": nq
}

grids = {
    "vn_dmft": mf.vn(n=niv_dmft),
    "vn_core": mf.vn(n=niv_core),
    "vn_urange": mf.vn(n=niv_urange),
    "vn_asympt": mf.vn(n=niv_asympt),
    "wn_core": mf.wn(n=niw_core),
    "wn_core_plus": mf.wn_plus(n=niw_core),
    "wn_rpa": mf.wn_outer(n_core=niw_core, n_outer=niw_urange),
    "wn_rpa_plus": mf.wn_outer_plus(n_core=niw_core, n_outer=niw_urange),
    "k_grid": k_grid,
    "q_grid": q_grid
}

config = {
    "options": options,
    "system": system,
    "names": names,
    "box_sizes": box_sizes,
    "grids": grids,
    "comm": comm,
    "dmft1p": dmft1p
}

config_dump = {
    "options": options,
    "system": system,
    "names": names,
    "box_sizes": box_sizes,
    "grids": grids,
    "dmft1p": dmft1p
}

# %%
# ------------------------------------------------ MAIN ----------------------------------------------------------------
if (comm.rank == 0):
    log = lambda s, *a: sys.stderr.write(str(s) % a + "\n")
    rerr = sys.stderr
else:
    log = lambda s, *a: None
    rerr = open(os.devnull, "w")

log("Running on %d core%s", comm.size, " s"[comm.size > 1])
log("Calculation started %s", time.strftime("%c"))

comm.Barrier()

if (comm.rank == 0):
    os.mkdir(output_path)
    np.save(output_path + 'config.npy', config_dump)

comm.Barrier()

dga_sde, dmft_sde, gamma_dmft = ldga.lambda_dga(config=config, verbose=verbose, outpfunc=log)
comm.Barrier()
log("Lambda-Dga finished %s", time.strftime("%c"))
# %%
if (comm.rank == 0):
    np.save(output_path + 'dmft_sde.npy', dmft_sde, allow_pickle=True)
    np.save(output_path + 'gamma_dmft.npy', gamma_dmft, allow_pickle=True)
    np.save(output_path + 'dga_sde.npy', dga_sde, allow_pickle=True)

    chi_lambda = np.load(output_path + 'chi_lambda.npy', allow_pickle=True).item()
    np.savetxt(output_path + 'lambda_values.txt', [chi_lambda['lambda_dens'], chi_lambda['lambda_magn']], delimiter=',',
               fmt='%.9f')



    # Create DMFT Green's function:
    gf_dict_dmft = twop.create_gk_dict(sigma=dmft1p['sloc'], kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'], n=dmft1p['n'],
                                  mu0=dmft1p['mu'], niv_cut=niv_urange)
    np.save(output_path + 'gk_dmft.npy', gf_dict_dmft, allow_pickle=True)

    # Create the Green's functions:
    gf_dict = twop.create_gk_dict(sigma=dga_sde['sigma'], kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'], n=dmft1p['n'],
                                  mu0=dmft1p['mu'])
    gf_dict_mu_dmft = twop.create_gk_dict(sigma=dga_sde['sigma'], kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'], n=dmft1p['n'],
                                  mu0=dmft1p['mu'],adjust_mu=False)
    np.save(output_path + 'gk_dga.npy', gf_dict, allow_pickle=True)
    np.save(output_path + 'gk_dga_mu_dmft.npy', gf_dict_mu_dmft, allow_pickle=True)
    np.savetxt(output_path + 'mu.txt', [[gf_dict['mu'],dmft1p['mu']], [gf_dict['n'], gf_dict_mu_dmft['n']]], delimiter=',',
               fmt='%.9f')




    gf_dict_nc = twop.create_gk_dict(sigma=dga_sde['sigma_nc'], kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'],
                                     n=dmft1p['n'], mu0=dmft1p['mu'])
    np.save(output_path + 'gk_dga_nc.npy', gf_dict_nc, allow_pickle=True)

    siw_dga_ksum_nc = dga_sde['sigma_nc'].mean(axis=(0, 1, 2))
    siw_dga_ksum = dga_sde['sigma'].mean(axis=(0, 1, 2))
    siw_dens_ksum = dga_sde['sigma_dens'].mean(axis=(0, 1, 2))
    siw_magn_ksum = dga_sde['sigma_magn'].mean(axis=(0, 1, 2))

    # Plot Siw-check:
    vn_list = [grids['vn_dmft'], grids['vn_urange'], grids['vn_urange'], grids['vn_urange']]
    siw_list = [dmft1p['sloc'], dmft_sde['siw'], siw_dga_ksum, siw_dga_ksum_nc]
    labels = [r'$\Sigma_{DMFT}(\nu)$', r'$\Sigma_{DMFT-SDE}(\nu)$', r'$\Sigma_{DGA}(\nu)$', r'$\Sigma_{DGA-NC}(\nu)$']
    plotting.plot_siw(vn_list=vn_list, siw_list=siw_list, labels_list=labels, plot_dir=output_path, niv_plot=100)

    # Plot Giw:
    vn_list = [grids['vn_urange'], grids['vn_urange']]
    giw_list = [gf_dict['gk'].mean(axis=(0, 1, 2)), gf_dict_nc['gk'].mean(axis=(0, 1, 2))]
    labels = [r'$G_{DGA}(\nu)$', r'$G_{DGA-NC}(\nu)$']
    plotting.plot_siw(vn_list=vn_list, siw_list=giw_list, labels_list=labels, plot_dir=output_path, niv_plot=100,
                      name='giw_loc')

    # Plot siw at important locations:

    ak_fs = -1. / np.pi * gf_dict['gk'][:, :, :, niv_urange].imag
    ak_fs_mu_dmft = -1. / np.pi * gf_dict_mu_dmft['gk'][:, :, :, niv_urange].imag
    ak_fs_dmft = -1. / np.pi * gf_dict_dmft['gk'][:, :, :, niv_urange].imag
    ak_fs_nc = -1. / np.pi * gf_dict_nc['gk'][:, :, :, niv_urange].imag

    ind_node = bz.find_arc_node(ak_fs=ak_fs, kgrid=k_grid)
    ind_anti_node = bz.find_arc_anti_node(ak_fs=ak_fs, kgrid=k_grid)
    ind_fs = bz.find_fermi_surface_peak(ak_fs=ak_fs, kgrid=k_grid)
    ind_fs_mu_dmft = bz.find_fermi_surface_peak(ak_fs=ak_fs_mu_dmft, kgrid=k_grid)
    ind_fs_dmft = bz.find_fermi_surface_peak(ak_fs=ak_fs_dmft, kgrid=k_grid)
    ind_arc = bz.find_arc_peaks(ak_fs=ak_fs, kgrid=k_grid)

    np.savetxt(output_path + 'loc_nodes_antinode.txt',
               [k_grid.kmesh.transpose((1, 2, 3, 0))[ind_node], k_grid.kmesh.transpose((1, 2, 3, 0))[ind_anti_node]],
               delimiter=',',
               fmt='%.9f')

    ind_node_nc = bz.find_arc_node(ak_fs=ak_fs_nc, kgrid=k_grid)
    ind_anti_node_nc = bz.find_arc_anti_node(ak_fs=ak_fs_nc, kgrid=k_grid)

    siw_dga_an = dga_sde['sigma'][ind_anti_node]
    siw_dga_n = dga_sde['sigma'][ind_node]
    vn_list = [grids['vn_dmft'], grids['vn_urange'], grids['vn_urange']]
    siw_list = [dmft1p['sloc'], siw_dga_n, siw_dga_an]
    labels = [r'$\Sigma_{DMFT}(\nu)$', r'$\Sigma_{DGA; Node}(\nu)$', r'$\Sigma_{DGA; Anti-Node}(\nu)$']
    plotting.plot_siw(vn_list=vn_list, siw_list=siw_list, labels_list=labels, plot_dir=output_path, niv_plot_min=0,
                      niv_plot=10, name='siw_at_bz_points', ms=5)

    siw_dga_an = dga_sde['sigma_nc'][ind_anti_node_nc]
    siw_dga_n = dga_sde['sigma_nc'][ind_node]
    vn_list = [grids['vn_dmft'], grids['vn_urange'], grids['vn_urange']]
    siw_list = [dmft1p['sloc'], siw_dga_n, siw_dga_an]
    labels = [r'$\Sigma_{DMFT}(\nu)$', r'$\Sigma_{DGA; Node}(\nu)$', r'$\Sigma_{DGA; Anti-Node}(\nu)$']
    plotting.plot_siw(vn_list=vn_list, siw_list=siw_list, labels_list=labels, plot_dir=output_path, niv_plot_min=0,
                      niv_plot=10, name='siw_at_bz_points_nc', ms=5)

    plotting.plot_siwk_fs(siwk=dga_sde['sigma'], plot_dir=output_path, kgrid=k_grid, do_shift=True)
    plotting.plot_siwk_fs(siwk=dga_sde['sigma_nc'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='nc')

    # DGA with adjusted mu:
    plotting.plot_giwk_fs(giwk=gf_dict['gk'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga',
                          ind_fs=ind_fs)
    plotting.plot_giwk_qpd(giwk=gf_dict['gk'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga')

    # DGA with mu from DMFT:
    plotting.plot_giwk_fs(giwk=gf_dict_mu_dmft['gk'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga_mu_dmft',
                          ind_fs=ind_fs_mu_dmft)
    plotting.plot_giwk_qpd(giwk=gf_dict_mu_dmft['gk'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga_mu_dmft')

    # DMFT:
    plotting.plot_giwk_fs(giwk=gf_dict_dmft['gk'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dmft',
                          ind_fs=ind_fs_dmft)
    plotting.plot_giwk_qpd(giwk=gf_dict_dmft['gk'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dmft')

    # Slight variant of the DGA:
    plotting.plot_giwk_fs(giwk=gf_dict_nc['gk'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga_nc')
    plotting.plot_giwk_qpd(giwk=gf_dict_nc['gk'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga_nc')

    plotting.plot_vertex_vvp(vertex=gamma_dmft['gamma_magn'].mat[niw_core, :, :].real, pdir=output_path,
                             name='gamma_magn')
    plotting.plot_vertex_vvp(vertex=gamma_dmft['gamma_dens'].mat[niw_core, :, :].real, pdir=output_path,
                             name='gamma_dens')

    plotting.plot_chi_fs(chi=chi_lambda['chi_magn_lambda'].mat.real, output_path=output_path, kgrid=q_grid,
                         name='magn_w0')
    plotting.plot_chi_fs(chi=chi_lambda['chi_dens_lambda'].mat.real, output_path=output_path, kgrid=q_grid,
                         name='dens_w0')

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(chi_lambda['chi_magn_lambda'].mat.real[ind_node])
    plt.plot(chi_lambda['chi_magn_lambda'].mat.real[ind_anti_node])
    plt.savefig(output_path + 'chi_magn_lambda_node_anti_node.png')
    plt.show()

    plt.figure()
    plt.plot(chi_lambda['chi_magn_lambda'].mat.imag[ind_node])
    plt.plot(chi_lambda['chi_magn_lambda'].mat.imag[ind_anti_node])
    plt.savefig(output_path + 'chi_magn_lambda_node_anti_node_imag.png')
    plt.show()

    plotting.plot_vrg_loc(vrg=dmft_sde['vrg_magn'].mat * dmft1p['beta'], niv_plot=niv_urange, pdir=output_path,
                          name='vrg_magn_loc')
    plotting.plot_vrg_loc(vrg=dmft_sde['vrg_dens'].mat * dmft1p['beta'], niv_plot=niv_urange, pdir=output_path,
                          name='vrg_dens_loc')

    oz_coeff, _ = ozfunc.fit_oz_spin(q_grid, chi_lambda['chi_magn_lambda'].mat[:, :, :, niw_core].flatten())

    np.savetxt(output_path + 'oz_coeff.txt', oz_coeff, delimiter=',', fmt='%.9f')
    plotting.plot_oz_fit(chi_w0=chi_lambda['chi_magn_lambda'].mat[:, :, :, niw_core], oz_coeff=oz_coeff, qgrid=q_grid,
                         pdir=output_path, name='oz_fit')

# --------------------------------------------- ANALYTIC CONTINUATION --------------------------------------------------
# %%

if(do_analytic_continuation):
    import AnalyticContinuation as a_cont
    v_real = a_cont.v_real_tan(wmax=wmax, nw=nwr)


# Do analytic continuation of local part:
if(do_analytic_continuation and comm.rank == 0):

    # DMFT Green's function:
    gloc_dmft_cont, gk_dmft = a_cont.max_ent_loc(v_real=v_real, sigma=dmft1p['sloc'], config=config, k_grid=k_grid,
                                                 niv_cut=niv_urange, use_preblur=use_preblur, bw=bw_dmft, err=err,
                                                 nfit=nfit, adjust_mu=True)
    plotting.plot_aw_loc(output_path=output_path, v_real=v_real, gloc=gloc_dmft_cont, name='aw-dmft')
    n_int = a_cont.check_filling(v_real=v_real, gloc_cont=gloc_dmft_cont)
    np.savetxt(output_path + 'n_dmft.txt', [n_int, gk_dmft['n']], delimiter=',', fmt='%.9f')
    np.save(output_path + 'gloc_cont_dmft.npy',gloc_dmft_cont, allow_pickle=True)

    # mu-adjusted DGA Green's function:
    gloc_dga_cont, gk_dga = a_cont.max_ent_loc(v_real=v_real, sigma=dga_sde['sigma'], config=config, k_grid=k_grid,
                                                 niv_cut=niv_urange, use_preblur=use_preblur, bw=bw, err=err,
                                                 nfit=nfit, adjust_mu=True)
    plotting.plot_aw_loc(output_path=output_path, v_real=v_real, gloc=gloc_dga_cont, name='aw-dga')
    n_int = a_cont.check_filling(v_real=v_real, gloc_cont=gloc_dga_cont)
    np.savetxt(output_path + 'n_dga.txt', [n_int, gk_dga['n']], delimiter=',', fmt='%.9f')
    np.save(output_path + 'gloc_cont_dga.npy',gloc_dga_cont, allow_pickle=True)

    # not mu-adjusted DGA Green's function:
    gloc_dga_cont_nma, gk_dga_nma = a_cont.max_ent_loc(v_real=v_real, sigma=dga_sde['sigma'], config=config, k_grid=k_grid,
                                                 niv_cut=niv_urange, use_preblur=use_preblur, bw=bw, err=err,
                                                 nfit=nfit, adjust_mu=False)
    plotting.plot_aw_loc(output_path=output_path, v_real=v_real, gloc=gloc_dga_cont_nma, name='aw-dga-no-mu-adjust')
    n_int = a_cont.check_filling(v_real=v_real, gloc_cont=gloc_dga_cont_nma)
    np.savetxt(output_path + 'n_dga_no_mu_adjust.txt', [n_int, gk_dga_nma['n']], delimiter=',', fmt='%.9f')
    np.save(output_path + 'gloc_cont_dga_no_mu_adjust.npy',gloc_dga_cont_nma, allow_pickle=True)

#%%
# Do analytic continuation along Fermi-surface:
if(do_analytic_continuation and comm.rank == 0):

    # DMFT Green's function:
    gk_fs_dmft_cont, ind_gf0_dmft, gk_dmft = a_cont.max_ent_on_fs(v_real=v_real, sigma=dmft1p['sloc'], config=config,
                                                         k_grid=k_grid,
                                                         niv_cut=niv_urange, use_preblur=use_preblur, bw=bw_dmft,
                                                         err=err, nfit=nfit)

    plotting.plot_ploints_on_fs(output_path=output_path, gk_fs=gk_dmft['gk'][:, :, 0, gk_dmft['niv']], k_grid=k_grid,
                                ind_fs=ind_gf0_dmft,
                                name='fermi_surface_dmft')

    plotting.plot_aw_ind(output_path=output_path, v_real=v_real, gk_cont=gk_fs_dmft_cont, ind=ind_gf0_dmft,
                         name='aw-dmft-fs-wide')
    plotting.plot_aw_ind(output_path=output_path, v_real=v_real, gk_cont=gk_fs_dmft_cont, ind=ind_gf0_dmft,
                         name='aw-dmft-fs-narrow', xlim=(-t, t))

    np.save(output_path + 'gk_fs_dmft_cont.npy',gk_fs_dmft_cont, allow_pickle=True)

    # mu-adjusted DGA Green's function:
    gk_fs_dga_cont, ind_gf0_dga, gk_dga = a_cont.max_ent_on_fs(v_real=v_real, sigma=dga_sde['sigma'], config=config,
                                                         k_grid=k_grid,
                                                         niv_cut=niv_urange, use_preblur=use_preblur, bw=bw,
                                                         err=err, nfit=nfit)

    plotting.plot_ploints_on_fs(output_path=output_path, gk_fs=gk_dga['gk'][:, :, 0, gk_dga['niv']], k_grid=k_grid,
                                ind_fs=ind_gf0_dga,
                                name='fermi_surface_dga')

    plotting.plot_aw_ind(output_path=output_path, v_real=v_real, gk_cont=gk_fs_dmft_cont, ind=ind_gf0_dga,
                         name='aw-dga-fs-wide')
    plotting.plot_aw_ind(output_path=output_path, v_real=v_real, gk_cont=gk_fs_dmft_cont, ind=ind_gf0_dga,
                         name='aw-dga-fs-narrow', xlim=(-t, t))
    np.save(output_path + 'gk_fs_dga_cont.npy', gk_fs_dga_cont, allow_pickle=True)

    # not mu-adjusted DGA Green's function:
    gk_fs_dga_cont_nma, ind_gf0_dgat_nma, gk_dgat_nma = a_cont.max_ent_on_fs(v_real=v_real, sigma=dga_sde['sigma'], config=config,
                                                         k_grid=k_grid,
                                                         niv_cut=niv_urange, use_preblur=use_preblur, bw=bw,
                                                         err=err, nfit=nfit)

    plotting.plot_ploints_on_fs(output_path=output_path, gk_fs=gk_dgat_nma['gk'][:, :, 0, gk_dgat_nma['niv']], k_grid=k_grid,
                                ind_fs=ind_gf0_dgat_nma,
                                name='fermi_surface_dmft-no-mu-adjust')

    plotting.plot_aw_ind(output_path=output_path, v_real=v_real, gk_cont=gk_fs_dga_cont_nma, ind=ind_gf0_dgat_nma,
                         name='aw-dga-fs-wide-no-mu-adjust')
    plotting.plot_aw_ind(output_path=output_path, v_real=v_real, gk_cont=gk_fs_dga_cont_nma, ind=ind_gf0_dgat_nma,
                         name='aw-dga-fs-narrow-no-mu-adjust', xlim=(-t, t))
    np.save(output_path + 'gk_fs_dga_cont_no_mu_adjust.npy', gk_fs_dga_cont_nma, allow_pickle=True)

#%%
# Do analytic continuation within the irreducible Brillouin Zone:
if(do_analytic_continuation):

    irrk_distributor = mpiaux.MpiDistributor(ntasks=k_grid.nk_irr, comm=comm,
                                            output_path=output_path,
                                            name='Qiw')

    index_grid_keys = ('irrk',)
    irrk_grid = ind.IndexGrids(grid_arrays=(k_grid.irrk_ind_lin,), keys=index_grid_keys,
                              my_slice=irrk_distributor.my_slice)
    ind_irrk = np.squeeze(np.array(np.unravel_index(k_grid.irrk_ind[irrk_grid.my_indizes], shape=k_grid.nk))).T
    if(np.size(ind_irrk.shape) > 1):
        ind_irrk = [tuple(ind_irrk[i, :]) for i in np.arange(ind_irrk.shape[0])]
    else:
        ind_irrk = tuple(ind_irrk)
    # DMFT Green's function:
    gk = twop.create_gk_dict(sigma=dmft1p['sloc'], kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'], n=dmft1p['n'],
                                  mu0=dmft1p['mu'], adjust_mu=True, niv_cut=niv_urange)



    gk_my_cont = a_cont.do_max_ent_on_ind_T(mat=gk['gk'], ind_list=ind_irrk, v_real=v_real,
                                                   beta=dmft1p['beta'],
                                                   n_fit=nfit, err=err, alpha_det_method='chi2kink', use_preblur = use_preblur, bw=bw_dmft)
    #print(f'{gk_my_cont=}')
    gk_cont = irrk_distributor.allgather(rank_result=gk_my_cont)
    print(f'{gk_cont[:,:]=}')
    print(f'{gk_cont[:,nwr//2]=}')
    print(f'{gk_cont.shape=}')
    if(comm.rank == 0):
        gk_cont_fbz = k_grid.irrk2fbz(mat=gk_cont)
        w_int = -0.2
        plotting.plot_cont_fs(output_path=output_path, name='fermi_surface_dmft_cont_wint-0.2',
                              gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=w_int)
        w_int = -0.1
        plotting.plot_cont_fs(output_path=output_path, name='fermi_surface_dmft_cont_wint-0.1',
                              gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=w_int)
        np.save('gk_dmft_cont_fbz.npy', gk_cont_fbz, allow_pickle=True)

    # mu-adjust DGA Green's function:
    gk = twop.create_gk_dict(sigma=dga_sde['sigma'], kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'], n=dmft1p['n'],
                                  mu0=dmft1p['mu'], adjust_mu=True, niv_cut=niv_urange)
    gk_my_cont = a_cont.do_max_ent_on_ind_T(mat=gk['gk'], ind_list=ind_irrk, v_real=v_real,
                                                   beta=dmft1p['beta'],
                                                   n_fit=nfit, err=err, alpha_det_method='chi2kink', use_preblur = use_preblur, bw=bw)

    gk_cont = irrk_distributor.allgather(rank_result=gk_my_cont)
    if(comm.rank == 0):
        gk_cont_fbz = k_grid.irrk2fbz(mat=gk_cont)
        w_int = -0.2
        plotting.plot_cont_fs(output_path=output_path, name='fermi_surface_dga_cont_wint-0.2',
                              gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=w_int)
        w_int = -0.1
        plotting.plot_cont_fs(output_path=output_path, name='fermi_surface_dga_cont_wint-0.1',
                              gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=w_int)
        np.save('gk_dga_cont_fbz.npy', gk_cont_fbz, allow_pickle=True)

    # not mu-adjust DGA Green's function:
    gk = twop.create_gk_dict(sigma=dga_sde['sigma'], kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'], n=dmft1p['n'],
                             mu0=dmft1p['mu'], adjust_mu=False, niv_cut=niv_urange)
    gk_my_cont = a_cont.do_max_ent_on_ind_T(mat=gk['gk'], ind_list=ind_irrk, v_real=v_real,
                                          beta=dmft1p['beta'],
                                          n_fit=nfit, err=err, alpha_det_method='chi2kink', use_preblur=use_preblur,
                                          bw=bw)

    gk_cont = irrk_distributor.allgather(rank_result=gk_my_cont)
    if (comm.rank == 0):
        gk_cont_fbz = k_grid.irrk2fbz(mat=gk_cont)
        w_int = -0.2
        plotting.plot_cont_fs(output_path=output_path, name='fermi_surface_dga_cont_wint-0.2_no_mu_adjust',
                              gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=w_int)
        w_int = -0.1
        plotting.plot_cont_fs(output_path=output_path, name='fermi_surface_dga_cont_wint-0.1_no_mu_adjust',
                              gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=w_int)
        np.save('gk_dga_cont_fbz_no_mu_adjust.npy', gk_cont_fbz, allow_pickle=True)

# ------------------------------------------------ PAIRING VERTEX ------------------------------------------------------
# %%
if (do_pairing_vertex and comm.rank == 0):
    import RealTime as rt
    import PairingVertex as pv


    realt = rt.real_time()

    log(realt.string_time('Start pairing vertex:'))

    qiw_distributor = mpiaux.MpiDistributor(ntasks=box_sizes['niw_core'] * q_grid.nk_irr, comm=comm,
                                            output_path=output_path,
                                            name='Qiw')

    qiw_grid = ind.IndexGrids(grid_arrays=q_grid.grid + (grids['wn_core'],),
                              keys=('qx', 'qy', 'qz', 'iw'),
                              my_slice=None)

    f1_magn, f2_magn, f1_dens, f2_dens = pv.load_pairing_vertex_from_rank_files(output_path=output_path, name='Qiw',
                                                                                mpi_size=comm.size, nq=q_grid.nk_irr,
                                                                                niv_pp=niv_pp)
    f1_magn = q_grid.irrk2fbz(mat=f1_magn)
    f2_magn = q_grid.irrk2fbz(mat=f2_magn)
    f1_dens = q_grid.irrk2fbz(mat=f1_dens)
    f2_dens = q_grid.irrk2fbz(mat=f2_dens)

    chi_dens_lambda_pp = pv.reshape_chi(chi=chi_lambda['chi_dens_lambda'].mat, niv_pp=niv_pp)
    chi_magn_lambda_pp = pv.reshape_chi(chi=chi_lambda['chi_magn_lambda'].mat, niv_pp=niv_pp)

    f_magn = f1_magn + (1 + dmft1p['u'] * chi_magn_lambda_pp) * f2_magn
    f_dens = f1_dens + (1 - dmft1p['u'] * chi_dens_lambda_pp) * f2_dens

    f_sing = -1.5 * f_magn + 0.5 * f_dens
    f_trip = -0.5 * f_magn - 0.5 * f_dens

    plotting.plot_vertex_vvp(vertex=f_dens.mean(axis=(0, 1, 2)).real, pdir=output_path, name='f_dens_loc')
    plotting.plot_vertex_vvp(vertex=f_magn.mean(axis=(0, 1, 2)).real, pdir=output_path, name='f_magn_loc')

    pairing_vertices = {
        'f_sing': f_sing,
        'f_trip': f_trip
    }

    np.save(output_path + 'pairing_vertices.npy', pairing_vertices)

    plotting.plot_vertex_vvp(vertex=f_sing.mean(axis=(0, 1, 2)).real, pdir=output_path, name='f_sing_loc')
    plotting.plot_vertex_vvp(vertex=f_trip.mean(axis=(0, 1, 2)).real, pdir=output_path, name='f_trip_loc')

    log(realt.string_time('End pairing vertex:'))
#
# ----------------------------------------------- Eliashberg Equation --------------------------------------------------
# %%

if (do_pairing_vertex and comm.rank == 0):
    import TwoPoint as twop
    import EliashbergEquation as eq

    log(realt.string_time('Start Eliashberg:'))
    gamma_sing = -f_sing
    gamma_trip = -f_trip
    #
    if (sym_sing):
        gamma_sing = 0.5 * (gamma_sing + np.flip(gamma_sing, axis=(-1)))

    if (sym_trip):
        gamma_trip = 0.5 * (gamma_trip - np.flip(gamma_trip, axis=(-1)))

    plotting.plot_vertex_vvp(vertex=gamma_trip.mean(axis=(0, 1, 2)).real, pdir=output_path, name='gamma_trip_loc')

    g_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=q_grid.grid, hr=hr,
                                               sigma=dga_sde['sigma'])
    mu_dga = g_generator.adjust_mu(n=dmft1p['n'], mu0=dmft1p['mu'])
    gk_dga = g_generator.generate_gk(mu=mu_dga, qiw=[0, 0, 0, 0], niv=niv_pp).gk

    gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=gap0_sing['k'], v_type=gap0_sing['v'],
                            k_grid=q_grid.grid)
    norm = np.prod(nq) * dmft1p['beta']
    n_eig = 2
    powiter_sing = eq.EliashberPowerIteration(gamma=gamma_sing, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
                                              n_eig=n_eig)

    gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=gap0_trip['k'], v_type=gap0_trip['v'],
                            k_grid=q_grid.grid)
    powiter_trip = eq.EliashberPowerIteration(gamma=gamma_trip, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
                                              n_eig=n_eig)

    eliashberg = {
        'lambda_sing': powiter_sing.lam,
        'lambda_trip': powiter_trip.lam,
        'delta_sing': powiter_sing.gap,
        'delta_trip': powiter_trip.gap,
    }
    np.save(output_path + 'eliashberg.npy', eliashberg)
    np.savetxt(output_path + 'eigenvalues.txt', [powiter_sing.lam.real, powiter_trip.lam.real], delimiter=',',
               fmt='%.9f')
    np.savetxt(output_path + 'eigenvalues_s.txt', [powiter_sing.lam_s.real, powiter_trip.lam_s.real], delimiter=',',
               fmt='%.9f')

    for i in range(len(powiter_sing.gap)):
        plotting.plot_gap_function(delta=powiter_sing.gap[i].real, pdir=output_path, name='sing_{}'.format(i),
                                   kgrid=q_grid,
                                   do_shift=True)
        plotting.plot_gap_function(delta=powiter_sing.gap[i].real, pdir=output_path, name='trip_{}'.format(i),
                                   kgrid=q_grid,
                                   do_shift=True)
    log(realt.string_time('End Eliashberg:'))
