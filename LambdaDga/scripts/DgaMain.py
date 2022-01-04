# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# WARNING: Currently there cannot be more processes than Niw*1+2. This is because my MpiDistibutor cannot handle slaves
# that have nothing to do.

# Warning: many of the files I prepared from Motoharu do not have smom stored.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import sys, os

sys.path.append('../src/')
sys.path.append(os.environ['HOME'] + "/Programs/dga/LambdaDga/src")
import Hr as hr
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

import Plotting as plotting
from mpi4py import MPI as mpi

# ----------------------------------------------- PARAMETERS -----------------------------------------------------------

# Define MPI communicator:
comm = mpi.COMM_WORLD

# Define paths of datasets:
input_path = './'
# input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
# input_path = '/mnt/c/users/pworm/Research/BEPS_Project/TriangularLattice/DGA/TriangularLattice_U8.0_tp1.0_tpp0.0_beta10_n1.0/'
input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/Testset1/LambdaDga_Python/'
# input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U8_b010_tp0_tpp0_n0.85/LambdaDgaPython/'
# input_path = '/mnt/c/users/pworm/Research/BEPS_Project/TriangularLattice/DGA/TriangularLattice_U9.5_tp1.0_tpp0.0_beta10_n1.0/'
# input_path = '/mnt/d/Research/BEPS_Project/TriangularLattice/TriangularLattice_U9.0_tp1.0_tpp0.0_beta10_n1.0/'
#input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/KonvergenceAnalysis/'
#input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
#input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta80_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
#input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/NdNiO2_U8_n0.85_b75/'
#input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.85/KonvergenceAnalysis/'
#input_path = '/mnt/c/users/pworm/Research/U2BenchmarkData/2DSquare_U2_tp-0.0_tpp0.0_beta15_mu1/'
output_path = input_path

fname_dmft = '1p-data.hdf5'
fname_g2 = 'g4iw_sym.hdf5'  # 'Vertex_sym.hdf5' #'g4iw_sym.hdf5'
fname_ladder_vertex = 'LadderVertex.hdf5'

# Define options:
do_pairing_vertex = False
keep_ladder_vertex = False
lambda_correction_type = 'sp' # Available: ['spch','sp','none','sp_only']
use_urange_for_lc = False # Use with care. This is not really tested and at least low k-grid samples don't look too good.
lattice = 'square'
verbose=True

#Set up real-space Wannier Hamiltonian:
t = 1.00
tp = -0.20 * t * 0
tpp = 0.10 * t * 0
t = 0.25
tp = -0.25 * t
tpp = 0.12 * t

# Define frequency box-sizes:
niw_core = 30
niw_urange = 50
niv_core = 30
niv_invbse = 30
niv_urange = 50
niv_asympt = 0 # Don't use this for now.

# Define k-ranges:
nkf = 16
nqf = 16
nk = (nkf, nkf, 1)
nq = (nqf, nqf, 1)

output_folder = 'LambdaDga_lc_{}_Nk{}_Nq{}_core{}_invbse{}_vurange{}_wurange{}'.format(lambda_correction_type,np.prod(nk), np.prod(nq), niw_core,niv_invbse, niv_urange, niw_urange)
output_path = output.uniquify(output_path + output_folder) + '/'
fname_ladder_vertex = output_path + fname_ladder_vertex

# Generate k-meshes:
k_grid = bz.KGrid(nk=nk, name='k')
q_grid = bz.KGrid(nk=nq, name='q')

if (lattice == 'square'):
    hr = hr.one_band_2d_t_tp_tpp(t=t, tp=tp, tpp=tpp)
elif (lattice == 'triangular'):
    hr = hr.one_band_2d_triangular_t_tp_tpp(t=t, tp=tp, tpp=tpp)
else:
    raise NotImplementedError('Only square or triangular lattice implemented at the moment.')

# load contents from w2dynamics DMFT file:
f1p = w2dyn_aux.w2dyn_file(fname=input_path + fname_dmft)
dmft1p = f1p.load_dmft1p_w2dyn()
f1p.close()
niv_dmft = dmft1p['niv']
if(dmft1p['n'] == 0.0): dmft1p['n'] = 1.0

# Define system paramters, like interaction or inverse temperature.
# Note: I want this to be decoupled from dmft1p, because of later RPA/FLEX stuff.

options = {
    'do_pairing_vertex': do_pairing_vertex,
    'lambda_correction_type':lambda_correction_type,
    'use_urange_for_lc':use_urange_for_lc,
    'lattice': lattice
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
    "nk": nk,
    "nq": nq
}

grids = {
    "vn_dmft": mf.vn(n=niv_dmft),
    "vn_core": mf.vn(n=niv_core),
    "vn_urange": mf.vn(n=niv_urange),
    "vn_asympt": mf.vn(n=niv_asympt),
    "wn_core": mf.wn(n=niw_core),
    "wn_rpa": mf.wn_outer(n_core=niw_core,n_outer=niw_urange),
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

# ------------------------------------------------ MAIN ----------------------------------------------------------------
if (comm.rank == 0):
    log = lambda s, *a: sys.stdout.write(str(s) % a + "\n")
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

dga_sde, dmft_sde, gamma_dmft, chi_lambda, chi_ladder = ldga.lambda_dga(config=config,verbose=verbose,outpfunc=log)
comm.Barrier()
log("Lambda-Dga finished %s", time.strftime("%c"))
if (comm.rank == 0):
    np.save(output_path + 'dmft_sde.npy', dmft_sde, allow_pickle=True)
    np.save(output_path + 'gamma_dmft.npy', gamma_dmft, allow_pickle=True)
    np.save(output_path + 'dga_sde.npy', dga_sde, allow_pickle=True)
    np.save(output_path + 'chi_lambda.npy', chi_lambda, allow_pickle=True)
    np.save(output_path + 'chi_ladder.npy', chi_ladder, allow_pickle=True)
    np.savetxt(output_path + 'lambda_values.txt', [chi_lambda['lambda_dens'], chi_lambda['lambda_magn']], delimiter=',', fmt='%.9f')

    siw_dga_ksum_nc = dga_sde['sigma_nc'].mean(axis=(0, 1, 2))
    siw_dga_ksum = dga_sde['sigma'].mean(axis=(0, 1, 2))
    siw_dens_ksum = dga_sde['sigma_dens'].mean(axis=(0, 1, 2))
    siw_magn_ksum = dga_sde['sigma_magn'].mean(axis=(0, 1, 2))

    qiw_grid = ind.IndexGrids(grid_arrays=q_grid.get_grid_as_tuple() + (grids['wn_core'],),
                              keys=('qx', 'qy', 'qz', 'iw'),
                              my_slice=None)

    # Plot Siw-check:
    vn_list = [grids['vn_dmft'], grids['vn_urange'], grids['vn_urange'], grids['vn_urange']]
    siw_list = [dmft1p['sloc'], dmft_sde['siw'], siw_dga_ksum, siw_dga_ksum_nc]
    labels = [r'$\Sigma_{DMFT}(\nu)$', r'$\Sigma_{DMFT-SDE}(\nu)$', r'$\Sigma_{DGA}(\nu)$', r'$\Sigma_{DGA-NC}(\nu)$']
    plotting.plot_siw(vn_list=vn_list, siw_list=siw_list, labels_list=labels, plot_dir=output_path, niv_plot=100)

    # Plot siw at important locations:

    siw_dga_an = dga_sde['sigma'][nk[0]//2,0,0,:]
    siw_dga_n = dga_sde['sigma'][nk[0]//4,nk[1]//4,0,:]
    vn_list = [grids['vn_dmft'], grids['vn_urange'], grids['vn_urange']]
    siw_list = [dmft1p['sloc'], siw_dga_n, siw_dga_an]
    labels = [r'$\Sigma_{DMFT}(\nu)$',r'$\Sigma_{DGA; Node}(\nu)$', r'$\Sigma_{DGA; Anti-Node}(\nu)$']
    plotting.plot_siw(vn_list=vn_list, siw_list=siw_list, labels_list=labels, plot_dir=output_path, niv_plot_min=0, niv_plot=10, name='siw_at_bz_points', ms=5)


    siw_dga_an = dga_sde['sigma_nc'][nk[0]//2,0,0,:]
    siw_dga_n = dga_sde['sigma_nc'][nk[0]//4,nk[1]//4,0,:]
    vn_list = [grids['vn_dmft'], grids['vn_urange'], grids['vn_urange']]
    siw_list = [dmft1p['sloc'], siw_dga_n, siw_dga_an]
    labels = [r'$\Sigma_{DMFT}(\nu)$',r'$\Sigma_{DGA; Node}(\nu)$', r'$\Sigma_{DGA; Anti-Node}(\nu)$']
    plotting.plot_siw(vn_list=vn_list, siw_list=siw_list, labels_list=labels, plot_dir=output_path, niv_plot_min=0, niv_plot=10, name='siw_at_bz_points_nc', ms=5)


    plotting.plot_siwk_fs(siwk=dga_sde['sigma'], plot_dir=output_path, kgrid=k_grid, do_shift=True)
    plotting.plot_siwk_fs(siwk=dga_sde['sigma_nc'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='nc')

    gk_dga_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=k_grid.get_grid_as_tuple(), hr=hr,
                                                    sigma=dga_sde['sigma'])
    mu_dga = gk_dga_generator.adjust_mu(n=dmft1p['n'], mu0=dmft1p['mu'])
    gk_dga = gk_dga_generator.generate_gk(mu=mu_dga)

    gf_dict = {
        'gk': gk_dga._gk,
        'mu': gk_dga._mu,
        'iv': gk_dga._iv,
        'beta': gk_dga._beta
    }

    np.save(output_path + 'gk_dga.npy', gk_dga, allow_pickle=True)

    plotting.plot_giwk_fs(giwk=gk_dga.gk, plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga')
    plotting.plot_giwk_qpd(giwk=gk_dga.gk, plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga')

    gk_dga_generator_nc = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=k_grid.get_grid_as_tuple(), hr=hr,
                                                       sigma=dga_sde['sigma_nc'])
    mu_dga_nc = gk_dga_generator_nc.adjust_mu(n=dmft1p['n'], mu0=dmft1p['mu'])
    gk_dga_nc = gk_dga_generator_nc.generate_gk(mu=mu_dga_nc)

    gf_dict_nc = {
        'gk': gk_dga_nc._gk,
        'mu': gk_dga_nc._mu,
        'iv': gk_dga_nc._iv,
        'beta': gk_dga_nc._beta
    }

    np.save(output_path + 'gk_dga_nc.npy', gk_dga_nc, allow_pickle=True)

    plotting.plot_giwk_fs(giwk=gk_dga_nc.gk, plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga_nc')

    chi_magn_lambda = chi_lambda['chi_magn_lambda'].mat.reshape(q_grid.nk + (niw_core * 2 + 1,))
    chi_dens_lambda = chi_lambda['chi_dens_lambda'].mat.reshape(q_grid.nk + (niw_core * 2 + 1,))

    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(chi_magn_lambda[:, :, 0, niw_core].real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'chi_magn_w0.png')
    plt.close()

    plt.figure()
    plt.imshow(chi_dens_lambda[:, :, 0, niw_core].real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'chi_dens_w0.png')
    plt.close()

    plt.figure()
    plt.imshow(gamma_dmft['gamma_magn'].mat[niw_core, :, :].real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'gamma_magn.png')
    plt.close()

    plt.figure()
    plt.imshow(gamma_dmft['gamma_dens'].mat[niw_core, :, :].real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'gamma_dens.png')
    plt.close()

# ------------------------------------------------ PAIRING VERTEX ----------------------------------------------------------------
# %%

# Collect Pairing vertex from subfiles:
if(do_pairing_vertex and comm.rank == 0):
    import MpiAux as mpiaux
    import h5py
    import re

    qiw_distributor = mpiaux.MpiDistributor(ntasks=box_sizes['niw_core'] * np.prod(nq), comm=comm,
                                            output_path=output_path,
                                            name='Qiw')

    # Collect data from subfiles (This is quite ugly, as it is hardcoded to my structure. This should be replaced by a general routine):
    f1_magn = np.zeros(nq+(niv_core,niv_core), dtype=complex)
    f2_magn = np.zeros(nq+(niv_core,niv_core), dtype=complex)
    f1_dens = np.zeros(nq+(niv_core,niv_core), dtype=complex)
    f2_dens = np.zeros(nq+(niv_core,niv_core), dtype=complex)
    if (qiw_distributor.my_rank == 0):
        file_out = h5py.File(fname_ladder_vertex, 'w')
        for ir in range(qiw_distributor.mpi_size):
            fname_input = output_path + 'QiwRank{:05d}'.format(ir) + '.hdf5'
            file_in = h5py.File(fname_input, 'r')
            for key1 in list(file_in.keys()):
                # extract the q indizes from the group name!
                qx = np.array(re.findall("\d+",key1), dtype=int)[0]
                qy = np.array(re.findall("\d+",key1), dtype=int)[1]
                qz = np.array(re.findall("\d+",key1), dtype=int)[2]
                condition = file_in[key1 + '/condition/'][()]
                f1_magn[qx,qy,qz,condition] = file_in[key1 +'/f1_magn/'][()]
                f2_magn[qx,qy,qz,condition] = file_in[key1 +'/f2_magn/'][()]
                f1_dens[qx,qy,qz,condition] = file_in[key1 +'/f1_dens/'][()]
                f2_dens[qx,qy,qz,condition] = file_in[key1 +'/f2_dens/'][()]

            file_in.close()
            os.remove(fname_input)
        file_out.close()

elif(not do_pairing_vertex and comm.rank == 0):
    import MpiAux as mpiaux
    import h5py
    import re
    qiw_distributor = mpiaux.MpiDistributor(ntasks=box_sizes['niw_core'] * np.prod(nq), comm=comm,
                                            output_path=output_path,
                                            name='Qiw')
    if (qiw_distributor.my_rank == 0):
        for ir in range(qiw_distributor.mpi_size):
            fname_input = output_path + 'QiwRank{:05d}'.format(ir) + '.hdf5'
            os.remove(fname_input)
else:
    pass

if(do_pairing_vertex and comm.rank == 0):
    import RealTime as rt

    realt = rt.real_time()

    log(realt.string_time('Start pairing vertex:'))

    import PairingVertex as pv

    niv_pp = niv_core // 2

    chi_dens_lambda = qiw_grid.reshape_matrix(mat=chi_lambda['chi_dens_lambda'].mat)
    chi_magn_lambda = qiw_grid.reshape_matrix(mat=chi_lambda['chi_magn_lambda'].mat)

    chi_dens_lambda_pp = pv.reshape_chi(chi=chi_dens_lambda, niv_pp=niv_pp)
    chi_magn_lambda_pp = pv.reshape_chi(chi=chi_magn_lambda, niv_pp=niv_pp)

    f_magn = f1_magn + (1 + dmft1p['u'] * chi_magn_lambda_pp) * f2_magn
    f_dens = f1_dens + (1 - dmft1p['u'] * chi_dens_lambda_pp) * f2_dens

    f_sing = -1.5 * f_magn + 0.5 * f_dens
    f_trip = -0.5 * f_magn - 0.5 * f_dens

    f_magn_loc = f_magn.mean(axis=(0, 1, 2))
    f1_magn_loc = f1_magn.mean(axis=(0, 1, 2))
    f2_magn_loc = f2_magn.mean(axis=(0, 1, 2))

    f_dens_loc = f_dens.mean(axis=(0, 1, 2))

    fig = plt.figure()
    plt.imshow(f_magn_loc.real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'f_magn_loc.png')
    plt.close()

    fig = plt.figure()
    plt.imshow(f_dens_loc.real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'f_dens_loc.png')
    plt.close()

    pairing_vertices = {
        'f_sing': f_sing,
        'f_trip': f_trip
    }

    np.save(output_path + 'pairing_vertices.npy', pairing_vertices)

    f_sing_loc = f_sing.mean(axis=(0, 1, 2))
    f_trip_loc = f_trip.mean(axis=(0, 1, 2))

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.imshow(f_sing_loc.real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'f_sing_loc.png')
    plt.close()

    fig = plt.figure()
    plt.imshow(f_trip_loc.real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'f_trip_loc.png')
    plt.close()

    log(realt.string_time('End pairing vertex:'))
#
# ----------------------------------------------- Eliashberg Equation --------------------------------------------------
# %%

if (do_pairing_vertex and comm.rank == 0):
    import TwoPoint as twop
    import EliashbergEquation as eq

    log(realt.string_time('Start Eliashberg:'))
    gamma_sing = f_sing
    gamma_trip = f_trip
    g_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=q_grid.get_grid_as_tuple(), hr=hr,
                                               sigma=dga_sde['sigma'])
    mu_dga = gk_dga_generator.adjust_mu(n=dmft1p['n'], mu0=dmft1p['mu'])
    gk_dga = g_generator.generate_gk(mu=mu_dga, qiw=[0, 0, 0, 0], niv=niv_core // 2).gk
    lambda_sing, delta_sing = eq.linear_eliashberg(gamma=gamma_sing, gk=gk_dga, eps=10 ** -6, max_count=10000,
                                                   norm=np.prod(nk) * dmft1p['beta'])
    lambda_trip, delta_trip = eq.linear_eliashberg(gamma=gamma_trip, gk=gk_dga, eps=10 ** -6, max_count=10000,
                                                   norm=np.prod(nk) * dmft1p['beta'])

    eliashberg = {
        'lambda_sing': lambda_sing[1].real,
        'lambda_trip': lambda_trip[1].real,
        'delta_sing': delta_sing[1].real,
        'delta_trip': delta_trip[1].real,
    }
    np.save(output_path + 'eliashberg.npy', eliashberg)
    np.savetxt(output_path + 'eigenvalues.txt', [lambda_sing[1].real, lambda_trip[1].real], delimiter=',', fmt='%.9f')

    plotting.plot_gap_function(delta=delta_sing[1].real, pdir=output_path, name='sing', kgrid=k_grid)
    plotting.plot_gap_function(delta=delta_trip[1].real, pdir=output_path, name='trip', kgrid=k_grid)

    log(realt.string_time('End Eliashberg:'))
