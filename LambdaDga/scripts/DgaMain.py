# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# WARNING: Currently there cannot be more processes than Niw*1+2. This is because my MpiDistibutor cannot handle slaves
# that have nothing to do.

# Warning: many of the files I prepared from Motoharu do not have smom stored.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import sys,os
sys.path.append('../src/')
sys.path.append(os.environ['HOME']+"/Programs/dga/LambdaDga/src")
import Hr as hr
import Hk as hamk
import Indizes as ind
import w2dyn_aux
import MatsubaraFrequencies as mf
import BrillouinZone as bz
import LambdaDga as ldga
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
#input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
#input_path = '/mnt/c/users/pworm/Research/BEPS_Project/TriangularLattice/DGA/TriangularLattice_U8.0_tp1.0_tpp0.0_beta10_n1.0/'
input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/Testset1/LambdaDga_Python/'
#input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U8_b010_tp0_tpp0_n0.85/LambdaDgaPython/'
#input_path = '/mnt/c/users/pworm/Research/BEPS_Project/TriangularLattice/DGA/TriangularLattice_U9.5_tp1.0_tpp0.0_beta10_n1.0/'
#input_path = '/mnt/d/Research/BEPS_Project/TriangularLattice/TriangularLattice_U9.0_tp1.0_tpp0.0_beta10_n1.0/'
#input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/KonvergenceAnalysis/'
#input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/NdNiO2_U8_n0.85_b75/'
#input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.85/KonvergenceAnalysis/'
output_path = input_path

fname_dmft = '1p-data.hdf5'
fname_g2 = 'g4iw_sym.hdf5' #'Vertex_sym.hdf5' #'g4iw_sym.hdf5'
fname_ladder_vertex = 'LadderVertex.hdf5'

# Define options:
do_pairing_vertex = False
keep_ladder_vertex = False
lattice = 'square'

# Set up real-space Wannier Hamiltonian:
# t = 1.00
# tp = -0.20 * t
# tpp = 0.10 * t
t = 0.25
tp = -0.25 * t
tpp = 0.12 * t

# Define frequency box-sizes:
niw_core = 20
niv_core = 20
niv_urange = 120

# Define k-ranges:
nkf = 32
nk = (nkf, nkf, 1)
nq = (nkf, nkf, 1)

output_folder = 'LambdaDga_Nk{}_Nq{}_core{}_urange{}'.format(np.prod(nk),np.prod(nq),niw_core,niv_urange)
output_path = output.uniquify(output_path+output_folder) + '/'
fname_ladder_vertex = output_path + fname_ladder_vertex

# Generate k-meshes:
k_grid = bz.KGrid(nk=nk, name='k')
q_grid = bz.KGrid(nk=nq, name='q')

if(lattice=='square'):
    hr = hr.one_band_2d_t_tp_tpp(t=t, tp=tp, tpp=tpp)
elif(lattice=='triangular'):
    hr = hr.one_band_2d_triangular_t_tp_tpp(t=t, tp=tp, tpp=tpp)
else:
    raise NotImplementedError('Only square or triangular lattice implemented at the moment.')

# load contents from w2dynamics DMFT file:
f1p = w2dyn_aux.w2dyn_file(fname=input_path + fname_dmft)
dmft1p = f1p.load_dmft1p_w2dyn()
f1p.close()
niv_dmft = dmft1p['niv']

# Define system paramters, like interaction or inverse temperature.
# Note: I want this to be decoupled from dmft1p, because of later RPA/FLEX stuff.

options = {
    'do_pairing_vertex': do_pairing_vertex
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
    "niv_core": niv_core,
    "niv_urange": niv_urange,
    "nk": nk,
    "nq": nq
}

grids = {
    "vn_dmft": mf.vn(n=niv_dmft),
    "vn_core": mf.vn(n=niv_core),
    "vn_urange": mf.vn(n=niv_urange),
    "wn_core": mf.wn(n=niw_core),
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
comm.Barrier()

if(comm.rank == 0):
    os.mkdir(output_path)
    np.save(output_path + 'config.npy', config_dump)

comm.Barrier()

dga_sde, dmft_sde, gamma_dmft = ldga.lambda_dga(config=config)
comm.Barrier()

if(comm.rank == 0):
    np.save(output_path + 'dmft_sde.npy',dmft_sde,allow_pickle=True)
    np.save(output_path + 'gamma_dmft.npy',gamma_dmft,allow_pickle=True)
    np.save(output_path + 'dga_sde.npy',dga_sde,allow_pickle=True)


    siw_dga_ksum_nc = dga_sde['sigma_nc'].mean(axis=(0, 1, 2))
    siw_dga_ksum = dga_sde['sigma'].mean(axis=(0, 1, 2))
    siw_dens_ksum = dga_sde['sigma_dens'].mean(axis=(0, 1, 2))
    siw_magn_ksum = dga_sde['sigma_magn'].mean(axis=(0, 1, 2))

    qiw_grid = ind.IndexGrids(grid_arrays=q_grid.get_grid_as_tuple() + (grids['wn_core'],), keys=('qx', 'qy', 'qz', 'iw'),
                              my_slice=None)

    vn_list = [grids['vn_dmft'], grids['vn_urange'], grids['vn_urange'], grids['vn_urange']]
    siw_list = [dmft1p['sloc'], dmft_sde['siw'], siw_dga_ksum,siw_dga_ksum_nc]
    labels = [r'$\Sigma_{DMFT}(\nu)$', r'$\Sigma_{DMFT-SDE}(\nu)$', r'$\Sigma_{DGA}(\nu)$', r'$\Sigma_{DGA-NC}(\nu)$']
    plotting.plot_siw(vn_list=vn_list, siw_list=siw_list, labels_list=labels, plot_dir=output_path, niv_plot=100)

    plotting.plot_siwk_fs(siwk=dga_sde['sigma'],plot_dir=output_path,kgrid=k_grid, do_shift=True)
    plotting.plot_siwk_fs(siwk=dga_sde['sigma_nc'],plot_dir=output_path,kgrid=k_grid, do_shift=True,name='nc')

    gk_dga_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'],kgrid=k_grid.get_grid_as_tuple(),hr=hr,sigma=dga_sde['sigma'])
    mu_dga = gk_dga_generator.adjust_mu(n=dmft1p['n'], mu0=dmft1p['mu'])
    gk_dga = gk_dga_generator.generate_gk(mu=mu_dga)

    gf_dict = {
        'gk': gk_dga._gk,
        'mu': gk_dga._mu,
        'iv': gk_dga._iv,
        'beta': gk_dga._beta
    }

    np.save(output_path + 'gk_dga.npy',gk_dga,allow_pickle=True)

    plotting.plot_giwk_fs(giwk=gk_dga.gk,plot_dir=output_path,kgrid=k_grid, do_shift=True, name='dga')

    gk_dga_generator_nc = twop.GreensFunctionGenerator(beta=dmft1p['beta'],kgrid=k_grid.get_grid_as_tuple(),hr=hr,sigma=dga_sde['sigma_nc'])
    mu_dga_nc = gk_dga_generator_nc.adjust_mu(n=dmft1p['n'], mu0=dmft1p['mu'])
    gk_dga_nc = gk_dga_generator_nc.generate_gk(mu=mu_dga_nc)

    gf_dict_nc = {
        'gk': gk_dga_nc._gk,
        'mu': gk_dga_nc._mu,
        'iv': gk_dga_nc._iv,
        'beta': gk_dga_nc._beta
    }

    np.save(output_path + 'gk_dga_nc.npy', gk_dga_nc, allow_pickle=True)


    plotting.plot_giwk_fs(giwk=gk_dga_nc.gk,plot_dir=output_path,kgrid=k_grid, do_shift=True, name='dga_nc')

    chi_magn_lambda = dga_sde['chi_magn_lambda'].mat.reshape(q_grid.nk + (niw_core*2+1,))
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(chi_magn_lambda[:,:,0,niw_core].real, cmap='RdBu')
    plt.savefig(output_path + 'chi_magn_w0.png')
    plt.show()

    plt.figure()
    plt.imshow(gamma_dmft['gamma_magn'].mat[niw_core,:, :].real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'gamma_magn.png')
    plt.show()

    plt.figure()
    plt.imshow(gamma_dmft['gamma_dens'].mat[niw_core,:, :].real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'gamma_dens.png')
    plt.show()


# Collect the ladder vertex from subfiles:
if(do_pairing_vertex and comm.rank==0):
    import MpiAux as mpiaux
    import h5py
    qiw_distributor = mpiaux.MpiDistributor(ntasks=box_sizes['niw_core'] * np.prod(nq), comm=comm, output_path=output_path,
                                            name='Qiw')

    # Collect data from subfiles (This is quite ugly, as it is hardcoded to my structure. This should be replaced by a general routine):
    if(do_pairing_vertex):
        if(qiw_distributor.my_rank == 0):
            file_out = h5py.File(fname_ladder_vertex,'w')
            for ir in range(qiw_distributor.mpi_size):
                fname_input = output_path+'QiwRank{:05d}'.format(ir) + '.hdf5'
                file_in = h5py.File(fname_input,'r')
                for key1 in list(file_in.keys()):
                    for key2 in list(file_in[key1].keys()):
                        file_out[key1+'/'+key2] = file_in[key1+'/'+key2][()]

                file_in.close()
                os.remove(fname_input)
            file_out.close()

# import matplotlib.pyplot as plt
# plt.imshow(dga_sde['sigma'][:,:,0,box_sizes['niv_core']], )

# wn_list = [grids['wn_core'], grids['wn_core'], grids['wn_core']]
# chi_magn_lambda_qsum = dga_sde['chi_magn_lambda'].mat
# chi_magn_lambda_qsum = qiw_grid.mean(mat=chi_magn_lambda_qsum, axes=(0, 1, 2))
# chi_magn_ladder_qsum = dga_sde['chi_magn_ladder'].mat
# chi_magn_ladder_qsum = qiw_grid.mean(mat=chi_magn_ladder_qsum, axes=(0, 1, 2))
# chiw_list_magn = [dmft_sde['chi_magn'].mat, chi_magn_lambda_qsum, chi_magn_ladder_qsum]
# labels = [r'$\chi_{magn;DMFT}(\omega)$', r'$\chi_{magn;\lambda}(\omega)$', r'$\chi_{magn;D\Gamma A}(\omega)$']
# plotting.plot_chiw(wn_list=wn_list, chiw_list=chiw_list_magn, labels_list=labels, channel='magn', plot_dir=output_path,
#                    niw_plot=20)
#
# wn_list = [grids['wn_core'], grids['wn_core'], grids['wn_core']]
# chi_dens_lambda_qsum = dga_sde['chi_dens_lambda'].mat
# chi_dens_lambda_qsum = qiw_grid.mean(mat=chi_dens_lambda_qsum, axes=(0, 1, 2))
# chi_dens_ladder_qsum = dga_sde['chi_dens_ladder'].mat
# chi_dens_ladder_qsum = qiw_grid.mean(mat=chi_dens_ladder_qsum, axes=(0, 1, 2))
# chiw_list_dens = [dmft_sde['chi_dens'].mat, chi_dens_lambda_qsum, chi_dens_ladder_qsum]
# labels = [r'$\chi_{magn;DMFT}(\omega)$', r'$\chi_{dens;\lambda}(\omega)$', r'$\chi_{dens;D\Gamma A}(\omega)$']
# plotting.plot_chiw(wn_list=wn_list, chiw_list=chiw_list_dens, labels_list=labels, channel='dens', plot_dir=output_path,
#                    niw_plot=20)
#
# wn_list = [grids['wn_core'], grids['wn_core']]
# chi0q_urange_qsum = dga_sde['chi0q_urange']
# chi0q_urange_qsum = qiw_grid.mean(mat=chi0q_urange_qsum, axes=(0, 1, 2))
# chi0q_list = [dmft_sde['chi0_urange'], chi0q_urange_qsum]
# labels = [r'$\chi_{0;loc}(\omega)$', r'$\chi_{0;q-sum}(\omega)$']
# plotting.plot_chiw(wn_list=wn_list, chiw_list=chi0q_list, labels_list=labels, channel='dens', plot_dir=output_path
#                    niw_plot=20)


# ------------------------------------------------ PAIRING VERTEX ----------------------------------------------------------------
#%%

if(do_pairing_vertex and comm.rank==0):
    import RealTime as rt
    realt = rt.real_time()

    realt.print_time('Start pairing vertex:')

    import PairingVertex as pv
    import h5py

    file = h5py.File(fname_ladder_vertex, 'r')

    def load_qiw(key1=None):
        arr = []
        for key2 in list(file[key1].keys()):
            arr.append(file[key1 + '/' + key2][()])
        return np.array(arr)

    gchi0 = load_qiw(key1='gchi0_core')

    gchi_aux_magn = load_qiw(key1='gchi_aux_magn')
    vrg_magn = load_qiw(key1='vrgq_magn_core')
    chi_magn_lambda = dga_sde['chi_magn_lambda'].mat

    gchi_aux_dens = load_qiw(key1='gchi_aux_dens')
    vrg_dens = load_qiw(key1='vrgq_dens_core')
    chi_dens_lambda = dga_sde['chi_dens_lambda'].mat

    f_magn = pv.ladder_vertex_from_chi_aux(gchi_aux=gchi_aux_magn, vrg=vrg_magn, chir=chi_magn_lambda, gchi0=gchi0, beta=dmft1p['beta']
                                           , u_r=dga_sde['chi_magn_lambda'].u_r)
    f_dens = pv.ladder_vertex_from_chi_aux(gchi_aux=gchi_aux_dens, vrg=vrg_dens, chir=chi_dens_lambda, gchi0=gchi0, beta=dmft1p['beta']
                                           , u_r=dga_sde['chi_dens_lambda'].u_r)

    f_magn = f_magn.reshape(-1,niw_core*2+1,niv_core*2,2*niv_core)
    f_dens = f_dens.reshape(-1,niw_core*2+1,niv_core*2,2*niv_core)

    f_dens_pp = pv.ph_to_pp_notation(mat_ph=f_dens)
    f_magn_pp = pv.ph_to_pp_notation(mat_ph=f_magn)

    f_sing = -1.5 * f_magn_pp + 0.5 * f_dens_pp
    f_trip = -0.5 * f_magn_pp - 0.5 * f_dens_pp

    pairing_vertices = {
        'f_sing':f_sing,
        'f_trip':f_trip
    }

    np.save(output_path + 'pairing_vertices.npy',pairing_vertices)


    f_sing_loc = f_sing.mean(axis=0)
    f_trip_loc = f_trip.mean(axis=0)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(f_sing_loc.real,cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'f_sing_loc.png')
    plt.close()

    fig = plt.figure()
    plt.imshow(f_trip_loc.real,cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'f_trip_loc.png')
    plt.close()

    realt.print_time('End pairing vertex:')
#
# ----------------------------------------------- Eliashberg Equation --------------------------------------------------
#%%

if(do_pairing_vertex and comm.rank == 0):
    import TwoPoint as twop
    import EliashbergEquation as eq
    gamma_sing = f_sing.reshape(nk + (niv_core, niv_core))
    gamma_trip = f_trip.reshape(nk + (niv_core, niv_core))
    g_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=q_grid.get_grid_as_tuple(), hr=hr,
                                               sigma=dga_sde['sigma'])
    mu_dga = gk_dga_generator.adjust_mu(n=dmft1p['n'], mu0=dmft1p['mu'])
    gk_dga = g_generator.generate_gk(mu=mu_dga, qiw=[0, 0, 0, 0], niv=niv_core // 2).gk
    lambda_sing, delta_sing = eq.linear_eliashberg(gamma=gamma_sing, gk=gk_dga, eps = 10**-6, max_count = 10000, norm=np.prod(nk)*dmft1p['beta'])
    lambda_trip, delta_trip = eq.linear_eliashberg(gamma=gamma_trip, gk=gk_dga, eps = 10**-6, max_count = 10000, norm=np.prod(nk)*dmft1p['beta'])

    eliashberg = {
        'lambda_sing': lambda_sing[1].real,
        'lambda_trip': lambda_trip[1].real,
        'delta_sing': delta_sing[1].real,
        'delta_trip': delta_trip[1].real,
    }
    np.save(output_path+'eliashberg.npy',eliashberg)

    plotting.plot_gap_function(delta=delta_sing[1].real, pdir = output_path, name='sing', kgrid=k_grid)
    plotting.plot_gap_function(delta=delta_trip[1].real, pdir = output_path, name='trip', kgrid=k_grid)

