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
#input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/Testset1/LambdaDga_Python/'
#input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U8_b010_tp0_tpp0_n0.85/LambdaDgaPython/'
#input_path = '/mnt/d/Research/BEPS_Project/TriangularLattice/TriangularLattice_U10.0_tp1.0_tpp0.0_beta10_n1.0/'
input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
output_path = input_path

fname_dmft = '1p-data.hdf5'
fname_g2 = 'g4iw_sym.hdf5' #'Vertex_sym.hdf5' #'g4iw_sym.hdf5'
fname_ladder_vertex = 'LadderVertex'

# Define options:
do_ladder_vertex = False
lattice = 'square'

# Set up real-space Wannier Hamiltonian:
t = 0.25
tp = 0.00 * t
tpp = 0.12 * t * 0

# Define frequency box-sizes:
niw_core = 20
niv_core = 20
niv_urange = 20
niv_asympt = 5000

# Define k-ranges:
nkf = 8
nk = (nkf, nkf, 1)
nq = (nkf, nkf, 1)

output_folder = 'LambdaDga_Nk{}_Nq{}'.format(np.prod(nk),np.prod(nq))
output_path = output.uniquify(output_path+output_folder) + '/'

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
    'do_ladder_vertex': do_ladder_vertex
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

dga_sde, dmft_sde, gamma_dmft, greens_functions = ldga.lambda_dga(config=config)

if(comm.rank == 0):
    np.save(output_path + 'dmft_sde.npy',dmft_sde,allow_pickle=True)
    np.save(output_path + 'gamma_dmft.npy',gamma_dmft,allow_pickle=True)
    np.save(output_path + 'dga_sde.npy',dga_sde,allow_pickle=True)

    gf_dict = {
        'gk': greens_functions['dga']['gk']._gk,
        'mu': greens_functions['dga']['gk']._mu,
        'iv': greens_functions['dga']['gk']._iv,
        'beta': greens_functions['dga']['gk']._beta
    }
    np.save(output_path + 'gk_dga.npy',gf_dict,allow_pickle=True)

    siw_dga_ksum = dga_sde['sigma'].mean(axis=(0, 1, 2))
    siw_dens_ksum = dga_sde['sigma_dens'].mean(axis=(0, 1, 2))
    siw_magn_ksum = dga_sde['sigma_magn'].mean(axis=(0, 1, 2))

    qiw_grid = ind.IndexGrids(grid_arrays=q_grid.get_grid_as_tuple() + (grids['wn_core'],), keys=('qx', 'qy', 'qz', 'iw'),
                              my_slice=None)

    vn_list = [grids['vn_dmft'], grids['vn_urange'], grids['vn_urange']]
    siw_list = [dmft1p['sloc'], dmft_sde['siw'], siw_dga_ksum]
    labels = [r'$\Sigma_{DMFT}(\nu)$', r'$\Sigma_{DMFT-SDE}(\nu)$', r'$\Sigma_{DGA}(\nu)$']
    plotting.plot_siw(vn_list=vn_list, siw_list=siw_list, labels_list=labels, plot_dir=output_path, niv_plot=100)


    plotting.plot_siwk_fs(siwk=dga_sde['sigma'],plot_dir=output_path,kgrid=k_grid, do_shift=True)

    gk_dga = greens_functions['dga']['gk']
    gk_dmft = greens_functions['dmft']['gk']
    gk_tb = greens_functions['tight_binding']['gk']
    plotting.plot_giwk_fs(giwk=gk_dga.gk,plot_dir=output_path,kgrid=k_grid, do_shift=True, name='dga')
    plotting.plot_giwk_fs(giwk=gk_dmft.gk,plot_dir=output_path,kgrid=k_grid, do_shift=True, name='dmft')
    plotting.plot_giwk_fs(giwk=gk_tb.gk,plot_dir=output_path,kgrid=k_grid, do_shift=True, name='tb')

    chi_magn_lambda = dga_sde['chi_magn_lambda'].mat.reshape(q_grid.nk + (niw_core*2+1,))
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(chi_magn_lambda[:,:,0,niw_core].real, cmap='RdBu')
    plt.savefig(output_path + 'chi_magn_w0.png')
    plt.show()

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

# import PairingVertex as pv
# import h5py
# fname = output_path + 'LadderVertex.hdf5'
# file = h5py.File(fname, 'r')
#
# def load_qiw(key1=None):
#     arr = []
#     for key2 in list(file[key1].keys()):
#         arr.append(file[key1 + '/' + key2][()])
#     return np.array(arr)
#
# gchi0 = load_qiw(key1='gchi0_core')
#
# gchi_aux_magn = load_qiw(key1='gchi_aux_magn')
# vrg_magn = load_qiw(key1='vrgq_magn_core')
# chi_magn_lambda = dga_sde['chi_magn_lambda'].mat
#
# gchi_aux_dens = load_qiw(key1='gchi_aux_dens')
# vrg_dens = load_qiw(key1='vrgq_dens_core')
# chi_dens_lambda = dga_sde['chi_dens_lambda'].mat
#
# f_magn = pv.ladder_vertex_from_chi_aux(gchi_aux=gchi_aux_magn, vrg=vrg_magn, chir=chi_magn_lambda, gchi0=gchi0, beta=dmft1p['beta']
#                                        , u_r=dga_sde['chi_magn_lambda'].u_r)
# f_dens = pv.ladder_vertex_from_chi_aux(gchi_aux=gchi_aux_dens, vrg=vrg_dens, chir=chi_dens_lambda, gchi0=gchi0, beta=dmft1p['beta']
#                                        , u_r=dga_sde['chi_dens_lambda'].u_r)
#
# f_magn = f_magn.reshape(-1,niw_core*2+1,niv_core*2,2*niv_core)
# f_dens = f_dens.reshape(-1,niw_core*2+1,niv_core*2,2*niv_core)
#
# f_dens_pp = pv.ph_to_pp_notation(mat_ph=f_dens)
# f_magn_pp = pv.ph_to_pp_notation(mat_ph=f_magn)
#
# f_sing = -1.5 * f_magn_pp + 0.5 * f_dens_pp
# f_trip = -0.5 * f_magn_pp - 0.5 * f_dens_pp
#
#
# f_sing_loc = f_sing.mean(axis=0)
# f_trip_loc = f_trip.mean(axis=0)
#
# import matplotlib.pyplot as plt
# plt.imshow(f_sing_loc.real,cmap='RdBu')
# plt.colorbar()
# plt.show()
#
# plt.imshow(f_trip_loc.real,cmap='RdBu')
# plt.colorbar()
# plt.show()
#
# # ----------------------------------------------- Eliashberg Equation --------------------------------------------------
# #%%
# import TwoPoint as twop
# gamma_sing = f_sing.reshape(8,8,1,20,20)
# gammax_sing = np.fft.fftn(gamma_sing, axes=(0,1,2))
# g_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=q_grid.get_grid_as_tuple(), hr=hr, sigma=dga_sde['sigma'])
# gk_dga = g_generator.generate_gk(mu=dmft1p['mu'], qiw=[0, 0, 0, 0], niv=niv_core//2).gk
# gmk_dga = np.flip(gk_dga)
#
# Delta = np.random.random_sample(np.shape(gmk_dga))
#
# Delta_old = Delta
# lambda_old = 10
# eps = 10**-6
# max_count = 10000
# converged = False
# count = 0
# while(not converged):
#     count += 1
#     Delta_tilde = np.fft.ifftn(Delta_old * gk_dga * gmk_dga, axes=(0, 1, 2))
#     Delta_new = 1./(8*8) * np.sum(gammax_sing * Delta_tilde[...,None,:],axis=-1)
#     Delta_new = np.fft.fftn(Delta_new,axes=(0,1,2))
#     lambda_r = np.sum(np.conj(Delta_old)*Delta_new)/np.sum((np.conj(Delta_old)*Delta_old))
#     Delta_old = Delta_new/lambda_r
#     if(np.abs(lambda_r-lambda_old)<eps or count > max_count):
#         converged = True
#     lambda_old = lambda_r
#
#
# plt.imshow(Delta_old[:,:,0,10].real,cmap='RdBu')
# plt.colorbar()
# plt.show()
