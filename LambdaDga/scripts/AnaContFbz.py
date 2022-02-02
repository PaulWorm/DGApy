# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Post-processing script to perform the analytic continuation of the local Green's function for a finished DGA run.
# The idea of this script is to find suitable settings for the analytic continuation so they can be used to perform a
# fbz run on the cluster.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import sys, os
sys.path.append('../src/')
sys.path.append(os.environ['HOME'] + "/Programs/dga/LambdaDga/src")
sys.path.append('../ana_cont/')
sys.path.append(os.environ['HOME'] + "/Programs/dga/LambdaDga/ana_cont")

import numpy as np

import matplotlib.pyplot as plt
import AnalyticContinuation as a_cont
from Plotting import plot_aw_loc
import Output as output
import TwoPoint as twop
from mpi4py import MPI as mpi
import MpiAux as mpiaux
import Indizes as ind
import Plotting
from Plotting import plot_cont_fs
from Plotting import plot_cont_edc_maps

# -------------------------------------------- LOAD INPUT DATA ---------------------------------------------------------
# Define MPI communicator:
comm = mpi.COMM_WORLD

# Define path for input files:
input_path = './'
#input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta130_n0.925/LambdaDga_lc_sp_Nk10000_Nq10000_core120_invbse120_vurange500_wurange120/'
input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.85/KonvergenceAnalysis/LambdaDga_lc_sp_Nk576_Nq576_core10_invbse10_vurange20_wurange10_1/'

config = np.load(input_path + 'config.npy', allow_pickle=True).item()
dga_sde = np.load(input_path + 'dga_sde.npy', allow_pickle=True).item()

k_grid = config['grids']['k_grid']
niv_urange = config['box_sizes']['niv_urange']
niv_core = config['box_sizes']['niv_core']
beta = config['dmft1p']['beta']
dmft1p = config['dmft1p']
# Set output path:
output_path = output.uniquify(input_path + 'AnaCont') + '/'

# Set ana-cont parameter:
t = 1.0
wmax = 15 * t
nwr = 1001
use_preblur = True
bw = 2*np.pi/beta
err = 1e-2
nfit = np.min((np.max((niv_core,int(beta * 4))),niv_urange))
v_real = a_cont.v_real_tan(wmax=wmax,nw=nwr)

comm.Barrier()
if (comm.rank == 0):
    os.mkdir(output_path)
    text_file = open(output_path + 'cont_settings.txt', 'w')
    text_file.write(f'nfit={nfit} \n')
    text_file.write(f'err={err} \n')
    text_file.write(f'wmax={wmax} \n')
    text_file.write(f'nwr={nwr} \n')
    text_file.close()

comm.Barrier()

# Perform analytical continuation in the full BZ:

irrk_distributor = mpiaux.MpiDistributor(ntasks=k_grid.nk_irr, comm=comm)

index_grid_keys = ('irrk',)
irrk_grid = ind.IndexGrids(grid_arrays=(k_grid.irrk_ind_lin,), keys=index_grid_keys,
                           my_slice=irrk_distributor.my_slice)
ind_irrk = np.squeeze(np.array(np.unravel_index(k_grid.irrk_ind[irrk_grid.my_indizes], shape=k_grid.nk))).T
if (np.size(ind_irrk.shape) > 1):
    ind_irrk = [tuple(ind_irrk[i, :]) for i in np.arange(ind_irrk.shape[0])]
else:
    ind_irrk = tuple(ind_irrk)
gk = twop.create_gk_dict(sigma=dga_sde['sigma'], kgrid=k_grid.grid, hr=config['system']['hr'], beta=dmft1p['beta'], n=dmft1p['n'],
                         mu0=dmft1p['mu'], adjust_mu=True, niv_cut=niv_urange)
gk_my_cont = a_cont.do_max_ent_on_ind_T(mat=gk['gk'], ind_list=ind_irrk, v_real=v_real,
                                        beta=dmft1p['beta'],
                                        n_fit=nfit, err=err, alpha_det_method='chi2kink', use_preblur=use_preblur,
                                        bw=bw)

comm.Barrier()
gk_cont = irrk_distributor.allgather(rank_result=gk_my_cont)
comm.Barrier()
if (comm.rank == 0):
    gk_cont_fbz = k_grid.irrk2fbz(mat=gk_cont)

    plot_cont_fs(output_path=output_path,
                          name='fermi_surface_dga_cont_wint-0.1-bw{}'.format(bw),
                          gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=-0.1)
    plot_cont_fs(output_path=output_path, name='fermi_surface_dga_cont_wint-0.05-bw{}'.format(bw),
                          gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=-0.05)
    plot_cont_fs(output_path=output_path, name='fermi_surface_dga_cont_w0-bw{}'.format(bw),
                          gk=gk_cont_fbz, v_real=v_real, k_grid=k_grid, w_int=None)
    np.save(output_path + 'gk_dga_cont_fbz_bw{}.npy'.format(bw), gk_cont_fbz, allow_pickle=True)
    plot_cont_edc_maps(v_real=v_real, gk_cont=gk_cont_fbz, k_grid=k_grid,
                                output_path=output_path,
                                name='fermi_surface_dga_cont_edc_maps_bw{}'.format(bw))
