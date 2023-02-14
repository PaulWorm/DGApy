 # ----------------------------------------------------------------------------------------------------
# File: MaxEntDGA.py
# Date: 07.11.2022
# Author: Paul Worm
# Short description: Interface for dga and the ana-cont package for analytic continuation.
# ----------------------------------------------------------------------------------------------------

import numpy as np
import h5py
import sys, os
sys.path.append('/home/fs71282/wormEco3P/Programs/dga_reimplementation/dga/src')
import continuation as cont
import matplotlib.pyplot as plt
import Output as output
import TwoPoint as tp
import MatsubaraFrequencies as mf
import BrillouinZone as bz
from mpi4py import MPI as mpi
import Config as conf

# Define MPI communicator:
comm = mpi.COMM_WORLD

# ------------------------------------------------ PARAMETERS -------------------------------------------------

# Set the path, where the input-data is located:
t = 1
bw_range = [0.0,]
# base = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.90/'
# path = base + 'LambdaDga_lc_sp_Nk14400_Nq14400_core60_invbse60_vurange500_wurange60_1/'
path = '../'


# ------------------------------------------------- LOAD DATA --------------------------------------------------

# Load the data of the preceeding DGA run:
dga_conf = np.load(path + 'config.npy',allow_pickle=True).item()
dmft_sde = np.load(path + 'dmft_sde.npy',allow_pickle=True).item()

# Update the output-path:
dga_conf.nam.output_path_ac = path + 'AnaContSigmaFBZ/'
dga_conf.nam.output_path_ac = output.uniquify(dga_conf.output_path_ac)
if(comm.rank == 0):
    os.mkdir(dga_conf.nam.output_path_ac)
comm.Barrier()

# Update the me_config object:
hartree = dmft_sde['hartree']
me_conf = conf.MaxEntConfig(t=t, beta=dga_conf.sys.beta, mesh_type='tan', nwr=1001,wmax=25)
sigma = np.load(path + 'sigma.npy', allow_pickle=True)
n_fit = 60

# Perform the analytic continuation:
print('Starting!')
# # Do analytic continuation within the irreducible Brillouin Zone:
output.max_ent_irrk_bw_range_sigma(comm=comm, dga_conf=dga_conf, me_conf=me_conf, bw_range=bw_range,
                          sigma=sigma,hartree=hartree, n_fit=n_fit, name='dga_sigma')
print('Finished!')