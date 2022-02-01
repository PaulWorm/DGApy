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
# -------------------------------------------- LOAD INPUT DATA ---------------------------------------------------------

input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta130_n0.925/LambdaDga_lc_sp_Nk10000_Nq10000_core120_invbse120_vurange500_wurange120/'

config = np.load(input_path + 'config.npy', allow_pickle=True).item()
dga_sde = np.load(input_path + 'dga_sde.npy', allow_pickle=True).item()

k_grid = config['grids']['k_grid']
niv_urange = config['box_sizes']['niv_urange']
niv_core = config['box_sizes']['niv_core']
beta = config['dmft1p']['beta']

# Set output path:
output_path = output.uniquify(input_path + 'AnaCont') + '/'
os.mkdir(output_path)

# Set ana-cont paramter:
t = 1.0
wmax = 15 * t
nw = 1001
use_preblur = True
bw = 2*np.pi/beta
err = 1e-2
nfit = np.min((np.max((niv_core,int(beta * 4))),niv_urange))
v_real = a_cont.v_real_tan(wmax=wmax,nw=nw)


# Perform analytic continuation for the local Green's function:
gloc_dga_cont, gk_dga = a_cont.max_ent_loc(v_real=v_real, sigma=dga_sde['sigma'], config=config, k_grid=k_grid,
                                           niv_cut=niv_urange, use_preblur=use_preblur, bw=bw, err=err,
                                           nfit=nfit, adjust_mu=True)
plot_aw_loc(output_path=output_path, v_real=v_real, gloc=gloc_dga_cont, name='aw-dga-bw{}'.format(bw))
n_int = a_cont.check_filling(v_real=v_real, gloc_cont=gloc_dga_cont)
np.savetxt(output_path + 'n_dga_bw{}.txt'.format(bw), [n_int, gk_dga['n']], delimiter=',', fmt='%.9f')
np.save(output_path + 'gloc_cont_dga_bw{}.npy'.format(bw), gloc_dga_cont, allow_pickle=True)
