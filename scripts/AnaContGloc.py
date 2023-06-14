# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Post-processing script to perform the analytic continuation of the local Green's function for a finished DGA run.
# The idea of this script is to find suitable settings for the analytic continuation so they can be used to perform a
# fbz run on the cluster.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import sys, os
sys.path.append('../src/')
sys.path.append(os.environ['HOME'] + "/Programs/dga/LambdaDga/src")

import numpy as np

import matplotlib.pyplot as plt
import AnalyticContinuation as a_cont
from Plotting import plot_aw_loc
import Output as output
import Config as conf
# -------------------------------------------- LOAD INPUT DATA ---------------------------------------------------------

input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.85/KonvergenceAnalysis/LambdaDga_lc_sp_Nk64_Nq64_core10_invbse10_vurange20_wurange10_5/'

config = np.load(input_path + 'config.npy', allow_pickle=True).item()
dga_sde = np.load(input_path + 'sigma_dga.npy', allow_pickle=True).item()

k_grid = config._k_grid
niv_urange = config.box.niv_urange
niv_core = config.box.niv_core
beta = config.sys.beta

# Set output path:
config.nam.output_path_ac = output.uniquify(input_path + 'AnaCont') + '/'
os.mkdir(config.nam.output_path_ac)

# Set ana-cont paramter:
t = 1.0
wmax = 15 * t
nw = 1001
use_preblur = True
bw = 2*np.pi/beta
err = 1e-2
nfit = np.min((np.max((niv_core,int(beta * 4))),niv_urange))

# Create ana cont config object:
me_conf = conf.MaxEntConfig(t=t,beta=beta,mesh_type='lorentzian')

v_real = me_conf.mesh


# Perform analytic continuation for the local Green's function:
gloc_dga_cont, gk_dga = a_cont.max_ent_loc(me_conf = me_conf, v_real=v_real, sigma=dga_sde['sigma'],dga_conf=config,niv_cut=niv_urange, bw=0.1, nfit=niv_urange, adjust_mu=True)
plot_aw_loc(output_path=config.nam.output_path_ac, v_real=v_real, gloc=gloc_dga_cont, name='aw-dga-bw{}'.format(bw))
n_int = a_cont.check_filling(v_real=v_real, gloc_cont=gloc_dga_cont)
np.savetxt(config.nam.output_path_ac + 'n_dga_bw{}.txt'.format(bw), [n_int, gk_dga['n']], delimiter=',', fmt='%.9f')
np.save(config.nam.output_path_ac + 'gloc_cont_dga_bw{}.npy'.format(bw), gloc_dga_cont, allow_pickle=True)
