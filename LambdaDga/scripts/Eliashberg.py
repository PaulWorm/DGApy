# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# WARNING: Currently there cannot be more processes than Niw*1+2. This is because my MpiDistibutor cannot handle slaves
# that have nothing to do.

# Warning: many of the files I prepared from Motoharu do not have smom stored.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import sys, os

sys.path.append('../src/')
sys.path.append(os.environ['HOME'] + "/Programs/dga/LambdaDga/src")
import BrillouinZone as bz
import EliashbergEquation as eq
import TwoPoint as twop

import matplotlib.pyplot as plt


# ----------------------------------------------- PARAMETERS -----------------------------------------------------------
input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.25_tpp0.12_beta12.5_n0.85/LambdaDga_lc_spch_Nk256_Nq256_core29_invbse30_vurange30_wurange29/'
output_path = input_path

config = np.load(input_path + 'config.npy', allow_pickle=True).item()
dmft1p = config['dmft1p']
nq = config['box_sizes']['nk']
hr = config['system']['hr']
q_grid = bz.KGrid(nk=nq, name='q')
niv_core = config['box_sizes']['niv_core']
dga_sde = np.load(input_path + 'dga_sde.npy', allow_pickle=True).item()
pairing_vertices = np.load(input_path + 'pairing_vertices.npy', allow_pickle=True).item()

f_sing = pairing_vertices['f_sing']
f_trip = pairing_vertices['f_trip']

gamma_sing = -f_sing  # - 2*f_sing.mean(axis=(0,1,2))
# gamma_sing = 0.5*(f_sing +  np.roll(np.flip(np.transpose(f_sing,axes=(0,1,2,4,3)),axis=(0,1)),shift=(1,1), axis=(0,1)))
gamma_trip = -f_trip  # - 2*f_trip.mean(axis=(0,1,2))
g_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=q_grid.get_grid_as_tuple(), hr=hr,
                                           sigma=dga_sde['sigma'])
mu_dga = g_generator.adjust_mu(n=dmft1p['n'], mu0=dmft1p['mu'])
gk_dga = g_generator.generate_gk(mu=mu_dga, qiw=[0, 0, 0, 0], niv=niv_core // 2).gk

gap_0 = eq.get_gap_start(shape=np.shape(gk_dga),k_type='d-wave',v_type='even',k_grid=q_grid.get_grid_as_tuple())

lambda_sing, delta_sing = eq.linear_eliashberg(gamma=gamma_sing, gk=gk_dga, eps=10 ** -7, max_count=10000,
                                               norm=np.prod(nq) * dmft1p['beta'], gap_0=gap_0)

gap_0 = eq.get_gap_start(shape=np.shape(gk_dga),k_type='d-wave',v_type='odd',k_grid=q_grid.get_grid_as_tuple())
lambda_trip, delta_trip = eq.linear_eliashberg(gamma=gamma_trip, gk=gk_dga, eps=10 ** -7, max_count=10000,
                                               norm=np.prod(nq) * dmft1p['beta'], gap_0=gap_0)

eliashberg = {
    'lambda_sing': lambda_sing,
    'lambda_trip': lambda_trip,
    'delta_sing': delta_sing,
    'delta_trip': delta_trip,
}
np.save(output_path + 'eliashberg.npy', eliashberg)
np.savetxt(output_path + 'eigenvalues.txt', [lambda_sing[1].real, lambda_trip[1].real], delimiter=',', fmt='%.9f')

import Plotting as plotting
plotting.plot_gap_function(delta=delta_sing[1].real, pdir=output_path, name='sing', kgrid=q_grid, do_shift=True)
plotting.plot_gap_function(delta=delta_trip[1].real, pdir=output_path, name='trip', kgrid=q_grid, do_shift=True)

