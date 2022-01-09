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
input_path = '/mnt/c/users/pworm/Research/Ba2CuO4/Plane1/U3.0eV_n0.93_b040/LambdaDga_lc_sp_Nk1024_Nq1024_core60_invbse100_vurange100_wurange100/'
#input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.25_tpp0.12_beta12.5_n0.85/LambdaDga_lc_spch_Nk4096_Nq4096_core25_invbse60_vurange100_wurange50/'
#input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.25_tpp0.12_beta12.5_n0.85/LambdaDga_lc_spch_Nk64_Nq64_core8_invbse10_vurange20_wurange10_4/'
output_path = input_path

# Starting gap functions:
gap0_sing = {
    'k': 'd-wave',
    'v': 'even'
}

gap0_trip = {
    'k': 'd-wave',
    'v': 'odd'
}

# Starting gap functions:
gap0_sing = {
    'k': 'p-wave-y',
    'v': 'odd'
}

gap0_trip = {
    'k': 'p-wave-y',
    'v': 'even'
}

config = np.load(input_path + 'config.npy', allow_pickle=True).item()
dmft1p = config['dmft1p']
nq = config['box_sizes']['nk']
hr = config['system']['hr']
q_grid = bz.KGrid(nk=nq, name='q')
niv_core = config['box_sizes']['niv_core']
niv_pp = config['box_sizes']['niv_pp']
dga_sde = np.load(input_path + 'dga_sde.npy', allow_pickle=True).item()
pairing_vertices = np.load(input_path + 'pairing_vertices.npy', allow_pickle=True).item()

f_sing = pairing_vertices['f_sing']
f_trip = pairing_vertices['f_trip']

gamma_sing = -f_sing   #- 2*f_sing.mean(axis=(0,1,2))
# gamma_sing = 0.5*(f_sing +  np.roll(np.flip(np.transpose(f_sing,axes=(0,1,2,4,3)),axis=(0,1)),shift=(1,1), axis=(0,1)))
gamma_trip = -f_trip  # - 2*f_trip.mean(axis=(0,1,2))
g_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=q_grid.get_grid_as_tuple(), hr=hr,
                                           sigma=dga_sde['sigma'])
mu_dga = g_generator.adjust_mu(n=dmft1p['n'], mu0=dmft1p['mu'])
gk_dga = g_generator.generate_gk(mu=mu_dga, qiw=[0, 0, 0, 0], niv=niv_pp).gk

gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=gap0_sing['k'], v_type=gap0_sing['v'],
                         k_grid=q_grid.get_grid_as_tuple())
norm = np.prod(nq) * dmft1p['beta']
n_eig = 2
powiter_sing = eq.EliashberPowerIteration(gamma=gamma_sing, gk=gk_dga, gap0=gap0, norm=norm,shift_mat=True, n_eig=n_eig)

gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=gap0_trip['k'], v_type=gap0_trip['v'],
                         k_grid=q_grid.get_grid_as_tuple())
powiter_trip = eq.EliashberPowerIteration(gamma=gamma_trip, gk=gk_dga, gap0=gap0, norm=norm,shift_mat=True, n_eig=n_eig)


eliashberg = {
    'lambda_sing': powiter_sing.lam,
    'lambda_trip': powiter_trip.lam,
    'delta_sing': powiter_sing.gap,
    'delta_trip': powiter_trip.gap,
}
np.save(output_path + 'eliashberg.npy', eliashberg)
np.savetxt(output_path + 'eigenvalues.txt', [powiter_sing.lam.real, powiter_trip.lam.real], delimiter=',', fmt='%.9f')
np.savetxt(output_path + 'eigenvalues_s.txt', [powiter_sing.lam_s.real, powiter_trip.lam_s.real], delimiter=',', fmt='%.9f')

import Plotting as plotting

for i in range(len(powiter_sing.gap)):
    plotting.plot_gap_function(delta=powiter_sing.gap[i].real, pdir=output_path, name='sing_{}'.format(i), kgrid=q_grid,
                               do_shift=True)
    plotting.plot_gap_function(delta=powiter_sing.gap[i].real, pdir=output_path, name='trip_{}'.format(i), kgrid=q_grid,
                               do_shift=True)
