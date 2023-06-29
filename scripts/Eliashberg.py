# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# WARNING: Currently there cannot be more processes than Niw*1+2. This is because my MpiDistibutor cannot handle slaves
# that have nothing to do.

# Warning: many of the files I prepared from Motoharu do not have smom stored.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI as mpi
from ruamel.yaml import YAML
import gc

import dga.config as config
import dga.matsubara_frequencies as mf
import dga.brillouin_zone as bz
import dga.loggers as loggers
import dga.dga_io as io
import dga.four_point as fp
import dga.local_four_point as lfp
import dga.plotting as plotting
import dga.lambda_correction as lc
import dga.two_point as twop
import dga.bubble as bub
import dga.hk as hamk
import dga.mpi_aux as mpi_aux
import dga.pairing_vertex as pv
import dga.util
import dga.eliashberg_equation as eq

# ----------------------------------------------- PARAMETERS -----------------------------------------------------------
input_path = '../test/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/LambdaDga_lc_spch_Nk64_Nq64_wcore30_vcore30_vshell200_23/'

# Set options:
update_mu = True
n_eig = 3
sym_sing = True
sym_trip = True

output_path = input_path

output_path = io.uniquify(output_path + 'Eliashberg1') + '/'
os.mkdir(output_path)

# Starting gap functions:
gap0_sing = {
    'k': 'p-wave-x',
    'v': 'random'
}

gap0_trip = {
    'k': 'p-wave-y',
    'v': 'even'
}

# # Starting gap functions:
# gap0_sing = {
#     'k': 'p-wave-y',
#     'v': 'odd'
# }
#
# gap0_trip = {
#     'k': 'p-wave-y',
#     'v': 'even'
# }

parser = config.create_dga_argparser(name='raw_config_file.yaml',path=input_path)
args = parser.parse_args()
assert hasattr(args, 'config'), 'Config file location must be provided.'
conf_file = YAML().load(open(args.path + args.config))
dga_config = config.DgaConfig(conf_file)
dga_config.output_path = input_path
pairing_config = config.EliashbergConfig(conf_file['pairing'])

f_sing = dga_config.load_data('F_sing_pp')
f_trip = dga_config.load_data('F_trip_pp')

gamma_sing = -f_sing   #- 2*f_sing.mean(axis=(0,1,2))
# gamma_sing = 0.5*(f_sing +  np.roll(np.flip(np.transpose(f_sing,axes=(0,1,2,4,3)),axis=(0,1)),shift=(1,1), axis=(0,1)))
gamma_trip = -f_trip  # - 2*f_trip.mean(axis=(0,1,2))

if (sym_sing):
    gamma_sing = 0.5 * (gamma_sing + np.flip(gamma_sing, axis=(-1)))

if (sym_trip):
    gamma_trip = 0.5 * (gamma_trip - np.flip(gamma_trip, axis=(-1)))

dmft_input = dga_config.load_data('dmft_input')
siwk_dmft = twop.SelfEnergy(dmft_input['siw'][None,None,None,:],dmft_input['beta'])
siwk = dga_config.load_data('siwk_dga')
siwk = twop.create_dga_siwk_with_dmft_as_asympt(siwk,siwk_dmft,dga_config.box_sizes.niv_shell)

hr = dga_config.lattice.set_hr()
ek = hamk.ek_3d(dga_config.lattice.k_grid.grid, hr)
g_generator = twop.GreensFunction(siwk,ek,n=dmft_input['n'], niv_asympt=dga_config.box_sizes.niv_pp)

gk_dga = mf.cut_v(g_generator.core,niv_cut=dga_config.box_sizes.niv_pp,axes=(-1,))
nq = dga_config.lattice.nq_tot
norm = np.prod(nq) * dmft_input['beta']
q_grid = dga_config.lattice.q_grid

gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=gap0_sing['k'], v_type=gap0_sing['v'],
                         k_grid=q_grid.grid)

powiter_sing = eq.EliashberPowerIteration(gamma=gamma_sing, gk=gk_dga, gap0=gap0, norm=norm,shift_mat=True, n_eig=n_eig)

gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=gap0_trip['k'], v_type=gap0_trip['v'],
                         k_grid=q_grid.grid)
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


for i in range(len(powiter_sing.gap)):
    plotting.plot_gap_function(delta=powiter_sing.gap[i].real, pdir=output_path, name='sing_{}'.format(i), kgrid=q_grid,
                               do_shift=True)
    plotting.plot_gap_function(delta=powiter_sing.gap[i].real, pdir=output_path, name='trip_{}'.format(i), kgrid=q_grid,
                               do_shift=True)
