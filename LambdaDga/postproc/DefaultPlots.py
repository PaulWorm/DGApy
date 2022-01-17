# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Generate default sanity plots for a given output from the DGA routine:


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import Plotting as plotting
import BrillouinZone as bz
import TwoPoint as twop
import Hr as hr

# Parameters:

input_path = '/mnt/c/users/pworm/Research/BEPS_Project/TriangularLattice/DGA/TriangularLattice_U11.5_tp1.0_tpp0.0_beta10_n1.0/LambdaDga_Nk4096_Nq4096/'
output_path = input_path

nk = (64,64,1)
k_grid = bz.NamedKGrid(nk=nk, name='k')
beta = 10
n = 1.0

t = 1.00
tp = 1.00 * t
tpp = 0.12 * t * 0
hr = hr.one_band_2d_triangular_t_tp_tpp(t=t, tp=tp, tpp=tpp)

# Load Data

dga_sde = np.load(input_path + 'dga_sde.npy', allow_pickle=True).item()
dga_sde = np.load(input_path + 'dga_sde.npy', allow_pickle=True).item()

plotting.plot_siwk_fs(siwk=dga_sde['sigma'], plot_dir=output_path, kgrid=k_grid, do_shift=True)

gk_dga_generator = twop.GreensFunctionGenerator(beta=beta, kgrid=k_grid.get_grid_as_tuple(), hr=hr, sigma=dga_sde['sigma'])
mu_dga = gk_dga_generator.adjust_mu(n=n, mu0=0.)
gk_dga = gk_dga_generator.generate_gk(mu=mu_dga)

plotting.plot_giwk_fs(giwk=gk_dga.gk, plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga')