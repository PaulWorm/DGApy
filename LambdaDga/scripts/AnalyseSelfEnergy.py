# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Currently this here should plot the Green's function without adjust the chemical potential.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import TwoPoint as twop
import Plotting as plotting
import BrillouinZone as bz

input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta60_n0.75/LambdaDga_lc_sp_Nk14400_Nq14400_core80_invbse80_vurange150_wurange80/'
output_path = input_path

config = np.load(input_path + 'config.npy', allow_pickle=True).item()
dmft1p = config['dmft1p']
k_grid = config['grids']['k_grid']
niv_urange = config['box_sizes']['niv_urange']
hr = config['system']['hr']
dga_sde  = np.load(input_path + 'dga_sde.npy', allow_pickle=True).item()

gf_dict = twop.create_gk_dict(sigma=dga_sde['sigma'], kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'], n=dmft1p['n'],
                                  mu0=dmft1p['mu'])
gf_dict_mu_dmft = twop.create_gk_dict(sigma=dga_sde['sigma'], kgrid=k_grid.grid, hr=hr, beta=dmft1p['beta'],
                                      n=dmft1p['n'],
                                      mu0=dmft1p['mu'], adjust_mu=False)

np.save(output_path + 'gk_dga.npy', gf_dict, allow_pickle=True)
np.save(output_path + 'gk_dga_mu_dmft.npy', gf_dict_mu_dmft, allow_pickle=True)

ak_fs_mu_dmft = -1. / np.pi * gf_dict_mu_dmft['gk'][:, :, :, niv_urange].imag
ind_fs_mu_dmft = bz.find_fermi_surface_peak(ak_fs=ak_fs_mu_dmft, kgrid=k_grid)

np.savetxt(output_path + 'mu.txt', [[gf_dict['mu'], dmft1p['mu']], [gf_dict['n'], gf_dict_mu_dmft['n']]], delimiter=',',
           fmt='%.9f')

# DGA with mu from DMFT:
plotting.plot_giwk_fs(giwk=gf_dict_mu_dmft['gk'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga_mu_dft',
                      ind_fs=ind_fs_mu_dmft)
plotting.plot_giwk_qpd(giwk=gf_dict_mu_dmft['gk'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga_mu_dft')

# # DGA with adjusted mu:
# plotting.plot_giwk_fs(giwk=gf_dict['gk'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga',
#                       ind_fs=ind_fs_mu_dmft)
# plotting.plot_giwk_qpd(giwk=gf_dict['gk'], plot_dir=output_path, kgrid=k_grid, do_shift=True, name='dga')
