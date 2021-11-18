#!/usr/bin/env python
# SBATCH -N 2
# SBATCH -J LambdaDga_Nk16
# SBATCH --ntasks-per-node=48
# SBATCH --partition=mem_0096
# SBATCH --qos=p71282_0096
# SBATCH --time=06:00:00

# SBATCH --mail-type=ALL                              # first have to state the type of event to occur
# SBATCH --mail-user=<p.worm@a1.net>   # and then your email address


# ------------------------------------------------ COMMENTS ------------------------------------------------------------

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import Hr as hr
import Indizes as ind
import w2dyn_aux
import MatsubaraFrequencies as mf
import BrillouinZone as bz
import LambdaDga as ldga
import Plotting as plotting
from mpi4py import MPI as mpi

# ----------------------------------------------- PARAMETERS -----------------------------------------------------------

# Define MPI communicator:
comm = mpi.COMM_WORLD

# Define paths of datasets:
input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta80_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
output_path = input_path
fname_dmft = '1p-data.hdf5'
fname_g2 = 'g4iw_sym.hdf5'
# Define frequency box-sizes:
niw_core = 59
niv_core = 40
niv_urange = 500
niv_asympt = 5000

# Define k-ranges:
nk = (16, 16, 1)
nq = (8, 8, 1)

# Generate k-meshes:
k_grid = bz.KGrid(nk=nk, name='k')
q_grid = bz.KGrid(nk=nq, name='q')

# Set up real-space Wannier Hamiltonian:
t = 0.25
tp = -0.25 * t * 0
tpp = 0.12 * t * 0
hr = hr.one_band_2d_t_tp_tpp(t=t, tp=tp, tpp=tpp)

# load contents from w2dynamics DMFT file:
f1p = w2dyn_aux.w2dyn_file(fname=input_path + fname_dmft)
dmft1p = f1p.load_dmft1p_w2dyn()
f1p.close()
niv_dmft = dmft1p['niv']

# Define system paramters, like interaction or inverse temperature.
# Note: I want this to be decoupled from dmft1p, because of later RPA/FLEX stuff.

system = {
    'u': dmft1p['u'],
    'beta': dmft1p['beta'],
    'n': dmft1p['n'],
    'hr': hr
}

names = {
    'input_path': input_path,
    'output_path': output_path,
    'fname_g2': fname_g2
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
    "system": system,
    "names": names,
    "box_sizes": box_sizes,
    "grids": grids,
    "comm": comm,
    "dmft1p": dmft1p
}

# ------------------------------------------------ MAIN ----------------------------------------------------------------

dga_sde, dmft_sde, gamma_dmft = ldga.lambda_dga(config=config)

siw_dga_ksum = dga_sde['sigma'].mean(axis=(0, 1, 2))
siw_dens_ksum = dga_sde['sigma_dens'].mean(axis=(0, 1, 2))
siw_magn_ksum = dga_sde['sigma_magn'].mean(axis=(0, 1, 2))

qiw_grid = ind.IndexGrids(grid_arrays=q_grid.get_grid_as_tuple() + (grids['wn_core'],), keys=('qx', 'qy', 'qz', 'iw'),
                          my_slice=None)

vn_list = [grids['vn_dmft'], grids['vn_urange'], grids['vn_urange'], grids['vn_urange'], grids['vn_urange']]
siw_list = [dmft1p['sloc'], dmft_sde['siw'], siw_dga_ksum, siw_dens_ksum, siw_magn_ksum]
labels = [r'$\Sigma_{DMFT}(\nu)$', r'$\Sigma_{DMFT-SDE}(\nu)$', r'$\Sigma_{DGA}(\nu)$', r'$\Sigma_{DGA-dens}(\nu)$',
          r'$\Sigma_{DGA-magn}(\nu)$']
plotting.plot_siw(vn_list=vn_list, siw_list=siw_list, labels_list=labels, plot_dir=output_path, niv_plot=40)

vn_list = [grids['vn_dmft'], grids['vn_urange'], grids['vn_urange']]
siw_list = [dmft1p['sloc'], dmft_sde['siw'], siw_dga_ksum]
labels = [r'$\Sigma_{DMFT}(\nu)$', r'$\Sigma_{DMFT-SDE}(\nu)$', r'$\Sigma_{DGA}(\nu)$']
plotting.plot_siw(vn_list=vn_list, siw_list=siw_list, labels_list=labels, plot_dir=output_path, niv_plot=40)

wn_list = [grids['wn_core'], grids['wn_core'], grids['wn_core']]
chi_magn_lambda_qsum = dga_sde['chi_magn_lambda'].mat
chi_magn_lambda_qsum = qiw_grid.mean(mat=chi_magn_lambda_qsum, axes=(0, 1, 2))
chi_magn_ladder_qsum = dga_sde['chi_magn_ladder'].mat
chi_magn_ladder_qsum = qiw_grid.mean(mat=chi_magn_ladder_qsum, axes=(0, 1, 2))
chiw_list_magn = [dmft_sde['chi_magn'].mat, chi_magn_lambda_qsum, chi_magn_ladder_qsum]
labels = [r'$\chi_{magn;DMFT}(\omega)$', r'$\chi_{magn;\lambda}(\omega)$', r'$\chi_{magn;D\Gamma A}(\omega)$']
plotting.plot_chiw(wn_list=wn_list, chiw_list=chiw_list_magn, labels_list=labels, channel='magn', plot_dir=output_path,
                   niw_plot=20)

wn_list = [grids['wn_core'], grids['wn_core'], grids['wn_core']]
chi_dens_lambda_qsum = dga_sde['chi_dens_lambda'].mat
chi_dens_lambda_qsum = qiw_grid.mean(mat=chi_dens_lambda_qsum, axes=(0, 1, 2))
chi_dens_ladder_qsum = dga_sde['chi_dens_ladder'].mat
chi_dens_ladder_qsum = qiw_grid.mean(mat=chi_dens_ladder_qsum, axes=(0, 1, 2))
chiw_list_dens = [dmft_sde['chi_dens'].mat, chi_dens_lambda_qsum, chi_dens_ladder_qsum]
labels = [r'$\chi_{magn;DMFT}(\omega)$', r'$\chi_{dens;\lambda}(\omega)$', r'$\chi_{dens;D\Gamma A}(\omega)$']
plotting.plot_chiw(wn_list=wn_list, chiw_list=chiw_list_dens, labels_list=labels, channel='dens', plot_dir=output_path,
                   niw_plot=20)

wn_list = [grids['wn_core'], grids['wn_core']]
chi0q_urange_qsum = dga_sde['chi0q_urange']
chi0q_urange_qsum = qiw_grid.mean(mat=chi0q_urange_qsum, axes=(0, 1, 2))
chi0q_list = [dmft_sde['chi0_urange'], chi0q_urange_qsum]
labels = [r'$\chi_{0;loc}(\omega)$', r'$\chi_{0;q-sum}(\omega)$']
plotting.plot_chiw(wn_list=wn_list, chiw_list=chi0q_list, labels_list=labels, channel='dens', plot_dir=output_path,
                   niw_plot=20)
