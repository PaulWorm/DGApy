import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI as mpi

import TestData as td
import dga.config as config
import dga.high_level_routines as hlr
import dga.hk as hamk
import dga.two_point as twop

import dga.matsubara_frequencies as mf
import dga.dga_io as dga_io
import dga.plot_specs as ps


def get_convergence_with_niv_shell(input_path, input_type, fname_1p, fname_2p, niv_shell_range, niv_core, niw_core):
    conf_dict = config.get_default_config_dict()
    dga_config = config.DgaConfig(conf_dict)

    dga_config.input_path = input_path
    dga_config.input_type = input_type
    dga_config.fname_1p = fname_1p
    dga_config.fname_2p = fname_2p
    dga_config.box_sizes.niv_core = niv_core
    dga_config.box_sizes.niw_core = niw_core
    dmft_input = dga_io.load_1p_data(dga_config.input_type, dga_config.input_path, dga_config.fname_1p, dga_config.fname_2p)
    g2_dens, g2_magn = dga_io.load_g2(dga_config.box_sizes, dmft_input)

    # Build the DMFT Green's function:
    ek = hamk.ek_3d(dga_config.lattice.k_grid.grid, dga_config.lattice.hr)

    siwk_dmft = twop.SelfEnergy(sigma=dmft_input['siw'][None, None, None, :], beta=dmft_input['beta'])
    giwk_dmft = twop.GreensFunction(siwk_dmft, ek, mu=dmft_input['mu_dmft'], niv_asympt=3000)

    siw_sde_range = []
    gamma_dens_range = []
    gamma_magn_range = []
    chi_dens_range = []
    chi_magn_range = []
    vrg_dens_range = []
    vrg_magn_range = []
    for niv_shell in niv_shell_range:
        dga_config.box_sizes.niv_shell = niv_shell
        gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, siw_sde_full = hlr.local_sde_from_g2(g2_dens, g2_magn,
                                                                                                       giwk_dmft,
                                                                                               dga_config,
                                                                                               dmft_input,
                                                                                               comm,
                                                                                               write_output=False)
        siw_sde_range.append(siw_sde_full)

        # Check convergence of eigenvalues and selected entries in Gamma_magn and Gamma_dens
        gamma_dens_range.append(gamma_dens.mat)
        gamma_magn_range.append(gamma_magn.mat)

        # Check convergence of the susceptibility:
        chi_dens_range.append(chi_dens)
        chi_magn_range.append(chi_magn)

        # Convergence of vrg:
        vrg_dens_range.append(vrg_dens.mat)
        vrg_magn_range.append(vrg_magn.mat)
    return siw_sde_range, gamma_dens_range, gamma_magn_range, chi_dens_range, chi_magn_range, vrg_dens_range, vrg_magn_range

comm = mpi.COMM_WORLD

# Load dataset:


# input_path = './2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
input_path = './2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
input_type = 'w2dyn'
fname_1p = '1p-data.hdf5'
fname_2p = 'g4iw_sym.hdf5'

# input_type = 'EDFermion'
# fname_1p = 'EDFermion_1p-data.hdf5'
# fname_2p = 'EDFermion_g4iw_sym.hdf5'
dmft_input = dga_io.load_1p_data(input_type, input_path, fname_1p, fname_2p)
# niv_shell_range = np.array([0,10,30,60,120,240,480])
niv_shell_range = np.array([400])

niv_core_1 = 30
niw_core_1 = 20
siw_sde_1, gamma_dens_1, gamma_magn_1, chi_dens_1, chi_magn_1, vrg_dens_range_1, vrg_magn_range_1 = \
    get_convergence_with_niv_shell(input_path, input_type, fname_1p,
                                                                                               fname_2p, niv_shell_range,
                                                                                               niv_core_1,
                                                                                               niw_core_1)

# %%
pdir = './TestPLots/'
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

line_colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, len(niv_shell_range)))
for i, siw_sde in enumerate(siw_sde_1):
    axes[0].loglog(mf.vn_from_mat(siw_sde), np.abs(siw_sde.real), color=line_colors[i], marker='o')
    axes[1].loglog(mf.vn_from_mat(siw_sde), np.abs(siw_sde.imag), color=line_colors[i], marker='o')


axes[0].loglog(mf.vn_from_mat(dmft_input['siw']), np.abs(dmft_input['siw'].real), 'k', marker='')
axes[1].loglog(mf.vn_from_mat(dmft_input['siw']), np.abs(dmft_input['siw'].imag), 'k', marker='')

for i, siw_sde in enumerate(siw_sde_1):
    niv_sde = np.size(siw_sde) // 2
    siw_dmft_clip = mf.cut_v(dmft_input['siw'], niv_cut=niv_sde)
    axes[2].plot(mf.vn_from_mat(siw_sde), (siw_sde.real), color=line_colors[i], marker='o')
    axes[3].plot(mf.vn_from_mat(siw_sde), (siw_sde.imag), color=line_colors[i], marker='o')
    # axes[2].loglog(mf.vn_from_mat(siw_sde), np.abs(siw_sde.real - siw_dmft_clip.real), color=line_colors[i], marker='o')
    # axes[3].loglog(mf.vn_from_mat(siw_sde), np.abs(siw_sde.imag - siw_dmft_clip.imag), color=line_colors[i], marker='o')
axes[2].plot(mf.vn_from_mat(dmft_input['siw']), (dmft_input['siw'].real), 'k', marker='')
axes[3].plot(mf.vn_from_mat(dmft_input['siw']), (dmft_input['siw'].imag), 'k', marker='')


# for ax in axes:
#     ax.set_xlim()
for ax in axes:
    ax.set_xlim(0,None)
axes[2].set_xlim(0,100)
axes[3].set_xlim(0,100)
axes[3].set_ylim(None,0)
plt.tight_layout()
# plt.savefig(fname=pdir + 'Siw_convergence_with_niv_shell.png')
plt.show()