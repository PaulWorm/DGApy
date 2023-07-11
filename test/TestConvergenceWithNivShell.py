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
input_path = './2DSquare_U8_tp-0.2_tpp0.1_beta50_n0.90/'
input_type = 'w2dyn'
fname_1p = '1p-data.hdf5'
fname_2p = 'g4iw_sym.hdf5'

# input_type = 'EDFermion'
# fname_1p = 'EDFermion_1p-data.hdf5'
# fname_2p = 'EDFermion_g4iw_sym.hdf5'
dmft_input = dga_io.load_1p_data(input_type, input_path, fname_1p, fname_2p)
# niv_shell_range = np.array([0,10,30,60,120,240,480])
niv_shell_range = np.array([0, 10, 30, 60, 120, 240])

niv_core_1 = 30
niw_core_1 = 2
siw_sde_1, gamma_dens_1, gamma_magn_1, chi_dens_1, chi_magn_1, vrg_dens_range_1, vrg_magn_range_1 = \
    get_convergence_with_niv_shell(input_path, input_type, fname_1p,
                                                                                               fname_2p, niv_shell_range,
                                                                                               niv_core_1,
                                                                                               niw_core_1)

niv_core_2 = 40
niw_core_2 = 2
siw_sde_2, gamma_dens_2, gamma_magn_2, chi_dens_2, chi_magn_2, vrg_dens_range_2, vrg_magn_range_2 = \
    get_convergence_with_niv_shell(input_path, input_type, fname_1p,
                                                                                               fname_2p, niv_shell_range,
                                                                                               niv_core_2,
                                                                                               niw_core_2)

niv_core_3 = 50
niw_core_3 = 2
siw_sde_3, gamma_dens_3, gamma_magn_3, chi_dens_3, chi_magn_3, vrg_dens_range_3, vrg_magn_range_3 = \
    get_convergence_with_niv_shell(input_path, input_type, fname_1p,
                                                                                               fname_2p, niv_shell_range,
                                                                                               niv_core_3,
                                                                                               niw_core_3)
niv_core_4 = 60
niw_core_4 = 2
siw_sde_4, gamma_dens_4, gamma_magn_4, chi_dens_4, chi_magn_4, vrg_dens_range_4, vrg_magn_range_4 = \
    get_convergence_with_niv_shell(input_path, input_type, fname_1p,
                                                                                               fname_2p, niv_shell_range,
                                                                                               niv_core_4,
                                                                                               niw_core_4)

# %%
pdir = './TestPLots/'
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

line_colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, len(niv_shell_range)))
for i, siw_sde in enumerate(siw_sde_1):
    axes[0].loglog(mf.vn_from_mat(siw_sde), np.abs(siw_sde.real), color=line_colors[i], marker='o')
    axes[1].loglog(mf.vn_from_mat(siw_sde), np.abs(siw_sde.imag), color=line_colors[i], marker='o')

for i, siw_sde in enumerate(siw_sde_2):
    axes[0].loglog(mf.vn_from_mat(siw_sde), np.abs(siw_sde.real), color=line_colors[i], marker='x')
    axes[1].loglog(mf.vn_from_mat(siw_sde), np.abs(siw_sde.imag), color=line_colors[i], marker='x')

axes[0].loglog(mf.vn_from_mat(dmft_input['siw']), np.abs(dmft_input['siw'].real), 'k', marker='')
axes[1].loglog(mf.vn_from_mat(dmft_input['siw']), np.abs(dmft_input['siw'].imag), 'k', marker='')

for i, siw_sde in enumerate(siw_sde_1):
    niv_sde = np.size(siw_sde) // 2
    siw_dmft_clip = mf.cut_v(dmft_input['siw'], niv_cut=niv_sde)
    axes[2].loglog(mf.vn_from_mat(siw_sde), np.abs(siw_sde.real - siw_dmft_clip.real), color=line_colors[i], marker='o')
    axes[3].loglog(mf.vn_from_mat(siw_sde), np.abs(siw_sde.imag - siw_dmft_clip.imag), color=line_colors[i], marker='o')

for i, siw_sde in enumerate(siw_sde_2):
    niv_sde = np.size(siw_sde) // 2
    siw_dmft_clip = mf.cut_v(dmft_input['siw'], niv_cut=niv_sde)
    axes[2].loglog(mf.vn_from_mat(siw_sde), np.abs(siw_sde.real - siw_dmft_clip.real), color=line_colors[i], marker='x')
    axes[3].loglog(mf.vn_from_mat(siw_sde), np.abs(siw_sde.imag - siw_dmft_clip.imag), color=line_colors[i], marker='x')

# for ax in axes:
#     ax.set_xlim()

plt.tight_layout()
plt.savefig(fname=pdir + 'Siw_convergence_with_niv_shell.png')
plt.show()


# %%
def get_gamma_range_mf(gamma_range, niv_use=0, niw_use=0):
    res = []
    niw = np.shape(gamma_range[0])[0] // 2
    niv = np.shape(gamma_range[1])[-1] // 2

    for gam in gamma_range:
        res.append(gam[niw + niw_use, niv, niv + niv_use])
    return res


pdir = './TestPLots/'
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

line_colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, len(niv_shell_range)))
x = np.minimum(1, 1 / niv_shell_range)
axes[0].semilogx(x, get_gamma_range_mf(gamma_dens_1,niw_use=1), label='Dens: (0,1,1)')
axes[2].semilogx(x, get_gamma_range_mf(gamma_dens_1, niv_use=-1), label='Dens: (0,1,-1)')

axes[0].semilogx(x, get_gamma_range_mf(gamma_dens_2,niw_use=1), label='Dens2: (0,1,1)')
axes[2].semilogx(x, get_gamma_range_mf(gamma_dens_2, niv_use=-1), label='Dens2: (0,1,-1)')

axes[0].semilogx(x, get_gamma_range_mf(gamma_dens_3,niw_use=1), label='Dens3: (0,1,1)')
axes[2].semilogx(x, get_gamma_range_mf(gamma_dens_3, niv_use=-1), label='Dens3: (0,1,-1)')

axes[0].semilogx(x, get_gamma_range_mf(gamma_dens_4,niw_use=1), label='Dens4: (0,1,1)')
axes[2].semilogx(x, get_gamma_range_mf(gamma_dens_4, niv_use=-1), label='Dens4: (0,1,-1)')

axes[1].semilogx(x, get_gamma_range_mf(gamma_magn_1), label='Magn: (0,1,1)')
axes[3].semilogx(x, get_gamma_range_mf(gamma_magn_1, niv_use=-1), label='Magn: (0,1,-1)')

axes[1].semilogx(x, get_gamma_range_mf(gamma_magn_2), label='Magn2: (0,1,1)')
axes[3].semilogx(x, get_gamma_range_mf(gamma_magn_2, niv_use=-1), label='Magn2: (0,1,-1)')

axes[1].semilogx(x, get_gamma_range_mf(gamma_magn_3), label='Magn3: (0,1,1)')
axes[3].semilogx(x, get_gamma_range_mf(gamma_magn_3, niv_use=-1), label='Magn3: (0,1,-1)')

axes[1].semilogx(x, get_gamma_range_mf(gamma_magn_4), label='Magn4: (0,1,1)')
axes[3].semilogx(x, get_gamma_range_mf(gamma_magn_4, niv_use=-1), label='Magn4: (0,1,-1)')

# axes[2].semilogx(x,gamma_dens_eigval,label='Dens: eigval')
# axes[3].semilogx(x,gamma_magn_eigval,label='Magn: eigval')

for ax in axes:
    ax.legend()
    ax.set_xlabel('1/n')

plt.tight_layout()
plt.savefig(fname=pdir + f'Gamma_convergence_with_niv_shell_{input_type}.png')
plt.show()

# %%
fig, axes = plt.subplots(5, len(niv_shell_range))
niv_plot = 10
for i, ax in enumerate(axes[0]):
    niw = np.shape(gamma_dens_1[0])[0] // 2
    ax.imshow(mf.cut_v(gamma_dens_1[i][niw, :, :], niv_cut=niv_plot, axes=(0, 1)).real, cmap='RdBu')

for i, ax in enumerate(axes[1]):
    niw = np.shape(gamma_dens_2[0])[0] // 2
    ax.imshow(mf.cut_v(gamma_dens_2[i][niw, :, :], niv_cut=niv_plot, axes=(0, 1)).real, cmap='RdBu')

for i, ax in enumerate(axes[2]):
    niw = np.shape(gamma_dens_3[0])[0] // 2
    ax.imshow(mf.cut_v(gamma_dens_3[i][niw, :, :], niv_cut=niv_plot, axes=(0, 1)).real, cmap='RdBu')

for i, ax in enumerate(axes[3]):
    niw = np.shape(gamma_dens_4[0])[0] // 2
    ax.imshow(mf.cut_v(gamma_dens_4[i][niw, :, :], niv_cut=niv_plot, axes=(0, 1)).real, cmap='RdBu')

for i, ax in enumerate(axes[4]):
    niw = np.shape(gamma_dens_4[0])[0] // 2
    ax.imshow((mf.cut_v(gamma_dens_4[i][niw, :, :], niv_cut=niv_plot, axes=(0, 1)) -
               mf.cut_v(gamma_dens_1[i][niw, :, :], niv_cut=niv_plot, axes=(0, 1))).real, cmap='RdBu')

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(4, len(niv_shell_range))
niv_plot = 10
for i, ax in enumerate(axes[0]):
    niw = np.shape(gamma_magn_1[0])[0] // 2
    ax.imshow(mf.cut_v(gamma_magn_1[i][niw, :, :], niv_cut=niv_plot, axes=(0, 1)).real, cmap='RdBu')

for i, ax in enumerate(axes[1]):
    niw = np.shape(gamma_magn_2[0])[0] // 2
    ax.imshow(mf.cut_v(gamma_magn_2[i][niw, :, :], niv_cut=niv_plot, axes=(0, 1)).real, cmap='RdBu')

for i, ax in enumerate(axes[2]):
    niw = np.shape(gamma_magn_3[0])[0] // 2
    ax.imshow(mf.cut_v(gamma_magn_3[i][niw, :, :], niv_cut=niv_plot, axes=(0, 1)).real, cmap='RdBu')

for i, ax in enumerate(axes[3]):
    niw = np.shape(gamma_magn_3[0])[0] // 2
    ax.imshow((mf.cut_v(gamma_magn_3[i][niw, :, :], niv_cut=niv_plot, axes=(0, 1)) -
               mf.cut_v(gamma_dens_1[i][niw, :, :], niv_cut=niv_plot, axes=(0, 1))).real, cmap='RdBu')

plt.tight_layout()
plt.show()

#%%
fig, axes = plt.subplots(4, 2)

for chi in chi_dens_1:
    axes[0][0].plot(chi)

for chi in chi_dens_2:
    axes[1][0].plot(chi)

for chi in chi_dens_3:
    axes[2][0].plot(chi)

for chi in chi_dens_4:
    axes[3][0].plot(chi)

for chi in chi_magn_1:
    axes[0][1].plot(chi)

for chi in chi_magn_2:
    axes[1][1].plot(chi)

for chi in chi_magn_3:
    axes[2][1].plot(chi)

for chi in chi_magn_4:
    axes[3][1].plot(chi)

plt.tight_layout()
plt.show()

#%%
fig, axes = plt.subplots(2, 1)
x = np.minimum(1, 1 / niv_shell_range)
axes[0].semilogx(x,[chi_dens_1[i][niw_core_1] for i in range(len(x))],label='Chi_dens: 1')
axes[0].semilogx(x,[chi_dens_2[i][niw_core_2] for i in range(len(x))],label='Chi_dens: 2')
axes[0].semilogx(x,[chi_dens_3[i][niw_core_3] for i in range(len(x))],label='Chi_dens: 3')
axes[0].semilogx(x,[chi_dens_4[i][niw_core_4] for i in range(len(x))],label='Chi_dens: 4')

axes[1].semilogx(x,[chi_magn_1[i][niw_core_1] for i in range(len(x))],label='Chi_magn: 1')
axes[1].semilogx(x,[chi_magn_2[i][niw_core_2] for i in range(len(x))],label='Chi_magn: 2')
axes[1].semilogx(x,[chi_magn_3[i][niw_core_3] for i in range(len(x))],label='Chi_magn: 3')
axes[1].semilogx(x,[chi_magn_4[i][niw_core_4] for i in range(len(x))],label='Chi_magn: 4')

axes[0].legend()
axes[1].legend()
plt.show()

#%%
fig, axes = plt.subplots(2, 1)
x = np.minimum(1, 1 / niv_shell_range)
axes[0].semilogx(x,[chi_dens_1[i][-1] for i in range(len(x))],label='Chi_dens: 1')
axes[0].semilogx(x,[chi_dens_2[i][-1] for i in range(len(x))],label='Chi_dens: 2')
axes[0].semilogx(x,[chi_dens_3[i][-1] for i in range(len(x))],label='Chi_dens: 3')
axes[0].semilogx(x,[chi_dens_4[i][-1] for i in range(len(x))],label='Chi_dens: 4')

axes[1].semilogx(x,[chi_magn_1[i][-1] for i in range(len(x))],label='Chi_magn: 1')
axes[1].semilogx(x,[chi_magn_2[i][-1] for i in range(len(x))],label='Chi_magn: 2')
axes[1].semilogx(x,[chi_magn_3[i][-1] for i in range(len(x))],label='Chi_magn: 3')
axes[1].semilogx(x,[chi_magn_4[i][-1] for i in range(len(x))],label='Chi_magn: 4')

axes[0].legend()
axes[1].legend()
plt.show()

#%%
fig, axes = plt.subplots(2, 1)
x = np.minimum(1, 1 / niv_shell_range)
axes[0].semilogx(x,[vrg_dens_range_1[i][niw_core_1,niv_core_1] for i in range(len(x))],label='Vrg_dens: 1')
axes[0].semilogx(x,[vrg_dens_range_2[i][niw_core_2,niv_core_2] for i in range(len(x))],label='Vrg_dens: 2')
axes[0].semilogx(x,[vrg_dens_range_3[i][niw_core_3,niv_core_3] for i in range(len(x))],label='Vrg_dens: 3')
axes[0].semilogx(x,[vrg_dens_range_4[i][niw_core_4,niv_core_4] for i in range(len(x))],label='Vrg_dens: 4')

axes[1].semilogx(x,[vrg_magn_range_1[i][niw_core_1,niv_core_1] for i in range(len(x))],label='Vrg_magn: 1')
axes[1].semilogx(x,[vrg_magn_range_2[i][niw_core_2,niv_core_2] for i in range(len(x))],label='Vrg_magn: 2')
axes[1].semilogx(x,[vrg_magn_range_3[i][niw_core_3,niv_core_3] for i in range(len(x))],label='Vrg_magn: 3')
axes[1].semilogx(x,[vrg_magn_range_4[i][niw_core_4,niv_core_4] for i in range(len(x))],label='Vrg_magn: 4')

axes[0].legend()
axes[1].legend()
plt.show()

#%%
fig, axes = plt.subplots(2, 1)
x = np.minimum(1, 1 / niv_shell_range)
axes[0].semilogx(x,[vrg_dens_range_1[i][niw_core_1+1,niv_core_1] for i in range(len(x))],label='Vrg_dens: 1')
axes[0].semilogx(x,[vrg_dens_range_2[i][niw_core_2+1,niv_core_2] for i in range(len(x))],label='Vrg_dens: 2')
axes[0].semilogx(x,[vrg_dens_range_3[i][niw_core_3+1,niv_core_3] for i in range(len(x))],label='Vrg_dens: 3')
axes[0].semilogx(x,[vrg_dens_range_4[i][niw_core_4+1,niv_core_4] for i in range(len(x))],label='Vrg_dens: 4')

axes[1].semilogx(x,[vrg_magn_range_1[i][niw_core_1+1,niv_core_1] for i in range(len(x))],label='Vrg_magn: 1')
axes[1].semilogx(x,[vrg_magn_range_2[i][niw_core_2+1,niv_core_2] for i in range(len(x))],label='Vrg_magn: 2')
axes[1].semilogx(x,[vrg_magn_range_3[i][niw_core_3+1,niv_core_3] for i in range(len(x))],label='Vrg_magn: 3')
axes[1].semilogx(x,[vrg_magn_range_4[i][niw_core_4+1,niv_core_4] for i in range(len(x))],label='Vrg_magn: 4')

axes[0].legend()
axes[1].legend()
plt.show()
