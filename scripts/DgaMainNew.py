# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# This code performs a DGA calculation starting from DMFT quantities as input.
# For the original paper look at: PHYSICAL REVIEW B 75, 045118 (2007)
# For a detailed review of the procedure read my thesis: "Numerical analysis of many-body effects in cuprate and nickelate superconductors"
# Asymptotics were adapted from Phys. Rev. B 106, 205101 (2022)
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import sys,os
import h5py
import numpy as np
import Output as out
from mpi4py import MPI as mpi
import numpy as np
import Input as input
import LocalFourPoint as lfp
import FourPoint as fp
import Hr as hamr
import Hk as hamk
import BrillouinZone as bz
import TwoPoint as twop
import Bubble as bub
import Plotting as plotting
import LambdaCorrection as lc
import matplotlib.pyplot as plt
import PlotSpecs


# Define MPI communicator:
comm = mpi.COMM_WORLD

# --------------------------------------------- CONFIGURATION ----------------------------------------------------------

# Momentum and frequency grids:
niw_core = 15
niv_core = 15
niv_shell = 2*niv_core
lambda_correction_type = 'spch'
nk = (16,16,1)
nq = (16,16,1)
symmetries = bz.two_dimensional_square_symmetries()
hr = hamr.one_band_2d_t_tp_tpp(1.0,-0.2,0.1)

# Input and output directories:
# input_type = 'EDFermion'#'w2dyn'
input_type = 'w2dyn'#'w2dyn'
input_dir = '../test/2DSquare_U8_tp-0.2_tpp0.1_beta2_n0.90/'
output_dir = input_dir + 'LambdaDga_lc_{}_Nk{}_Nq{}_wcore{}_vcore{}_vshell{}'.format(lambda_correction_type,np.prod(nk),np.prod(nq),
                                                                            niw_core,niv_core,niv_shell)
output_dir = out.uniquify(output_dir)

# Create output directory:
comm.barrier()
if (comm.rank == 0): os.mkdir(output_dir)
comm.barrier()

#%%
# ------------------------------------------- LOAD THE INPUT --------------------------------------------------------
dmft_input = input.load_1p_data(input_type,input_dir)

# cut the two-particle Green's functions:
g2_dens = lfp.LocalFourPoint(channel='dens',matrix=dmft_input['g4iw_dens'],beta=dmft_input['beta'])
g2_magn = lfp.LocalFourPoint(channel='magn',matrix=dmft_input['g4iw_magn'],beta=dmft_input['beta'])

# Cut frequency ranges:
g2_dens.cut_iv(niv_core)
g2_dens.cut_iw(niw_core)

g2_magn.cut_iv(niv_core)
g2_magn.cut_iw(niw_core)

if(comm.rank == 0):
    g2_dens.plot(0,pdir=output_dir,name='G2_dens')
    g2_magn.plot(0,pdir=output_dir,name='G2_magn')

    g2_magn.plot(10,pdir=output_dir,name='G2_magn')
    g2_magn.plot(-10,pdir=output_dir,name='G2_magn')

# Build Green's function and susceptibility:
k_grid = bz.KGrid(nk=nk,symmetries=symmetries)
ek = hamk.ek_3d(k_grid.grid,hr)

siwk_dmft = twop.SelfEnergy(sigma=dmft_input['siw'][None,None,None,:],beta=dmft_input['beta'])
giwk_dmft = twop.GreensFunction(siwk_dmft,ek,mu = dmft_input['mu_dmft'])


gchi_dens = lfp.gchir_from_g2(g2_dens,giwk_dmft.g_loc)
gchi_magn = lfp.gchir_from_g2(g2_magn,giwk_dmft.g_loc)

# Create Bubble generator:

bubble_gen = bub.LocalBubble(wn=g2_dens.wn,giw=giwk_dmft)

# --------------------------------------------- LOCAL PART ----------------------------------------------------------

# Perform the local SDE for box-size checks:

vrg_dens, chi_dens = lfp.get_vrg_and_chir_tilde_from_chir(gchi_dens, bubble_gen, dmft_input['u'], niv_core=niv_core, niv_shell=niv_shell)
vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_chir(gchi_magn, bubble_gen, dmft_input['u'], niv_core=niv_core, niv_shell=niv_shell)

# Create checks of the susceptibility:
if(comm.rank == 0): plotting.chi_checks(chi_dens,chi_magn,giwk_dmft,output_dir,verbose=False,do_plot=True,name='loc')

siw_sde_full = lfp.schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, giwk_dmft.g_loc, dmft_input['u'], dmft_input['n'], niv_shell=niv_shell)

# Create checks of the self-energy:
if(comm.rank == 0): plotting.siw_sde_local_checks(siw_sde_full,dmft_input['siw'],dmft_input['beta'],output_dir,verbose=False,do_plot=True)


# --------------------------------------------- NON-LOCAL PART --------------------------------------------------------
#%% Extract the local irreducible vertex Gamma:
gchi0_core = bubble_gen.get_gchi0(niv_core)
gamma_dens = lfp.gamob2_from_chir(gchi_dens,gchi0_core)
gamma_magn = lfp.gamob2_from_chir(gchi_magn,gchi0_core)

# gchi_magn_test = lfp.gchir_from_gamob2(gamma_magn,gchi0_core)
#
# if(comm.rank == 0):
#     gchi_magn_test.plot(pdir=output_dir,name='Gchi_test')


#%%
if(comm.rank == 0):
    gamma_dens.plot(0,pdir=output_dir,niv=min(niv_core,2*int(dmft_input['beta'])),name='Gamma_dens')
    gamma_magn.plot(0,pdir=output_dir,niv=min(niv_core,2*int(dmft_input['beta'])),name='Gamma_magn')
    gamma_magn.plot(10,pdir=output_dir,niv=min(niv_core,2*int(dmft_input['beta'])),name='Gamma_magn')
    gamma_magn.plot(-10,pdir=output_dir,niv=min(niv_core,2*int(dmft_input['beta'])),name='Gamma_magn')
    gamma_dens.plot(10,pdir=output_dir,niv=min(niv_core,2*int(dmft_input['beta'])),name='Gamma_dens')

#%% Build the non-local susceptibility:
q_grid = bz.KGrid(nk=nq,symmetries=symmetries)
my_q_list = q_grid.irrk_mesh_ind.T # Currently nk and nq have to be equal. For inequal grids the q-list would have to be adjusted.

gchi0_q_core = bubble_gen.get_gchi0_q_list(niv_core,my_q_list)
chi0_q_core = 1/dmft_input['beta']**2 * np.sum(gchi0_q_core,axis=-1)
chi0q_shell = bubble_gen.get_chi0q_shell(chi0_q_core,niv_core,niv_shell,my_q_list)

#%%
if(comm.rank==0):
    chi0_core = bubble_gen.get_chi0(niv_core)
    chi0_shell = bubble_gen.get_chi0_shell(niv_core,niv_shell)
    chi0q_shell_loc = np.mean(q_grid.map_fbz2irrk(chi0q_shell),axis=(0,1,2))
    chi0q_core_loc = np.mean(q_grid.map_fbz2irrk(chi0_q_core),axis=(0,1,2))
    plt.figure()
    plt.plot(chi0q_core_loc.real+chi0q_shell_loc.real,label='Full')
    plt.plot(chi0q_core_loc.real,label='Core')
    plt.plot(chi0q_shell_loc.real,label='Shell')
    plt.plot(chi0_core.real+chi0_shell.real,label='BM')
    plt.plot(chi0_core.real,label='BM-Core')
    plt.legend()
    plt.savefig(output_dir+'/TestChi0.png')
    plt.show()
#%%

gchi_lad_dens = fp.get_gchir_from_gamma_loc_q(gammar=gamma_dens,gchi0=gchi0_q_core)
gchi_lad_magn = fp.get_gchir_from_gamma_loc_q(gammar=gamma_magn,gchi0=gchi0_q_core)


chi_lad_dens = 1/dmft_input['beta']**2* np.sum(gchi_lad_dens,axis=(-1,-2))
chi_lad_magn = 1/dmft_input['beta']**2* np.sum(gchi_lad_magn,axis=(-1,-2))

#%%
if(comm.rank == 0):
    chi_lad_dens_loc = np.mean(q_grid.map_fbz2irrk(chi_lad_dens),axis=(0,1,2))
    chi_lad_magn_fbz = q_grid.map_fbz2irrk(chi_lad_magn)
    chi_lad_magn_loc = np.mean(chi_lad_magn_fbz,axis=(0,1,2))
    plotting.chi_checks(chi_lad_dens_loc, chi_lad_magn_loc, giwk_dmft, output_dir, verbose=False, do_plot=True,name='q_core')

    plt.figure()
    plt.imshow(chi_lad_magn_fbz[:, :, 0, niw_core].real, cmap='RdBu')
    plt.colorbar()
    plt.show()

    gchi_lad_dens_fbz = np.mean(q_grid.map_fbz2irrk(gchi_lad_dens),axis=(0,1,2))
    plt.figure()
    plt.imshow(gchi_lad_dens_fbz[niw_core,...].real, cmap='RdBu')
    plt.colorbar()
    plt.show()

#%%
# Lambda core:
lam_dens = fp.lam_from_chir_q(gchi_lad_dens,gchi0_q_core,'dens')
lam_magn = fp.lam_from_chir_q(gchi_lad_magn,gchi0_q_core,'magn')

# Lambda tilde:
lam_dens = fp.lam_tilde(lam_dens,chi0q_shell,dmft_input['u'],'dens')
lam_magn = fp.lam_tilde(lam_magn,chi0q_shell,dmft_input['u'],'magn')

# chi_tilde:
chi_lad_dens = fp.chir_tilde(chi_lad_dens,lam_dens,chi0q_shell,gchi0_q_core,dmft_input['beta'],dmft_input['u'],'dens')
chi_lad_magn = fp.chir_tilde(chi_lad_magn,lam_magn,chi0q_shell,gchi0_q_core,dmft_input['beta'],dmft_input['u'],'magn')

# vrg_tilde:
vrg_q_dens = fp.vrg_q_tilde(lam_dens,chi_lad_dens,dmft_input['u'],'dens')
vrg_q_magn = fp.vrg_q_tilde(lam_magn,chi_lad_magn,dmft_input['u'],'magn')


chi_lad_dens = q_grid.map_fbz2irrk(chi_lad_dens)
chi_lad_magn = q_grid.map_fbz2irrk(chi_lad_magn)

#%%

if(comm.rank == 0):
    chi_lad_dens_loc = np.mean(chi_lad_dens,axis=(0,1,2))
    chi_lad_magn_loc = np.mean(chi_lad_magn,axis=(0,1,2))
    plotting.chi_checks(chi_lad_dens_loc, chi_lad_magn_loc, giwk_dmft, output_dir, verbose=False, do_plot=True,name='q_tilde')

    plt.figure()
    plt.imshow(chi_lad_magn[:, :, 0, niw_core].real, cmap='RdBu')
    plt.colorbar()
    plt.show()

#%% Lambda-corrected susceptibility:
lambda_dens = -np.min(1. / (chi_lad_dens[...,niw_core].real))
lambda_dens = lc.lambda_correction_single(dmft_input['beta'],lambda_start=lambda_dens, chir=chi_lad_dens,
                             chi_loc_sum=1/dmft_input['beta']*np.sum(chi_dens))
chi_lad_dens = 1/(1/chi_lad_dens + lambda_dens)

lambda_magn = -np.min(1. / (chi_lad_magn[...,niw_core].real))
lambda_dens = lc.lambda_correction_single(dmft_input['beta'],lambda_start=lambda_magn, chir=chi_lad_magn,
                             chi_loc_sum=1/dmft_input['beta']*np.sum(chi_magn))
chi_lad_magn = 1/(1/chi_lad_magn + lambda_magn)

#%%
if(comm.rank == 0):
    chi_lad_dens_loc = np.mean(chi_lad_dens,axis=(0,1,2))
    chi_lad_magn_loc = np.mean(chi_lad_magn,axis=(0,1,2))
    plotting.chi_checks(chi_lad_dens_loc, chi_lad_magn_loc, giwk_dmft, output_dir, verbose=False, do_plot=True,name='q_lam')

    plt.figure()
    plt.imshow(chi_lad_magn[:, :, 0, niw_core].real, cmap='RdBu')
    plt.colorbar()
    plt.show()