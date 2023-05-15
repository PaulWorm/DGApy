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
import Hr as hamr
import Hk as hamk
import BrillouinZone as bz
import TwoPoint as twop
import Bubble as bub
import Plotting as plotting
import PlotSpecs

# Define MPI communicator:
comm = mpi.COMM_WORLD

# --------------------------------------------- CONFIGURATION ----------------------------------------------------------

# Momentum and frequency grids:
niw_core = 60
niv_core = 60
niv_shell = 500
lambda_correction_type = 'spch'
nk = (64,64,1)
nq = (16,16,1)
symmetries = bz.two_dimensional_square_symmetries()
hr = hamr.one_band_2d_t_tp_tpp(1.0,-0.2,0.1)

# Input and output directories:
input_type = 'EDFermion'
input_dir = '../test/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
output_dir = input_dir + 'LambdaDga_lc_{}_Nk{}_Nq{}_wcore{}_vcore{}_vshell{}'.format(lambda_correction_type,np.prod(nk),np.prod(nq),
                                                                            niw_core,niv_core,niv_shell)
output_dir = out.uniquify(output_dir)

# Create output directory:
comm.barrier()
if (comm.rank == 0): os.mkdir(output_dir)
comm.barrier()

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
if(comm.rank == 0): plotting.chi_checks(chi_dens,chi_magn,giwk_dmft,output_dir,verbose=False,do_plot=True)

siw_sde_full = lfp.schwinger_dyson_full(vrg_dens, vrg_magn, chi_dens, chi_magn, giwk_dmft.g_loc, dmft_input['u'], dmft_input['n'], niv_shell=niv_shell)

# Create checks of the self-energy:
if(comm.rank == 0): plotting.siw_sde_local_checks(siw_sde_full,dmft_input['siw'],dmft_input['beta'],output_dir,verbose=False,do_plot=True)