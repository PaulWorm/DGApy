
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI as mpi
from ruamel.yaml import YAML
import gc

import dga.config as config
import dga.matsubara_frequencies as mf
import dga.brillouin_zone as bz
import dga.loggers as loggers
import dga.dga_io as dga_io
import dga.four_point as fp
import dga.local_four_point as lfp
import dga.plotting as plotting
import dga.lambda_correction as lc
import dga.two_point as twop
import dga.bubble as bub
import dga.hk as hamk
import dga.hr as hamr
import dga.mpi_aux as mpi_aux
import dga.pairing_vertex as pv
import dga.eliashberg_equation as eq
import dga.util as util
import dga.high_level_routines as hlr


hr = hamr.one_band_2d_t_tp_tpp(1,-0.2,0.1)
nk = (16,16,1)
k_grid = bz.KGrid(nk=nk,symmetries=bz.two_dimensional_square_symmetries())
ek = hamk.ek_3d(k_grid.grid, hr)

input_path = './2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
fname_1p = '1p-data.hdf5'
fname_2p = 'g4iw_sym.hdf5'
niv_asympt = 1000

dmft_input = dga_io.load_1p_data('w2dyn', input_path, fname_1p, fname_2p)


siwk_dmft = twop.SelfEnergy(sigma=dmft_input['siw'][None, None, None, :], beta=dmft_input['beta'])
giwk_dmft = twop.GreensFunction(siwk_dmft, ek, mu=dmft_input['mu_dmft'], niv_asympt=niv_asympt)

plt.figure()
plt.imshow(giwk_dmft.g_full()[:,:,0,giwk_dmft.niv_full].imag,'RdBu')
plt.colorbar()
plt.show()
