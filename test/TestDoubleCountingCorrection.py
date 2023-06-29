import numpy as np
import matplotlib.pyplot as plt
import TwoPoint as twop
import FourPoint as fp
import LocalFourPoint as lfp
import Bubble as bub
import BrillouinZone as bz
import MatsubaraFrequencies as mf
import Hr as hamr
import Hk as hamk
import Input as input


# Momentum and frequency grids:
niw_core = 40
niv_core = 40
niv_shell = 10
niv_full = niv_core + niv_shell
nk = (16, 16, 1)
nq = nk  # Currently nk and nq have to be equal. For inequal grids the q-list would have to be adjusted.
symmetries = bz.two_dimensional_square_symmetries()
q_grid = bz.KGrid(nk=nq, symmetries=symmetries)
hr = hamr.one_band_2d_t_tp_tpp(1.0, -0.2, 0.1)
input_type = 'w2dyn'  # 'w2dyn'
input_dir = '../test/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'

# ------------------------------------------- LOAD THE INPUT --------------------------------------------------------
dmft_input = input.load_1p_data(input_type, input_dir)

# cut the two-particle Green's functions:
g2_dens = lfp.LocalFourPoint(channel='dens', matrix=dmft_input['g4iw_dens'], beta=dmft_input['beta'])
g2_magn = lfp.LocalFourPoint(channel='magn', matrix=dmft_input['g4iw_magn'], beta=dmft_input['beta'])

# Cut frequency ranges:
g2_dens.cut_iv(niv_core)
g2_dens.cut_iw(niw_core)

g2_magn.cut_iv(niv_core)
g2_magn.cut_iw(niw_core)

k_grid = bz.KGrid(nk=nk, symmetries=symmetries)
ek = hamk.ek_3d(k_grid.grid, hr)

siwk_dmft = twop.SelfEnergy(sigma=dmft_input['siw'][None, None, None, :], beta=dmft_input['beta'])
giwk_dmft = twop.GreensFunction(siwk_dmft, ek, mu=dmft_input['mu_dmft'], niv_asympt=niv_full * 2 + niw_core * 2)

gchi_dens = lfp.gchir_from_g2(g2_dens, giwk_dmft.g_loc)
gchi_magn = lfp.gchir_from_g2(g2_magn, giwk_dmft.g_loc)

# Create Bubble generator:
bubble_gen = bub.BubbleGenerator(wn=g2_dens.wn, giw=giwk_dmft)
gchi0_core = bubble_gen.get_gchi0(niv_core)
gchi0_urange = bubble_gen.get_gchi0(niv_full)
gamma_dens = lfp.gammar_from_gchir(gchi_dens, gchi0_urange, dmft_input['u'])
gamma_magn = lfp.gammar_from_gchir(gchi_magn, gchi0_urange, dmft_input['u'])

u = dmft_input['u']
beta = dmft_input['beta']

F_magn = lfp.Fob2_from_chir(gchi_magn,gchi0_core)
F_magn.plot(do_save=False,name='F_magn',verbose=True, niv = -1)

q_list_irr = q_grid.irrk_mesh_ind.T
q_list = np.array([q_grid.kmesh_ind[i].flatten() for i in range(3)]).T

F_magn_2 = lfp.Fob2_from_gamob2_urange(gamma_magn, gchi0_urange, u)
F_magn_2.plot(do_save=False,name='F_dens',verbose=True, niv = niv_core)
vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, bubble_gen, dmft_input['u'],
                                                                      niv_shell=niv_shell)
#%%
F_dens = lfp.Fob2_from_gamob2_urange(gamma_dens, gchi0_urange, u)
# F_dc = -0.5 * (F_dens.mat-F_magn.mat)
# F_dc = F_magn.mat
F_dc = F_magn.mat
F_dc_2 = -0.5 * (F_dens.mat-F_magn_2.mat)
gchi0_q_urange = bubble_gen.get_gchi0_q_list(niv_full, q_list_irr)
gchi0_q_urange = q_grid.map_irrk2fbz(gchi0_q_urange,shape='list')
gchi0_q_core = mf.cut_v(gchi0_q_urange,niv_core,axes=-1)
kernel_dc = fp.get_kernel_dc(F_dc, gchi0_q_core, u, 'magn')
#%%
chi0q_shell = bubble_gen.get_asymptotic_correction_q(niv_core, q_list)
kernel_dc += u/dmft_input['beta'] * (1 + u * chi_magn[None,:,None])*vrg_magn.mat[None,:,:]*chi0q_shell[:,:,None]
kernel_dc_2 = fp.get_kernel_dc(F_dc_2, gchi0_q_urange, u, 'magn')

siwk_dc = fp.schwinger_dyson_kernel_q(kernel_dc,giwk_dmft.g_full(),beta,q_list,gamma_magn.wn,np.prod(q_grid.nk))
siwk_dc_2 = fp.schwinger_dyson_kernel_q(kernel_dc_2,giwk_dmft.g_full(),beta,q_list,gamma_magn.wn,np.prod(q_grid.nk))


siwk_dc_loc = np.mean(siwk_dc,axis=(0,1,2))
siwk_dc_loc_2 = np.mean(siwk_dc_2,axis=(0,1,2))
siwk_dmft_loc = np.mean(siwk_dmft.get_siw(niv_full), axis=(0,1,2))

#%%
plt.figure()
plt.plot(mf.vn_from_mat(siwk_dc_loc),siwk_dc_loc.imag,'-o',color='cornflowerblue', markeredgecolor='k',ms=4)
plt.plot(mf.vn_from_mat(siwk_dc_loc_2),siwk_dc_loc_2.imag,'-o',color='seagreen', markeredgecolor='k',ms=4)
plt.plot(mf.vn_from_mat(siwk_dmft_loc),siwk_dmft_loc.imag,'-h',color='firebrick', markeredgecolor='k',ms=1)
plt.xlim(0,niv_full)
plt.ylim(None,0)
plt.show()
















