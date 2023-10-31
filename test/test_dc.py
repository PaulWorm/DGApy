import numpy as np
import matplotlib.pyplot as plt

import dga.matsubara_frequencies as mf
import dga.brillouin_zone as bz
import dga.bubble as bub
import dga.hk as hamk
import dga.two_point as twop
import TestData as td
import dga.local_four_point as lfp
import dga.four_point as fp

ddict = td.get_data_set_6(True, True)

siw = ddict['siw']
beta = ddict['beta']
u = ddict['u']
n = ddict['n']
g2_dens = ddict['g2_dens']
g2_magn = ddict['g2_magn']

chi_dens_dmft = ddict['chi_dens']
chi_magn_dmft = ddict['chi_magn']

niv_core = 30
niw_core = 30
niv_shell = 0
niv_full = niv_core + niv_shell

g2_dens.cut_iv(niv_core)
g2_magn.cut_iv(niv_core)

g2_dens.is_full_w = True
g2_magn.is_full_w = True

g2_dens.cut_iw(niw_core)
g2_magn.cut_iw(niw_core)

nk = (16, 16, 1)
k_grid = bz.KGrid(nk=nk, symmetries=bz.two_dimensional_square_symmetries())
ek = hamk.ek_3d(k_grid.grid, hr=ddict['hr'])

sigma = twop.SelfEnergy(siw[None, None, None, :], beta, pos=False)
giwk = twop.GreensFunction(sigma, ek, n=n)
giwk.set_g_asympt(niv_asympt=np.max(niv_core) + 1000)
g_loc = giwk.k_mean(iv_range='full')

gchi_dens = lfp.gchir_from_g2(g2_dens, g_loc)
gchi_magn = lfp.gchir_from_g2(g2_magn, g_loc)

chi0_gen = bub.BubbleGenerator(gchi_magn.wn, giwk)
gchi0_shell = chi0_gen.get_gchi0(niv_full)
gchi0_core = chi0_gen.get_gchi0(niv_core)

gamma_magn = lfp.gammar_from_gchir(gchi_magn, gchi0_shell, u)

F_dc = lfp.Fob2_from_gamob2_urange(gamma_magn, gchi0_shell, u)

vrg_magn, chi_magn = lfp.get_vrg_and_chir_tilde_from_gammar_uasympt(gamma_magn, chi0_gen, u,
                                                                    niv_shell=niv_shell)

# Build the different non-local Bubbles:
my_q_list = k_grid.irrk_mesh_ind.T
gchi0_q_urange = chi0_gen.get_gchi0_q_list(niv_full, my_q_list)
chi0q_shell_dc = chi0_gen.get_asymptotic_correction_q(niv_full, my_q_list)
kernel_dc = mf.cut_v(fp.get_kernel_dc(F_dc.mat, gchi0_q_urange, u, 'magn'), niv_core, axes=(-1,))
u_r = fp.get_ur(u, gamma_magn.channel)
# kernel_dc += u_r / gamma_magn.beta * (1 - u_r * chi_magn[None, :, None]) * vrg_magn.mat[None, :, :] * chi0q_shell_dc[
#                                                                                                       :, :, None]
kernel_dc = k_grid.map_irrk2fbz(kernel_dc,'list')
full_q_list = np.array([k_grid.kmesh_ind[i].flatten() for i in range(3)]).T
siwk_sde = fp.schwinger_dyson_kernel_q(kernel_dc,giwk.g_full(),beta,full_q_list,gamma_magn.wn,k_grid.nk_tot)



#%%
siw_magn = lfp.schwinger_dyson_core_urange(vrg_magn, chi_magn, giwk.g_loc, u, niv_shell)
F_magn = lfp.Fob2_from_chir(gchi_magn,gchi0_core)
siw_magn_f = lfp.schwinger_dyson_F(F_magn,gchi0_core,giwk.g_loc,u)
siwk_sde_loc = k_grid.k_mean(siwk_sde,'fbz-mesh')

vn = mf.vn_from_mat(siwk_sde_loc)
vn_loc = mf.vn_from_mat(siw_magn)
vn_f = mf.vn_from_mat(siw_magn_f)

plt.figure()
plt.plot(vn,siwk_sde_loc,'-o',color='cornflowerblue')
plt.plot(vn_loc,2*siw_magn,'-h',color='firebrick')
plt.plot(vn_f,-siw_magn_f,'-h',color='forestgreen')
plt.xlim(0,10)
plt.show()

plt.figure()
plt.plot(vn,np.abs(siwk_sde_loc-2*mf.cut_v(siw_magn,niv_core)),'-o',color='cornflowerblue')
plt.xlim(0,10)
plt.show()

