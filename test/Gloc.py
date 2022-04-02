# Check the convergence of \sum_k G(k) = G(DMFT)

import numpy as np
import matplotlib.pyplot as plt
import Input as input
import Config as conf
import Hk as hamk
import Hr as hamr
import TwoPoint as twop
import BrillouinZone as bz
import MatsubaraFrequencies as mf

names = conf.Names()
box = conf.BoxSizes()
opt = conf.Options()
sys = conf.SystemParamter()
box.nk = (1,1,1)
box.niv_core = 60
sys.hr =hamr.one_band_2d_t_tp_tpp(t=1.0,tp=-0.2,tpp=0.1)
dga_conf = conf.DgaConfig(BoxSizes=box,Names=names,Options=opt,SystemParameter=sys,ek_funk=hamk.ek_3d)
names.input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.95/'

dmft1p = input.load_1p_data(dga_conf=dga_conf)
gk_mean = []
nk_grid = [32,64,80,100,120,200]
mu_kmean = []
ek_kmean = []
for nkx in nk_grid:
    print(nkx)
    nk = (nkx,nkx,1)
    k_grid =bz.KGrid(nk=nk)
    g_gen = twop.GreensFunctionGenerator(beta=dmft1p['beta'],kgrid=k_grid,hr=sys.hr,sigma=dmft1p['sloc'])
    #mu = g_gen.adjust_mu(n=0.95,mu0=dmft1p['mu'])
    #mu_kmean.append(mu)
    mu = dmft1p['mu']
    gk = g_gen.generate_gk(mu=mu,niv=box.niv_core)
    ek = g_gen.get_ekpq()
    ek_kmean.append(ek)
    gk_mean.append(gk.k_mean())

iv_core = mf.vn(box.niv_core)
g_loc_dmft = mf.cut_v_1d(dmft1p['gloc'],niv_cut=box.niv_core)



plt.figure()
for i, nkx in enumerate(nk_grid):
    plt.plot(iv_core,gk_mean[i].imag-g_loc_dmft.imag,'-o', label=f'Nk={nkx}')
plt.legend()
plt.xlim(-60,60)
plt.show()

plt.figure()
for i, nkx in enumerate(nk_grid):
    plt.plot(iv_core,gk_mean[i].real-g_loc_dmft.real,'-o', label=f'Nk={nkx}')
plt.legend()
plt.xlim(-60,60)
plt.show()



nk = (200,200,1)
hk_dmft, kmesh_dmft = hamk.read_Hk_w2k(fname='2dhubbard_tp_-0.2_tpp_0.1_Nk_200_Nkz001.hk')
#hk_dmft, kmesh_dmft = hamk.read_Hk_w2k(fname='../input/2dhubbard_tp_-0.2_tpp_0.1_Nk_200_Nkz001.hk')
hk_dmft = np.reshape(hk_dmft,nk)[:,:,0]
k_grid = bz.KGrid(nk=nk)
hk = hamk.ek_3d(kgrid=k_grid.grid,hr=sys.hr)[:,:,0]

plt.figure()
plt.imshow(hk_dmft.real-ek_kmean[-1][:,:,0].real, cmap='RdBu')
plt.colorbar()
plt.show()
