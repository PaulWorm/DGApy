import sys,os
sys.path.append(os.environ['HOME'] + "/Programs/dga/src")
import numpy as np
import matplotlib.pyplot as plt
import Hr as hamr
import Hk as hamk
import BrillouinZone as bz
import w2dyn_aux
import TwoPoint as tp
import MatsubaraFrequencies as mf
import FourPoint as fp
import Indizes as ind
import MpiAux
import mpi4py.MPI as mpi
import matplotlib
import socket
if(socket.gethostname() != 'LAPTOP-SB9HJ48I'):
    matplotlib.use('agg') # non GUI backend since VSC has no display

comm = mpi.COMM_WORLD

input_path = './'
input_path = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.925/LambdaDga_lc_sp_Nk10000_Nq10000_core80_invbse80_vurange250_wurange80/'

niv = 100
niv_core = 40
niw_core = 20
nkx = 32
nky = nkx
nk = (nkx,nky,1)
kgrid = bz.KGrid(nk=nk)
hr = hamr.one_band_2d_t_tp_tpp(t=1.0,tp=-0.25,tpp=0.12)
hk = hamk.ek_3d(kgrid=kgrid.grid,hr=hr)
kgrid.get_irrk_from_ek(ek=hk)
dmft1p = np.load(input_path+'dmft1p.npy', allow_pickle=True).item()


vn = mf.vn(niv)
delta = 0.2
sigma = -np.ones(np.shape(vn))*delta*1j * np.sign(vn)
gk_gen = tp.GreensFunctionGenerator(beta=dmft1p['beta'],kgrid=kgrid,hr=hr,sigma=sigma)
mu = gk_gen.adjust_mu(n=dmft1p['n'],mu0=-0.7254370956737395)
n_test = gk_gen.get_fill(mu=mu)
gk = gk_gen.generate_gk(mu=mu)

gk_shift = bz.shift_mat_by_pi(gk.gk,nk=nk)
plt.imshow(gk_shift.imag[:,:,0,niv], cmap='terrain')
plt.colorbar()
plt.show()

def get_qiw(iqw=None, qiw_mesh=None):
    wn = qiw_mesh[iqw][-1]
    q_ind = qiw_mesh[iqw][0]
    q = kgrid.irr_kmesh[:, q_ind]
    qiw = np.append(-q, wn)  # WARNING: Here I am not sure if it should be +q or -q.
    return qiw, wn


wn_core_plus = mf.wn_plus(niw_core)
index_grid_keys = ('irrq', 'iw')
qiw_distributor = MpiAux.MpiDistributor(ntasks=(niw_core+1)*kgrid.nk_irr,comm=comm)
qiw_grid = ind.IndexGrids(grid_arrays=(kgrid.irrk_ind_lin,) + (wn_core_plus,),
                               keys=index_grid_keys,
                               my_slice=qiw_distributor.my_slice)

chi0 = fp.LadderSusceptibility(channel='0', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw_grid.my_mesh)

for iqw in range(qiw_grid.my_n_tasks):
    qiw, wn = get_qiw(iqw=iqw, qiw_mesh=qiw_grid.my_mesh)

    gkpq_urange = gk_gen.generate_gk(mu=mu, qiw=qiw)

    chi0q = fp.Bubble(gk=gk.get_gk_cut_iv(niv_cut=niv_core),gkpq=gkpq_urange.get_gk_cut_iv(niv_cut=niv_core), beta=dmft1p['beta'], wn=wn)

    chi0.mat[iqw] = chi0q.chi0

chi0.mat_to_array()
chi0 = qiw_distributor.allgather(chi0.mat)

if(comm.rank == 0):
    chi0 = kgrid.irrk2fbz(mat=qiw_grid.reshape_matrix(chi0))
    chi0 = mf.wplus2wfull(mat=chi0)
    np.save(input_path+ f'chi0_tight_binding_d_{delta}_niv_{niv}_niw_{niw_core}.npy', chi0, allow_pickle=True)
#%%

plt.figure()
plt.imshow(chi0[:,:,0,niw_core].real,cmap='terrain')
plt.colorbar()
plt.show()

plt.figure()
plt.plot(chi0[nkx//2,nkx//2,0,:].real)
plt.show()