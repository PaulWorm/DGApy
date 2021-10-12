# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Dga conda distribution has to be loaded, otherwise the mpirun does not work properly!

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import copy
from mpi4py import MPI as mpi
import MpiAux as mpiaux
import importlib
import time
import numpy as np
import w2dyn_aux
import Indizes as ind
import FourPoint as fp
import TwoPoint as twop
import LambdaCorrection as lc
import SDE as sde
import matplotlib.pyplot as plt
import Plotting as plot

importlib.reload(twop)
importlib.reload(fp)
importlib.reload(lc)
importlib.reload(plot)
importlib.reload(sde)


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

class real_time():
    ''' simple class to keep track of real time '''

    def __init__(self):
        self._ts = time.time()
        self._tm = []
        self._tm.append(0)

    def measure_time(self):
        self._tm.append(time.time() - self._ts - self._tm[-1])

    def print_time(self, string=''):
        self.measure_time()
        print(string + 'took {} seconds'.format(self._tm[-1]))


# ----------------------------------------------- PARAMETERS -----------------------------------------------------------
path = './'
dataset = ''
# path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta80_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta80_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
# path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/Testset1/LambdaDga_Python/'

fname_g2 = 'g4iw_sym.hdf5'
fname_dmft = '1p-data.hdf5'

do_pairing_vertex = True
t0 = time.time()
Nk = 16
Nq = 8
t_x = 1. / 4.
t_y = t_x
t_z = 0
t_xy = -0.25 * t_x
t_yx = t_xy
t2_x = 0.12 * t_x
t2_y = t2_x
t2_z = 0
nivp = 5
t_mat = np.array([[t_x, t_y, t_z], [t_xy, t_yx, 0.], [t2_x, t2_y, t2_z]])

niw_core = 19
niv_core = 20
niv_urange = 100
niv_asympt = 5000
iw_core = np.arange(-niw_core, niw_core + 1)
iv_urange = np.arange(-niv_urange, niv_urange)

Nkx = Nk
Nky = Nk
Nkz = 1
Nqx = Nq
Nqy = Nq
Nqz = 1
Nqtot = Nqx * Nqy * Nqz

box_sizes = {
    "niw_core": niw_core,
    "niv_core": niv_core,
    "niv_urange": niv_urange,
    "niv_asympt": niv_asympt,
    "Nkx": Nkx,
    "Nky": Nky,
    "Nkz": Nkz,
    "Nqx": Nqx,
    "Nqy": Nqy,
    "Nqz": Nqz
}

kx = np.arange(0, Nkx) * 2 * np.pi / Nkx
ky = np.arange(0, Nky) * 2 * np.pi / Nky
kz = np.arange(0, Nkz) * 2 * np.pi / Nkz
kgrid = [kx, ky, kz]

qx = np.arange(0, Nqx) * 2 * np.pi / Nqx
qy = np.arange(0, Nqy) * 2 * np.pi / Nqy
qz = np.arange(0, Nqz) * 2 * np.pi / Nqz
qgrid = [qx, qy, qz]

f1p = w2dyn_aux.w2dyn_file(fname=path + fname_dmft)
dmft1p = f1p.load_dmft1p_w2dyn()
f1p.close()

giw = dmft1p['gloc']

# ----------------------------------------------- MPI SETUP ------------------------------------------------------------

comm = mpi.COMM_WORLD
iw_distributor = mpiaux.MpiDistributor(ntasks=iw_core.size, comm=comm)
my_iw = iw_core[iw_distributor.my_slice]
print(f'My rank is {iw_distributor.my_rank} and I am doing: {my_iw=}')

realt = real_time()

# -------------------------------------------LOAD G2 FROM W2DYN --------------------------------------------------------


g2_file = w2dyn_aux.g4iw_file(fname=path + fname_g2)
niw_dmft_full = g2_file.get_niw(channel='dens')

g2_dens_loc = fp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='dens', iw=my_iw), giw=giw, channel='dens',
                                beta=dmft1p['beta'], iw=my_iw)
g2_magn_loc = fp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='magn', iw=my_iw), giw=giw, channel='magn',
                                beta=dmft1p['beta'], iw=my_iw)

g2_dens_loc.cut_iv(niv_cut=box_sizes['niv_core'])
g2_magn_loc.cut_iv(niv_cut=box_sizes['niv_core'])

dmft1p['g2_dens'] = g2_dens_loc
dmft1p['g2_magn'] = g2_magn_loc

# --------------------------------------------- LOCAL DMFT SDE ---------------------------------------------------------

dmft_sde = sde.local_dmft_sde_from_g2(dmft_input=dmft1p, box_sizes=box_sizes)

chi_dens_loc_mat = iw_distributor.allgather(rank_result=dmft_sde['chi_dens'].mat)
chi_magn_loc_mat = iw_distributor.allgather(rank_result=dmft_sde['chi_magn'].mat)

chi_dens_loc = fp.LocalSusceptibility(matrix=chi_dens_loc_mat,giw=dmft_sde['chi_dens'].giw,channel=dmft_sde['chi_dens'].channel,
                                      beta = dmft_sde['chi_dens'].beta,iw=iw_core)

chi_magn_loc = fp.LocalSusceptibility(matrix=chi_magn_loc_mat,giw=dmft_sde['chi_magn'].giw,channel=dmft_sde['chi_magn'].channel,
                                      beta = dmft_sde['chi_magn'].beta,iw=iw_core)

siw_sde_dens = dmft_sde['siw_dens']
siw_sde_dens = dmft_sde['siw_magn']
siw_sde = dmft_sde['siw']
siw_sde_reduce = np.zeros(np.shape(siw_sde), dtype=complex)
comm.Allreduce(siw_sde, siw_sde_reduce)
siw_sde_reduce = siw_sde_reduce + dmft_sde['hartree']

niv_sde = siw_sde_dens.size // 2
v_sde = np.arange(-niv_sde, niv_sde)
niv_dmft = dmft1p["sloc"].size // 2
v_dmft = np.arange(-niv_dmft, niv_dmft)

plt.subplot(211)
plt.plot(v_dmft, dmft1p["sloc"].real)
plt.plot(v_sde, siw_sde_reduce.real, 'o', ms=2,
         label='asympt-range')
plt.xlim([0, dmft1p['beta']])
plt.xlabel('w')
plt.ylabel('Sigma-real')
plt.subplot(212)
plt.plot(v_dmft, dmft1p["sloc"].imag)
plt.plot(v_sde, siw_sde_reduce.imag, 'o', ms=2, label='asympt-range')
plt.xlim([0, dmft1p['beta']])
plt.legend()
plt.xlabel('w')
plt.ylabel('Sigma-imag')
plt.savefig(path + 'dga_local_sde_check_{}.png'.format(iw_distributor.my_rank))
plt.show()

realt.print_time('Local Part ')

# ------------------------------------------------ NON-LOCAL PART  -----------------------------------------------------
# ======================================================================================================================


qiw_distributor = mpiaux.MpiDistributor(ntasks=iw_core.size * Nqtot, comm=comm)

qiw = ind.qiw(qgrid=qgrid, iw=iw_core, my_slice=qiw_distributor.my_slice)

# ----------------------------------------- NON-LOCAL LADDER SUCEPTIBILITY  --------------------------------------------

dga_susc = fp.dga_susceptibility(dmft_input=dmft1p, local_sde=dmft_sde, hr=t_mat, kgrid=kgrid, box_sizes=box_sizes,
                                 qiw=qiw)
realt.print_time('Non-local Susceptibility: ')

chi_dens_ladder_mat = qiw_distributor.allgather(rank_result = dga_susc['chi_dens_asympt'].mat)
chi_magn_ladder_mat = qiw_distributor.allgather(rank_result = dga_susc['chi_magn_asympt'].mat)

chi_dens_ladder = fp.LadderSusceptibility(qiw=qiw.qiw,channel='dens',u=dmft1p['u'], beta=dmft1p['beta'])
chi_dens_ladder.mat = chi_dens_ladder_mat

chi_magn_ladder = fp.LadderSusceptibility(qiw=qiw.qiw,channel='magn',u=dmft1p['u'], beta=dmft1p['beta'])
chi_magn_ladder.mat = chi_magn_ladder_mat

lambda_dens = lc.lambda_correction(lambda_start=lc.get_lambda_start(chi_magn_ladder), chir=chi_magn_ladder,
                                   chi_loc=chi_dens_loc, qiw=qiw)
lambda_magn = lc.lambda_correction(lambda_start=lc.get_lambda_start(chi_magn_ladder), chir=chi_magn_ladder,
                                    chi_loc=chi_magn_loc, qiw=qiw)

chi_dens_lambda = fp.LadderSusceptibility(qiw=qiw.qiw,channel='dens',u=dmft1p['u'], beta=dmft1p['beta'])
chi_dens_lambda.mat = 1./(1./chi_dens_ladder_mat + lambda_dens)

chi_magn_lambda = fp.LadderSusceptibility(qiw=qiw.qiw,channel='magn',u=dmft1p['u'], beta=dmft1p['beta'])
chi_magn_lambda.mat = 1./(1./chi_magn_ladder_mat + lambda_magn)

chi_magn_lambda_qmean = qiw.q_mean_full(chi_magn_lambda.mat)
chi_magn_ladder_qmean = qiw.q_mean_full(chi_magn_ladder.mat)

plt.plot(iw_core, chi_magn_loc.mat.real, label='dmft')
plt.plot(iw_core,chi_magn_lambda_qmean.real, 'o',ms=6, label=r'$\lambda: q-mean$')
plt.plot(iw_core,chi_magn_ladder_qmean.real, 's',ms=3, label=r'$ladder: q-mean$')
plt.legend()
plt.title(r'$\chi_{magn}$')
plt.xlim([-2,20])
plt.show()

# --------------------------------------------- LAMBDA CORRECTIONS -----------------------------------------------------

#
#
# lambda_dens = lc.lambda_correction(lambda_start=lc.get_lambda_start(chi_dens_asympt), chir=chi_dens_asympt,
#                                    chi_loc=chi_dens_asympt_loc)
# lambda_magn = lc.lambda_correction(lambda_start=lc.get_lambda_start(chi_magn_asympt), chir=chi_magn_asympt,
#                                    chi_loc=chi_magn_asympt_loc)
#
# for iqw in range(qiw.my_size):
#     chi_dens_asympt_lambda.mat[qiw.my_indizes[iqw]] = 1. / (1. / chi_dens_asympt.mat[qiw.my_indizes[iqw]] + lambda_dens)
#     chi_magn_asympt_lambda.mat[qiw.my_indizes[iqw]] = 1. / (1. / chi_magn_asympt.mat[qiw.my_indizes[iqw]] + lambda_magn)
#
# chi_dens_asympt_lambda.mat_to_array()
# chi_magn_asympt_lambda.mat_to_array()
#
# sigma_dens = sde.sde_dga(vrg=vrg_dens, chir=chi_dens_asympt_lambda, g_generator=g_generator, mu=dmft1p['mu'])
# sigma_magn = sde.sde_dga(vrg=vrg_magn, chir=chi_magn_asympt_lambda, g_generator=g_generator, mu=dmft1p['mu'])
#
# realt.print_time('Non-local SDE ')
#
# sigma_dens_ksum = sigma_dens.mean(axis=(0,1,2))
# sigma_magn_ksum = sigma_magn.mean(axis=(0,1,2))
#
# s_dmft = dmft1p['sloc'][niv_dmft-niv_urange:niv_dmft+niv_urange]
# sigma_dga =  2.*(dmft1p['u'] / 2. * dmft1p['n']) + sigma_dens +  3. * sigma_magn - siw_sde  #- (-s_dmft + siw_sde)
# sigma_dga_loc = np.mean(sigma_dga, axis=(0, 1, 2))
#
# niv_dga = sigma_dga_loc.size // 2
# v_dga = np.arange(-niv_dga, niv_dga)
#
# plt.subplot(211)
# plt.plot(v_dmft, dmft1p["sloc"].real)
# plt.plot(v_sde, siw_sde.real, 'o', ms=2, label='asympt-range')
# plt.plot(v_dga, sigma_dga_loc.real, 'o', ms=2, label='dga-loc')
# plt.xlim([0, niv_urange])
# plt.xlabel('w')
# plt.ylabel('Sigma-real')
# plt.subplot(212)
# plt.plot(v_dmft, dmft1p["sloc"].imag)
# plt.plot(v_sde, siw_sde.imag, 'o', ms=2, label='asympt-range')
# plt.plot(v_dga, sigma_dga_loc.imag, 'o', ms=2, label='dga-loc')
# plt.xlim([0, niv_urange])
# plt.legend()
# plt.xlabel('w')
# plt.ylabel('Sigma-imag')
# plt.savefig(path + 'dga_ksum_sde_check.png')
# plt.show()
#
#
# plt.subplot(211)
# plt.plot(v_sde, siw_sde_magn.real, 'o', ms=2, label='asympt-range')
# plt.plot(v_dga, sigma_magn_ksum.real, 'o', ms=2, label='dga-loc')
# plt.xlim([0, dmft1p['beta']])
# plt.xlabel('w')
# plt.ylabel('Sigma-real')
# plt.subplot(212)
# plt.plot(v_sde, sigma_magn_ksum.imag, 'o', ms=2, label='asympt-range')
# plt.plot(v_dga, sigma_dga_loc.imag, 'o', ms=2, label='dga-loc')
# plt.xlim([0, dmft1p['beta']])
# plt.legend()
# plt.xlabel('w')
# plt.ylabel('Sigma-imag')
# plt.show()
#
# plt.plot(iw_core, chi_magn_asympt_loc.mat.real, 'o', label='loc-asympt')
# plt.plot(iw_core, chi_magn_asympt_lambda.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1).mean(axis=(0, 1, 2)).real, 'o',
#          label='lambda-corrected')
# plt.plot(iw_core, chi_magn_asympt.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1).mean(axis=(0, 1, 2)).real, 'o',
#          label='non-local')
# plt.legend()
# plt.xlim(-2)
# plt.xlabel(r'$\omega$')
# plt.ylabel(r'$\chi_{magn}$')
# plt.show()
#
# plt.plot(iw_core, chi_dens_asympt_loc.mat.real, 'o', label='loc-asympt')
# plt.plot(iw_core, chi_dens_asympt_lambda.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1).mean(axis=(0, 1, 2)).real, 'o',
#          label='lambda-corrected')
# plt.legend()
# plt.xlim(-2)
# plt.xlabel(r'$\omega$')
# plt.ylabel(r'$\chi_{dens}$')
# plt.show()
#
# plt.imshow(vrg_magn.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1, 2 * niv_urange).mean(axis=(0, 1, 2)).real, cmap='RdBu')
# plt.colorbar()
# plt.show()
#
# plt.imshow(vrg_magn.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1, 2 * niv_urange)[:,:,0,niw_core,niv_urange].real, cmap='RdBu')
# plt.colorbar()
# plt.show()
#
# plot.plot_tp(twop=vrg_magn_loc, niv_cut=niv_core, name=r'$\gamma$')
#
# plt.imshow(chi_magn_asympt_lambda.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1)[:,:,0,niw_core].real, cmap='RdBu')
# plt.colorbar()
# plt.show()
#
#
# plt.plot(iw_core,chi0q_core_full.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1).mean(axis=(0, 1, 2)).real, 'o', label='q-sum: core')
# plt.plot(iw_core,chi0q_urange_full.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1).mean(axis=(0, 1, 2)).real, 'o', label='q-sum: urange')
# plt.plot(iw_core,chi0q_asympt_full.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1).mean(axis=(0, 1, 2)).real, 'o', label='q-sum: asympt')
# plt.legend()
# plt.show()
#
# # plot.plot_tp(twop=vrg_magn_loc, niv_cut=niv_core, name=r'$\gamma$')
# #
# # iv_dmft = np.arange(-niv_dmft,niv_dmft)
# # plt.plot(iv_dmft,dmft1p['gloc'].real, 'o', label='loc')
# # plt.plot(iv_urange,gk_urange.get_giw().real, 's', label='k-sum')
# # plt.legend()
# # plt.xlim([-10,10])
# # plt.xlabel(r'$\nu$')
# # plt.ylabel(r'$\Re G$')
# # plt.show()
# # #
