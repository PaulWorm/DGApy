# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import sys
import copy
import importlib
import time
import numpy as np
import w2dyn_aux
import Indizes as ind
import Hk as hk
import FourPoint as fp
import TwoPoint as twop
import LambdaCorrection as lc
import SDE as sde
import h5py
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
# path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta80_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/Testset1/LambdaDga_Python/'

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

niw_core = 99
niv_core = 100
niv_urange = 500
niv_asympt = 5000
iw = np.arange(-niw_core, niw_core + 1)
iv_urange = np.arange(-niv_urange, niv_urange)

Nkx = Nk
Nky = Nk
Nkz = 1
Nqx = Nq
Nqy = Nq
Nqz = 1

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

f1p = w2dyn_aux.w2dyn_file(fname=path + '1p-data.hdf5')
dmft1p = f1p.load_dmft1p_w2dyn()
f1p.close()

giw = dmft1p['gloc']
# --------------------------------------------- LOCAL BUBBLE -----------------------------------------------------------

realt = real_time()

chi0_core = fp.LocalBubble(giw=giw, beta=dmft1p['beta'], niv_sum=box_sizes['niv_core'], iw=iw)
chi0_urange = fp.LocalBubble(giw=giw, beta=dmft1p['beta'], niv_sum=box_sizes['niv_urange'], iw=iw)
chi0_asympt = copy.deepcopy(chi0_urange)
chi0_asympt.add_asymptotic(niv_asympt=box_sizes['niv_asympt'])

# -------------------------------------------LOAD G2 FROM W2DYN --------------------------------------------------------


g2_file = w2dyn_aux.g4iw_file(fname=path + 'g4iw_sym.hdf5')
niw_dmft_full = g2_file.get_niw(channel='dens')

g2_dens_loc = fp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='dens', iw=iw), giw=giw, channel='dens',
                                beta=dmft1p['beta'], iw=iw)
g2_magn_loc = fp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='magn', iw=iw), giw=giw, channel='magn',
                                beta=dmft1p['beta'], iw=iw)

g2_dens_loc.cut_iv(niv_cut=box_sizes['niv_core'])
g2_magn_loc.cut_iv(niv_cut=box_sizes['niv_core'])

gchi_dens_loc = fp.LocalFourPoint(matrix=fp.vec_chir_from_g2(g2=g2_dens_loc), giw=g2_dens_loc.giw,
                                  channel=g2_dens_loc.channel,
                                  beta=g2_dens_loc.beta, iw=g2_dens_loc.iw)
gchi_magn_loc = fp.LocalFourPoint(matrix=fp.vec_chir_from_g2(g2=g2_magn_loc), giw=g2_magn_loc.giw,
                                  channel=g2_magn_loc.channel,
                                  beta=g2_magn_loc.beta, iw=g2_magn_loc.iw)

gamma_dens_loc = fp.gammar_from_gchir(gchir=gchi_dens_loc, gchi0_urange=chi0_urange, u=dmft1p['u'])
gamma_magn_loc = fp.gammar_from_gchir(gchir=gchi_magn_loc, gchi0_urange=chi0_urange, u=dmft1p['u'])

gchi_aux_dens_loc = fp.local_gchi_aux_from_gammar(gammar=gamma_dens_loc, gchi0_core=chi0_core, u=dmft1p['u'])
gchi_aux_magn_loc = fp.local_gchi_aux_from_gammar(gammar=gamma_magn_loc, gchi0_core=chi0_core, u=dmft1p['u'])

chi_aux_dens_loc = fp.local_susceptibility_from_four_point(four_point=gchi_aux_dens_loc)
chi_aux_magn_loc = fp.local_susceptibility_from_four_point(four_point=gchi_aux_magn_loc)

chi_dens_urange_loc = fp.local_chi_phys_from_chi_aux(chi_aux=chi_aux_dens_loc, chi0_urange=chi0_urange,
                                                     chi0_core=chi0_core,
                                                     u=dmft1p['u'])
chi_dens_asympt_loc = copy.deepcopy(chi_dens_urange_loc)
chi_dens_asympt_loc.add_asymptotic(chi0_asympt=chi0_asympt, chi0_urange=chi0_urange)
chi_magn_urange_loc = fp.local_chi_phys_from_chi_aux(chi_aux=chi_aux_magn_loc, chi0_urange=chi0_urange,
                                                     chi0_core=chi0_core,
                                                     u=dmft1p['u'])
chi_magn_asympt_loc = copy.deepcopy(chi_magn_urange_loc)
chi_magn_asympt_loc.add_asymptotic(chi0_asympt=chi0_asympt, chi0_urange=chi0_urange)

vrg_dens_loc = fp.local_fermi_bose_from_chi_aux_asympt(gchi_aux=gchi_aux_dens_loc, gchi0=chi0_core,
                                                       chi_asympt=chi_dens_asympt_loc,
                                                       chi_urange=chi_dens_urange_loc,
                                                       niv_urange=box_sizes['niv_urange'],
                                                       u=dmft1p["u"])

vrg_magn_loc = fp.local_fermi_bose_from_chi_aux_asympt(gchi_aux=gchi_aux_magn_loc, gchi0=chi0_core,
                                                       chi_asympt=chi_dens_asympt_loc,
                                                       chi_urange=chi_dens_urange_loc,
                                                       niv_urange=box_sizes['niv_urange'],
                                                       u=dmft1p["u"])

# ---------------------------------------- LOCAL SCHWINGER DYSON EQUATION ----------------------------------------------

siw_sde_dens = sde.local_dmft_sde(vrg=vrg_dens_loc, chir=chi_dens_asympt_loc, u=dmft1p["u"])
siw_sde_magn = sde.local_dmft_sde(vrg=vrg_magn_loc, chir=chi_magn_asympt_loc, u=dmft1p["u"])

siw_sde = siw_sde_dens + siw_sde_magn + dmft1p['u'] / 2. * dmft1p['n']

niv_sde = siw_sde_dens.size // 2
v_sde = np.arange(-niv_sde, niv_sde)
niv_dmft = dmft1p["sloc"].size // 2
v_dmft = np.arange(-niv_dmft, niv_dmft)

plt.subplot(211)
plt.plot(v_dmft, dmft1p["sloc"].real)
plt.plot(v_sde, siw_sde.real, 'o', ms=2,
         label='asympt-range')
plt.xlim([0, dmft1p['beta']])
plt.xlabel('w')
plt.ylabel('Sigma-real')
plt.subplot(212)
plt.plot(v_dmft, dmft1p["sloc"].imag)
plt.plot(v_sde, siw_sde.imag, 'o', ms=2, label='asympt-range')
plt.xlim([0, dmft1p['beta']])
plt.legend()
plt.xlabel('w')
plt.ylabel('Sigma-imag')
plt.savefig(path + 'dga_local_sde_check.png')
plt.show()

realt.print_time('Local Part ')

# ------------------------------------------------ NON-LOCAL PART  -----------------------------------------------------
importlib.reload(twop)
qiw = ind.qiw(qgrid=qgrid, iw=iw)

chi0q_core_full = fp.FullQ(channel='dens', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)
chi0q_urange_full = fp.FullQ(channel='dens', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)
chi0q_asympt_full = fp.FullQ(channel='dens', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)

chi_dens_asympt = fp.FullQ(channel='dens', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)
chi_magn_asympt = fp.FullQ(channel='magn', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)

chi_dens_asympt_lambda = fp.FullQ(channel='dens', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)
chi_magn_asympt_lambda = fp.FullQ(channel='magn', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=True)

vrg_dens = fp.FullQ(channel='dens', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=False)
vrg_magn = fp.FullQ(channel='magn', beta=dmft1p['beta'], u=dmft1p['u'], qiw=qiw, is_master=False)

g_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=kgrid, hr=t_mat, sigma=dmft1p['sloc'])

gk_urange = g_generator.generate_gk(mu=dmft1p['mu'], qiw=[0, 0, 0, 0], niv=niv_urange)
gk_core = copy.deepcopy(gk_urange)
gk_core.cut_self_iv(niv_cut=niv_core)

for iqw in range(qiw.my_size):
    gkpq_urange = g_generator.generate_gk(mu=dmft1p['mu'], qiw=qiw.my_qiw[iqw], niv=niv_urange)

    gkpq_core = copy.deepcopy(gkpq_urange)
    gkpq_core.cut_self_iv(niv_cut=niv_core)

    chi0q_core = fp.Bubble(gk=gk_core.gk, gkpq=gkpq_core.gk, beta=gk_core.beta)
    chi0q_urange = fp.Bubble(gk=gk_urange.gk, gkpq=gkpq_urange.gk, beta=gk_urange.beta)
    chi0q_asympt = copy.deepcopy(chi0q_urange)
    chi0q_asympt.add_asymptotic(niv_asympt=box_sizes['niv_asympt'], wn=qiw.my_iw[iqw])

    gchi_aux_dens = fp.construct_gchi_aux(gammar=gamma_dens_loc, gchi0=chi0q_core, u=dmft1p['u'], wn=qiw.wn(iqw))
    gchi_aux_magn = fp.construct_gchi_aux(gammar=gamma_magn_loc, gchi0=chi0q_core, u=dmft1p['u'], wn=qiw.wn(iqw))

    chi_aux_dens = fp.susceptibility_from_four_point(four_point=gchi_aux_dens)
    chi_aux_magn = fp.susceptibility_from_four_point(four_point=gchi_aux_magn)

    chiq_dens_urange = fp.chi_phys_from_chi_aux(chi_aux=chi_aux_dens, chi0_urange=chi0q_urange, chi0_core=chi0q_core)
    chiq_magn_urange = fp.chi_phys_from_chi_aux(chi_aux=chi_aux_magn, chi0_urange=chi0q_urange, chi0_core=chi0q_core)

    chiq_dens_asympt = copy.deepcopy(chiq_dens_urange)
    chiq_dens_asympt.add_asymptotic(chi0_asympt=chi0q_asympt, chi0_urange=chi0q_urange)

    chiq_magn_asympt = copy.deepcopy(chiq_magn_urange)
    chiq_magn_asympt.add_asymptotic(chi0_asympt=chi0q_asympt, chi0_urange=chi0q_urange)

    vrgq_dens = fp.fermi_bose_from_chi_aux_asympt(gchi_aux=gchi_aux_dens, gchi0=chi0q_core, chi_asympt=chiq_dens_asympt
                                                  , chi_urange=chiq_dens_urange, niv_urange=box_sizes['niv_urange'])

    vrgq_magn = fp.fermi_bose_from_chi_aux_asympt(gchi_aux=gchi_aux_magn, gchi0=chi0q_core, chi_asympt=chiq_magn_asympt
                                                  , chi_urange=chiq_magn_urange, niv_urange=box_sizes['niv_urange'])

    chi_dens_asympt.mat[qiw.my_indizes[iqw]] = chiq_dens_asympt.mat
    chi_magn_asympt.mat[qiw.my_indizes[iqw]] = chiq_magn_asympt.mat

    vrg_dens.mat[qiw.my_indizes[iqw]] = vrgq_dens.mat
    vrg_magn.mat[qiw.my_indizes[iqw]] = vrgq_magn.mat

    chi0q_core_full.mat[qiw.my_indizes[iqw]] = chi0q_core.chi0
    chi0q_urange_full.mat[qiw.my_indizes[iqw]] = chi0q_urange.chi0
    chi0q_asympt_full.mat[qiw.my_indizes[iqw]] = chi0q_asympt.chi0

realt.print_time('Non-local chi ')

chi_dens_asympt.mat_to_array()
chi_magn_asympt.mat_to_array()

vrg_dens.mat_to_array()
vrg_magn.mat_to_array()

chi0q_urange_full.mat_to_array()
chi0q_core_full.mat_to_array()
chi0q_asympt_full.mat_to_array()


lambda_dens = lc.lambda_correction(lambda_start=lc.get_lambda_start(chi_dens_asympt), chir=chi_dens_asympt,
                                   chi_loc=chi_dens_asympt_loc)
lambda_magn = lc.lambda_correction(lambda_start=lc.get_lambda_start(chi_magn_asympt), chir=chi_magn_asympt,
                                   chi_loc=chi_magn_asympt_loc)

for iqw in range(qiw.my_size):
    chi_dens_asympt_lambda.mat[qiw.my_indizes[iqw]] = 1. / (1. / chi_dens_asympt.mat[qiw.my_indizes[iqw]] + lambda_dens)
    chi_magn_asympt_lambda.mat[qiw.my_indizes[iqw]] = 1. / (1. / chi_magn_asympt.mat[qiw.my_indizes[iqw]] + lambda_magn)

chi_dens_asympt_lambda.mat_to_array()
chi_magn_asympt_lambda.mat_to_array()

sigma_dens = sde.sde_dga(vrg=vrg_dens, chir=chi_dens_asympt_lambda, g_generator=g_generator, mu=dmft1p['mu'])
sigma_magn = sde.sde_dga(vrg=vrg_magn, chir=chi_magn_asympt_lambda, g_generator=g_generator, mu=dmft1p['mu'])

realt.print_time('Non-local SDE ')

sigma_dens_ksum = sigma_dens.mean(axis=(0,1,2))
sigma_magn_ksum = sigma_magn.mean(axis=(0,1,2))

s_dmft = dmft1p['sloc'][niv_dmft-niv_urange:niv_dmft+niv_urange]
sigma_dga =  2.*(dmft1p['u'] / 2. * dmft1p['n']) + sigma_dens +  3. * sigma_magn - siw_sde  #- (-s_dmft + siw_sde)
sigma_dga_loc = np.mean(sigma_dga, axis=(0, 1, 2))

niv_dga = sigma_dga_loc.size // 2
v_dga = np.arange(-niv_dga, niv_dga)

plt.subplot(211)
plt.plot(v_dmft, dmft1p["sloc"].real)
plt.plot(v_sde, siw_sde.real, 'o', ms=2, label='asympt-range')
plt.plot(v_dga, sigma_dga_loc.real, 'o', ms=2, label='dga-loc')
plt.xlim([0, niv_urange])
plt.xlabel('w')
plt.ylabel('Sigma-real')
plt.subplot(212)
plt.plot(v_dmft, dmft1p["sloc"].imag)
plt.plot(v_sde, siw_sde.imag, 'o', ms=2, label='asympt-range')
plt.plot(v_dga, sigma_dga_loc.imag, 'o', ms=2, label='dga-loc')
plt.xlim([0, niv_urange])
plt.legend()
plt.xlabel('w')
plt.ylabel('Sigma-imag')
plt.savefig(path + 'dga_ksum_sde_check.png')
plt.show()


plt.subplot(211)
plt.plot(v_sde, siw_sde_magn.real, 'o', ms=2, label='asympt-range')
plt.plot(v_dga, sigma_magn_ksum.real, 'o', ms=2, label='dga-loc')
plt.xlim([0, dmft1p['beta']])
plt.xlabel('w')
plt.ylabel('Sigma-real')
plt.subplot(212)
plt.plot(v_sde, sigma_magn_ksum.imag, 'o', ms=2, label='asympt-range')
plt.plot(v_dga, sigma_dga_loc.imag, 'o', ms=2, label='dga-loc')
plt.xlim([0, dmft1p['beta']])
plt.legend()
plt.xlabel('w')
plt.ylabel('Sigma-imag')
plt.show()

plt.plot(iw, chi_magn_asympt_loc.mat.real, 'o', label='loc-asympt')
plt.plot(iw, chi_magn_asympt_lambda.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1).mean(axis=(0, 1, 2)).real, 'o',
         label='lambda-corrected')
plt.plot(iw, chi_magn_asympt.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1).mean(axis=(0, 1, 2)).real, 'o',
         label='non-local')
plt.legend()
plt.xlim(-2)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\chi_{magn}$')
plt.show()

plt.plot(iw, chi_dens_asympt_loc.mat.real, 'o', label='loc-asympt')
plt.plot(iw, chi_dens_asympt_lambda.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1).mean(axis=(0, 1, 2)).real, 'o',
         label='lambda-corrected')
plt.legend()
plt.xlim(-2)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\chi_{dens}$')
plt.show()

plt.imshow(vrg_magn.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1, 2 * niv_urange).mean(axis=(0, 1, 2)).real, cmap='RdBu')
plt.colorbar()
plt.show()

plt.imshow(vrg_magn.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1, 2 * niv_urange)[:,:,0,niw_core,niv_urange].real, cmap='RdBu')
plt.colorbar()
plt.show()

plot.plot_tp(tp=vrg_magn_loc, niv_cut=niv_core, name=r'$\gamma$')

plt.imshow(chi_magn_asympt_lambda.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1)[:,:,0,niw_core].real, cmap='RdBu')
plt.colorbar()
plt.show()


plt.plot(iw,chi0q_core_full.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1).mean(axis=(0, 1, 2)).real, 'o', label='q-sum: core')
plt.plot(iw,chi0q_urange_full.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1).mean(axis=(0, 1, 2)).real, 'o', label='q-sum: urange')
plt.plot(iw,chi0q_asympt_full.mat.reshape(Nqx, Nqy, Nqz, 2 * niw_core + 1).mean(axis=(0, 1, 2)).real, 'o', label='q-sum: asympt')
plt.legend()
plt.show()

# plot.plot_tp(tp=vrg_magn_loc, niv_cut=niv_core, name=r'$\gamma$')
#
# iv_dmft = np.arange(-niv_dmft,niv_dmft)
# plt.plot(iv_dmft,dmft1p['gloc'].real, 'o', label='loc')
# plt.plot(iv_urange,gk_urange.get_giw().real, 's', label='k-sum')
# plt.legend()
# plt.xlim([-10,10])
# plt.xlabel(r'$\nu$')
# plt.ylabel(r'$\Re G$')
# plt.show()
# #
