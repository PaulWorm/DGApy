#!/usr/bin/env python
# SBATCH -N 1
# SBATCH -J Susc_cont
# SBATCH --ntasks-per-node=48
# SBATCH --partition=mem_0096
# SBATCH --qos=p71282_0096
# SBATCH --time=06:00:00

# SBATCH --mail-type=ALL                              # first have to state the type of event to occur
# SBATCH --mail-user=<p.worm@a1.net>   # and then your email address

# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import mpi4py as mpi
import MatsubaraFrequencies as mf
import continuation as cont
import BrillouinZone as bz


# ToDo: Scan alpha for single q=(pi,pi)
# ToDo: Plot Matsubara for q=(pi,pi)

# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def cont_k(iw, data_1, w, err_val=0.002, verbose=False, preblur=False, blur_width=0.0, alpha_determination='chi2kink'):
    #     model = np.ones_like(w)
    data_1 = np.squeeze(data_1)
    model = np.exp(-w ** 2 / 0.15 ** 2)
    model *= data_1[0].real / np.trapz(model, w)
    err = err_val * np.ones_like(iw)
    probl = cont.AnalyticContinuationProblem(re_axis=w, im_axis=iw,
                                             im_data=data_1, kernel_mode='freq_bosonic')
    sol, _ = probl.solve(method='maxent_svd', optimizer='newton', preblur=preblur, blur_width=blur_width,
                         alpha_determination=alpha_determination,
                         model=model, stdev=err, interactive=False, verbose=verbose,
                         alpha_start=1e12, alpha_end=1e-3, fit_position=2.)  # fit_psition smaller -> underfitting

    if (verbose):
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 4))
        ax[0].plot(w, sol.A_opt * w)
        ax[1].plot(iw, data_1)
        ax[1].plot(iw, sol.backtransform)
        ax[2].plot(iw, data_1 - sol.backtransform)
        plt.show()
    return (sol.A_opt, sol.backtransform)


# ------------------------------------------------ OBJECTS -------------------------------------------------------------
class KPath(object):
    def __init__(self, nk1=None):
        self.nk1 = nk1
        self.ikx = self.get_ikx()
        self.iky = self.get_iky()
        self.k_axis = self.get_k_axis()
        self.ik = np.vstack((self.ikx, self.iky)).T
        self.nk, = self.k_axis.shape

    def get_ikx(self):
        ikx = np.concatenate((np.arange(self.nk1 // 2),
                              self.nk1 // 2 * np.ones((self.nk1 // 2), dtype=int),
                              np.arange(self.nk1 // 2 + 1)[::-1]))
        return ikx

    def get_iky(self):
        iky = np.concatenate((np.zeros((self.nk1 // 2), dtype=int),
                              np.arange(self.nk1 // 2),
                              np.arange(self.nk1 // 2 + 1)[::-1]))
        return iky

    def get_k_axis(self):
        k_axis = np.concatenate((np.linspace(0., 1., num=self.nk1, endpoint=False),
                                 1. + np.linspace(0., 0.5 * np.sqrt(2.), num=self.nk1 // 2 + 1, endpoint=True)))
        return k_axis


class KPath2(object):
    def __init__(self, nk1=None):
        self.nk1 = nk1
        self.ikx = self.get_ikx()
        self.iky = self.get_iky()
        self.k_axis = self.get_k_axis()
        self.ik = np.vstack((self.ikx, self.iky)).T
        self.nk, = self.k_axis.shape

    def get_ikx(self):
        ikx = np.concatenate((np.arange(self.nk1 // 2), np.arange(self.nk1 // 4, self.nk1 // 2 + 1)[::-1],
                              np.arange(self.nk1 // 4 + 1)[::-1]))
        return ikx

    def get_iky(self):
        iky = np.concatenate((np.zeros((self.nk1 // 2), dtype=int),
                              np.arange(self.nk1 // 4+1), np.arange(self.nk1 // 4 + 1)[::-1]))
        return iky

    def get_k_axis(self):
        k_axis = np.concatenate((np.linspace(0., 0.5, num=self.nk1 // 2, endpoint=False),
                                 0.5 + np.linspace(0., 0.25 * np.sqrt(2.), num=self.nk1 // 4 + 1, endpoint=True),
                                 0.5 + 0.25 * np.sqrt(2.) + np.linspace(0., 0.25 * np.sqrt(2.), num=self.nk1 // 4 + 1, endpoint=True)))
        return k_axis

class KPath3(object):
    def __init__(self, nk1=None):
        self.nk1 = nk1
        self.ikx = self.get_ikx()
        self.iky = self.get_iky()
        self.k_axis = self.get_k_axis()
        self.ik = np.vstack((self.ikx, self.iky)).T
        self.nk, = self.k_axis.shape

    def get_ikx(self):
        ikx = np.concatenate((np.arange(self.nk1 // 2),
                              self.nk1 // 2 * np.ones((self.nk1 // 2), dtype=int),
                              np.arange(self.nk1 // 2 + 1)[::-1]))
        return ikx

    def get_iky(self):
        iky = np.arange(self.nk1 // 4+1,self.nk1 // 2+1)
        return iky

    def get_k_axis(self):
        k_axis = np.concatenate((np.linspace(0., 1., num=self.nk1//4, endpoint=False)))
        return k_axis


# -------------------------------------------------- MAIN --------------------------------------------------------------
tev = 0.4
t_scal = 0.25
err = 0.002
w_max = 10
nw = 501
use_preblur = False
bw = 0.0
alpha_det = 'chi2kink'
channel = 'magn'
name = '_pipi'
# input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.90/LambdaDga_lc_sp_Nk10000_Nq10000_core60_invbse60_vurange500_wurange60_2/'
#input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta50_n0.95/LambdaDga_lc_sp_Nk10000_Nq10000_core60_invbse60_vurange500_wurange60/'
input_path = '/mnt/d/Research/HoleDopedNickelates/2DSquare_U8_tp-0.25_tpp0.12_beta25_n0.95/LambdaDgaPython/LambdaDga_lc_sp_Nk6400_Nq6400_core59_invbse60_vurange500_wurange59/'
output_path = input_path

config = np.load(input_path + 'config.npy', allow_pickle=True).item()
chi_data = np.load(input_path + 'chi_lambda.npy', allow_pickle=True).item()
chi = chi_data['chi_magn_lambda'].mat.real * t_scal
# beta = config.sys.beta
# n = config.sys.n
# u = config.sys.u
# niw = config.box.niw_core
beta = config['system']['beta']*t_scal
n = config['system']['n']
u = config['system']['u']/t_scal
niw = config['box_sizes']['niw_core']
nk = config['box_sizes']['nk']
iw = mf.w(beta=beta, n=niw)
#q_grid = config.q_grid
q_grid = bz.KGrid(nk=nk)

# Cut negative frequencies:
chi = chi[..., niw:]

fac = 1
q_path = KPath(nk[0])
chi_qpath = chi[q_path.ikx[::fac], q_path.iky[::fac], 0]
w = w_max * np.tan(np.linspace(0., np.pi / 2.1, num=nw)) / np.tan(np.pi / 2.1)
s = np.zeros((len(q_path.ikx[::fac]), nw))
bt = np.zeros((len(q_path.ikx[::fac]), niw + 1))
for ik in range(len(q_path.ikx[::fac])):
    print(ik)
    a, b = cont_k(iw[niw:], chi_qpath[ik, :], w, err, preblur=use_preblur, blur_width=bw, alpha_determination=alpha_det)
    s[ik] = a[:]
    bt[ik] = b[:]

out_dir = output_path + channel + name + '/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if not os.path.exists(out_dir + 'Plots/'):
    os.mkdir(out_dir + 'Plots/')

# %%
plt.figure()
plt.plot(q_path.k_axis[::fac], chi_qpath[:, 0], label='real')
plt.plot(q_path.k_axis[::fac], chi_qpath[:, 1], label='imag')
plt.xlabel('q-path')
plt.ylabel('chi')
plt.savefig(
    out_dir + 'Plots/' + 'chi_qpath_matsubara_err{}_usepreblur_{}_bw{}_alpha_{}_.pdf'.format(err, use_preblur, bw,
                                                                                             alpha_det), dpi=300)
plt.savefig(
    out_dir + 'Plots/' + 'chi_qpath_matsubara_err{}_usepreblur_{}_bw{}_alpha_{}_.png'.format(err, use_preblur, bw,
                                                                                             alpha_det), dpi=300)
plt.show()
plt.close()

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))
pl0 = ax[0].pcolormesh(q_path.k_axis[::fac], w * tev, w[:, None] * s.T, vmin=0, vmax=0.5/t_scal, shading='auto',
                       rasterized=True, cmap='terrain')
plt.colorbar(pl0, ax=ax[0])
ax[0].set_title('chi(q, w)')
pl1 = ax[1].pcolormesh(q_path.k_axis[::fac], w * tev, np.log10(w[:, None] * s.T + 1e-8), vmin=-1.0, vmax=0,
                       shading='auto', rasterized=True, cmap='terrain')
plt.colorbar(pl1, ax=ax[1])
ax[1].set_title('log10 chi(q, w)')
ax[0].set_ylabel('$\omega [eV]$')
ax[1].set_ylabel('$\omega [eV]$')

ax[0].vlines([0.5,0.5+0.25 * np.sqrt(2.)],0,1.0,'k')
ax[1].vlines([0.5,0.5+0.25 * np.sqrt(2.)],0,1.0,'k')
ax[0].hlines([0.2],0,0.5+0.5 * np.sqrt(2.),'k')
ax[1].hlines([0.2],0,0.5+0.5 * np.sqrt(2.),'k')
ax[0].set_ylim(0., 0.5)
ax[1].set_ylim(0., 0.5)
plt.savefig(
    out_dir + 'Plots/' + 'chi_qpath_err{}_usepreblur_{}_bw{}_alpha_{}_.pdf'.format(err, use_preblur, bw, alpha_det),
    dpi=300)
plt.savefig(
    out_dir + 'Plots/' + 'chi_qpath_err{}_usepreblur_{}_bw{}_alpha_{}_.png'.format(err, use_preblur, bw, alpha_det),
    dpi=300)
plt.show()
plt.close()

np.save(out_dir + 'chi_cont_err{}_usepreblur_{}_bw{}_alpha_{}_'.format(err, use_preblur, bw, alpha_det), s)
np.save(out_dir + 'q_path_err{}_usepreblur_{}_bw{}_alpha_{}_'.format(err, use_preblur, bw, alpha_det), q_path)
np.save(out_dir + 'w_err{}_usepreblur_{}_bw{}_alpha_{}_'.format(err, use_preblur, bw, alpha_det), w)

plt.figure()
plt.plot(q_grid.kx[q_path.ikx])
plt.plot(q_grid.ky[q_path.iky])
plt.savefig(out_dir + 'Plots/' + 'q_path.png'.format(err, use_preblur, bw, alpha_det), dpi=300)
plt.show()
plt.close()
