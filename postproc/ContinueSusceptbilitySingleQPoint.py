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

def cont_k(iw, data_1, w, err_val=0.002,sigma=0.4, verbose=False, preblur=False, blur_width=0.0, alpha_determination='chi2kink'):
    #     model = np.ones_like(w)
    data_1 = np.squeeze(data_1)
    model = np.exp(-w ** 2 / sigma** 2)
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
    return (sol.A_opt, sol.backtransform, sol.chi2)



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
chip = chi[nk[0]//4,nk[1]//4,0]

sigma = np.array([1.0,2.0,4.0])
fac = 1
w = w_max * np.tan(np.linspace(0., np.pi / 2.1, num=nw)) / np.tan(np.pi / 2.1)
s = np.zeros((len(sigma), nw))
bt = np.zeros((len(sigma), niw + 1))
chi2 = np.zeros((len(sigma)))
for ik in range(len(sigma)):
    print(ik)
    a, b, c = cont_k(iw[niw:], chip, w, err,sigma=sigma[ik], preblur=use_preblur, blur_width=bw, alpha_determination=alpha_det)
    s[ik] = a[:]
    bt[ik] = b[:]
    chi2[ik] = c

out_dir = output_path + channel + name + '/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if not os.path.exists(out_dir + 'Plots/'):
    os.mkdir(out_dir + 'Plots/')

# %%
plt.figure()
for ik in range(len(sigma)):
    plt.plot(w*tev,w * s[ik], label=f'$\sigma = {sigma[ik]}')
plt.legend()
plt.show()

plt.figure()
plt.plot(sigma,chi2,'-o')
plt.ylabel('\chi^2')
plt.xlabel('\sigma')
plt.legend()
plt.show()