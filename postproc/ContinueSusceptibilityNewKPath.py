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
import MatsubaraFrequencies as mf
import continuation as cont
import BrillouinZone as bz
import Output as output


# ToDo: Scan alpha for single q=(pi,pi)
# ToDo: Plot Matsubara for q=(pi,pi)

# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def cont_k(iw, data_1, w, err_val=0.002, sigma=2.0, verbose=False, preblur=False, blur_width=0.0, alpha_determination='chi2kink'):
    data_1 = np.squeeze(data_1)
    model = np.exp(-w ** 2 / sigma ** 2)
    # model = np.ones_like(w)
    model *= data_1[0].real / np.trapz(model, w)
    # model *= 1 / np.trapz(model, w)
    err = err_val * np.ones_like(iw)
    probl = cont.AnalyticContinuationProblem(re_axis=w, im_axis=iw,
                                             im_data=data_1, kernel_mode='freq_bosonic')
    sol, _ = probl.solve(method='maxent_svd', optimizer='newton', preblur=preblur, blur_width=blur_width,
                         alpha_determination=alpha_determination,
                         model=model, stdev=err, interactive=False, verbose=verbose,
                         alpha_start=1e12, alpha_end=1e-3, fit_position=2.)  # fit_psition smaller -> underfitting

    # sol, _ = probl.solve_preblur(model=model, stdev=err, interactive=False, verbose=verbose,alpha_start=1e12, alpha_end=1e-3,)  # fit_psition smaller -> underfitting

    if (verbose):
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 4))
        ax[0].plot(w, sol.A_opt * w)
        ax[1].plot(iw, data_1)
        ax[1].plot(iw, sol.backtransform)
        ax[2].plot(iw, data_1 - sol.backtransform)
        plt.show()
    return (sol.A_opt, sol.backtransform)


# ------------------------------------------------ OBJECTS -------------------------------------------------------------
def add_lambda(chi,lambda_add):
    chi = 1./(1./chi+lambda_add)
    return chi

# -------------------------------------------------- MAIN --------------------------------------------------------------
tev = 1.0
t_scal = 1
err = 0.02
w_max = 5
nw = 501
use_preblur = False
ncut = 30
nmin = 0
bw = 0.0
sigma = 1.0
lambda_add = 0.0 #-0.7504336211373053
# alpha_det = 'historic'
alpha_det = 'chi2kink'
channel = 'magn'
name = '_pipi'
bz_k_path = 'Gamma-X-M2-Gamma'
# bz_k_path = 'Gamma-X-M-Gamma'
input_path = 'D:/Research/HoleDopedNickelates/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.80/LambdaDgaPython/LambdaDga_lc_sp_Nk6400_Nq6400_core59_invbse60_vurange500_wurange59/'
input_path = 'D:/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.95/LambdaDga_lc_spch_Nk14400_Nq14400_core80_invbse80_vurange500_wurange80/'
input_path = 'D:/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.95/LambdaDga_lc_sp_Nk10000_Nq10000_core80_invbse80_vurange500_wurange80/'
output_path = input_path
output_folder = f'ChiCont_nw_{nw}_err_{err}_sigma_{sigma}_lambda_{lambda_add}'
output_path = output.uniquify(output_path + output_folder) + '/'

result = 'old'

if(result!='adga' and result!='fort'):
    config = np.load(input_path + 'config.npy', allow_pickle=True).item()
    chi_data = np.load(input_path + 'chi_lambda.npy', allow_pickle=True).item()
    # chi_data = np.load(input_path + 'chi_ladder.npy', allow_pickle=True).item()
if(result=='old'):
    chi = chi_data['chi_magn_lambda'].mat.real * t_scal
    # chi = chi_data['chi_magn_ladder'].mat.real * t_scal
    beta = config['system']['beta'] * t_scal
    n = config['system']['n']
    u = config['system']['u'] / t_scal
    niw = config['box_sizes']['niw_core']
    nk = config['box_sizes']['nk']
    q_grid = bz.KGrid(nk=nk)
elif(result=='chi0'):
    chi = np.load(input_path + fname_chi0, allow_pickle=True)
    beta = config.sys.beta
    n = config.sys.n
    u = config.sys.u
    niw = config.box.niw_core
    nk = config._k_grid.nk
    q_grid = config._q_grid
elif(result=='adga'):
    import h5py
    hfile = h5py.File(input_path + 'adga.hdf5')
    chi = hfile['/susceptibility/nonloc/magn'][()]
    chi = chi[0,0,...]
    beta = hfile['input/beta']
    n = hfile['input/n_dmft']
    niw = chi.shape[-1] // 2
    nk = chi.shape[0:-1]
    q_grid = bz.KGrid(nk=nk)

elif(result=='fort'):
    chi = np.load(input_path + 'chisp_1.npy')
    chi = chi.T[:,:,None,:]
    niw = chi.shape[-1] // 2
    nk = chi.shape[0:-1]
    beta = 25 * 4
    n = 1.0
    chi = np.roll(chi,(nk[0]//2,nk[1]//2),(0,1))
    q_grid = bz.KGrid(nk=nk)

else:
    chi = chi_data['magn'].mat.real * t_scal
    beta = config.sys.beta * t_scal
    n = config.sys.n
    u = config.sys.u / t_scal
    niw = config.box.niw_core
    nk = config._k_grid.nk
    q_grid = config._q_grid


chi = add_lambda(chi,lambda_add)
if ncut == -1:
    ncut = niw
iw = mf.w(beta=beta, n=niw)
#_q_grid = config._q_grid

# Cut negative frequencies:
chi = chi[..., niw+nmin:niw+nmin+ncut]
# chi[...,0] = chi[...,0] * 0.03

# create q-path:
q_path = bz.KPath(nk=nk,path=bz_k_path)

fac = 1
out_dir = output_path
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

q_dict ={
    'kpts':q_path.kpts,
    'cind':q_path.cind,
    'k_axis':q_path.k_axis,
    'k_val': q_path.k_val,
    'path': q_path.path,
    'path_split': q_path.ckps
}
np.save(out_dir + 'q_path', q_dict,allow_pickle=True)


np.savetxt(out_dir + 'q_path_string.txt', [bz_k_path], delimiter=" ", fmt="%s")

chi_qpath = chi[q_path.ikx[::fac], q_path.iky[::fac], 0]
# w = w_max * np.tan(np.linspace(0., np.pi / 2.1, num=nw)) / np.tan(np.pi / 2.1)
w = np.linspace(0,w_max,num=nw)
s = np.zeros((len(q_path.ikx[::fac]), nw))
bt = np.zeros((len(q_path.ikx[::fac]), ncut))
for ik in range(len(q_path.ikx[::fac])):
    print(ik)
    a, b = cont_k(iw[niw+nmin:niw+nmin+ncut], chi_qpath[ik, :], w, err,sigma=sigma, preblur=use_preblur, blur_width=bw, alpha_determination=alpha_det)
    s[ik] = a[:]
    bt[ik] = b[:]



np.save(out_dir + 'chi', s)

np.save(out_dir + 'w', w)



plt.figure()
plt.plot(q_path.k_axis[::fac], chi_qpath[:, 0], label='real')
plt.plot(q_path.k_axis[::fac], chi_qpath[:, 1], label='imag')
plt.xlabel('q-path')
plt.ylabel('chi')
plt.legend()
plt.savefig(out_dir + 'chi_qpath_matsubara.pdf', dpi=300)
plt.savefig(out_dir +  'chi_qpath_matsubara.png', dpi=300)
plt.show()
plt.close()

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))
pl0 = ax[0].pcolormesh(q_path.k_axis[::fac], w * tev, w[:, None] * s.T, shading='auto',
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
ax[0].set_ylim(0., 1.0)
ax[1].set_ylim(0., 1.0)
plt.savefig(
    out_dir + 'chi_qpath.pdf',
    dpi=300)
plt.savefig(
    out_dir + 'chi_qpath.png',
    dpi=300)
plt.show()
plt.close()


plt.figure()
plt.plot(q_grid.kx[q_path.ikx])
plt.plot(q_grid.ky[q_path.iky])
plt.savefig(out_dir + 'q_path.png'.format(err, use_preblur, bw, alpha_det), dpi=300)
plt.show()
plt.close()
print('Finsihed!')
