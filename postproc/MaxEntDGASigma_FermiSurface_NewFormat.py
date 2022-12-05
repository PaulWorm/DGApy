 # ----------------------------------------------------------------------------------------------------
# File: MaxEntDGA.py
# Date: 07.11.2022
# Author: Paul Worm
# Short description: Interface for dga and the ana-cont package for analytic continuation.
# ----------------------------------------------------------------------------------------------------

import numpy as np
import h5py
import sys, os
import continuation as cont
import matplotlib.pyplot as plt
import Output as output
import TwoPoint as tp
import MatsubaraFrequencies as mf
import BrillouinZone as bz
import dga_aux as daux

def plot_imag_data(iw,giw=None,plot_dir=None, fname=None,niv=-1):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax = ax.flatten()
    ax0 = ax[0]
    ax1 = ax[1]

    if(niv==-1):
        niv=giw.size

    ax0.plot(iw, giw.imag[:niv])
    ax1.plot(iw, giw.imag[:niv])


    ax0.set_xlabel(r'$i \omega_n$')
    ax1.set_xlabel(r'$i \omega_n$')

    ax0.set_ylabel(r'$ \Re G$')
    ax1.set_ylabel(r'$ \Im G$')

    plt.tight_layout()

    plt.savefig(plot_dir + fname + '.png')
    plt.savefig(plot_dir + fname + '.pdf')
    plt.show()
# ------------------------------------------------ PARAMETERS -------------------------------------------------

# Set the path, where the input-data is located:

base = '/mnt/d/Research/HubbardModel_tp-0.25_tpp0.12/2DSquare_U8_tp-0.25_tpp0.12_beta25_n0.95/'
base = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta50_n0.85/'
path = base + 'LambdaDga_lc_sp_Nk10000_Nq10000_core60_invbse60_vurange500_wurange60/'

base = 'D:/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.875/'
path = base + 'LambdaDga_lc_sp_Nk10000_Nq10000_core60_invbse60_vurange500_wurange60/'


base = 'D:/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.85/'
path = base + 'LambdaDga_lc_sp_Nk10000_Nq10000_core60_invbse60_vurange500_wurange60/'

base = 'D:/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.80/'
path = base + 'LambdaDga_lc_sp_Nk10000_Nq10000_core60_invbse60_vurange500_wurange60/'

base = 'D:/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
path = base + 'LambdaDga_lc_sp_Nk19600_Nq19600_core60_invbse60_vurange200_wurange60_2/'

base = 'D:/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.90/'
path = base + 'LambdaDga_lc_sp_Nk14400_Nq14400_core30_invbse30_vurange500_wurange30/'

base = 'D:/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta7.5_n0.90/'
path = base + 'LambdaDga_lc_sp_Nk19600_Nq19600_core30_invbse30_vurange250_wurange30/'

base = 'D:/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.90/'
path = base + 'LambdaDga_lc_sp_Nk14400_Nq14400_core60_invbse60_vurange500_wurange60_1/'

base = 'D:/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.90/'
path = base + 'LambdaDga_lc_sp_Nk14400_Nq14400_core60_invbse60_vurange500_wurange60_1/'

base = 'D:/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.75/'
path = base + 'LambdaDga_lc_sp_Nk19600_Nq19600_core60_invbse60_vurange200_wurange60/'

# ------------------------------------------------- LOAD DATA --------------------------------------------------
is_old = False

dga_conf = np.load(path + 'config.npy', allow_pickle=True).item()
dmft_sde = np.load(path + 'dmft_sde.npy', allow_pickle=True).item()
hartree = dmft_sde['hartree']

if(is_old):
    raise NotImplementedError
    beta = dga_conf['system']['beta']
    k_grid = dga_conf['system']['beta']
    hr = dga_conf.sys.hr
    n = dga_conf.sys.n
else:
    sigma = np.load(path + 'sigma.npy', allow_pickle=True)
    beta = dga_conf.sys.beta
    k_grid = dga_conf.k_grid
    hr = dga_conf.sys.hr
    n = dga_conf.sys.n

g_fac = tp.GreensFunctionGenerator(beta=beta,kgrid=k_grid,hr=hr,sigma=sigma)
mu = g_fac.adjust_mu(n,mu0=0)
gk = g_fac.generate_gk(mu=mu)
niv = gk.niv
vn = mf.v(beta,gk.niv)
dga_data = daux.DgaDataManager(path = path)
ind = 0
ind_fs = dga_data.get_fs_ind_1q()[0]



# Set the output folder and file names:
path_out = path

folder_out = 'MaxEntSigma_FermiSurface'
folder_out = output.uniquify(path_out+folder_out) + '/'
fname_g = 'ContinuedSigma.hdf5'

# Set parameter for the analytic continuation:
interactive = True
w_grid_type = 'tan'
alpha_det_method = "chi2kink"
wmax = 15
Nwr = 501
nf = 60 # Number of frequencies to keep
use_preblur = True
bw = 0.01  # preblur width
aerr_g = 1e-3
aerr_s = 1e-3
if(w_grid_type == 'lin'):
    w = np.linspace(-wmax, wmax, num=Nwr)
elif(w_grid_type == 'tan'):
    w = np.tan(np.linspace(-np.pi/2.5,np.pi/2.5,num=Nwr,endpoint=True))*wmax/np.tan(np.pi/2.5)


# ------------------------------------------ GET SYSTEM CONFIGURATION ------------------------------------------
if nf > niv:
    nf = niv - 1

# Cut frequency:
# -----------------

vn_cut = vn[niv:niv + nf]

os.mkdir(folder_out)

# -------------------------------------------- ANALYTIC CONTINUATION -----------------------------------------------

model = np.ones_like(w)
model /= np.trapz(model, w)
error_g = np.ones((nf,), dtype=np.float64) * aerr_g

fac = 1
sigma_kpath = sigma[ind_fs[:,0], ind_fs[:,1], 0,niv:niv+nf] - hartree
nk_use = np.shape(sigma_kpath)[0]
Sw = np.zeros((Nwr,nk_use), dtype=complex)
alpha_g = np.zeros((nk_use,), dtype=complex)
chi2_g = np.zeros((nk_use,), dtype=complex)
print('Continuing G:')
for i in range(nk_use):
    problem = cont.AnalyticContinuationProblem(im_axis=vn_cut, re_axis=w, im_data=sigma_kpath[i,:], kernel_mode="freq_fermionic",
                                               beta=beta)
    sol, _ = problem.solve(method="maxent_svd", model=model, stdev=error_g, alpha_determination=alpha_det_method,
                           optimizer="newton", preblur=use_preblur, blur_width=bw)
    Sw[:,i] = cont.GreensFunction(spectrum=sol.A_opt, wgrid=w, kind='fermionic').kkt()
    alpha_g[i] = sol.__dict__['alpha']
    chi2_g[i] = sol.chi2

Sw = Sw + hartree
# -------------------------------------------- OUTPUT -----------------------------------------------
Nw = w.shape[0]

# Print Green-function continuation:
Outputfile = h5py.File(folder_out + fname_g, 'w')
Outputfile['beta'] = beta
Outputfile['ind_fs'] = ind_fs
Outputfile['alpha'] = alpha_g
Outputfile['chi2'] = chi2_g
Outputfile['Sw'] = Sw
Outputfile['w'] = w
Outputfile['aerr'] = aerr_g
Outputfile['mu'] = mu
Outputfile['nf_keep'] = nf
Outputfile['.config/use_preblur'] = use_preblur
Outputfile['.config/bw'] = bw
Outputfile['.system/Nw'] = Nw
Outputfile['.config/alpha_det_method'] = alpha_det_method
Outputfile['.config/w_grid_type'] = w_grid_type
Outputfile.close()

try:
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    axes = np.atleast_1d(axes)
    im1 = axes[0].pcolormesh(np.linspace(0,1,nk_use),w,Sw.imag,cmap='magma',shading='nearest')
    plt.colorbar(im1)
    plt.ylim(-5,5)
    plt.tight_layout()
    plt.savefig(folder_out + 'sigma_dispersion.png')
    plt.show()
except:
    pass

print('Finished!')

# try:
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6,6))
#     axes = axes.flatten()
#     axes[0].plot(w, Gw.real, color='cornflowerblue', label='$ \Re G(\omega)$')
#     axes[1].plot(w, -1/np.pi* Gw.imag, color='cornflowerblue', label='$ A(\omega)$')
#     axes[2].plot(w, Gw.real, '-o',color='cornflowerblue', label='$ \Re G(\omega)$')
#     axes[3].plot(w, -1/np.pi* Gw.imag,'-o', color='cornflowerblue', label='$ A(\omega)$')
#     for ax in axes:
#         ax.legend()
#         ax.set_xlabel('$\omega$')
#         ax.set_xlim(-15, 15)
#     axes[2].set_xlim(-2, 2)
#     axes[3].set_xlim(-2, 2)
#     axes[0].set_ylabel('$ \Re G(\omega)$')
#     axes[1].set_ylabel('$ A(\omega)$')
#     axes[2].set_ylabel('$ \Re G(\omega)$')
#     axes[3].set_ylabel('$ A(\omega)$')
#     plt.tight_layout()
#     plt.savefig(folder_out + 'ContinuedData.png')
#     plt.show()
# except:
#     pass
