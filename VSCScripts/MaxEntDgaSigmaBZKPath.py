#!/bin/bash
#SBATCH -N 1
#SBATCH -J MaxEntBZKPathSigma
#SBATCH --ntasks-per-node=48
#SBATCH --partition=skylake_0096
#SBATCH --qos=skylake_0096
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL                              # first have to state the type of event to occur
#SBATCH --mail-user=<p.worm@a1.net>   # and then your email address


import numpy as np
import h5py
import sys, os
sys.path.append('/home/fs71282/wormEco3P/Programs/dga/src')
import continuation as cont
import matplotlib.pyplot as plt
import Output as output
import TwoPoint as tp
import MatsubaraFrequencies as mf
import BrillouinZone as bz

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
path = './'

# ------------------------------------------------- LOAD DATA --------------------------------------------------

sigma = np.load(path + 'sigma.npy',allow_pickle=True)
dga_conf = np.load(path + 'config.npy', allow_pickle=True).item()
dmft_sde = np.load(path + 'dmft_sde.npy', allow_pickle=True).item()
hartree = dmft_sde['hartree']

beta = dga_conf.sys.beta
k_grid = dga_conf.k_grid
hr = dga_conf.sys.hr
g_fac = tp.GreensFunctionGenerator(beta=beta,kgrid=k_grid,hr=hr,sigma=sigma)
n = dga_conf.sys.n
mu = g_fac.adjust_mu(n,mu0=0)
gk = g_fac.generate_gk(mu=mu)
niv = gk.niv
vn = mf.v(beta,gk.niv)
bz_path='Gamma-X-M-Gamma'
k_path = bz.KPath(nk=k_grid.nk,path=bz_path)


# sigma_loc = np.mean(sigma,axis=(0,1,2))
# vn = mf.vn(n=niv)
# niv_plot = 100
# plt.figure()
# plt.plot(vn[niv:niv+niv_plot],sigma_loc[niv:niv+niv_plot].real,'-o',color='cornflowerblue',markeredgecolor='k')
# plt.show()


# Set the output folder and file names:
path_out = path

folder_out = 'MaxEntSigma_BZKPath'
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
sigma_kpath = sigma[k_path.ikx[::fac], k_path.iky[::fac], 0,niv:niv+nf] - hartree
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
    im1 = axes[0].pcolormesh(k_path.k_axis[::fac],w,Sw.imag,cmap='magma',shading='nearest')
    plt.colorbar(im1)
    plt.ylim(-5,5)
    plt.tight_layout()
    plt.savefig(folder_out + 'sigma_dispersion.png')
    plt.show()
except:
    pass

print('Finished!')