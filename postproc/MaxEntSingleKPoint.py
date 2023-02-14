import numpy as np
import matplotlib.pyplot as plt
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

base = 'D:/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta5_n0.90/'
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

# Set the output folder and file names:
path_out = path

folder_out = 'MaxEntSingleKPoint'
folder_out = output.uniquify(path_out+folder_out) + '/'
fname_g = 'ContinuedSigma.hdf5'

# Set parameter for the analytic continuation:
interactive = True
w_grid_type = 'tan'
alpha_det_method = "chi2kink"
wmax = 15
Nwr = 4000
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
niv = sigma.shape[-1] // 2
vn = mf.v(beta,niv)
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
# sigma_kpath = sigma[ind_fs[:,0], ind_fs[:,1], 0,niv:niv+nf] - hartree
nkx = sigma.shape[0]
nky = sigma.shape[1]
sigma_kpath = sigma[nkx-1, nky//2, 0,niv:niv+nf] - hartree
# sigma_kpath = sigma[nkx//2, nky//2, 0,niv:niv+nf] - hartree
nk_use = 1
Sw = np.zeros((Nwr,nk_use), dtype=complex)
alpha_g = np.zeros((nk_use,), dtype=complex)
chi2_g = np.zeros((nk_use,), dtype=complex)
print('Continuing G:')
for i in range(nk_use):
    problem = cont.AnalyticContinuationProblem(im_axis=vn_cut, re_axis=w, im_data=sigma_kpath[:], kernel_mode="freq_fermionic",
                                               beta=beta)
    sol, _ = problem.solve(method="maxent_svd", model=model, stdev=error_g, alpha_determination=alpha_det_method,
                           optimizer="newton", preblur=use_preblur, blur_width=bw,verbose=True,interactive=True)
    Sw[:,i] = cont.GreensFunction(spectrum=sol.A_opt, wgrid=w, kind='fermionic').kkt()
    alpha_g[i] = sol.__dict__['alpha']
    chi2_g[i] = sol.chi2
    plt.show()
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(18,5))
    ax[0].plot(w, sol.A_opt)
    ax[1].plot(vn_cut, sigma_kpath-sol.backtransform)
    #ax[2].errorbar(x=iw, y=giw.real, yerror=err)
    #ax[2].errorbar(x=iw, y=giw.imag, yerror=err)
    ax[2].plot(sol.backtransform)
    ax[2].plot(sigma_kpath, marker='x')
    plt.savefig(folder_out + f'sanity_check_{i}.png')
    plt.show()

Sw = Sw + hartree
# -------------------------------------------- OUTPUT -----------------------------------------------
Nw = w.shape[0]

# Print Green-function continuation:
Outputfile = h5py.File(folder_out + fname_g, 'w')
Outputfile['beta'] = beta
# Outputfile['ind_fs'] = ind_fs
Outputfile['alpha'] = alpha_g
Outputfile['chi2'] = chi2_g
Outputfile['Sw'] = Sw
Outputfile['w'] = w
Outputfile['aerr'] = aerr_g
# Outputfile['mu'] = mu
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

#%%

iv_test = 1j * np.linspace(0.005,4,30)

backtransform = np.zeros_like(iv_test).astype(complex)
for i,iv in enumerate(iv_test):
    kernel = 1/(iv - w)
    backtransform[i] = np.trapz(kernel*sol.A_opt,w, axis=-1)

plt.figure()
plt.plot(iv_test.imag,backtransform.imag,'-o',color='cornflowerblue')
plt.plot(vn_cut,sigma_kpath.imag,'-x',color='firebrick')
plt.plot(vn_cut,sol.backtransform.imag,'-x',color='goldenrod')
plt.xlim(0,10)
plt.show()