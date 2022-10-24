# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Plot the local irreducible vertex (Gamma) for several frequency box sizes.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os, sys

# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def get_dirs(root=None):
    dirs = []
    for item in os.listdir(root):
        if os.path.isdir(os.path.join(root, item)):
            dirs.append(os.path.join(root, item))
    return dirs

def get_gammas(dirs=None, name='gamma_dmft.npy', key='gamma_magn'):
    gamma = []
    for dir in dirs:
        try:
            dict = np.load(os.path.join(dir, name), allow_pickle=True).item()
            gamma.append(dict[key])
        except:
            pass
    return gamma

def get_boxsizes(dirs=None, name='config.npy', key='box_sizes'):
    box_sizes = []
    for dir in dirs:
        try:
            dict = np.load(os.path.join(dir, name), allow_pickle=True).item()
            box_sizes.append(dict[key])
        except:
            pass
    return box_sizes

def get_quantity(dirs=None, name='', key=''):
    quant = []
    for dir in dirs:
        try:
            dict = np.load(os.path.join(dir, name), allow_pickle=True).item()
            quant.append(dict[key])
        except:
            pass
    return quant

def plot_gammas(gammas=None, box_sizes=None, title='',niw_off=0):

    fig = plt.figure()
    for i, gamma in enumerate(gammas):
        niv = gamma.niv
        niw = gamma.niw // 2
        iv = np.arange(-niv,niv)
        plt.plot(iv,gamma.mat[niw+niw_off,:,niv].real,'o', ms = 1, label='Core:{} ;Urange{} '.format(box_sizes[i]['niv_core'],box_sizes[i]['niv_urange']))
    plt.legend()
    plt.xlabel(r'$\nu_n$')
    plt.ylabel(r'$\Gamma$')
    plt.title(r'{}'.format(title))
    return fig

def plot_chi(chis=None, box_sizes=None,title=''):

    fig = plt.figure()
    for i, chi in enumerate(chis):
        iw = chi['iw']
        chi_loc = chi['chi_loc']
        plt.plot(iw,chi_loc.real,'o', ms = 1, label='Core:{} ;Urange{} '.format(box_sizes[i]['niv_core'],box_sizes[i]['niv_urange']))
        #plt.scatter(iw,chi_loc.real, label='Core:{} ;Urange{} '.format(box_sizes[i]['niv_core'],box_sizes[i]['niv_urange']))
    plt.legend()
    plt.xlabel(r'$\nu_n$')
    plt.ylabel(r'$\chi_{loc}$')
    plt.title(r'{}'.format(title))
    return fig

def reshape_chi(chis=None, box_sizes=None):
    chi_reshape = []
    for i, chi in enumerate(chis):
        niw = box_sizes[i]['niw_core']
        nq = box_sizes[i]['nq']
        iw = np.arange(-niw,niw+1)
        chi = chi.mat.reshape(nq+(2*niw+1,))
        dict = {
            'iw':iw,
            'chi':chi,
        'chi_loc':chi.mean(axis=(0,1,2))
        }
        chi_reshape.append(dict)
    return chi_reshape

def plot_chi_dmft(chis=None, box_sizes=None, title=''):

    fig = plt.figure()
    for i, chi in enumerate(chis):
        plt.plot(chi.wn, chi.mat.real, 'o', ms = 1, label='Core:{} ;Urange{} '.format(box_sizes[i]['niv_core'], box_sizes[i]['niv_urange']))
    plt.legend()
    plt.xlabel(r'$\nu_n$')
    plt.ylabel(r'$\chi_{loc}$')
    plt.title(r'{}'.format(title))
    return fig



# ----------------------------------------------- Parameters ------------------------------------------------------------

input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.85/KonvergenceAnalysis/'
data = 'LambdaDga_lc_sp_Nk1024_Nq1024_core20_invbse20_vurange50_wurange50/'

chi = np.load(input_path+data+'chi_lambda.npy', allow_pickle=True).item()
box = np.load(input_path+data+'config.npy', allow_pickle=True).item()['box_sizes']
chi_magn_mat = chi['chi_magn_lambda'].mat

dirs = get_dirs(root=input_path)

gammas_magn = get_gammas(dirs=dirs)
chi_dmft_magn = get_quantity(dirs=dirs, name='dmft_sde.npy', key='chi_magn')
box_sizes = get_boxsizes(dirs=dirs)
chi_magn_lambda_data = get_quantity(dirs=dirs, name='dga_sde.npy', key='chi_magn_lambda')
#chi_magn_lambda = reshape_chi(chis=chi_magn_lambda_data, box_sizes=box_sizes)

# Plot Gamma:
fig = plot_gammas(gammas=gammas_magn,box_sizes=box_sizes, title='$\Gamma_{DMFT;magn}$', niw_off=2)
plt.savefig(input_path + 'Gamma_magn' + '.png')
plt.show()
plt.close()

# Plot Gamma:
fig = plot_gammas(gammas=gammas_magn,box_sizes=box_sizes, title='$\Gamma_{DMFT;magn}$', niw_off=1)
plt.savefig(input_path + 'Gamma_magn' + '.png')
plt.show()
plt.close()

# Plot Gamma:
fig = plot_gammas(gammas=gammas_magn,box_sizes=box_sizes, title='$\Gamma_{DMFT;magn}$', niw_off=0)
plt.savefig(input_path + 'Gamma_magn' + '.png')
plt.show()
plt.close()

# # Plot Chi:
# plot_chi(chis=chi_magn_lambda, box_sizes=box_sizes, title='$\chi_{magn,loc}$')
# plt.savefig(input_path + 'Chi_lambda_loc_magn' + '.png')
# plt.show()
# plt.close()
#
# # Plot Chi:
# plot_chi_dmft(chis=chi_dmft_magn, box_sizes=box_sizes, title='$\chi_{magn,loc}$')
# plt.savefig(input_path + 'Chi_dmft_loc_magn' + '.png')
# plt.show()
# plt.close()

#%%

plt.imshow(gammas_magn[0].mat[15].real, cmap='RdBu', extent=(-15,14,14,-15))
plt.plot(np.arange(-15,15)*0,np.arange(-15,15),'k')
plt.plot(np.arange(-15,15),0*np.arange(-15,15),'k')
plt.colorbar()
plt.show()

plt.imshow(gammas_magn[0].mat[-1].real, cmap='RdBu', extent=(-15,14,14,-15))
plt.plot(np.arange(-15,15)*0,np.arange(-15,15),'k')
plt.plot(np.arange(-15,15),0*np.arange(-15,15),'k')
plt.colorbar()
plt.show()

plt.imshow(gammas_magn[0].mat[0].real, cmap='RdBu', extent=(-15,14,14,-15))
plt.plot(np.arange(-15,15)*0,np.arange(-15,15),'k')
plt.plot(np.arange(-15,15),0*np.arange(-15,15),'k')
plt.colorbar()
plt.show()


