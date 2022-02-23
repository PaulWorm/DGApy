''' Create plots to check the convergence of local routines with the used frequency box'''
import numpy as np
import matplotlib.pyplot as plt
import MatsubaraFrequencies as mf

input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.85/KonvergenceAnalysis/'
beta = 10

def load_data(paths=None,name=None,key=None):
    data = []
    for path in paths:
        if(key is not None):
            data.append(np.load(path+name, allow_pickle=True).item()[key])
        else:
            data.append(np.load(path + name, allow_pickle=True).item())
    return data

def get_folder_name(niv_urange=None,n_core=None):
    return f'LambdaDga_lc_spch_local_core{n_core}_invbse{n_core}_vurange{niv_urange}_wurange{n_core}/'

def plot_iv(dataset=None, niv_plot=100, name='', box_sizes=None, pdir=None):
    for i, data in enumerate(dataset):
        iv = mf.vn(n=data.shape[-1] // 2)
        plt.plot(iv, data.imag, 'o', ms=2, label=f'N={box_sizes[i]}')
        plt.legend()
        plt.xlabel('$v_n$')
        plt.ylabel(name)
        plt.xlim(-1, niv_plot)
        plt.ylim(None, 0)
    plt.savefig(pdir + name + '_vs_iv.png')
    plt.show()

    datv0 = []
    for data in dataset:
        datv0.append(data[data.shape[-1] // 2])
    plt.plot(box_sizes, np.imag(datv0), 'o')
    plt.xlabel('N')
    plt.ylabel(name)
    plt.savefig(pdir + name + '_vs_bs.png')
    plt.show()

def plot_iw(dataset=None, niw_plot=20, name='', box_sizes=None, pdir=None):
    for i, data in enumerate(dataset):
        iw = mf.wn(n=data.shape[-1] // 2)
        plt.plot(iw, data.real, 'o', ms=2, label=f'N={box_sizes[i]}')
        plt.legend()
        plt.xlabel('$w_n$')
        plt.ylabel(name)
        plt.xlim(-1, niw_plot)
        #plt.ylim(0, Non)
    plt.savefig(pdir + name + '_vs_iw.png')
    plt.show()

    datv0 = []
    for data in dataset:
        datv0.append(data[data.shape[-1] // 2])
    plt.plot(box_sizes, np.real(datv0), 'o')
    plt.xlabel('N')
    plt.ylabel(name)
    plt.savefig(pdir + name + '_vs_bs.png')
    plt.show()

niv_urange = [20,40,80,160,320,640,900]
n_core = 20
data_paths = []
for n in niv_urange:
    data_paths.append(input_path + get_folder_name(n,n_core))



sigma = load_data(paths=data_paths, name='dmft_sde.npy',key='siw')
chi0_data = load_data(paths=data_paths, name='chi0_urange.npy', key=None)
chi0 = [chi0.chi0 for chi0 in chi0_data]
chi0_asympt = [chi0.chi0_asympt for chi0 in chi0_data]
chi_magn_data = load_data(paths=data_paths, name='chi_dmft.npy',key='magn')
chi_magn = [chi.mat for chi in chi_magn_data]
chi_magn_asympt = [chi.mat_asympt for i, chi in enumerate(chi_magn_data)]
vrg_magn_data = load_data(paths=data_paths, name='vrg_dmft.npy',key='magn')


plot_iv(dataset=sigma, niv_plot=100, name=r'$\Sigma$', box_sizes=niv_urange,pdir=input_path)
plot_iw(dataset=chi0, niw_plot=20, name=r'$\chi_0$', box_sizes=niv_urange,pdir=input_path)
plot_iw(dataset=chi0_asympt, niw_plot=20, name=r'$\chi_{0,asympt}$', box_sizes=niv_urange,pdir=input_path)
plot_iw(dataset=chi_magn, niw_plot=20, name=r'$\chi_{magn}$', box_sizes=niv_urange,pdir=input_path)
plot_iw(dataset=chi_magn_asympt, niw_plot=20, name=r'$\chi_{magn, asympt}$', box_sizes=niv_urange,pdir=input_path)


#
# niw_plot = 30
# fig = plt.figure()
# for chi in chi_magn:
#     iw = mf.wn(n=chi.mat.shape[-1]//2)
#     plt.plot(iw,chi.mat.real, 'o', ms=2)
#     plt.xlim(-niw_plot,niw_plot)
#     #plt.ylim(None,0)
# plt.show()
#
# fig = plt.figure()
# chiw0 = []
# for chi in chi_magn:
#     chiw0.append(chi.mat[chi.mat.shape[-1]//2])
# plt.plot(niv_urange,np.real(chiw0), 'o')
# plt.show()
#
# niw_plot = 30
# fig = plt.figure()
# for chi in chi_magn:
#     iw = mf.wn(n=chi.shape[-1]//2)
#     plt.plot(iw,chi.mat.real, 'o', ms=2)
#     plt.xlim(-niw_plot,niw_plot)
#     #plt.ylim(None,0)
# plt.show()
#
# fig = plt.figure()
# chiw0 = []
# for chi in chi_magn:
#     chiw0.append(chi.mat[chi.mat.shape[-1]//2])
# plt.plot(niv_urange,np.real(chiw0), 'o')
# plt.show()
#
# fig = plt.figure()
# for vrg in vrg_magn:
#     iv = mf.vn(n=vrg.mat.shape[-1]//2)
#     plt.plot(iv,vrg.mat[vrg.mat.shape[0]//2].real, 'o', ms=2)
#     plt.xlim(-niv_plot,niv_plot)
#     #plt.ylim(None,0)
# plt.show()
#
# fig = plt.figure()
# vrgw0v0 = []
# for vrg in vrg_magn:
#     vrgw0v0.append(vrg.mat[vrg.mat.shape[0]//2,vrg.mat.shape[-1]//2])
# plt.plot(niv_urange,np.real(vrgw0v0), 'o')
# plt.show()