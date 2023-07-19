# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Benchmark against the data from Thomas Schäfers Spin-fluctuation paper
# (https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.011058)


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import MatsubaraFrequencies as mf


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def load_ldga_data(path=None):
    dga_sde = np.load(path + 'dga_sde.npy', allow_pickle=True).item()
    config = np.load(path + 'config.npy', allow_pickle=True).item()
    return dga_sde, config


def load_ldga_chi_lambda(path=None):
    chi = np.load(path + 'chi_lambda.npy', allow_pickle=True).item()
    return chi


def load_ldga_chi_ladder(path=None):
    chi = np.load(path + 'chi_ladder.npy', allow_pickle=True).item()
    return chi

def load_ldga_n_an(path=None,sigma_type='normal'):
    dmft_sde = np.load(path + 'dmft_sde.npy', allow_pickle=True).item()
    config = np.load(path + 'config.npy', allow_pickle=True).item()
    dmft1p = np.load(path + 'dmft1p.npy', allow_pickle=True).item()

    nk = config.k_grid.nk
    niv_urange = config.box.niv_urange
    niv_padded = config.box.niv_padded

    sigma_dga = np.load(path + 'sigma_dga.npy', allow_pickle=True).item()
    siw_dmft_clip = dmft1p['sloc'][dmft1p['niv']-niv_urange:dmft1p['niv']+niv_urange]
    #sigma = sigma_dga['dens'] + 3*sigma_dga['magn']  - 2*dmft_sde['magn'] + 0*dmft_sde['dens'] - dmft_sde['siw'] + siw_dmft_clip
    if(sigma_type == 'nc'):
        sigma = np.load(path + 'sigma_nc.npy', allow_pickle=True)
    else:
        sigma = np.load(path + 'sigma.npy', allow_pickle=True)
    #sigma = -sigma_dga['dens'] + 3*sigma_dga['magn']  -2* dmft_sde['magn'] +2* dmft_sde['dens']# - dmft_sde['siw'] + siw_dmft_clip

    sigma_node = sigma[nk[0] // 4, nk[1] // 4, 0, niv_padded:]
    sigma_anti_node = sigma[nk[0] // 2, 0, 0, niv_padded:]
    w = (config.box.vn_padded[niv_padded:] * 2 + 1) * np.pi / config.sys.beta
    #w = (config.box.vn_urange[niv_urange:] * 2 + 1) * np.pi / config.sys.beta
    return w, sigma_node, sigma_anti_node

def load_ldga_n_an_old(path=None, sloc= None):
    dmft_sde = np.load(path + 'dmft_sde.npy', allow_pickle=True).item()
    config = np.load(path + 'config.npy', allow_pickle=True).item()
    nk = config.k_grid.nk
    niv_urange = config.box.niv_urange
    sigma_dga = np.load(path + 'sigma_dga.npy', allow_pickle=True).item()
    niv = sloc.size // 2
    niv_urange = dmft_sde['siw'].size // 2
    dmft_sigma = sloc[niv - niv_urange:niv + niv_urange]
    #sigma = sigma_dga['dens'] + 3*sigma_dga['magn'] - 2*dmft_sde['magn'] - 0*dmft_sde['dens'] - dmft_sde['siw']+dmft_sigma
    sigma = -1*sigma_dga['dens'] + 3*sigma_dga['magn'] - 2*dmft_sde['magn']+ 2*dmft_sde['dens'] - dmft_sde['siw'] + dmft_sigma

    #sigma = np.load(path + 'sigma.npy', allow_pickle=True)
    sigma_node = sigma[nk[0] // 4, nk[1] // 4, 0, niv_urange:]
    sigma_anti_node = sigma[nk[0] // 2, 0, 0, niv_urange:]
    w = (config.box.vn_urange[niv_urange:] * 2 + 1) * np.pi / config.sys.beta
    return w, sigma_node, sigma_anti_node

def append_dga_output(w,s_n,s_an,path):
    a, b, c = load_ldga_n_an(path=path)
    w.append(a)
    s_n.append(b)
    s_an.append(c)
    return w, s_n, s_an

def append_dga_output_old(w,s_n,s_an,path,sloc):
    a, b, c = load_ldga_n_an_old(path=path,sloc=sloc)
    w.append(a)
    s_n.append(b)
    s_an.append(c)
    return w, s_n, s_an


# ----------------------------------------------- LOAD DATA ------------------------------------------------------------
input_path = '/mnt/d/Research/U2BenchmarkData/'
bm_path = input_path + 'DgaTsSpinOnly/'
bm_path_qmc = input_path + 'DiagMcBm/'


# There has been a mistake when extracting the data. It has hence to be scaled by:

def rescal(siw=None):
    return siw - (0.16 + siw) * 0.12 / 0.22


# Dga Spin-only Benchmark data:
dga_sp_n_065 = np.array(np.genfromtxt(bm_path + 'N_T_0.067.csv', delimiter=','))
dga_sp_n_063 = np.array(np.genfromtxt(bm_path + 'N_T_0.059.csv', delimiter=','))

dga_sp_an_065 = np.array(np.genfromtxt(bm_path + 'AN_T_0.067.csv', delimiter=','))
dga_sp_an_063 = np.array(np.genfromtxt(bm_path + 'AN_T_0.059.csv', delimiter=','))

# DiagMc Benchmark data:
dmc_n_065 = np.array(np.genfromtxt(bm_path_qmc + 'N_T_0.065.csv', delimiter=','))
dmc_n_063 = np.array(np.genfromtxt(bm_path_qmc + 'N_T_0.063.csv', delimiter=','))

dmc_an_065 = np.array(np.genfromtxt(bm_path_qmc + 'AN_T_0.065.csv', delimiter=','))
dmc_an_063 = np.array(np.genfromtxt(bm_path_qmc + 'AN_T_0.063.csv', delimiter=','))

# Lambda-Dda Data:
input_path_2 = '/mnt/d/Research/U2BenchmarkData/'
input_path_3 = '/mnt/d/Research/U2BenchmarkData/BenchmarkSchaefer_beta_15/LambdaDgaPython/'
w_dga_ed = []
sldga_n_ed = []
sldga_an_ed = []

nk_ed = [32,64,100,64,64,64,100]
nv_ed = [30,30,30,50,70,70,80]
w_dga_ed, sldga_n_ed, sldga_an_ed = append_dga_output(w_dga_ed, sldga_n_ed, sldga_an_ed,path=input_path_3+'LambdaDga_lc_sp_Nk1024_Nq1024_core30_invbse30_vurange30_wurange30/')
w_dga_ed, sldga_n_ed, sldga_an_ed = append_dga_output(w_dga_ed, sldga_n_ed, sldga_an_ed,path=input_path_3+'LambdaDga_lc_sp_Nk4096_Nq4096_core30_invbse30_vurange30_wurange30/')
w_dga_ed, sldga_n_ed, sldga_an_ed = append_dga_output(w_dga_ed, sldga_n_ed, sldga_an_ed,path=input_path_3+'LambdaDga_lc_sp_Nk10000_Nq10000_core30_invbse30_vurange30_wurange30/')
w_dga_ed, sldga_n_ed, sldga_an_ed = append_dga_output(w_dga_ed, sldga_n_ed, sldga_an_ed,path=input_path_3+'LambdaDga_lc_sp_Nk4096_Nq4096_core50_invbse50_vurange50_wurange50/')
w_dga_ed, sldga_n_ed, sldga_an_ed = append_dga_output(w_dga_ed, sldga_n_ed, sldga_an_ed,path=input_path_3+'LambdaDga_lc_sp_Nk4096_Nq4096_core70_invbse70_vurange70_wurange70/')
w_dga_ed, sldga_n_ed, sldga_an_ed = append_dga_output(w_dga_ed, sldga_n_ed, sldga_an_ed,path=input_path_3+'LambdaDga_lc_spch_Nk4096_Nq4096_core70_invbse70_vurange70_wurange70/')

path_bm_ts = input_path + 'BenchmarkSchaefer_beta_15/LambdaDgaPython/' + 'LambdaDga_lc_sp_Nk1024_Nq1024_core27_invbse27_vurange27_wurange27/'
dmft_sloc = np.load(path_bm_ts + 'config.npy', allow_pickle=True).item()['dmft1p']['sloc']
w_dga_ed, sldga_n_ed, sldga_an_ed = append_dga_output_old(w_dga_ed, sldga_n_ed, sldga_an_ed,path=input_path_3+'LambdaDga_lc_sp_Nk10000_Nq10000_core79_invbse80_vurange80_wurange79/',sloc=dmft_sloc)

# ------------------------------------------------- PLOTS --------------------------------------------------------------

shade_colors = ['blueviolet', 'firebrick', 'palegreen', 'dimgray', 'dimgray', 'cornflowerblue']
line_colors = ['indigo', 'green', 'black', 'blue', 'darkgray', 'k', 'cornflowerblue']
my_colors = ['#FFBF33', '#FF8033', '#E666A4', '#8A71F4', '#5BD0E9', 'dimgray']
colors = ['firebrick', '#FF8033', '#E666A4', '#5BD0E9', 'blueviolet', 'indigo']

fig, ax = plt.subplots(ncols=2, nrows=2, sharey=True, sharex=True, figsize=(8, 6))
ax = ax.flatten()

ax0 = ax[0]
ax1 = ax[1]
ax2 = ax[2]
ax3 = ax[3]

shift = 0.000
# Anti-Node:
ax0.plot(dga_sp_an_065[:, 0], rescal(dga_sp_an_065[:, 1]), '-o', color=colors[3])
for i, w in enumerate(w_dga_ed):
    ax0.plot(w*4, sldga_an_ed[i].imag*4+shift,'-s',ms=2,label=f'nk={nk_ed[i]}; nv={nv_ed[i]}')
ax0.set_xlim(0, 5)
ax0.legend()

# Anti-Node:
ax2.plot(dmc_an_065[:, 0], dmc_an_065[:, 1], '-o', color=colors[3])
for i, w in enumerate(w_dga_ed):
    ax2.plot(w*4, sldga_an_ed[i].imag*4+shift,'-s',ms=2,label=f'nk={nk_ed[i]}; nv={nv_ed[i]}')
ax2.set_xlim(0, 5)
ax2.legend()


# Node:
ax1.plot(dga_sp_n_065[:, 0], rescal(dga_sp_n_065[:, 1]), '-o', color=colors[3])
for i, w in enumerate(w_dga_ed):
    ax1.plot(w*4, sldga_n_ed[i].imag*4+shift,'-s',ms=2,label=f'nk={nk_ed[i]}; nv={nv_ed[i]}')
ax1.set_xlim(0, 5)
ax0.grid()
ax1.grid()
ax0.set_xlabel(r'$\omega_n$')
ax1.set_xlabel(r'$\omega_n$')
ax0.set_ylabel(r'$\Im \Sigma$')

# Node:
ax3.plot(dmc_n_065[:, 0], dmc_n_065[:, 1], '-o', color=colors[3])
for i, w in enumerate(w_dga_ed):
    ax3.plot(w*4, sldga_n_ed[i].imag*4+shift,'-s',ms=2,label=f'nk={nk_ed[i]}; nv={nv_ed[i]}')
ax3.set_xlim(0, 5)
ax3.legend()

ax0.set_title(r'Anti-Node')
ax1.set_title(r'Node')

plt.savefig('DGA_sp_benchmark_b15.png')
plt.savefig('DGA_sp_benchmark_b15.pdf')
plt.tight_layout()
plt.show()


# fig = plt.figure()
# x = np.linspace(-0.16,0.06,100)
# plt.plot(x,rescal(x))
# plt.show()


# fig = plt.figure()
# plt.imshow(ldga_066['sigma_dens'][:,:,0,config_066['box_sizes']['niv_urange']].imag,cmap='RdBu')
# plt.colorbar()
# plt.show()
#
#
# fig = plt.figure()
# plt.imshow(ldga_066['sigma_magn'][:,:,0,config_066['box_sizes']['niv_urange']].imag,cmap='RdBu')
# plt.colorbar()
# plt.show()
