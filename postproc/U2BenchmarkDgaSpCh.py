# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Benchmark against the data from Thomas Schäfers Spin-fluctuation paper
# (https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.011058)


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import MatsubaraFrequencies as mf
import h5py
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

def load_ldga_n_an(path=None):
    dga_sde = np.load(path + 'dga_sde.npy', allow_pickle=True).item()
    dmft_sde = np.load(path + 'dmft_sde.npy', allow_pickle=True).item()
    config = np.load(path + 'config.npy', allow_pickle=True).item()
    nk = config['box']['nk']
    niv_urange = config['box']['niv_urange']
    #sigma = dga_sde['sigma']
    sigma = dga_sde['sigma']
    dmft1p = config['dmft1p']
    dmft_sigma = config['dmft1p']['sloc'][dmft1p['niv']-niv_urange:dmft1p['niv']+niv_urange]
    sigma = -1*dga_sde['sigma_dens'] + 3*dga_sde['sigma_magn'] - 2*dmft_sde['siw_magn'] + 2*dmft_sde['siw_dens'] - dmft_sde['siw']+dmft_sigma
    #sigma = dga_sde['sigma_dens'] + 3*dga_sde['sigma_magn'] - 2*dmft_sde['siw_magn'] - 0*dmft_sde['siw_dens'] - dmft_sde['siw']+dmft_sigma
    sigma_node = sigma[nk[0]//4,nk[1]//4,0,niv_urange:]
    sigma_anti_node = sigma[nk[0]//2,0,0,niv_urange:]
    w = (config['grids']['vn_urange'][niv_urange:] * 2+1) * np.pi/config['dmft1p']['beta']
    return w, sigma_node, sigma_anti_node

# ----------------------------------------------- LOAD DATA ------------------------------------------------------------
input_path = '/mnt/c/users/pworm/Research/U2BenchmarkData/'
bm_path = input_path + 'DgaTsBoth/'

# DiagMc Benchmark data:
dmc_n_1 = np.array(np.genfromtxt(bm_path + 'N_T_1.0.csv', delimiter=','))
dmc_n_03 = np.array(np.genfromtxt(bm_path + 'N_T_0.33.csv', delimiter=','))
dmc_n_01 = np.array(np.genfromtxt(bm_path + 'N_T_0.1.csv', delimiter=','))
dmc_n_065 = np.array(np.genfromtxt(bm_path + 'N_T_0.067.csv', delimiter=','))
dmc_n_063 = np.array(np.genfromtxt(bm_path + 'N_T_0.05.csv', delimiter=','))

dmc_an_1 = np.array(np.genfromtxt(bm_path + 'AN_T_1.0.csv', delimiter=','))
dmc_an_03 = np.array(np.genfromtxt(bm_path + 'AN_T_0.33.csv', delimiter=','))
dmc_an_01 = np.array(np.genfromtxt(bm_path + 'AN_T_0.1.csv', delimiter=','))
dmc_an_065 = np.array(np.genfromtxt(bm_path + 'AN_T_0.067.csv', delimiter=','))
dmc_an_063 = np.array(np.genfromtxt(bm_path + 'AN_T_0.05.csv', delimiter=','))


# Lambda-Dda Data:

path_1 = input_path + '2DSquare_U2_tp-0.0_tpp0.0_beta1_mu1/' + 'LambdaDga_Nk14400_Nq14400_core30_urange60/'
ldga_1, config_1 = load_ldga_data(path_1)
w1, sldga_n_1, sldga_an_1= load_ldga_n_an(path_1)

path_03 = input_path + '2DSquare_U2_tp-0.0_tpp0.0_beta3_mu1/' + 'LambdaDga_Nk14400_Nq14400_core30_urange60/'
ldga_03, config_03 = load_ldga_data(path_03)
w03, sldga_n_03, sldga_an_03= load_ldga_n_an(path_03)

path_01 = input_path + '2DSquare_U2_tp-0.0_tpp0.0_beta10_mu1/' + 'LambdaDga_Nk14400_Nq14400_core30_urange60/'
ldga_01, config_01 = load_ldga_data(path_01)
chi_lambda = load_ldga_chi_lambda(path_01)
chi_ladder = load_ldga_chi_ladder(path_01)
w01, sldga_n_01, sldga_an_01= load_ldga_n_an(path_01)


# nk = config_01['box']['nk']
# niw_core = config_01['box']['niw_core']
# chi_magn_ladder = chi_ladder['chi_magn_ladder'].mat.reshape(nk+(niw_core*2+1,))
# fig = plt.figure()
# plt.imshow(chi_magn_ladder[:,:,0,niw_core].real, cmap='RdBu')
# plt.colorbar()
# plt.show()



path_066 = input_path + '2DSquare_U2_tp-0.0_tpp0.0_beta15_mu1/' + 'LambdaDga_Nk14400_Nq14400_core30_urange60/'
path_066 = input_path + '2DSquare_U2_tp-0.0_tpp0.0_beta15_mu1/' + 'LambdaDga_lc_spch_Nk6400_Nq6400_core30_invbse30_vurange100_wurange100/'
ldga_066, config_066 = load_ldga_data(path_066)
w066, sldga_n_066, sldga_an_066= load_ldga_n_an(path_066)


path_063 = input_path + '2DSquare_U2_tp-0.0_tpp0.0_beta17_mu1/' + 'LambdaDga_Nk14400_Nq14400_core30_urange60/'
ldga_063, config_0v = load_ldga_data(path_063)
w063, sldga_n_063, sldga_an_063= load_ldga_n_an(path_063)

# ------------------------------------------------- PLOTS --------------------------------------------------------------

shade_colors = ['blueviolet', 'firebrick', 'palegreen', 'dimgray', 'dimgray','cornflowerblue']
line_colors = ['indigo', 'green', 'black', 'blue', 'darkgray', 'k','cornflowerblue']
my_colors = ['#FFBF33','#FF8033','#E666A4','#8A71F4','#5BD0E9','dimgray']
colors = ['firebrick','#FF8033','#E666A4','#5BD0E9','blueviolet','indigo']


fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=True, figsize=(10,4))

ax0 = ax[0]
ax1 = ax[1]

# Anti-Node:
ax0.plot(dmc_an_1[0],dmc_an_1[1], '-o', color = colors[0])
ax0.plot(dmc_an_03[:,0],dmc_an_03[:,1], '-o', color = colors[1])
ax0.plot(dmc_an_01[:,0],dmc_an_01[:,1], '-o', color = colors[2])
ax0.plot(dmc_an_065[:,0],dmc_an_065[:,1], '-o', color = colors[3])
ax0.plot(dmc_an_063[:,0],dmc_an_063[:,1], '-o', color = colors[4])

ax0.plot(w1,sldga_an_1.imag,'-s', color=colors[0], ms=2, markeredgecolor='k')
ax0.plot(w03,sldga_an_03.imag,'-s', color=colors[1], ms=2, markeredgecolor='k')
ax0.plot(w01,sldga_an_01.imag,'-s', color=colors[2], ms=2, markeredgecolor='k')
ax0.plot(w066,sldga_an_066.imag,'-s', color=colors[3], ms=2, markeredgecolor='k')
#ax0.plot(w063,sldga_an_063.imag,'-s', color=colors[4], ms=2, markeredgecolor='k')
ax0.set_xlim(0,5)

# Node:
ax1.plot(dmc_n_1[0],dmc_n_1[1], '-o', color = colors[0])
ax1.plot(dmc_n_03[:,0],dmc_n_03[:,1], '-o', color = colors[1])
ax1.plot(dmc_n_01[:,0],dmc_n_01[:,1], '-o', color = colors[2])
ax1.plot(dmc_n_065[:,0],dmc_n_065[:,1], '-o', color = colors[3])
ax1.plot(dmc_n_063[:,0],dmc_n_063[:,1], '-o', color = colors[4])

ax1.plot(w1,sldga_n_1.imag,'-s', color=colors[0], ms=2, markeredgecolor='k')
ax1.plot(w03,sldga_n_03.imag,'-s', color=colors[1], ms=2, markeredgecolor='k')
ax1.plot(w01,sldga_n_01.imag,'-s', color=colors[2], ms=2, markeredgecolor='k')
ax1.plot(w066,sldga_n_066.imag,'-s', color=colors[3], ms=2, markeredgecolor='k')
#ax1.plot(w063,sldga_n_063.imag,'-s', color=colors[4], ms=2, markeredgecolor='k')
ax1.set_xlim(0,5)

ax0.set_xlabel(r'$\omega_n$')
ax1.set_xlabel(r'$\omega_n$')
ax0.set_ylabel(r'$\Im \Sigma$')

ax0.set_title(r'Anti-Node')
ax1.set_title(r'Node')

plt.tight_layout()
plt.show()


# fig = plt.figure()
# plt.imshow(ldga_066['sigma_dens'][:,:,0,config_066['box']['niv_urange']].imag,cmap='RdBu')
# plt.colorbar()
# plt.show()
#
#
# fig = plt.figure()
# plt.imshow(ldga_066['sigma_magn'][:,:,0,config_066['box']['niv_urange']].imag,cmap='RdBu')
# plt.colorbar()
# plt.show()







