# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Check the convergence of the results with the number of bosonic momenta points.


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import MatsubaraFrequencies as mf
import h5py
import pandas as pd
# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def load_lambdas(path=None):
    file = np.load(path + 'chi_lambda.npy', allow_pickle=True).item()
    lambda_magn = file['lambda_magn']
    lambda_dens = file['lambda_dens']
    return lambda_dens,lambda_magn

def load_sigma(path=None):
    dga_sde = np.load(path + 'dga_sde.npy', allow_pickle=True).item()
    dmft_sde = np.load(path + 'dmft_sde.npy', allow_pickle=True).item()
    config = np.load(path + 'config.npy', allow_pickle=True).item()
    nk = config['box_sizes']['nk']
    niv_urange = config['box_sizes']['niv_urange']
    # sigma = dga_sde['sigma']
    dmft1p = config['dmft1p']
    dmft_sigma = config['dmft1p']['sloc'][dmft1p['niv'] - niv_urange:dmft1p['niv'] + niv_urange]
    sigma = -1 * dga_sde['sigma_dens'] + 3 * dga_sde['sigma_magn'] - 2 * dmft_sde['siw_magn'] + 2 * dmft_sde[
        'siw_dens'] - dmft_sde['siw'] + dmft_sigma
    # sigma = dga_sde['sigma_dens'] + 3*dga_sde['sigma_magn'] - 2*dmft_sde['siw_magn'] - 0*dmft_sde['siw_dens'] - dmft_sde['siw']+dmft_sigma
    sigma_magn_node = dga_sde['sigma_magn'][nk[0] // 4, nk[1] // 4, 0, niv_urange:]
    sigma_magn_anti_node = dga_sde['sigma_magn'][nk[0] // 2, 0, 0, niv_urange:]
    sigma_dens_node = dga_sde['sigma_dens'][nk[0] // 4, nk[1] // 4, 0, niv_urange:]
    sigma_dens_anti_node = dga_sde['sigma_dens'][nk[0] // 2, 0, 0, niv_urange:]
    w = (config['grids']['vn_urange'][niv_urange:] * 2 + 1) * np.pi / config['dmft1p']['beta']
    return w, sigma_dens_node, sigma_dens_anti_node, sigma_magn_node, sigma_magn_anti_node
# ----------------------------------------------- LOAD DATA ------------------------------------------------------------

input_path = '/mnt/c/users/pworm/Research/U2BenchmarkData/2DSquare_U2_tp-0.0_tpp0.0_beta15_mu1/'


data_1 = 'LambdaDga_Nk64_Nq64_core8_urange100/'
lambda_dens_1, lambda_magn_1 = load_lambdas(input_path+data_1)
w1, sigma_dens_node_1, sigma_dens_anti_node_1, sigma_magn_node_1, sigma_magn_anti_node_1 = load_sigma(input_path+data_1)

data_2 = 'LambdaDga_Nk256_Nq256_core8_urange100/'
lambda_dens_2, lambda_magn_2 = load_lambdas(input_path+data_2)
w2, sigma_dens_node_2, sigma_dens_anti_node_2, sigma_magn_node_2, sigma_magn_anti_node_2 = load_sigma(input_path+data_2)

data_3 = 'LambdaDga_Nk576_Nq576_core8_urange100/'
lambda_dens_3, lambda_magn_3 = load_lambdas(input_path+data_3)
w3, sigma_dens_node_3, sigma_dens_anti_node_3, sigma_magn_node_3, sigma_magn_anti_node_3 = load_sigma(input_path+data_3)


data_4 = 'LambdaDga_Nk1024_Nq1024_core8_urange100/'
lambda_dens_4, lambda_magn_4 = load_lambdas(input_path+data_4)
w4, sigma_dens_node_4, sigma_dens_anti_node_4, sigma_magn_node_4, sigma_magn_anti_node_4 = load_sigma(input_path+data_4)


nq = [64,256,576,1024]
lambda_magn = [lambda_magn_1,lambda_magn_2,lambda_magn_3,lambda_magn_4]
lambda_dens = [lambda_dens_1,lambda_dens_2,lambda_dens_3,lambda_dens_4]

fig = plt.figure()
plt.plot(nq,lambda_magn, label=r'$\Lambda_{magn}$')
plt.plot(nq,lambda_dens,label=r'$\Lambda_{dens}$')
plt.legend()
plt.show()

fig = plt.figure()
plt.plot(w1,sigma_magn_node_1.imag,'-o', label='Node Nq:{}'.format(nq[0]))
plt.plot(w2,sigma_magn_node_2.imag,'-o', label='Node Nq:{}'.format(nq[1]))
plt.plot(w3,sigma_magn_node_3.imag,'-o', label='Node Nq:{}'.format(nq[2]))
plt.plot(w4,sigma_magn_node_4.imag,'-o', label='Node Nq:{}'.format(nq[3]))
plt.legend()
plt.xlim(-1,10)
plt.show()

fig = plt.figure()
plt.plot(w1,sigma_magn_anti_node_1.imag,'-o', label='Anti-Node Nq:{}'.format(nq[0]))
plt.plot(w2,sigma_magn_anti_node_2.imag,'-o', label='Anti-Node Nq:{}'.format(nq[1]))
plt.plot(w3,sigma_magn_anti_node_3.imag,'-o', label='Anti-Node Nq:{}'.format(nq[2]))
plt.plot(w4,sigma_magn_anti_node_4.imag,'-o', label='Anti-Node Nq:{}'.format(nq[3]))
plt.legend()
plt.xlim(-1,10)
plt.show()

fig = plt.figure()
plt.plot(w1,sigma_dens_node_1.imag,'-o', label='Node Nq:{}'.format(nq[0]))
plt.plot(w2,sigma_dens_node_2.imag,'-o', label='Node Nq:{}'.format(nq[1]))
plt.plot(w3,sigma_dens_node_3.imag,'-o', label='Node Nq:{}'.format(nq[2]))
plt.plot(w4,sigma_dens_node_4.imag,'-o', label='Node Nq:{}'.format(nq[3]))
plt.legend()
plt.xlim(-1,10)
plt.show()

fig = plt.figure()
plt.plot(w1,sigma_dens_anti_node_1.imag,'-o', label='Anti-Node Nq:{}'.format(nq[0]))
plt.plot(w2,sigma_dens_anti_node_2.imag,'-o', label='Anti-Node Nq:{}'.format(nq[1]))
plt.plot(w3,sigma_dens_anti_node_3.imag,'-o', label='Anti-Node Nq:{}'.format(nq[2]))
plt.plot(w4,sigma_dens_anti_node_4.imag,'-o', label='Anti-Node Nq:{}'.format(nq[3]))
plt.legend()
plt.xlim(-1,10)
plt.show()




