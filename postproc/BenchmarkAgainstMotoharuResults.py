# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Load results from DGA and Motoharu and constructs Benchmark plots:




# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import dga.matsubara_frequencies as mf


# Parameters:

input_path_m = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/RawDataMotoharu/'
input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/LambdaDga_Nk64_Nq64_core59_urange500/'
output_path = input_path

# Load Python Data:

config = np.load(input_path+'config.npy', allow_pickle=True).item()
dga_sde = np.load(input_path+'dga_sde.npy', allow_pickle=True).item()
dmft_sde = np.load(input_path+'dmft_sde.npy', allow_pickle=True).item()
gk_dga = np.load(input_path + 'gk_dga.npy', allow_pickle=True).item()

chi_dmft_dens = dmft_sde['chi_dens']
chi_dmft_magn = dmft_sde['chi_magn']

niv = config['box']['niv_urange']
iv = config['grids']['vn_urange']
nk = config['grids']['_k_grid'].nk[0]
sigma = dga_sde['sigma']
hartree = config['dmft1p']['n'] * config['dmft1p']['u']/2
sigma = np.roll(sigma, nk // 2, 0)
sigma = np.roll(sigma, nk // 2, 1)
sigma_node = sigma[nk//4,nk//4,0,:]
sigma_anti_node = sigma[nk//2,0,0,:]

# Load Motoharu Data:
dict_moto = np.load(input_path_m + 'siwk.npy', allow_pickle=True).item()
sigma_moto = dict_moto['siwk']
sigma_moto = np.concatenate((np.flip(np.conj(sigma_moto)),sigma_moto),axis=0)
nk_moto = dict_moto['Nk']
sigma_moto_node = sigma_moto[:,nk_moto//4,nk_moto//4]
sigma_moto_anti_node = sigma_moto[:,nk_moto//2,0]


niv_moto = dict_moto['Niv']
iv_moto = mf.vn(niv_moto)

# Plots:

vn_list = [config['grids']['vn_urange'],config['grids']['vn_urange'],iv_moto,iv_moto ]
siw_list = [sigma_node,sigma_anti_node,sigma_moto_node,sigma_moto_anti_node]
labels = [r'$\Sigma_{N,P}(\nu)$', r'$\Sigma_{AN,P}(\nu)$', r'$\Sigma_{N,M}(\nu)$', r'$\Sigma_{AN,M}(\nu)$',]
#plotting.plot_siw(vn_list=vn_list, siw_list=siw_list, labels_list=labels, plot_dir=output_path, niv_plot=10, name='siwk_check_with_Motoharu_data', ncol=2)

# Compare Node and Anti-node:
#%%
fig = plt.figure()
plt.plot(iv, sigma_node.imag, 'ro', label=labels[0], ms=4)
plt.plot(iv, sigma_anti_node.imag, 'bo', label=labels[1], ms=4)
plt.plot(iv_moto, sigma_moto_node.imag, 'gs', label=labels[2], ms=2)
plt.plot(iv_moto, sigma_moto_anti_node.imag, 'gs', label=labels[3],ms=2)
plt.xlim([0,10])
plt.legend()
plt.savefig(output_path + 'BM_Motoharu_imag.png')
plt.show()

fig = plt.figure()
plt.plot(iv, sigma_node.real, 'ro', label=labels[0], ms=4)
plt.plot(iv, sigma_anti_node.real, 'bo', label=labels[1], ms=4)
plt.plot(iv_moto, sigma_moto_node.real+hartree, 'gs', label=labels[2], ms=2)
plt.plot(iv_moto, sigma_moto_anti_node.real+hartree, 'gs', label=labels[3],ms=2)
plt.xlim([0,10])
plt.legend()
plt.savefig(output_path + 'BM_Motoharu_real.png')
plt.show()


#%%
plt.imshow(sigma[:,:,0,niv].imag, cmap='RdBu', origin='lower')
plt.colorbar()
plt.show()

plt.imshow(sigma_moto[niv_moto,:,:].imag, cmap='RdBu', origin='lower')
plt.colorbar()
plt.savefig(output_path + 'siwk_Motoharu.png')
plt.show()

print(f'{np.sum(chi_dmft_dens.mat)/16=}')
print(f'{np.sum(chi_dmft_magn.mat)/16=}')
print((np.sum(chi_dmft_dens.mat)+np.sum(chi_dmft_magn.mat))/(16*2))