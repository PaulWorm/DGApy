# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Load results from DGA and Motoharu and constructs Benchmark plots:




# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import MatsubaraFrequencies as mf

# ---------------------------------------------- FUNCTIONS -------------------------------------------------------------

def read_wn_groups(gname=None,file=None):
    count = 1
    group = gname.format(count)
    mat = []
    while(group in file):
        mat.append(file[group][()])
        count +=1
        group = gname.format(count)
    return np.array(mat)

def load_fortran_chi_dmft(fname=None):
    file = h5py.File(fname)
    gname = './chi/dens/loc/wn{:05}'
    chi_dens = read_wn_groups(gname=gname,file=file)

    gname = './chi/magn/loc/wn{:05}'
    chi_magn = read_wn_groups(gname=gname,file=file)
    file.close()
    return chi_dens, chi_magn

def load_fortran_chi_ladder(fname=None):
    file = h5py.File(fname)
    chi_dens_ladder = file['./chi/dens/non_loc'][()]
    chi_magn_ladder = file['./chi/magn/non_loc'][()]

    file.close()
    return chi_dens_ladder, chi_magn_ladder

def load_fortran_chi_lambda(fname=None):
    file = h5py.File(fname)
    dens = file['./chi_lambda/dens/'][()]
    magn = file['./chi_lambda/magn/'][()]

    file.close()
    return dens, magn


# Parameters:

input_path_f = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/LambdaDga_Nk32_Nq32_new/'
input_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/LambdaDga_Nk64_Nq64_core59_urange200/'
output_path = input_path

dmft_sde = np.load(input_path+'dmft_sde.npy',allow_pickle=True).item()
gamma_dmft = np.load(input_path+'gamma_dmft.npy',allow_pickle=True).item()
chi_ladder = np.load(input_path+'chi_ladder.npy',allow_pickle=True).item()
chi_lambda = np.load(input_path+'chi_lambda.npy',allow_pickle=True).item()
config = np.load(input_path+'config.npy',allow_pickle=True).item()
nk = config['box_sizes']['nk']
niw_core = config['box_sizes']['niw_core']
chi_dens = dmft_sde['chi_dens']
chi_magn = dmft_sde['chi_magn']

chi_dens_ladder = chi_ladder['chi_dens_ladder'].mat.reshape(nk+(niw_core*2+1,))
chi_dens_ladder_loc = chi_dens_ladder.mean(axis=(0,1,2))
chi_magn_ladder = chi_ladder['chi_magn_ladder'].mat.reshape(nk+(niw_core*2+1,))
chi_magn_ladder_loc = chi_magn_ladder.mean(axis=(0,1,2))

chi_dens_lambda = chi_lambda['chi_dens_lambda'].mat.reshape(nk+(niw_core*2+1,))
chi_dens_lambda_loc = chi_dens_lambda.mean(axis=(0,1,2))
chi_magn_lambda = chi_lambda['chi_magn_lambda'].mat.reshape(nk+(niw_core*2+1,))
chi_magn_lambda_loc = chi_magn_lambda.mean(axis=(0,1,2))


chi_dens_f, chi_magn_f = load_fortran_chi_dmft(fname=input_path_f+'LadderDgaOut.hdf5')

chi_dens_ladder_f, chi_magn_ladder_f = load_fortran_chi_ladder(fname=input_path_f+'LadderDgaOut.hdf5')
chi_dens_lambda_f, chi_magn_lambda_f = load_fortran_chi_lambda(fname=input_path_f+'LadderDgaOut.hdf5')
chi_dens_ladder_loc_f =  np.mean(chi_dens_ladder_f, axis = 0)
chi_magn_ladder_loc_f =  np.mean(chi_magn_ladder_f, axis = 0)

chi_dens_lambda_loc_f =  np.mean(chi_dens_lambda_f, axis = 0)
chi_magn_lambda_loc_f =  np.mean(chi_magn_lambda_f, axis = 0)


fig = plt.figure()
plt.plot(chi_dens_lambda_loc, '-o')
plt.plot(chi_dens_lambda_loc_f, '-^')
plt.show()

fig = plt.figure()
plt.plot(chi_dens_ladder_loc, '-o')
plt.plot(chi_dens_ladder_loc_f, '-^')
plt.show()


fig = plt.figure()
plt.plot(chi_dens.mat.real,'-o')
plt.plot(chi_dens_f.real,'-^')
plt.show()

fig = plt.figure()
plt.plot(chi_magn.mat.real,'-o')
plt.plot(chi_magn_f.real,'-^')
plt.show()




