# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# WARNING: Currently there cannot be more processes than Niw*1+2. This is because my MpiDistibutor cannot handle slaves
# that have nothing to do.

# Warning: many of the files I prepared from Motoharu do not have smom stored.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import sys, os

sys.path.append('../src/')
sys.path.append(os.environ['HOME'] + "/Programs/dga/LambdaDga/src")
import BrillouinZone as bz
import EliashbergEquation as eq
import TwoPoint as twop

import matplotlib.pyplot as plt


# ----------------------------------------------- PARAMETERS -----------------------------------------------------------
input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.25_tpp0.12_beta12.5_n0.85/LambdaDga_lc_spch_Nk256_Nq256_core29_invbse30_vurange30_wurange29/'
output_path = input_path

# Collect Pairing vertex from subfiles:

import MpiAux as mpiaux
import h5py
import re

qiw_distributor = mpiaux.MpiDistributor(ntasks=box_sizes['niw_core'] * np.prod(nq), comm=comm,
                                        output_path=output_path,
                                        name='Qiw')

# Collect data from subfiles (This is quite ugly, as it is hardcoded to my structure. This should be replaced by a general routine):
f1_magn = np.zeros(nq + (niv_core, niv_core), dtype=complex)
f2_magn = np.zeros(nq + (niv_core, niv_core), dtype=complex)
f1_dens = np.zeros(nq + (niv_core, niv_core), dtype=complex)
f2_dens = np.zeros(nq + (niv_core, niv_core), dtype=complex)
if (qiw_distributor.my_rank == 0):
    file_out = h5py.File(fname_ladder_vertex, 'w')
    for ir in range(qiw_distributor.mpi_size):
        fname_input = output_path + 'QiwRank{:05d}'.format(ir) + '.hdf5'
        file_in = h5py.File(fname_input, 'r')
        for key1 in list(file_in.keys()):
            # extract the q indizes from the group name!
            qx = np.array(re.findall("\d+", key1), dtype=int)[0]
            qy = np.array(re.findall("\d+", key1), dtype=int)[1]
            qz = np.array(re.findall("\d+", key1), dtype=int)[2]
            condition = file_in[key1 + '/condition/'][()]
            f1_magn[qx, qy, qz, condition] = file_in[key1 + '/f1_magn/'][()]
            f2_magn[qx, qy, qz, condition] = file_in[key1 + '/f2_magn/'][()]
            f1_dens[qx, qy, qz, condition] = file_in[key1 + '/f1_dens/'][()]
            f2_dens[qx, qy, qz, condition] = file_in[key1 + '/f2_dens/'][()]

        file_in.close()
        os.remove(fname_input)
    file_out.close()


    import MpiAux as mpiaux
    import h5py
    import re

    qiw_distributor = mpiaux.MpiDistributor(ntasks=box_sizes['niw_core'] * np.prod(nq), comm=comm,
                                            output_path=output_path,
                                            name='Qiw')
    if (qiw_distributor.my_rank == 0):
        for ir in range(qiw_distributor.mpi_size):
            fname_input = output_path + 'QiwRank{:05d}'.format(ir) + '.hdf5'
            os.remove(fname_input)
else:
    pass


    import RealTime as rt

    realt = rt.real_time()

    log(realt.string_time('Start pairing vertex:'))

    import PairingVertex as pv

    niv_pp = niv_core // 2

    chi_dens_lambda = qiw_grid.reshape_matrix(mat=chi_lambda['chi_dens_lambda'].mat)
    chi_magn_lambda = qiw_grid.reshape_matrix(mat=chi_lambda['chi_magn_lambda'].mat)

    chi_dens_lambda_pp = pv.reshape_chi(chi=chi_dens_lambda, niv_pp=niv_pp)
    chi_magn_lambda_pp = pv.reshape_chi(chi=chi_magn_lambda, niv_pp=niv_pp)

    f_magn = f1_magn + (1 + dmft1p['u'] * chi_magn_lambda_pp) * f2_magn
    f_dens = f1_dens + (1 - dmft1p['u'] * chi_dens_lambda_pp) * f2_dens

    f_sing = -1.5 * f_magn + 0.5 * f_dens
    f_trip = -0.5 * f_magn - 0.5 * f_dens

    f_magn_loc = f_magn.mean(axis=(0, 1, 2))
    f1_magn_loc = f1_magn.mean(axis=(0, 1, 2))
    f2_magn_loc = f2_magn.mean(axis=(0, 1, 2))

    f_dens_loc = f_dens.mean(axis=(0, 1, 2))

    fig = plt.figure()
    plt.imshow(f_magn_loc.real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'f_magn_loc.png')
    plt.close()

    fig = plt.figure()
    plt.imshow(f_dens_loc.real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'f_dens_loc.png')
    plt.close()

    pairing_vertices = {
        'f_sing': f_sing,
        'f_trip': f_trip
    }

    np.save(output_path + 'pairing_vertices.npy', pairing_vertices)

    f_sing_loc = f_sing.mean(axis=(0, 1, 2))
    f_trip_loc = f_trip.mean(axis=(0, 1, 2))

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.imshow(f_sing_loc.real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'f_sing_loc.png')
    plt.close()

    fig = plt.figure()
    plt.imshow(f_trip_loc.real, cmap='RdBu')
    plt.colorbar()
    plt.savefig(output_path + 'f_trip_loc.png')
    plt.close()
