# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Prepare the ED-DMFT input from Motoharu so it can be used by me as input.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys, os
import re
# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def load_config(fname=None):
    file = open(fname)
    file.readline()
    line = file.readline()
    #params = np.fromstring(line, dtype=float, sep='\t')
    params = re.findall("\d+\.\d+|\d+\.", line)
    params = [float(p) for p in params]
    return np.array(params)


def load_giw(path=None):
    input_path = find_input(path=path)
    giw = np.loadtxt(path + input_path + "gm_wim")
    giw = giw[:, 1] + 1j * giw[:, 2]
    giw = np.concatenate((np.flip(np.conj(giw)), giw))
    return giw

def prepare_1p_data(params=None, path=None, niv_plot=100):
    niv_plot = niv_plot
    input_path = find_input(path=path)
    giw = np.loadtxt(path + input_path + "gm_wim")
    giw = giw[:, 1] + 1j * giw[:, 2]
    g0mand = np.loadtxt(path + input_path + "g0mand")
    g0mand = g0mand[:, 1] + 1j * g0mand[:, 2]
    siw = g0mand - giw ** (-1.0)
    siw = np.concatenate((np.flip(np.conj(siw)), siw))
    giw = np.concatenate((np.flip(np.conj(giw)), giw))
    Niv1 = np.size(giw) // 2
    print(Niv1)
    f = h5py.File(path + '1p-data.hdf5', 'w')
    f['/dmft-last/ineq-001/giw/value'] = giw[None, None, :] * np.ones((1, 2, 1))
    f['/dmft-last/ineq-001/siw/value'] = siw[None, None, :] * np.ones((1, 2, 1))
    f.create_group('/.config')
    f['.config'].attrs['general.beta'] = params[2]
    f['.config'].attrs['atoms.1.udd'] = params[0]
    f['.config'].attrs['general.totdens'] = params[3]
    f['dmft-last/mu/value'] = params[1]
    f.close()

    plt.figure()
    plt.plot(giw.imag[Niv1 - niv_plot:Niv1 + niv_plot])
    plt.savefig(path + 'giw_imag.png')
    plt.show()

    plt.figure()
    plt.plot(siw.imag[Niv1 - niv_plot:Niv1 + niv_plot])
    plt.savefig(path + 'siw_imag.png')
    plt.show()

def create_chi_f_sanity_plots(path=None,giw=None,beta=None,Nv=60,Nw=59):
    Niv1 = np.size(giw) // 2
    iw = Nw
    iwb = (np.arange(-Nw, Nw + 1))
    chi_dir = find_chi_dir(path=path)
    chi = np.loadtxt(path + chi_dir + 'chi{:03}'.format(iw))
    chim = chi[:, 7].reshape((2 * Nv, 2 * Nv)) + 1j * chi[:, 8].reshape((2 * Nv, 2 * Nv)) - chi[:, 9].reshape(
        (2 * Nv, 2 * Nv)) - 1j * chi[:, 10].reshape((2 * Nv, 2 * Nv))
    Fm = chim / beta + giw[Niv1 - Nv:Niv1 + Nv, None] * giw[Niv1 - Nv - iwb[iw]:Niv1 + Nv - iwb[iw], None] * np.eye(
        2 * Nv)
    Fm = -1.0 * beta * Fm / (
                giw[Niv1 - Nv:Niv1 + Nv, None] * giw[Niv1 - Nv - iwb[iw]:Niv1 + Nv - iwb[iw], None] * giw[None,Niv1 - Nv:Niv1 + Nv] * giw[None,Niv1 - Nv -iwb[iw]:Niv1 + Nv -iwb[iw]])
    chim = chim / beta

    plt.figure()
    plt.subplot(221)
    plt.imshow(chim.real, cmap='RdBu', origin='lower')
    plt.colorbar()

    plt.subplot(222)
    plt.imshow(Fm.real, cmap='RdBu', origin='lower')
    plt.colorbar()
    plt.savefig(path + 'chim_fm_real.png')
    plt.show()
    chid = chi[:, 7].reshape((2 * Nv, 2 * Nv)) + 1j * chi[:, 8].reshape((2 * Nv, 2 * Nv)) + chi[:, 9].reshape(
        (2 * Nv, 2 * Nv)) + 1j * chi[:, 10].reshape((2 * Nv, 2 * Nv))
    chid = chid / beta + giw[Niv1 - Nv:Niv1 + Nv, None] * giw[None,
                                                          Niv1 - Nv:Niv1 + Nv]  # - giw[Niv1-Nv:Niv1+Nv,None] * giw[Niv1-Nv-iwb[iw]:Niv1+Nv-iwb[iw],None] * np.eye(2*Nv)

    plt.figure()
    plt.imshow(chid.real, cmap='RdBu', origin='lower')
    plt.colorbar()
    plt.savefig(path + 'chid_real.png')
    plt.show()




def prepare_g4iw_sym(params=None,path=None,Nv=60,Nw=59):
    giw = load_giw(path)
    Niv1 = np.size(giw) // 2
    beta = params[2]

    create_chi_f_sanity_plots(path=path, giw=giw, beta=beta, Nv=Nv, Nw=Nw)
    chi_dir = find_chi_dir(path=path)
    f = h5py.File(path + 'g4iw_sym.hdf5', 'w')
    iwb = (np.arange(-Nw, Nw + 1))
    f['/.axes/iwb-g4'] = iwb * 2
    f['/.axes/iwf-g4'] = (np.arange(-Nv, Nv))
    for iw in range(2 * Nw + 1):
        print(iw)
        chi = np.loadtxt(path + chi_dir + 'chi{:03}'.format(iw))
        chid = chi[:, 7].reshape((2 * Nv, 2 * Nv)) + 1j * chi[:, 8].reshape((2 * Nv, 2 * Nv)) + chi[:, 9].reshape(
            (2 * Nv, 2 * Nv)) + 1j * chi[:, 10].reshape((2 * Nv, 2 * Nv))
        chid = chid / beta  # - giw[Niv1-Nv:Niv1+Nv,None] * giw[Niv1-Nv-iwb[iw]:Niv1+Nv-iwb[iw],None] * np.eye(2*Nv)
        chim = chi[:, 7].reshape((2 * Nv, 2 * Nv)) + 1j * chi[:, 8].reshape((2 * Nv, 2 * Nv)) - chi[:, 9].reshape(
            (2 * Nv, 2 * Nv)) - 1j * chi[:, 10].reshape((2 * Nv, 2 * Nv))
        chim = chim / beta  # - giw[Niv1-Nv:Niv1+Nv,None] * giw[Niv1-Nv-iwb[iw]:Niv1+Nv-iwb[iw],None] * np.eye(2*Nv)
        if (iw == Nw):
            f['/ineq-001/dens/{:05}/00001/value'.format(2 * Nw - iw)] = chid + 2.0 * giw[Niv1 - Nv:Niv1 + Nv,None] * giw[None,Niv1 - Nv:Niv1 + Nv]
        else:
            f['/ineq-001/dens/{:05}/00001/value'.format(2 * Nw - iw)] = chid
        f['/ineq-001/magn/{:05}/00001/value'.format(2 * Nw - iw)] = chim
    f.close()

def get_nw_motoharu(path=None):
    chi_dir = find_chi_dir(path=path)
    chi_files = os.listdir(path + chi_dir)
    nw = len(chi_files) // 2
    nv = nw + 1
    return nw, nv

def find_chi_dir(path):
    if(os.path.exists(path + '/INPUT-for-DGA/chi_dir/')):
        chi_dir = '/INPUT-for-DGA/chi_dir/'
    elif(os.path.exists(path + '/chi_dir/')):
        chi_dir = '/chi_dir/'
    else:
        raise ValueError('Chi dir not found')

    return chi_dir

def find_input(path=None):
    if (os.path.exists(path + 'INPUT-for-DGA/chi_dir/')):
        input_path = 'INPUT-for-DGA/'
    else:
        input_path = ''
    return input_path
# -------------------------------- local Green's function ----------------------------------------------

path = '/mnt/d/Research/HoleDopedNickelates/'
subdirs = os.listdir(path)
for dir in subdirs:
    if(os.path.isdir(path+dir)):
        new_path = path+dir+'/from_Motoharu/'
        params = load_config(new_path + 'ladderDGA.in')
        if(os.path.exists(new_path+'1p-data.hdf5')):
            pass
        else:
            prepare_1p_data(params=params, path=new_path)

        if(os.path.exists(new_path + 'g4iw_sym.hdf5')):
            pass
        else:
            Nw, Nv = get_nw_motoharu(path=new_path)
            prepare_g4iw_sym(params=params, path=new_path, Nv=60, Nw=59)

