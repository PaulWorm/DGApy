import numpy as np
import matplotlib.pyplot as plt
import w2dyn_aux_dga
import MatsubaraFrequencies as mf
import FourPoint as fp
import TwoPoint_old as tp
import sys, os
import Output

path = '/mnt/d/Research/VertexConventionsTestCase/TestInputPaul/'
path = '/mnt/d/Research/HoleDopedPalladates/2DSquare_U7_tp-0.184_tpp0.014_beta400_n0.95/'
# path = '/mnt/d/Research/VertexConventionsTestCase/TestInputMotoharu/'
fname_1p = path + '1p-data.hdf5'
fname_2p = path + 'g4iw_sym.hdf5'

dmft_file = w2dyn_aux.w2dyn_file(fname=fname_1p)
g2_file = w2dyn_aux_dga.g4iw_file(fname=fname_2p)

convert_unit = 1.0
# Extract system parameter:
beta = dmft_file.get_beta()
giw_raw = np.mean(dmft_file.get_giw()[0], axis=0)
giw_dmft = tp.LocalGreensFunction(mat=giw_raw, beta=beta)
niw = g2_file.get_niw(channel='dens')
wn_core = mf.wn(n=niw)
# g2_dens = g2_file.read_g2_iw(channel='dens', iw=wn_core)
# g2_magn = g2_file.read_g2_iw(channel='magn', iw=wn_core)

g2_dens = fp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='dens', iw=wn_core), channel='dens',
                            beta=beta, wn=wn_core)
gchi_dens = fp.chir_from_g2(g2=g2_dens, giw=giw_dmft)

g2_magn = fp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='magn', iw=wn_core), channel='magn',
                            beta=beta, wn=wn_core)
gchi_magn = fp.chir_from_g2(g2=g2_magn, giw=giw_dmft)

chi0_inv = fp.LocalBubble(wn=wn_core, giw=giw_dmft, niv=g2_dens.niv, is_inv=True, do_chi0=False, do_shell=False)
f_dens = fp.Fob2_from_chir(chir=gchi_dens, chi0_inv=chi0_inv)
f_magn = fp.Fob2_from_chir(chir=gchi_magn, chi0_inv=chi0_inv)

f_dens._mat = f_dens._mat * convert_unit ** 3
f_dens._beta = f_dens._beta / convert_unit
f_magn._beta = f_magn._beta / convert_unit
f_magn._mat = f_magn._mat * convert_unit ** 3
# gamma_dens = fp.gamob2_from_chir(chir = gchi_dens, chi0_inv = chi0_inv)
# gamma_magn = fp.gamob2_from_chir(chir = gchi_magn, chi0_inv = chi0_inv)

# gamma_dens._mat = gamma_dens.mat - dmft_file.get_udd()
# gamma_magn._mat = gamma_magn.mat + dmft_file.get_udd()

g2_file.close()
dmft_file.close()

# Generate default plots:
gchi_dens.plot(name='Chi-dens', pdir=path)
gchi_magn.plot(name='Chi-magn', pdir=path)

f_dens.plot(name='F-dens', pdir=path)
f_magn.plot(name='F-magn', pdir=path)


# gamma_dens.plot(name='Gamma-dens',pdir=path)
# gamma_magn.plot(name='Gamma-magn',pdir=path)



def write_files(data_dens, data_magn, path, name):
    full = []
    w_core = mf.w(beta=data_dens.beta, n=niw)
    for i, _ in enumerate(wn_core):
        print(i)
        vn_base = mf.v(beta=data_dens.beta, n=data_dens.niv)
        vn = np.repeat(vn_base, data_dens.niv * 2)
        vnp = np.tile(vn_base, data_dens.niv * 2)
        iwn = w_core[i] * np.ones_like(vn)
        dens = data_dens.mat[i, ...].flatten()
        magn = data_magn.mat[i, ...].flatten()
        tmp = np.stack((iwn, vn, vnp, dens.real, dens.imag, magn.real, magn.imag), axis=1)
        if (i == 0):
            full = np.copy(tmp)
        else:
            full = np.concatenate((full, tmp), axis=0)
    np.savetxt(fname=path + name, X=full, fmt='%.10f', delimiter='\t')


write_files(data_dens=f_dens, data_magn=f_magn, path=path, name='F_DM')
print('File written!')
#
# chi_dir = path + '/chi_dir/'
# chi_dir = Output.uniquify(chi_dir)
# if(os.path.isdir(chi_dir)):
#     print('Chi directory already exists. Skipping chi.')
# else:
#     os.mkdir(chi_dir)
#     write_files(gchi_dens,gchi_magn,path=chi_dir,name='chi')
#
#
# gamma_dir = path + '/gamma_dir/'
# gamma_dir = Output.uniquify(gamma_dir)
# if(os.path.isdir(gamma_dir)):
#     print('Chi directory already exists. Skipping gamma.')
# else:
#     os.mkdir(gamma_dir)
#     write_files(gamma_dens,gamma_magn,path=gamma_dir,name='gamma')
