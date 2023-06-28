import numpy as np
import dga.w2dyn_aux_dga as w2dyn_aux_dga
import dga.hr as hamr
import h5py
import dga.matsubara_frequencies as mf
import dga.local_four_point as lfp

# class InputDataFromDMFT():
#     ''' Simple container that contains necessary input data.'''
#     def __init__(self,giw,sigma,hr,mu,n,u,beta):
#         self.sigma = sigma
#         self.hr = hr
#         self.mu = mu
#         self.u = u
#         self.beta = beta
#         self.n = n
#         self.giw = giw

def load_w2dyn_data_set(path,file,hr,g2_file=None,load_g2=False):
    dmft_file = w2dyn_aux_dga.w2dyn_file(fname=path+file)
    siw = dmft_file.get_siw()[0, 0, :]
    ddict = {}
    ddict['beta'] = dmft_file.get_beta()
    ddict['u'] = dmft_file.get_udd()
    ddict['n'] = dmft_file.get_totdens()
    ddict['mu'] = dmft_file.get_mu()
    ddict['siw'] = siw
    ddict['giw'] = dmft_file.get_giw()[0,0,:]
    ddict['hr'] = hr
    if(load_g2):
        ddict['niw'] = g2_file.get_niw(channel='dens')
        ddict['wn'] = mf.wn(ddict['niw'])
        ddict['g2_dens'] = lfp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='dens', iw=ddict['wn']), beta=ddict['beta'], wn=ddict['wn'], channel='dens')
        ddict['g2_magn'] = lfp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='magn', iw=ddict['wn']), beta=ddict['beta'], wn=ddict['wn'], channel='magn')
    return ddict

def load_edfermion_data_set(path,file,hr,load_g2=False):
    file = h5py.File(path + file, 'r')

    ddict = {}
    ddict['beta'] = file['config/beta'][()]
    ddict['u'] = file['config/U'][()]
    ddict['n'] = file['config/totdens'][()]
    ddict['mu'] = file['dmft/mu'][()]
    ddict['giw'] = file['dmft/giw'][()]
    ddict['siw'] = 1/file['dmft/g0iw'][()] - 1/file['dmft/giw'][()]
    ddict['hr'] = hr
    if(load_g2):
        ddict['wn'] = mf.wn(np.size(file['iw4'][()]) // 2)
        ddict['g2_dens'] = lfp.LocalFourPoint(matrix=file['g4iw_dens'][()], beta=ddict['beta'], wn=ddict['wn'], channel='dens')
        ddict['g2_magn'] = lfp.LocalFourPoint(matrix=file['g4iw_magn'][()], beta=ddict['beta'], wn=ddict['wn'], channel='magn')
    return ddict



def get_data_set_1(load_g2=False):
    path = '../test/2DSquare_U8_tp-0.2_tpp0.1_beta17.5_n0.90/'
    file = '1p-data.hdf5'
    g2_file = 'g4iw_sym.hdf5'
    hr = hamr.one_band_2d_t_tp_tpp(1, -0.2, 0.1)
    return load_w2dyn_data_set(path,file,g2_file=g2_file,load_g2=load_g2,hr=hr)

def get_data_set_2(load_g2=False):
    path = '../test/2DSquare_U8_tp-0.25_tpp0.12_beta12.5_n0.85/'
    file = '1p-data.hdf5'
    g2_file = 'g4iw_sym.hdf5'
    hr = hamr.one_band_2d_t_tp_tpp(1, -0.25, 0.12)# Motoharus data
    return load_w2dyn_data_set(path,file,g2_file=g2_file,load_g2=load_g2,hr=hr)

def get_data_set_3(load_g2=False):
    path = '../test/2DSquare_U8_tp-0.2_tpp0.1_beta22.5_n0.90/'
    file = '1p-data.hdf5'
    g2_file = 'g4iw_sym.hdf5'
    hr = hamr.one_band_2d_t_tp_tpp(1, -0.2, 0.1)# Motoharus data
    return load_w2dyn_data_set(path,file,g2_file=g2_file,load_g2=load_g2,hr=hr)

def get_data_set_4(load_g2=False):
    path = '../test/2DSquare_U8_tp-0.2_tpp0.1_beta5_n0.90/'
    file = 'EDFermion_GreenFunctions_nbath_3.hdf5'
    hr = hamr.one_band_2d_t_tp_tpp(1, -0.2, 0.1)
    return load_edfermion_data_set(path,file,load_g2=load_g2,hr=hr)

def get_data_set_5(load_g2=False):
    path = '../test/2DSquare_U8_tp-0.2_tpp0.1_beta2_n0.90/'
    file = 'EDFermion_1p-data.hdf5'
    hr = hamr.one_band_2d_t_tp_tpp(1, -0.2, 0.1)
    return load_edfermion_data_set(path,file,load_g2=load_g2,hr=hr)

def get_data_set_6(load_g2=False):
    path = '../test/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
    file = 'EDFermion_1p-data.hdf5'
    hr = hamr.one_band_2d_t_tp_tpp(1, -0.2, 0.1)
    return load_edfermion_data_set(path,file,load_g2=load_g2,hr=hr)

def get_data_set_7(load_g2=False):
    path = '../test/2DSquare_U2_tp-0.0_tpp0.0_beta15_mu1/'
    file = '1p-data.hdf5'
    hr = hamr.one_band_2d_t_tp_tpp(1, -0.0, 0.0)
    return load_w2dyn_data_set(path,file,load_g2=load_g2,hr=hr)

def get_data_set_8(load_g2=False):
    path = '../test/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.85/'
    file = '1p-data.hdf5'
    hr = hamr.one_band_2d_t_tp_tpp(0.25, -0.0625, 0.03)
    return load_w2dyn_data_set(path,file,load_g2=load_g2,hr=hr)