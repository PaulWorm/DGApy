import numpy as np
import w2dyn_aux
import Hr as hamr

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
    dmft_file = w2dyn_aux.w2dyn_file(fname=path+file)
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
        ddict['g2_file'] = w2dyn_aux.g4iw_file(fname=path + g2_file)
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
    hr = hamr.one_band_2d_t_tp_tpp(0.25, -0.25*0.25, 0.12*0.25)# Motoharus data
    return load_w2dyn_data_set(path,file,g2_file=g2_file,load_g2=load_g2,hr=hr)

def get_data_set_3(load_g2=False):
    path = '../test/2DSquare_U8_tp-0.2_tpp0.1_beta22.5_n0.90/'
    file = '1p-data.hdf5'
    g2_file = 'g4iw_sym.hdf5'
    hr = hamr.one_band_2d_t_tp_tpp(1, -0.2, 0.1)# Motoharus data
    return load_w2dyn_data_set(path,file,g2_file=g2_file,load_g2=load_g2,hr=hr)