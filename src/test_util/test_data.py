import os
import numpy as np
import dga.w2dyn_aux_dga as w2dyn_aux_dga
import dga.wannier as wannier
import h5py
from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import local_four_point as lfp

CURRDIR = os.path.split(__file__)[0]
PATH_FOR_TEST_HR_HK = os.path.dirname(__file__) + '/../../tests/TestHRAndHkFiles/'

def load_w2dyn_data_set(path,file,hr,g2_file=None,chi_file=None,load_g2=False, load_chi=False):
    dmft_file = w2dyn_aux_dga.W2dynFile(fname=path + file)
    siw = dmft_file.get_siw()[0, 0, :]
    ddict = {}
    ddict['beta'] = dmft_file.get_beta()
    ddict['u'] = dmft_file.get_udd()
    ddict['n'] = dmft_file.get_totdens()
    ddict['mu'] = dmft_file.get_mu()
    ddict['siw'] = siw
    ddict['giwk_obj'] = dmft_file.get_giw()[0,0,:]
    ddict['hr'] = hr
    if(load_g2):
        g2_file = w2dyn_aux_dga.W2dynG4iwFile(fname=path + g2_file)
        ddict['niw'] = g2_file.get_niw(channel='dens')
        ddict['wn'] = mf.wn(ddict['niw'])
        ddict['g2_dens'] = lfp.LocalFourPoint(channel='dens', mat=g2_file.read_g2_iw(channel='dens', iw=ddict['wn']),
                                              beta=ddict['beta'], wn=ddict['wn'])
        ddict['g2_magn'] = lfp.LocalFourPoint(channel='magn', mat=g2_file.read_g2_iw(channel='magn', iw=ddict['wn']),
                                              beta=ddict['beta'], wn=ddict['wn'])
    if(load_chi):
        chi_file = w2dyn_aux_dga.W2dynFile(fname=path + chi_file)
        ddict['chi_dens'] = chi_file.get_chi(channel='dens')
        ddict['chi_magn'] = chi_file.get_chi(channel='magn')
    return ddict

def load_edfermion_data_set(path,file,hr,load_g2=False,g2_file=None,chi_file=None, load_chi=False):
    file = h5py.File(path + file, 'r')

    ddict = {}
    ddict['beta'] = file['config/beta'][()]
    ddict['u'] = file['config/U'][()]
    ddict['n'] = file['config/totdens'][()]
    ddict['mu'] = file['dmft/mu'][()]
    ddict['giwk_obj'] = file['dmft/giwk_obj'][()]
    ddict['siw'] = 1/file['dmft/g0iw'][()] - 1/file['dmft/giwk_obj'][()]
    ddict['hr'] = hr

    if(load_g2):
        g2_file = h5py.File(path + g2_file,'r')
        ddict['wn'] = mf.wn(len(g2_file['iw4'])//2)
        ddict['g2_dens'] = lfp.LocalFourPoint(channel='dens', mat=g2_file['g4iw_dens'][()], beta=ddict['beta'])
        ddict['g2_magn'] = lfp.LocalFourPoint(channel='magn', mat=g2_file['g4iw_magn'][()], beta=ddict['beta'])

    if(load_chi):
        chi_file = h5py.File(path + chi_file,'r')
        ddict['chi_dens'] = chi_file['chi_dens'][()]
        ddict['chi_magn'] = chi_file['chi_magn'][()]

    return ddict



def get_data_set_1(load_g2=False):
    path = '../../tests/2DSquare_U8_tp-0.2_tpp0.1_beta17.5_n0.90/'
    file = '1p-data.hdf5'
    g2_file = 'g4iw_sym.hdf5'
    hr = wannier.one_band_2d_t_tp_tpp(1, -0.2, 0.1)
    return load_w2dyn_data_set(path,file,g2_file=g2_file,load_g2=load_g2,hr=hr)

def get_data_set_2(load_g2=False):
    path = '../../tests/2DSquare_U8_tp-0.25_tpp0.12_beta12.5_n0.85/'
    file = '1p-data.hdf5'
    g2_file = 'g4iw_sym.hdf5'
    hr = wannier.one_band_2d_t_tp_tpp(1, -0.25, 0.12)# Motoharus data
    return load_w2dyn_data_set(path,file,g2_file=g2_file,load_g2=load_g2,hr=hr)

def get_data_set_3(load_g2=False):
    path = '../../tests/2DSquare_U8_tp-0.2_tpp0.1_beta22.5_n0.90/'
    file = '1p-data.hdf5'
    g2_file = 'g4iw_sym.hdf5'
    hr = wannier.one_band_2d_t_tp_tpp(1, -0.2, 0.1)# Motoharus data
    return load_w2dyn_data_set(path,file,g2_file=g2_file,load_g2=load_g2,hr=hr)

def get_data_set_4(load_g2=False):
    path = '../../tests/2DSquare_U8_tp-0.2_tpp0.1_beta5_n0.90/'
    file = 'EDFermion_GreenFunctions_nbath_3.hdf5'
    hr = wannier.one_band_2d_t_tp_tpp(1, -0.2, 0.1)
    return load_edfermion_data_set(path,file,load_g2=load_g2,hr=hr)

def get_data_set_5(load_g2=False):
    path = '../../tests/2DSquare_U8_tp-0.2_tpp0.1_beta2_n0.90/'
    file = 'EDFermion_1p-data.hdf5'
    hr = wannier.one_band_2d_t_tp_tpp(1, -0.2, 0.1)
    return load_edfermion_data_set(path,file,load_g2=load_g2,hr=hr)

def get_data_set_6(load_g2=False,load_chi=False):
    path = '../../tests/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
    file = 'EDFermion_1p-data.hdf5'
    hr = wannier.one_band_2d_t_tp_tpp(1, -0.2, 0.1)
    chi_file = 'EDFermion_chi.hdf5'
    g2_file = 'EDFermion_g4iw_sym.hdf5'
    return load_edfermion_data_set(path,file,load_g2=load_g2,hr=hr,load_chi=load_chi,chi_file=chi_file,g2_file=g2_file)

def get_data_set_7(load_g2=False):
    path = '../../tests/2DSquare_U2_tp-0.0_tpp0.0_beta15_mu1/'
    file = '1p-data.hdf5'
    hr = wannier.one_band_2d_t_tp_tpp(1, -0.0, 0.0)
    return load_w2dyn_data_set(path,file,load_g2=load_g2,hr=hr)

def get_data_set_8(load_g2=False):
    path = '../../tests/2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.85/'
    file = '1p-data.hdf5'
    hr = wannier.one_band_2d_t_tp_tpp(0.25, -0.0625, 0.03)
    return load_w2dyn_data_set(path,file,load_g2=load_g2,hr=hr)

def get_data_set_9(load_g2=False,load_chi=False):
    path = CURRDIR + '/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
    file = '1p-data.hdf5'
    g2_file = 'g4iw_sym.hdf5'
    chi_file = 'chi.hdf5'
    t,tp,tpp = 1,-0.2,0.1
    hr = wannier.WannierHr(*wannier.wannier_one_band_2d_t_tp_tpp(t,tp,tpp))
    return load_w2dyn_data_set(path,file,g2_file=g2_file,load_g2=load_g2,load_chi=load_chi,chi_file = chi_file,hr=hr)

def load_minimal_dataset():
    # path = '../tests/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
    path = '/../../tests/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
    path = os.path.dirname(__file__) + path
    dmft_data = np.load(path + 'minimal_dataset.npy',allow_pickle=True).item()
    dmft_data['sym'] = bz.two_dimensional_square_symmetries()
    hr = wannier.create_wannier_hr_from_file(path + 'wannier_hr.dat')
    return dmft_data, hr

def load_quasi_1d_dataset():
    path = '/../../tests/LaSr2NiO3_beta80_n0.85/'
    path = os.path.dirname(__file__) + path
    dmft_data = np.load(path + 'quasi_1d_dataset.npy',allow_pickle=True).item()
    dmft_data['sym'] = bz.simultaneous_x_y_inversion()
    hr = wannier.create_wannier_hr_from_file(path + 'wannier_hr.dat')
    return dmft_data, hr

def load_minimal_dataset_ed():
    path = '/../../tests/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
    path = os.path.dirname(__file__) + path
    dmft_data = np.load(path + 'minimal_dataset_ed_fermion.npy',allow_pickle=True).item()
    dmft_data['sym'] = bz.two_dimensional_square_symmetries()
    hr = wannier.create_wannier_hr_from_file(path + 'wannier_hr.dat')
    return dmft_data, hr

def load_ht_eliashberg_input():
    path = '/../../tests/2DSquare_U2_tp-0.0_tpp0.0_beta3_mu1/'
    path = os.path.dirname(__file__) + path
    dmft_data = np.load(path + 'eliashberg_input.npy',allow_pickle=True).item()
    hr = wannier.WannierHr(*wannier.wannier_one_band_2d_t_tp_tpp(1,0,0))
    return dmft_data, hr

def load_minimal_eliashberg_input():
    path = '/../../tests/2DSquare_U8_tp-0.2_tpp0.1_beta12.5_n0.90/'
    path = os.path.dirname(__file__) + path
    dmft_data = np.load(path + 'eliashberg_input.npy',allow_pickle=True).item()
    hr = wannier.create_wannier_hr_from_file(path + 'wannier_hr.dat')
    return dmft_data, hr

def load_testdataset(input_type='minimal'):
    if input_type == 'minimal':
        ddict, hr = load_minimal_dataset()
    elif input_type == 'ed_minimal':
        ddict, hr = load_minimal_dataset_ed()
    elif input_type == 'quasi_1d':
        ddict, hr = load_quasi_1d_dataset()
    else:
        raise ValueError(f'Unknown input type: {input_type}')
    return ddict, hr