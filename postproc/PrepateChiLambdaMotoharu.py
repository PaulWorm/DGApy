import numpy as np
import matplotlib.pyplot as plt

def get_chi_folder(path,channel):
    return path + f'chi{channel}_omega/'

path_base = '/mnt/d/Research/HubbardModel_tp-0.0_tpp0.0/from_Motoharu/2DHubbard-U2.0-tp0.0-tpp0.0-beta400-ntot1.0/RESULT/'
channel = 'sp'
dirname = get_chi_folder(path_base,channel)
print(dirname)

tmp = np.loadtxt(dirname + 'chi001')
