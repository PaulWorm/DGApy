import numpy as np
import matplotlib.pyplot as plt
import postproc.plot_siwk_cont as swk_anal
import dga.config as config


def get_folder_name(beta,totdens):
    return '/2DSquare_U8_tp-0.2_tpp0.1_beta{}_n{}'.format(beta,totdens)

base_path = '/mnt/d/Research/HoleDopedCuprates'
max_ent_dir = '/MaxEntSiwk_2/'
lambda_corr = 'spch'
beta = '27.5'
n = '0.925'
niv_core = 80
niw_core = niv_core
niv_shell = 600
nk = [100,100,1]
nq = nk

path = base_path + get_folder_name(beta,n)+\
       config.get_dga_output_folder_name(lambda_corr, np.prod(nk), np.prod(nq), niw_core,niv_core,niv_shell) + '/'

swk_anal.main(path,max_ent_dir)



