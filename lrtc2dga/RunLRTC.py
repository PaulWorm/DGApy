import numpy as np
import matplotlib.pyplot as plt
import h5py
import os, sys
import shutil
from  Output import uniquify

if __name__ == '__main__':

    input_folder = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.85/LambdaDga_lc_sp_Nk10000_Nq10000_core60_invbse60_vurange250_wurange60/'
    output_folder = input_folder + 'LinReTraCe/'
    #output_folder = uniquify(output_folder)
    try:
        os.mkdir(output_folder)
    except:
        pass

    # Parameters:
    nk = (100,100,1)
    n = 0.85

    # Tight-binding file: Just copy for now. Generate  automatically later
    tb_file = 'tight_binding_tp-0.2_tpp0.1.txt'
    shutil.copy(tb_file,output_folder)

    # Create the lrtc tight-binding file on a predefined k-mesh:
    ltb_dga.main()
    # os.system('conda activate lrtc')
    # os.system('ltb ' + output_folder + tb_file + f'{nk[0]} {nk[1]} {nk[2]} {n} ' + '--red')