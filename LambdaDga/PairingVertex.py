# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import w2dyn_aux
import numpy as np
import TwoPoint as tp
import FourPoint as fp


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

# ======================================================================================================================

# --------------------------------------- FUNCTIONS TO CREATE THE LADDER VERTEX ----------------------------------------

def ladder_vertex_from_chi_aux(gchi_aux=None, vrg=None, chir=None, gchi0=None, beta=None, u_r=None):
    f = beta ** 2 * 1. / gchi0[...,:,None] * (np.eye(gchi0.shape[-1]) - gchi_aux * 1. / gchi0[...,None,:]) + u_r * (
                1.0 - u_r * chir[...,None,None]) * beta * vrg[...,:,None] * beta * vrg[...,None,:]
    return f

def ph_to_pp_notation(mat_ph=None,wn=0):
    if(wn != 0):
        raise NotImplementedError('wn!=0 not yet implemented')
    assert len(np.shape(mat_ph))>=3, 'Input does not have more than or equal three dimensions as required.'
    niw = mat_ph.shape[-3] // 2
    niv = mat_ph.shape[-1] // 2

    niv_pp = np.min((niw//2,niv//2))
    iv = np.arange(-niv_pp,niv_pp)
    mat_pp = np.zeros(mat_ph.shape[:-3]+(2*niv_pp,2*niv_pp), dtype=complex)
    for i, vi in enumerate(iv):
        for j, vip in enumerate(iv):
            wn = niw + vi-vip
            vn = niv + vi
            vnp = niv - (vip+1)
            mat_pp[...,i,j] = mat_ph[...,wn,vn,vnp]

    return mat_pp


# def get_chi_aux_asympt(chi_aux: fp.FourPoint = None, chi_r_urange: , chi_r_asympt=None, u=1):
#     niv = np.shape(chi_aux)[-1] // 2
#     u_mat = u * np.ones((2 * niv, 2 * niv), dtype=complex)
#     return chi_aux - np.matmul(chi_aux, np.matmul(u_mat, chi_aux)) * (
#                 (1 - u * chi_r_urange) - (1 - u * chi_r_urange) ** 2 / (1 - u * chi_r_asympt))


# def get_f_ladder_from_chir(chi_r:, chi_aux_r=None, vrg=None, gchi0=None, u=1, beta=1):
#     niv = chi_aux_r.shape[-1] // 2
#     unity = np.eye(2*niv, dtype=complex)
#     gchi0_inv = 1./ gchi0
#     return beta ** 2 * gchi0_inv[:,None] * (unity - chi_aux_r * gchi0_inv[None,:]) + u * (1.0 - u * chi_r) * beta * \
#            vrg[:,None] * beta * vrg[None,:]


if __name__ == '__main__':
    import h5py
    import matplotlib.pyplot as plt
    test_path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/LambdaDga_4/'
    fname = test_path + 'LadderVertex.hdf5'
    file = h5py.File(fname, 'r')

    def load_qiw(key1=None):
        arr = []
        for key2 in list(file[key1].keys()):
            arr.append(file[key1 + '/' + key2][()])
        return np.array(arr)


    gchi_aux_magn = load_qiw(key1='gchi_aux_magn')

    gchi0 = load_qiw(key1='gchi0')
    vrg_magn = load_qiw(key1='vrg_magn')
    dga_sde = np.load(test_path+'dga_sde.npy', allow_pickle=True).item()
    chi_magn_lambda = dga_sde['chi_magn_lambda'].mat
    beta = dga_sde['chi_magn_lambda'].beta
    u_r = dga_sde['chi_magn_lambda'].u_r
    file.close()

    niv_core = gchi_aux_magn.shape[-1] // 2
    niv_urange = gchi0.shape[-1] // 2
    gchi0 = gchi0[:,niv_urange-niv_core:niv_urange+niv_core]
    vrg_magn = vrg_magn[:,niv_urange-niv_core:niv_urange+niv_core]

    f_magn = ladder_vertex_from_chi_aux(gchi_aux=gchi_aux_magn, vrg=vrg_magn, chir=chi_magn_lambda, gchi0=gchi0, beta=beta, u_r=u_r)
    f_magn = f_magn.reshape(8,8,1,41,40,40)
    f_magn_loc = f_magn.reshape(8,8,1,41,40,40).mean(axis=(0,1,2))
    f_magn_loc_pp = ph_to_pp_notation(mat_ph=f_magn_loc)
    f_magn_pp = ph_to_pp_notation(mat_ph=f_magn)

    plt.imshow(f_magn_loc_pp.real,cmap='RdBu')
    plt.colorbar()
    plt.show()

    vrg_magn_loc = vrg_magn.reshape(-1,41,40).mean(axis=0)
    gchi_aux_magn_loc = gchi_aux_magn.reshape(-1,41,40,40).mean(axis=0)

    plt.imshow(f_magn_loc[20,:,:].real,cmap='RdBu')
    plt.colorbar()
    plt.show()

    plt.imshow(vrg_magn_loc[:,:].real,cmap='RdBu')
    plt.colorbar()
    plt.show()

    plt.imshow(gchi_aux_magn_loc[20,:,:].real,cmap='RdBu')
    plt.colorbar()
    plt.show()
