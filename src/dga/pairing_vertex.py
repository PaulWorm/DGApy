# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import dga.matsubara_frequencies as mf


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

# ======================================================================================================================

# --------------------------------------- FUNCTIONS TO CREATE THE LADDER VERTEX ----------------------------------------

def ladder_vertex_from_chi_aux(gchi_aux=None, vrg=None, chir=None, gchi0=None, beta=None, u_r=None):
    f = beta ** 2 * 1. / gchi0[...,:,None] * (np.eye(gchi0.shape[-1]) - gchi_aux * 1. / gchi0[...,None,:]) + u_r * (
                1.0 - u_r * chir[...,None,None]) * beta * vrg[...,:,None] * beta * vrg[...,None,:]
    return f

def ladder_vertex_from_chi_aux_components(gchi_aux=None, vrg=None, gchi0=None, beta=None, u_r=None):
    f1 = beta**2*1. / gchi0[...,:,None] * (np.eye(gchi0.shape[-1]) - gchi_aux * 1. / gchi0[...,None,:])
    f2 =  u_r * vrg[...,:,None] * vrg[...,None,:]
    return f1, f2

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
            iwn = niw + vi-vip
            # iwn = niw + vi+vip+1-wn
            vn = niv + vi
            vnp = niv - (vip+1)
            # vnp = niv + vip
            mat_pp[...,i,j] = mat_ph[...,iwn,vn,vnp]

    return mat_pp

def get_pp_slice_4pt(mat=None, condition=None, niv_pp=None):
    mat = np.flip(mat,axis=-1)
    mat_cut = mf.cut_iv_2d(arr=mat,niv_cut=niv_pp)
    slice = mat_cut[condition].flatten()
    return slice

# def get_pp_slice_4pt(mat=None, wn = None, niv_pp=None):
#     niv = np.shape(mat)[-1] // 2
#     ivn = np.arange(-niv_pp, niv_pp)
#     slice = []
#
#     for vi in ivn:
#         for vip in ivn:
#             vn = niv + vi
#             vnp = niv - (vip+1)
#             if((vi-vip) == wn):
#                 slice.append(mat[...,vn,vnp])
#     return np.array(slice)

def reshape_chi(chi=None, niv_pp=None):
    iv = np.arange(-niv_pp,niv_pp)
    niw = np.shape(chi)[-1] // 2
    chi_pp = np.zeros(chi.shape[:-1]+(niv_pp*2,niv_pp*2), dtype=complex)
    for i, ivn in enumerate(iv):
        for j, ivnp in enumerate(iv):
            wn = ivn - ivnp + niw
            chi_pp[:,:,:,i,j] = chi[:,:,:,wn]
    return chi_pp


def load_pairing_vertex_from_rank_files(output_path=None,name=None, mpi_size=None, nq=None, niv_pp=None):

    import h5py
    import re

    # Collect data from subfiles (This is quite ugly, as it is hardcoded to my structure. This should be replaced by a general routine):
    f1_magn = np.zeros((nq, 2*niv_pp, 2*niv_pp), dtype=complex)
    f2_magn = np.zeros((nq, 2*niv_pp, 2*niv_pp), dtype=complex)
    f1_dens = np.zeros((nq, 2*niv_pp, 2*niv_pp), dtype=complex)
    f2_dens = np.zeros((nq, 2*niv_pp, 2*niv_pp), dtype=complex)

    for ir in range(mpi_size):
        fname = output_path + name + 'Rank{0:05d}'.format(ir) + '.hdf5'
        file_in = h5py.File(fname, 'r')
        for key1 in list(file_in.file.keys()):
            # extract the q indizes from the group name!
            irrq = np.array(re.findall("\d+", key1), dtype=int)[0]
            condition = file_in.file[key1 + '/condition/'][()]
            f1_magn[irrq, condition] = file_in.file[key1 + '/f1_magn/'][()]
            f1_magn[irrq, condition.T] = file_in.file[key1 + '/f1_magn/'][()]
            f2_magn[irrq, condition] = file_in.file[key1 + '/f2_magn/'][()]
            f2_magn[irrq, condition.T] = file_in.file[key1 + '/f2_magn/'][()]
            f1_dens[irrq, condition] = file_in.file[key1 + '/f1_dens/'][()]
            f1_dens[irrq, condition.T] = file_in.file[key1 + '/f1_dens/'][()]
            f2_dens[irrq, condition] = file_in.file[key1 + '/f2_dens/'][()]
            f2_dens[irrq, condition.T] = file_in.file[key1 + '/f2_dens/'][()]

        file_in.close()

    #f1_magn = f1_magn + np.transpose(f1_magn,axes=(0,2,1)) - np.diagonal(f1_magn, axis1=1, axis2=2)

    return f1_magn, f2_magn, f1_dens, f2_dens

def get_omega_condition(niv_pp=None):
    ivn = np.arange(-niv_pp, niv_pp)
    omega = np.zeros((2 * niv_pp, 2 * niv_pp))
    for i, vi in enumerate(ivn):
        for j, vip in enumerate(ivn):
            omega[i, j] = vi - vip
    return omega

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

    slice = get_pp_slice_4pt(mat=f_magn[0,0,0,21,:,:],wn=mf.wnfind(niw=20,n=21))