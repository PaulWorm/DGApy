# ------------------------------------------------ COMMENTS ------------------------------------------------------------
'''
    Module that contained routines to construct the pairing vertex from the ladder vertex.
'''

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import gc  # garbage collector
import re

import h5py
import numpy as np

from dga import matsubara_frequencies as mf
from dga import config as conf
from dga import mpi_aux
from dga import four_point as fp
from dga import plotting
from dga import local_four_point as lfp


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

# ======================================================================================================================

# --------------------------------------- FUNCTIONS TO CREATE THE LADDER VERTEX ----------------------------------------

def ladder_vertex_from_chi_aux(gchi_aux=None, vrg=None, chir=None, gchi0=None, beta=None, u_r=None):
    f = beta ** 2 * 1. / gchi0[..., :, None] * (np.eye(gchi0.shape[-1]) - gchi_aux * 1. / gchi0[..., None, :]) + u_r * (
            1.0 - u_r * chir[..., None, None]) * beta * vrg[..., :, None] * beta * vrg[..., None, :]
    return f





def get_pp_slice_4pt(mat, condition, niv_pp):
    '''
        mat: (iw,v,vp) slice of 4pt function at given iw with v,vp as axis
    '''
    mat = np.flip(mat, axis=-1)  # flip to get mat(v,-vp)
    mat_cut = mf.cut_v(arr=mat, niv_cut=niv_pp, axes=(-2, -1))
    # minus sign comes from the crossing symmetry that is used
    mat_slice = -mat_cut[condition].flatten()  # Take mat(v-v'[from condition],v,-v'); use only v,v' where v-v' = w
    return mat_slice


def reshape_chi(chi, niv_pp):
    '''
        Index transformation: chi(iw) -> chi(v-vp) needed for F(v-vp,v,-vp) = ... (1 - u_r chi(v-vp)) ...
        (See PHYSICAL REVIEW B 99, 041115(R) (2019) S.5 in combination with S.8 - S.10)
    '''

    iv = np.arange(-niv_pp, niv_pp)
    niw = np.shape(chi)[-1] // 2
    chi_pp = np.zeros(chi.shape[:-1] + (niv_pp * 2, niv_pp * 2), dtype=complex)
    for i, ivn in enumerate(iv):
        for j, ivnp in enumerate(iv):
            wn = ivn - ivnp + niw
            chi_pp[:, :, :, i, j] = chi[:, :, :, wn]  # [kx,ky,kz,iw = v-vp] -> [kx,ky,kz,v,vp]
    return chi_pp


def get_omega_condition(niv_pp):
    '''
        Look in supplemental of Phys. Rev. B 99, 041115(R) (S.8 - S.10). This takes care of the index shift for
        Gamma(q=0,k,kp) = F(k-kp,v-vp,v,-vp)
     '''
    ivn = np.arange(-niv_pp, niv_pp)
    omega = np.zeros((2 * niv_pp, 2 * niv_pp))
    for i, vi in enumerate(ivn):
        for j, vip in enumerate(ivn):
            omega[i, j] = vi - vip
    return omega

def write_pairing_vertex_components(d_cfg: conf.DgaConfig, mpi_distributor: mpi_aux.MpiDistributor, channel, gchiq_aux,
                                    vrg_q, gchi0_q_core):
    niv_pp = d_cfg.box.niv_pp
    omega = get_omega_condition(niv_pp=niv_pp)
    with mpi_distributor as file:
        for i, iq in enumerate(mpi_distributor.my_tasks):
            for j, iw in enumerate(d_cfg.box.wn):
                if np.abs(iw) < 2 * niv_pp:
                    condition = omega == iw

                    f1_slice, f2_slice = fp.ladder_vertex_from_chi_aux_components(gchi_aux=gchiq_aux[i, j],
                                                                                  vrg=vrg_q[i, j],
                                                                                  gchi0=gchi0_q_core[i, j],
                                                                                  beta=d_cfg.sys.beta,
                                                                                  u_r=lfp.get_ur(d_cfg.sys.u, channel))
                    group = f'/pv_irrq{iq:03d}wn{iw:04d}/'

                    file[group + f'f1_{channel}/'] = get_pp_slice_4pt(mat=f1_slice, condition=condition,
                                                                      niv_pp=niv_pp)
                    file[group + f'f2_{channel}/'] = get_pp_slice_4pt(mat=f2_slice, condition=condition,
                                                                      niv_pp=niv_pp)
                    if group + 'condition' not in file:
                        file[group + 'condition/'] = condition


def load_pairing_vertex_from_rank_files(output_path=None, name=None, mpi_size=None, nq=None, niv_pp=None):

    # Collect data from subfiles (This is quite ugly, as it is hardcoded to my structure.
    # This should be replaced by a general routine):
    f1_magn = np.zeros((nq, 2 * niv_pp, 2 * niv_pp), dtype=complex)
    f2_magn = np.zeros((nq, 2 * niv_pp, 2 * niv_pp), dtype=complex)
    f1_dens = np.zeros((nq, 2 * niv_pp, 2 * niv_pp), dtype=complex)
    f2_dens = np.zeros((nq, 2 * niv_pp, 2 * niv_pp), dtype=complex)

    for ir in range(mpi_size):
        fname = output_path + name + f'Rank{ir:05d}' + '.hdf5'
        with h5py.File(fname, 'r') as file_in:
            keys = (key for key in file_in.keys() if key.startswith('pv_'))
            for key1 in keys:
                # extract the q indizes from the group name!
                irrq = np.array(re.findall(r'\d+', key1), dtype=int)[0]
                condition = file_in.file[key1 + '/condition/'][()]
                f1_magn[irrq, condition] += 0.5 * file_in.file[key1 + '/f1_magn/'][()]
                f2_dens[irrq, condition] += 0.5 * file_in.file[key1 + '/f2_dens/'][()]
                f1_dens[irrq, condition] += 0.5 * file_in.file[key1 + '/f1_dens/'][()]
                f2_magn[irrq, condition] += 0.5 * file_in.file[key1 + '/f2_magn/'][()]

                # transpose condition to get vp-v = w instead of v-vp = w
                # this is a symmetrization with respect to v-vp
                # if g4iw is already symmetric, this should not be necessary
                f1_magn[irrq, condition.T] += 0.5 * np.conj(file_in.file[key1 + '/f1_magn/'][()])
                f2_magn[irrq, condition.T] += 0.5 * np.conj(file_in.file[key1 + '/f2_magn/'][()])
                f1_dens[irrq, condition.T] += 0.5 * np.conj(file_in.file[key1 + '/f1_dens/'][()])
                f2_dens[irrq, condition.T] += 0.5 * np.conj(file_in.file[key1 + '/f2_dens/'][()])

    # f1_magn = f1_magn + np.transpose(f1_magn,axes=(0,2,1)) - np.diagonal(f1_magn, axis1=1, axis2=2)

    return f1_magn, f2_magn, f1_dens, f2_dens


def build_pairing_vertex(d_cfg: conf.DgaConfig, comm, chi_lad_magn, chi_lad_dens):
    f1_magn, f2_magn, f1_dens, f2_dens = load_pairing_vertex_from_rank_files(output_path=d_cfg.output_path, name='Q',
                                                                             mpi_size=comm.size,
                                                                             nq=d_cfg.lattice.q_grid.nk_irr,
                                                                             niv_pp=d_cfg.box.niv_pp)
    chi_magn_lambda_pp = reshape_chi(chi_lad_magn, d_cfg.box.niv_pp)
    f1_magn = d_cfg.lattice.q_grid.map_irrk2fbz(f1_magn, shape='mesh')
    f2_magn = d_cfg.lattice.q_grid.map_irrk2fbz(f2_magn, shape='mesh')
    f_magn = f1_magn + (1 + d_cfg.sys.u * chi_magn_lambda_pp) * f2_magn

    chi_dens_lambda_pp = reshape_chi(chi_lad_dens, d_cfg.box.niv_pp)
    f1_dens = d_cfg.lattice.q_grid.map_irrk2fbz(f1_dens, shape='mesh')
    f2_dens = d_cfg.lattice.q_grid.map_irrk2fbz(f2_dens, shape='mesh')
    f_dens = f1_dens + (1 - d_cfg.sys.u * chi_dens_lambda_pp) * f2_dens

    # Build singlet and triplet vertex:
    f_sing = +1.5 * f_magn - 0.5 * f_dens
    f_trip = +0.5 * f_magn + 0.5 * f_dens

    plotting.plot_kx_ky(f_sing[:, :, 0, d_cfg.box.niv_pp, d_cfg.box.niv_pp], d_cfg.lattice.k_grid.kx,
                        d_cfg.lattice.k_grid.ky, pdir=d_cfg.eliash.output_path, name='F_sing_pp')

    plotting.plot_kx_ky(f_trip[:, :, 0, d_cfg.box.niv_pp, d_cfg.box.niv_pp], d_cfg.lattice.k_grid.kx,
                        d_cfg.lattice.k_grid.ky, pdir=d_cfg.eliash.output_path, name='F_trip_pp')

    f_sing_loc = d_cfg.lattice.q_grid.k_mean(f_sing, 'fbz-mesh')
    f_trip_loc = d_cfg.lattice.q_grid.k_mean(f_trip, 'fbz-mesh')

    lfp.plot_fourpoint_nu_nup(f_sing_loc, pdir=d_cfg.eliash.output_path, name='F_sing_pp_loc')
    lfp.plot_fourpoint_nu_nup(f_trip_loc, pdir=d_cfg.eliash.output_path, name='F_trip_pp_loc')

    d_cfg.eliash.save_data(f_sing, 'F_sing_pp')
    d_cfg.eliash.save_data(f_trip, 'F_trip_pp')
    del f_sing, f_trip, f_dens, f_magn
    gc.collect()


if __name__ == '__main__':
    pass
