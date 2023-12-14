'''
    Module for computing the optical conductivity.
'''

import gc  # garbage collector

import matplotlib.pyplot as plt
import numpy as np
import h5py
import re
from scipy.interpolate import interp1d


from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import wannier
from dga import two_point as twop
from dga import local_four_point as lfp
from dga import config as conf
from dga import plotting
from dga import four_point as fp
from dga import mpi_aux


def get_chijj_bubble(giwk, lv_a, lv_b, beta, iwn, niv_sum,
                     freq_notation='minus'):
    '''
        iwn: single bosonic Matsubara frequency index
        lv_a = del e_k/ del k_a
        sigma_0[q=0,omega] = - 2/beta sum_k lv_a(k) lv_b(k)  G(v-w,k) * G(v,k)
        Note: at the moment only q=0 is implemented
    '''

    iws, iws2 = mf.get_freq_shift(iwn, freq_notation)
    niv_giw = mf.niv_from_mat(giwk, axis=-1, pos=False)
    return - 2 / beta * np.mean((lv_a * lv_b) * np.sum(giwk[..., niv_giw - niv_sum + iws:niv_giw + niv_sum + iws] *
                                                       giwk[..., niv_giw - niv_sum + iws2:niv_giw + niv_sum + iws2],
                                                       axis=-1))


def get_chijj_bubble_realf():
    pass


def fermi_function(w, beta):
    return 1 / (np.exp(beta * w) + 1)


def vec_get_chijj_bubble_realf(gwk_obj: twop.RealFrequencyGF, hr: wannier.WannierHr, k_grid: bz.KGrid, beta, w_max_bub, der_a=0,
                               der_b=0):
    '''
        der: index of momentum for derivative (0: kx, 1: ky, 2: kz)
        w_max_bub: chijj is computed up to w_max_bub
        Equation from Eq. (5) of P. Worm et al. Phys. Rev. B 104, 115153 (2021):
        chi_jj(w) = - 2 sum_k lv_a(k) lv_b(k) int dv nf(v) A(k,v) [GR(k,v+w) + GA(k,v-w)]
    '''
    assert np.max(gwk_obj.w) > w_max_bub, r'w_max_bub is larger than the maximum frequency in the Green\'s function'
    lv_a = hr.get_light_vertex(k_grid, der=der_a)
    lv_b = hr.get_light_vertex(k_grid, der=der_b)
    w_0_ind = np.argmin(np.abs(gwk_obj.w))
    w_mbub_ind = np.argmin(np.abs(gwk_obj.w - w_max_bub))
    w_int_ind = len(gwk_obj.w) - w_mbub_ind
    nu = w_0_ind + w_int_ind
    nl = w_0_ind - w_int_ind + len(gwk_obj.w) % 2
    lv2 = lv_a * lv_b
    awk = gwk_obj.awk()[..., nl:nu]
    w_int = gwk_obj.w[nl:nu]
    nf = fermi_function(w_int, beta)[None, None, None, :]
    w_bub = gwk_obj.w[w_0_ind + 1:w_mbub_ind]

    def get_chijj_bubble_realf_iw(i):
        return -2 * np.mean(
            lv2 * np.trapz(nf * awk * (gwk_obj.gwk()[..., nl + i:nu + i] + np.conj(gwk_obj.gwk())[..., nl - i:nu - i])
                           , w_int, axis=-1))

    chi_jj = np.array([get_chijj_bubble_realf_iw(i + 1) for i, _ in enumerate(w_bub)])
    return chi_jj, w_bub


def get_sigma_from_chijj_realf(chi_jj, w):
    ''' Get sigma = chi/w including interpolation for the w=0 value.'''
    w_full = np.concatenate((-w[::-1], w))
    chi_jj_full = np.concatenate((np.conj(chi_jj)[::-1].imag, chi_jj.imag))
    interp = interp1d(w_full, chi_jj_full / w_full, kind='cubic')
    w_new = np.concatenate(((0,), w))
    return np.array(interp(w_new)), w_new


def vec_get_sigma_bub_realf(gwk_obj: twop.RealFrequencyGF, hr: wannier.WannierHr, k_grid: bz.KGrid, beta, w_max_bub, der_a=0,
                            der_b=0):
    '''
        der: index of momentum for derivative (0: kx, 1: ky, 2: kz)
        w_max_bub: chijj is computed up to w_max_bub
        Equation from Eq. (2) of P. Worm et al. Phys. Rev. B 104, 115153 (2021):
        sigma(w) = Im(chi_jj(w)) / w
    '''
    chi_jj_bub, w_bub = vec_get_chijj_bubble_realf(gwk_obj, hr, k_grid, beta, w_max_bub, der_a, der_b)
    return get_sigma_from_chijj_realf(chi_jj_bub, w_bub)


def vec_get_chijj_bubble(giwk_gen: twop.GreensFunction, hr: wannier.WannierHr, k_grid: bz.KGrid, wn, niv_sum, der_a=0,
                         der_b=0,
                         freq_notation='minus'):
    '''
        der: index of momentum for derivative (0: kx, 1: ky, 2: kz)
        wn: list of Matsubara frequencies
    '''
    lv_a = hr.get_light_vertex(k_grid, der=der_a)
    lv_b = hr.get_light_vertex(k_grid, der=der_b)
    beta = giwk_gen.beta
    giwk = giwk_gen.g_full()
    return np.array([get_chijj_bubble(giwk, lv_a, lv_b, beta, iwn, niv_sum, freq_notation=freq_notation) for iwn in wn])


def get_chijj_vertex(f_cond, lv_a, lv_b, giwk, beta, iwn, wpn, q_list, nk_tot):
    '''
        Equation from Matthias Pickem's notes:
        sigma_vert(w)  = -2/beta^2 * sum_{q,k,wp,v} lv_a(k-q) lv_b(k) G(v-w,k) G(v,k) G(v-wp,k-q) G(v-wp-w,k-q)
                                                 * [-1/2 F_d(q,wp,v-w,v) - 3/2 F_m(q,wp,v-w,v)]
         The additional minus sign stems from the different sign conventions in the vertex
    '''
    sigma_vert = 0.0
    niv_vert = mf.niv_from_mat(f_cond, axis=-1, pos=False)
    niv_g = mf.niv_from_mat(giwk, axis=-1, pos=False)
    nl, nu = niv_g - niv_vert, niv_g + niv_vert  # lower and upper bounds
    for i, q in enumerate(q_list):
        lv_b_kpq = bz.shift_mat_by_ind(lv_b, ind=[-iq for iq in q])
        giwk_kpq = bz.shift_mat_by_ind(giwk, ind=[-iq for iq in q])
        gk_vmw = giwk[..., nl - iwn:nu - iwn]
        gk_v = giwk[..., nl:nu]
        lv2_gg = (lv_a * lv_b_kpq)[..., None] * gk_vmw * gk_v
        gg_fcond = 0
        for j, iwpn in enumerate(wpn):
            gkpq_vmwp = giwk_kpq[..., nl - iwpn:nu - iwpn]
            gkpq_vmwpmw = giwk_kpq[..., nl - iwpn - iwn:nu - iwpn - iwn]
            gg_fcond += gkpq_vmwp * gkpq_vmwpmw * f_cond[i, j, None, None, None, ...]
        sigma_vert += np.sum(np.mean(lv2_gg * gg_fcond, axis=(0, 1, 2)), axis=-1)

    # replace q_list with nq; The additional minus sign stems from the different sign conventions in the vertex
    return -2 / (beta ** 2 * nk_tot) * sigma_vert  # one n_k is contained in the mean


def vec_get_chijj_vert(f_cond, giwk_gen: twop.GreensFunction, hr: wannier.WannierHr, k_grid: bz.KGrid, wn_vert, wn_cond, q_list,
                       der_a=0, der_b=0):
    '''
        f_cond -> [q,wp,w,v]
    '''
    assert np.shape(wn_vert)[0] == np.shape(f_cond)[1], 'Number of internal bosonic frequencies in ' \
                                                        'f_cond must be equal to wn_vert'
    assert np.shape(wn_cond)[0] == np.shape(f_cond)[2], 'Number of transformed bosonic frequencies in ' \
                                                        'f_cond must be equal to wn_cond'
    niv_cut = f_cond.shape[-1] // 2 + mf.niw_from_mat(wn_vert) + mf.niw_from_mat(wn_cond, pos=True)
    giwk = mf.cut_v(giwk_gen.g_full(), niv_cut, axes=(-1,))
    lv_a = hr.get_light_vertex(k_grid, der=der_a)
    lv_b = hr.get_light_vertex(k_grid, der=der_b)
    beta = giwk_gen.beta
    sigma_vert = np.array([get_chijj_vertex(f_cond[:, :, i, :], lv_a, lv_b, giwk, beta, iwn, wn_vert, q_list, k_grid.nk_tot)
                           for i, iwn in enumerate(wn_cond)])
    return sigma_vert


def build_vertex_for_optical_conductivity(d_cfg: conf.DgaConfig, comm, chi_lad_magn, chi_lad_dens):
    '''
        Build the vertex for the optical conductivity from the parts previously written to files.
    '''
    f1_magn, f2_magn, f1_dens, f2_dens = load_optics_vertex_from_rank_files(output_path=d_cfg.output_path, name='Q',
                                                                            mpi_size=comm.size,
                                                                            nq=d_cfg.lattice.q_grid.nk_irr,
                                                                            optics_config=d_cfg.optics)
    d_cfg.logger.log_memory_usage('f1_magn', f1_magn, n_exists=4)
    f1_magn = d_cfg.lattice.q_grid.map_irrk2fbz(f1_magn, shape='mesh')
    f2_magn = d_cfg.lattice.q_grid.map_irrk2fbz(f2_magn, shape='mesh')
    chi_lad_magn_optics = mf.cut_w(chi_lad_magn, d_cfg.optics.niw_vert, axes=(-1,))
    f_magn = f1_magn + (1 + d_cfg.sys.u * chi_lad_magn_optics[..., None, None]) * f2_magn

    f1_dens = d_cfg.lattice.q_grid.map_irrk2fbz(f1_dens, shape='mesh')
    f2_dens = d_cfg.lattice.q_grid.map_irrk2fbz(f2_dens, shape='mesh')
    chi_lad_dens_optics = mf.cut_w(chi_lad_dens, d_cfg.optics.niw_vert, axes=(-1,))

    f_dens = f1_dens + (1 - d_cfg.sys.u * chi_lad_dens_optics[..., None, None]) * f2_dens
    del f1_magn, f2_magn, f1_dens, f2_dens
    gc.collect()
    niw = f_dens.shape[-3] // 2
    plotting.plot_kx_ky(f_dens[:, :, 0, niw, 0, d_cfg.optics.niv_vert],
                        d_cfg.lattice.k_grid.kx,
                        d_cfg.lattice.k_grid.ky, pdir=d_cfg.optics.output_path, name='F_dens_niw0')

    plotting.plot_kx_ky(f_magn[:, :, 0, niw, 0, d_cfg.optics.niv_vert],
                        d_cfg.lattice.k_grid.kx,
                        d_cfg.lattice.k_grid.ky, pdir=d_cfg.optics.output_path, name='F_magn_niw0')

    f_cond = -1 / 2 * f_dens - 3 / 2 * f_magn
    del f_dens, f_magn
    gc.collect()

    plotting.plot_kx_ky(f_cond[:, :, 0, niw, 0, d_cfg.optics.niv_vert],
                        d_cfg.lattice.k_grid.kx,
                        d_cfg.lattice.k_grid.ky, pdir=d_cfg.optics.output_path, name='F_cond_niw0')

    d_cfg.optics.save_data(f_cond, 'F_cond')
    del f_cond
    gc.collect()


def write_optics_vertex_components(d_cfg: conf.DgaConfig, mpi_distributor: mpi_aux.MpiDistributor, channel, gchiq_aux,
                                   vrg_q, gchi0_q_core, name='f_'):
    '''
         Write the components of the vertex in the correct frequency slice to file:
         Look in supplemental of Phys. Rev. B 99, 041115(R) (S.5).

        [f1]:eV = beta^2 * gchi0(v))^(-1) [delta(v,vp) - chi_aux(v,vp) gchi0(v)^(-1)]
        [f2]:eV = u_r vrg(v) vrg(vp)

        Consiquently the slice F(v,vp) -> F(v,v-w) is saved to file (use v -> v-w). Only positive wp values are computed
        and stored.
    '''
    niv_vert = d_cfg.optics.niv_vert
    wn_vert = d_cfg.optics.wn_vert()
    wn_cond_pos = d_cfg.optics.wn_cond(pos=True)
    with mpi_distributor as file:
        for i, iq in enumerate(mpi_distributor.my_tasks):
            for iw in wn_vert:
                ind_w = d_cfg.box.niw_core + iw  # get corresponding omega index in quantities with the full range
                f1_slice, f2_slice = fp.ladder_vertex_from_chi_aux_components(gchi_aux=gchiq_aux[i, ind_w],
                                                                              vrg=vrg_q[i, ind_w],
                                                                              gchi0=gchi0_q_core[i, ind_w],
                                                                              beta=d_cfg.sys.beta,
                                                                              u_r=lfp.get_ur(d_cfg.sys.u, channel))
                group = '/' + name + f'irrq{iq:03d}wn{iw:04d}/'

                # Get F(v,vp) to F(v,v-w) transformation
                file[group + f'f1_{channel}/'] = get_vmw_v_slice(f1_slice, wn_cond_pos, niv_vert)
                file[group + f'f2_{channel}/'] = get_vmw_v_slice(f2_slice, wn_cond_pos, niv_vert)


def load_optics_vertex_from_rank_files(output_path, name, mpi_size, nq, optics_config: conf.OpticsConfig):
    # Collect data from subfiles (This is quite ugly, as it is hardcoded to my structure.
    # This should be replaced by a general routine):

    niw_vert = optics_config.niw_vert
    niw_cond = optics_config.niw_cond
    niv_vert = optics_config.niv_vert
    # NOTE: they need to be predefined due to the different mpi files
    # store only positive values of the second index
    f1_magn = np.zeros((nq, 2 * niw_vert + 1, niw_cond + 1, 2 * niv_vert), dtype=complex)
    f2_magn = np.zeros((nq, 2 * niw_vert + 1, niw_cond + 1, 2 * niv_vert), dtype=complex)
    f1_dens = np.zeros((nq, 2 * niw_vert + 1, niw_cond + 1, 2 * niv_vert), dtype=complex)
    f2_dens = np.zeros((nq, 2 * niw_vert + 1, niw_cond + 1, 2 * niv_vert), dtype=complex)

    for ir in range(mpi_size):
        fname = output_path + name + f'Rank{ir:05d}' + '.hdf5'
        with h5py.File(fname, 'r') as file_in:
            keys = (key for key in file_in.keys() if key.startswith('f_'))  # the group is still hard-coded!
            for key1 in keys:
                # extract the q indizes from the group name!
                irrq = np.array(re.findall(r'\d+', key1), dtype=int)[0]
                iwn = np.array(re.findall(r'\d+', key1), dtype=int)[1] + niw_vert
                f1_magn[irrq, iwn, ...] = file_in.file[key1 + '/f1_magn/'][()]
                f2_dens[irrq, iwn, ...] = file_in.file[key1 + '/f2_dens/'][()]
                f1_dens[irrq, iwn, ...] = file_in.file[key1 + '/f1_dens/'][()]
                f2_magn[irrq, iwn, ...] = file_in.file[key1 + '/f2_magn/'][()]
    return f1_magn, f2_magn, f1_dens, f2_dens


def get_vmw_v_slice(mat, wn, niv_new):
    ''''
        map from F(v,v') to F(v-w,v)
    '''
    mat_new = np.zeros((len(wn), niv_new * 2), dtype=mat.dtype)  # v-wp is still a fermionic frequency
    niv_old = mf.niv_from_mat(mat)
    for i, iwn in enumerate(wn):
        nl, nu = niv_old - niv_new, niv_old + niv_new
        mat_new[i, :] = np.diag(mat[nl:nu, nl - iwn:nu - iwn])
    return mat_new


def plot_opt_cond_matsubara(chijj_bubble, chijj_vert, verbose=False, do_save=False, pdir=None):
    chijj_full = mf.add_bosonic(chijj_bubble, chijj_vert)

    plt.figure()
    plt.plot(mf.wn(chijj_bubble), chijj_bubble.real, '-o', markeredgecolor='k', color='cornflowerblue', label='bubble')
    plt.plot(mf.wn(chijj_vert), chijj_vert.real, '-o', markeredgecolor='k', color='firebrick', label='vertex')
    plt.plot(mf.wn(chijj_full), chijj_full.real, '-o', markeredgecolor='k', color='seagreen',
             label='full')
    plt.legend()
    plt.xlim(0, None)
    plt.xlabel(r'$\omega_n$')
    plt.ylabel(r'$\chi_{jj}(\omega_n)$')
    if do_save: plt.savefig(pdir + 'optical_conductivity_matsubara.png')
    if verbose:
        plt.show()
    else:
        plt.close()


def plot_opt_cond_realf(w, sigma_bubble, sigma_full, verbose=False, do_save=False, pdir=None):
    assert sigma_bubble.shape == sigma_full.shape, ' Optical conductivity bubble and vertex part have different shapes!'

    _, axes = plt.subplots(ncols=2, nrows=1, figsize=(8, 3), dpi=500)
    axes[0].plot(w, sigma_bubble.real, '-', color='cornflowerblue', label='bubble', alpha=0.8)
    axes[0].plot(w, sigma_full.real, '-', color='firebrick', label='full', alpha=0.8)

    axes[1].plot(w, sigma_full.real - sigma_bubble.real, '-o', markeredgecolor='k', color='cornflowerblue', label='vertex')

    for ax in axes:
        ax.legend()
        ax.set_xlim(0, None)
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'$\sigma(\omega)$')
    plt.tight_layout()
    if do_save: plt.savefig(pdir + 'optical_conductivity_realf.png')
    if verbose:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    pass
