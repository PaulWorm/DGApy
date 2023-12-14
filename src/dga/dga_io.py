''' Input-Ouput routines for dga'''

import os
import numpy as np
import h5py

from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import two_point as twop
from dga import local_four_point as lfp
from dga import ornstein_zernicke_function as oz_func
from dga import plotting
from dga import config
from dga import analytic_continuation as a_cont
from dga import util
from dga import w2dyn_aux_dga

# ----------------------------------------------- LOAD DATA ------------------------------------------------------------
KNOWNINPUTTYPES = ['EDFermion', 'w2dyn']


def set_output_path(base_name, comm=None):
    ''' comm is for mpi applications '''
    output_path = util.uniquify(base_name)

    if not os.path.exists(output_path):
        if comm is not None:
            if comm.rank == 0: os.mkdir(output_path)
        else:
            os.mkdir(output_path)
    return output_path


def create_dmft_ddict(dmft_input, g2_dens: lfp.LocalFourPoint, g2_magn: lfp.LocalFourPoint):
    ddict = {
        'giw': dmft_input['giw'],
        'siw': dmft_input['siw'],
        'n': dmft_input['n'],
        'beta': dmft_input['beta'],
        'u': dmft_input['u'],
        'mu_dmft': dmft_input['mu_dmft'],
        'g4iw_dens': g2_dens.mat,
        'g4iw_magn': g2_magn.mat
    }
    return ddict


def load_1p_data(input_type, path, fname_1p, fname_2p=None):
    if input_type == 'EDFermion':
        return load_1p_data_ed(path, fname_1p, fname_2p=fname_2p)
    elif input_type == 'w2dyn':
        return load_1p_data_w2dyn(path, fname_1p, fname_2p=fname_2p)
    elif input_type == 'test':
        return np.load(path + fname_1p, allow_pickle=True).item()
    else:
        raise NotImplementedError(f'Requested input type format not yet implemented. Currently known is: {KNOWNINPUTTYPES}')


def load_1p_data_ed(path, fname_1p, fname_2p=None):
    '''Load the single particle quantities from the ED calculation'''
    ddict = {}
    # Load the single particle Data:
    f = h5py.File(path + fname_1p, 'r')
    ddict['giw'] = f['giw'][()]
    ddict['siw'] = f['siw_hat'][()]
    ddict['n'] = f['/config/totdens'][()]
    ddict['beta'] = f['/config/beta'][()]
    ddict['u'] = f['/config/U'][()]
    ddict['mu_dmft'] = f['/dmft/mu'][()]
    f.close()

    if fname_2p is not None:
        f = h5py.File(path + fname_2p, 'r')
        ddict['g4iw_dens'] = f['g4iw_dens'][()]
        ddict['g4iw_magn'] = f['g4iw_magn'][()]
        f.close()

    return ddict


def load_1p_data_w2dyn(path, fname_1p, fname_2p=None):
    '''Load the single particle quantities from a w2dynamics calculation'''
    ddict = {}
    # Load the single particle Data:
    file = w2dyn_aux_dga.W2dynFile(fname=path + fname_1p)
    ddict['giw'] = np.mean(file.get_giw()[0, :, :], axis=0) # band spin niv
    ddict['siw'] = np.mean(file.get_siw()[0, :, :], axis=0)
    ddict['n'] = file.get_totdens()
    if ddict['n'] == 0:
        occ = np.sum(np.diag(file.get_occ()[0, :, 0, :]))  # band spin band spin
        ddict['n'] = occ
    ddict['beta'] = file.get_beta()
    ddict['u'] = file.get_udd()
    ddict['mu_dmft'] = file.get_mu()
    file.close()

    if fname_2p is not None:
        file = w2dyn_aux_dga.W2dynG4iwFile(fname=path + fname_2p)
        ddict['g4iw_dens'] = file.read_g2_full(channel='dens')
        ddict['g4iw_magn'] = file.read_g2_full(channel='magn')
        file.close()

    return ddict


def build_g2_obj(d_cfg: config.DgaConfig, dmft_input):
    ''' Create the two-particle Green's function object and determine the box-sizes if not set'''
    # cut the two-particle Green's functions:
    g2_dens = lfp.get_g2_from_dmft_input(dmft_input, channel='dens')
    g2_magn = lfp.get_g2_from_dmft_input(dmft_input, channel='magn')

    d_cfg.box.set_from_lfp(g2_dens)

    # Cut frequency ranges:
    g2_dens.cut_iv(d_cfg.box.niv_core)
    g2_dens.cut_iw(d_cfg.box.niw_core)

    g2_magn.cut_iv(d_cfg.box.niv_core)
    g2_magn.cut_iw(d_cfg.box.niw_core)

    # symmetrize v-vp:
    if (d_cfg.do_sym_v_vp):
        g2_dens.symmetrize_v_vp()
        g2_magn.symmetrize_v_vp()

    return g2_dens, g2_magn


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def fit_and_plot_oz(output_path, q_grid):
    chi_lambda_magn = np.load(output_path + '/chi_magn_lam.npy', allow_pickle=True)
    niw0 = np.shape(chi_lambda_magn)[-1] // 2
    try:
        # pylint: disable=unbalanced-tuple-unpacking
        oz_coeff, _ = oz_func.fit_oz_spin(q_grid, chi_lambda_magn[:, :, :, niw0].real.flatten())
        # pylint: enable=unbalanced-tuple-unpacking
    except:  # pylint: disable=bare-except
        oz_coeff = [-1, -1]

    np.savetxt(output_path + '/oz_coeff.txt', oz_coeff, delimiter=',', fmt='%.9f', header='A xi')
    plotting.plot_oz_fit(chi_w0=chi_lambda_magn[:, :, :, niw0], oz_coeff=oz_coeff,
                         qgrid=q_grid,
                         pdir=output_path + '/', name='oz_fit')


def dmft_poly_fit(giwk_obj: twop.GreensFunction, d_cfg: config.DgaConfig):
    comm = d_cfg.comm
    if comm.rank == 0 and d_cfg.do_poly_fitting:
        poly_fit(giwk_obj.g_full(), d_cfg.sys.beta, d_cfg.lattice.k_grid, d_cfg.n_fit,
                 d_cfg.o_fit, name='Giwk_dmft_poly_cont_', output_path=d_cfg.poly_fit_dir)

        niv_dmft = mf.niv_from_mat(d_cfg.sys.siw)
        vn_dmft_pos = mf.vn(d_cfg.sys.beta, niv_dmft, pos=True)
        siw_pos = d_cfg.sys.siw[niv_dmft:]
        gam, bandshift, z = a_cont.get_gamma_bandshift_z(vn_dmft_pos, siw_pos, order=d_cfg.o_fit, n_fit=d_cfg.n_fit)
        np.savetxt(d_cfg.poly_fit_dir + 'Siw_DMFT_polyfit.txt', np.array([[bandshift, ], [gam, ], [z, ]]).T,
                   header='bandshift gamma Z', fmt='%.6f')


def poly_fit(mat_data, beta, k_grid: bz.KGrid, n_fit, order, name='poly_cont', output_path='./'):
    v = mf.vn(beta, mat_data.shape[-1] // 2, pos=True)

    re_fs, im_fs, z = a_cont.extract_coeff_on_ind(
        siwk=np.squeeze(mat_data.reshape(-1, mat_data.shape[-1])[:, mat_data.shape[-1] // 2:]), indizes=k_grid.irrk_ind, v=v,
        n_fit=n_fit, order=order)
    re_fs = k_grid.map_irrk2fbz(mat=re_fs)
    im_fs = k_grid.map_irrk2fbz(mat=im_fs)
    z = k_grid.map_irrk2fbz(mat=z)

    extrap = {
        're_fs': re_fs,
        'im_fs': im_fs,
        'Z': z
    }
    np.save(output_path + f'{name}.npy', extrap, allow_pickle=True)
    plotting.plot_siwk_extrap(siwk_re_fs=re_fs, siwk_im_fs=im_fs, siwk_z=z, output_path=output_path, name=name, k_grid=k_grid)


def chiq_checks(d_cfg: config.DgaConfig, chi_dens, chi_magn, chi_lad_dens, chi_lad_magn, giwk_dmft, name='ladder'):
    ''' Create checks of the susceptibility: '''
    if d_cfg.comm.rank == 0:
        chi_lad_dens_loc = d_cfg.lattice.q_grid.k_mean(chi_lad_dens, shape='fbz-mesh')
        chi_lad_magn_loc = d_cfg.lattice.q_grid.k_mean(chi_lad_magn, shape='fbz-mesh')
        plotting.chi_checks([chi_dens, chi_lad_dens_loc], [chi_magn, chi_lad_magn_loc], ['loc', f'{name}-sum'], giwk_dmft,
                            d_cfg.pdir,
                            verbose=False,
                            do_plot=True, name='lad_q_tilde')

        plotting.plot_kx_ky(chi_lad_dens[..., 0, d_cfg.box.niw_core], d_cfg.lattice.q_grid.kx,
                            d_cfg.lattice.q_grid.ky, pdir=d_cfg.pdir, name=f'Chi_{name}_dens_kz0')
        plotting.plot_kx_ky(chi_lad_magn[..., 0, d_cfg.box.niw_core], d_cfg.lattice.q_grid.kx,
                            d_cfg.lattice.q_grid.ky, pdir=d_cfg.pdir, name=f'Chi_{name}_magn_kz0')


def default_siwk_checks(d_cfg: config.DgaConfig, sigma_dga, siw_sde_loc, siwk_dmft):
    if d_cfg.comm.rank == 0:
        siwk_dga_shift = sigma_dga.get_siw(d_cfg.box.niv_full, pi_shift=True)
        siw_dga_loc = np.mean(siwk_dga_shift, axis=(0, 1, 2))
        siw_dga_loc_raw = np.mean(mf.fermionic_full_nu_range(sigma_dga.sigma), axis=(0, 1, 2))

        if d_cfg.verbosity > 0:
            plotting.plot_kx_ky(siwk_dga_shift[..., 0, d_cfg.box.niv_full], d_cfg.lattice.k_grid.kx_shift,
                                d_cfg.lattice.k_grid.ky_shift,
                                pdir=d_cfg.pdir,
                                name='Siwk_dga_kz0')
            plotting.sigma_loc_checks([siw_sde_loc, siw_dga_loc, siw_dga_loc_raw,
                                       siwk_dmft.get_siw(d_cfg.box.niv_full)[0, 0, 0, :]],
                                      ['SDE-loc', 'DGA-loc', 'DGA-loc-raw', 'DMFT-input', 'Magn-loc'], d_cfg.sys.beta,
                                      d_cfg.pdir, verbose=False, do_plot=True, name='dga_loc',
                                      xmax=d_cfg.box.niv_full)

            plotting.sigma_loc_checks([siw_sde_loc, siw_dga_loc, siw_dga_loc_raw,
                                       siwk_dmft.get_siw(d_cfg.box.niv_full)[0, 0, 0, :]],
                                      ['SDE-loc', 'DGA-loc', 'DGA-loc-raw', 'DMFT-input', 'Magn-loc'], d_cfg.sys.beta,
                                      d_cfg.pdir, verbose=False, do_plot=True, name='dga_loc_core',
                                      xmax=d_cfg.box.niv_core)

        d_cfg.save_data(sigma_dga.get_siw(d_cfg.box.niv_full, pi_shift=False), 'siwk_dga')


def default_giwk_checks(d_cfg: config.DgaConfig, giwk_dga, sigma_dga):
    if d_cfg.comm.rank == 0:
        giwk_shift = giwk_dga.g_full(pi_shift=False)[..., 0, giwk_dga.niv_full][:d_cfg.lattice.nk[0] // 2,
                     :d_cfg.lattice.nk[1] // 2]
        fs_ind = bz.find_zeros(giwk_shift)
        n_fs = np.shape(fs_ind)[0]
        fs_ind = fs_ind[:n_fs // 2]

        fs_points = np.stack((d_cfg.lattice.k_grid.kx[fs_ind[:, 0]], d_cfg.lattice.k_grid.ky[fs_ind[:, 1]]), axis=1)
        plotting.plot_kx_ky(giwk_dga.g_full(pi_shift=True)[..., 0, giwk_dga.niv_full], d_cfg.lattice.k_grid.kx_shift,
                            d_cfg.lattice.k_grid.ky_shift,
                            pdir=d_cfg.pdir, name='Giwk_dga_kz0', scatter=fs_points)

        d_cfg.save_data(giwk_dga.g_full(), 'giwk_dga')
        np.savetxt(d_cfg.output_path + '/mu.txt', [[giwk_dga.mu, d_cfg.sys.mu_dmft]], header='mu_dga, mu_dmft',
                   fmt='%1.6f')

        # Plots along Fermi-Luttinger surface:
        plotting.plot_along_ind(sigma_dga.get_siw(d_cfg.box.niv_full), fs_ind, pdir=d_cfg.pdir,
                                niv_plot_min=0,
                                niv_plot=20, name='Sigma_{dga}')


def dga_poly_fit(d_cfg: config.DgaConfig, sigma_dga, giwk_dga):
    if d_cfg.do_poly_fitting and d_cfg.comm.rank == 0:
        # Extrapolate the self-energy to the Fermi-level via polynomial fit:
        poly_fit(sigma_dga.get_siw(d_cfg.box.niv_full, pi_shift=True), d_cfg.sys.beta,
                 d_cfg.lattice.k_grid, d_cfg.n_fit,
                 d_cfg.o_fit, name='Siwk_poly_cont_', output_path=d_cfg.poly_fit_dir)

        poly_fit(giwk_dga.g_full(pi_shift=True), d_cfg.sys.beta, d_cfg.lattice.k_grid, d_cfg.n_fit,
                 d_cfg.o_fit,
                 name='Giwk_dga_poly_cont_',
                 output_path=d_cfg.poly_fit_dir)

        d_cfg.logger.log_cpu_time(task=' Poly-fits ')


if __name__ == '__main__':
    pass
