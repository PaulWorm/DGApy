# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Contains output related routines.


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import sys, os
import numpy as np
import Plotting as plotting
import Config as conf
import FourPoint as fp
import OrnsteinZernickeFunction as ozfunc
import MatsubaraFrequencies as mf
import AnalyticContinuation as a_cont
import TwoPoint_old as twop
import MpiAux as mpiaux
import Indizes as ind
import PairingVertex as pv
import BrillouinZone as bz
import gc
# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def uniquify(path=None):
    '''

    path: path to be checked for uniqueness
    return: updated unique path
    '''
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path


def spin_fermion_contributions_output(dga_conf=None, sigma_dga_contributions=None):
    output_path = dga_conf.nam.output_path_sp
    np.save(output_path + 'dga_sde_sf_contrib.npy', sigma_dga_contributions, allow_pickle=True)
    plotting.plot_siwk_fs(siwk=sigma_dga_contributions['magn_re'], plot_dir=output_path, kgrid=dga_conf.k_grid,
                          do_shift=True, name='magn_spre')
    plotting.plot_siwk_fs(siwk=sigma_dga_contributions['magn_im'], plot_dir=output_path, kgrid=dga_conf.k_grid,
                          do_shift=True, name='magn_spim')
    plotting.plot_siwk_fs(siwk=sigma_dga_contributions['dens_re'], plot_dir=output_path, kgrid=dga_conf.k_grid,
                          do_shift=True, name='dens_spre')
    plotting.plot_siwk_fs(siwk=sigma_dga_contributions['dens_im'], plot_dir=output_path, kgrid=dga_conf.k_grid,
                          do_shift=True, name='dens_spim')


def prepare_and_plot_vrg_dga(dga_conf: conf.DgaConfig = None, distributor=None):
    output_path = dga_conf.nam.output_path_sp
    vrg_dens, vrg_magn = fp.load_spin_fermion(output_path=dga_conf.nam.output_path, name='Qiw',
                                              mpi_size=distributor.comm.size,
                                              nq=dga_conf.q_grid.nk_irr, niv=dga_conf.box.niv_vrg_save,
                                              niw=dga_conf.box.niw_vrg_save)
    vrg_dens = dga_conf.q_grid.irrk2fbz(mat=vrg_dens)
    vrg_magn = dga_conf.q_grid.irrk2fbz(mat=vrg_magn)

    plotting.plot_spin_fermion_fs(output_path=output_path, name='spin_fermion_dens_fs',
                                  vrg_fs=vrg_dens[..., 0, dga_conf.box.niv_vrg_save], q_grid=dga_conf.q_grid)
    plotting.plot_spin_fermion_fs(output_path=output_path, name='spin_fermion_magn_fs',
                                  vrg_fs=vrg_magn[..., 0, dga_conf.box.niv_vrg_save], q_grid=dga_conf.q_grid)
    spin_fermion_vertex = {
        'vrg_dens': vrg_dens,
        'vrg_magn': vrg_magn
    }

    np.save(output_path + 'spin_fermion_vertex.npy', spin_fermion_vertex)


def fit_and_plot_oz(output_path, q_grid):
    chi_lambda_magn = np.load(output_path + '/chi_magn_lam.npy', allow_pickle=True)
    niw0 = np.shape(chi_lambda_magn)[-1] // 2
    try:

        oz_coeff, _ = ozfunc.fit_oz_spin(q_grid,chi_lambda_magn[:, :, :, niw0].real.flatten())
    except:
        oz_coeff = [-1, -1]

    np.savetxt(output_path + '/oz_coeff.txt', oz_coeff, delimiter=',', fmt='%.9f', header='A xi')
    plotting.plot_oz_fit(chi_w0=chi_lambda_magn[:, :, :, niw0], oz_coeff=oz_coeff,
                         qgrid=q_grid,
                         pdir=output_path + '/', name='oz_fit')


def poly_fit(mat_data, beta, k_grid: bz.KGrid, n_fit, order, name='poly_cont', output_path='./'):
    v = mf.v_plus(beta=beta, n=mat_data.shape[-1] // 2)

    re_fs, im_fs, Z = a_cont.extract_coeff_on_ind(
        siwk=np.squeeze(mat_data.reshape(-1, mat_data.shape[-1])[:, mat_data.shape[-1] // 2:]),
        indizes=k_grid.irrk_ind, v=v, N=n_fit, order=order)
    re_fs = k_grid.map_irrk2fbz(mat=re_fs)
    im_fs = k_grid.map_irrk2fbz(mat=im_fs)
    Z = k_grid.map_irrk2fbz(mat=Z)

    extrap = {
        're_fs': re_fs,
        'im_fs': im_fs,
        'Z': Z
    }

    np.save(output_path + '{}.npy'.format(name), extrap, allow_pickle=True)
    plotting.plot_siwk_extrap(siwk_re_fs=re_fs, siwk_im_fs=im_fs, siwk_Z=Z, output_path=output_path,
                              name=name, k_grid=k_grid)


def max_ent_loc_bw_range(dga_conf: conf.DgaConfig = None, me_conf: conf.MaxEntConfig = None, bw_range=None, sigma=None,
                         n_fit=None, adjust_mu=True, name=''):
    v_real = me_conf.mesh
    chi2 = []
    for bw in bw_range:
        # mu-adjusted DGA Green's function:
        g_cont, gk_dga, chi2_tmp = a_cont.max_ent_loc(me_conf=me_conf, v_real=v_real, sigma=sigma, dga_conf=dga_conf,
                                            niv_cut=dga_conf.box.niv_urange, bw=bw, nfit=n_fit, adjust_mu=adjust_mu, return_chi2=True)
        chi2.append(chi2_tmp)
        plotting.plot_aw_loc(output_path=dga_conf.nam.output_path_ac, v_real=v_real, gloc=g_cont,
                             name=name + '-bw{}'.format(bw))
        n_int = a_cont.check_filling(v_real=v_real, gloc_cont=g_cont)
        np.savetxt(dga_conf.nam.output_path_ac + 'n_' + name + '_bw{}.txt'.format(bw), [n_int, gk_dga['n']],
                   delimiter=',', fmt='%.9f')
        np.save(dga_conf.nam.output_path_ac + 'gloc_cont_' + name + '_bw{}.npy'.format(bw), g_cont, allow_pickle=True)

    chi2 = np.array(chi2)
    bw_opt_ind,fit = a_cont.fit_piecewise(np.log10(np.flip(bw_range)), np.log10(np.flip(chi2)), p2_deg=1)
    bw_opt = np.flip(bw_range)[bw_opt_ind]
    plotting.plot_bw_fit(bw_opt=bw_opt, bw=np.flip(bw_range), chi2=np.flip(chi2), fit=fit, output_path=dga_conf.nam.output_path_ac, name='chi2_bw_{}'.format(name))
    return bw_opt



def max_ent_irrk_bw_range(comm=None, dga_conf: conf.DgaConfig = None, me_conf: conf.MaxEntConfig = None, bw_range=None,
                          sigma=None, n_fit=None, adjust_mu=True, name='', logger = None):
    bw_range = np.atleast_1d(bw_range)
    v_real = me_conf.mesh
    irrk_distributor = mpiaux.MpiDistributor(ntasks=dga_conf.k_grid.nk_irr, comm=comm)
    index_grid_keys = ('irrk',)
    irrk_grid = ind.IndexGrids(grid_arrays=(dga_conf.k_grid.irrk_ind_lin,), keys=index_grid_keys,
                               my_slice=irrk_distributor.my_slice)
    ind_irrk = np.squeeze(
        np.array(np.unravel_index(dga_conf.k_grid.irrk_ind[irrk_grid.my_indizes], shape=dga_conf.k_grid.nk))).T
    if (np.size(ind_irrk.shape) > 1):
        ind_irrk = [tuple(ind_irrk[i, :]) for i in np.arange(ind_irrk.shape[0])]
    else:
        ind_irrk = tuple(ind_irrk)

    for bw in bw_range:
        if (bw == 0):
            use_preblur = False
        else:
            use_preblur = me_conf.use_preblur

        # mu-adjust DGA Green's function:
        gk = twop.create_gk_dict(dga_conf=dga_conf, sigma=sigma, mu0=dga_conf.sys.mu_dmft, adjust_mu=adjust_mu,
                                 niv_cut=dga_conf.box.niv_urange)
        gk_my_cont = a_cont.do_max_ent_on_ind_T(mat=gk['gk'], ind_list=ind_irrk, v_real=v_real,
                                                beta=me_conf.beta,
                                                n_fit=n_fit, err=me_conf.err, alpha_det_method=me_conf.alpha_det_method,
                                                use_preblur=use_preblur, bw=bw, optimizer=me_conf.optimizer)
        if(logger is not None): logger.log_cpu_time(task=' for {} MaxEnt done left are plots and gather '.format(name))
        comm.Barrier()
        gk_cont = irrk_distributor.allgather(rank_result=gk_my_cont)
        comm.Barrier()
        if(logger is not None):  logger.log_cpu_time(task=' for {} Gather done left are plots '.format(name))
        if (comm.rank == 0):
            gk_cont_fbz = dga_conf.k_grid.irrk2fbz(mat=gk_cont)

            plotting.plot_cont_fs(output_path=dga_conf.nam.output_path_ac,
                                  name='fermi_surface_' + name + '_cont_w0-bw{}'.format(bw),
                                  gk=gk_cont_fbz, v_real=v_real, k_grid=dga_conf.k_grid, w_plot=0)
            plotting.plot_cont_fs(output_path=dga_conf.nam.output_path_ac,
                                  name='fermi_surface_' + name + '_cont_w-0.1-bw{}'.format(bw),
                                  gk=gk_cont_fbz, v_real=v_real, k_grid=dga_conf.k_grid, w_plot=-0.1 * me_conf.t)
            plotting.plot_cont_fs(output_path=dga_conf.nam.output_path_ac,
                                  name='fermi_surface_' + name + '_cont_w0.1-bw{}'.format(bw),
                                  gk=gk_cont_fbz, v_real=v_real, k_grid=dga_conf.k_grid, w_plot=0.1 * me_conf.t)
            np.save(dga_conf.nam.output_path_ac + 'gk_' + name + '_cont_fbz_bw{}.npy'.format(bw), gk_cont_fbz,
                    allow_pickle=True)
            plotting.plot_cont_edc_maps(v_real=v_real, gk_cont=gk_cont_fbz, k_grid=dga_conf.k_grid,
                                        output_path=dga_conf.nam.output_path_ac,
                                        name='fermi_surface_' + name + '_cont_edc_maps_bw{}'.format(bw))

def max_ent_irrk_bw_range_sigma(comm=None, dga_conf: conf.DgaConfig = None, me_conf: conf.MaxEntConfig = None, bw_range=None,
                          sigma=None, hartree = None, n_fit=None, name='', logger = None):
    bw_range = np.atleast_1d(bw_range)
    v_real = me_conf.mesh
    irrk_distributor = mpiaux.MpiDistributor(ntasks=dga_conf.k_grid.nk_irr, comm=comm)
    index_grid_keys = ('irrk',)
    irrk_grid = ind.IndexGrids(grid_arrays=(dga_conf.k_grid.irrk_ind_lin,), keys=index_grid_keys,
                               my_slice=irrk_distributor.my_slice)
    ind_irrk = np.squeeze(
        np.array(np.unravel_index(dga_conf.k_grid.irrk_ind[irrk_grid.my_indizes], shape=dga_conf.k_grid.nk))).T
    if (np.size(ind_irrk.shape) > 1):
        ind_irrk = [tuple(ind_irrk[i, :]) for i in np.arange(ind_irrk.shape[0])]
    else:
        ind_irrk = tuple(ind_irrk)

    for bw in bw_range:
        if (bw == 0):
            use_preblur = False
        else:
            use_preblur = me_conf.use_preblur

        sigma_cont = a_cont.do_max_ent_on_ind_T(mat=sigma-hartree, ind_list=ind_irrk, v_real=v_real,
                                                beta=me_conf.beta,
                                                n_fit=n_fit, err=me_conf.err, alpha_det_method=me_conf.alpha_det_method,
                                                use_preblur=use_preblur, bw=bw, optimizer=me_conf.optimizer)
        if(logger is not None): logger.log_cpu_time(task=' for {} MaxEnt done left are plots and gather '.format(name))
        comm.Barrier()
        sk_cont = irrk_distributor.allgather(rank_result=sigma_cont+hartree)
        comm.Barrier()
        if(logger is not None):  logger.log_cpu_time(task=' for {} Gather done left are plots '.format(name))
        if (comm.rank == 0):
            gk_cont_fbz = dga_conf.k_grid.irrk2fbz(mat=sk_cont)

            plotting.plot_cont_fs(output_path=dga_conf.nam.output_path_ac,
                                  name='swk_fermi_surface_' + name + '_cont_w0-bw{}'.format(bw),
                                  gk=gk_cont_fbz, v_real=v_real, k_grid=dga_conf.k_grid, w_plot=0)
            plotting.plot_cont_fs(output_path=dga_conf.nam.output_path_ac,
                                  name='swk_fermi_surface_' + name + '_cont_w-0.1-bw{}'.format(bw),
                                  gk=gk_cont_fbz, v_real=v_real, k_grid=dga_conf.k_grid, w_plot=-0.1 * me_conf.t)
            plotting.plot_cont_fs(output_path=dga_conf.nam.output_path_ac,
                                  name='swk_fermi_surface_' + name + '_cont_w0.1-bw{}'.format(bw),
                                  gk=gk_cont_fbz, v_real=v_real, k_grid=dga_conf.k_grid, w_plot=0.1 * me_conf.t)
            np.save(dga_conf.nam.output_path_ac + 'swk_' + name + '_cont_fbz_bw{}.npy'.format(bw), gk_cont_fbz,
                    allow_pickle=True)
            np.save(dga_conf.nam.output_path_ac + 'me_config_' + name + '_bw{}.npy'.format(bw), me_conf, allow_pickle=True)
            plotting.plot_cont_edc_maps(v_real=v_real, gk_cont=gk_cont_fbz, k_grid=dga_conf.k_grid,
                                        output_path=dga_conf.nam.output_path_ac,
                                        name='swk_fermi_surface_' + name + '_cont_edc_maps_bw{}'.format(bw))

def load_and_construct_pairing_vertex(dga_conf:conf.DgaConfig = None, comm=None):

    f1_magn, f2_magn, f1_dens, f2_dens = pv.load_pairing_vertex_from_rank_files(output_path=dga_conf.nam.output_path, name='Qiw',
                                                                                mpi_size=comm.size, nq=dga_conf.q_grid.nk_irr,
                                                                                niv_pp=dga_conf.box.niv_pp)

    chi_lambda = np.load(dga_conf.nam.output_path + 'chi_lambda.npy', allow_pickle=True).item()

    # Create f_magn:
    f1_magn = dga_conf.q_grid.irrk2fbz(mat=f1_magn)
    f2_magn = dga_conf.q_grid.irrk2fbz(mat=f2_magn)
    chi_magn_lambda_pp = pv.reshape_chi(chi=chi_lambda['magn'].mat, niv_pp=dga_conf.box.niv_pp)
    f_magn = f1_magn + (1 + dga_conf.sys.u * chi_magn_lambda_pp) * f2_magn
    del f1_magn, f2_magn, chi_magn_lambda_pp
    gc.collect()
    plotting.plot_vertex_vvp(vertex=f_magn.mean(axis=(0, 1, 2)).real, pdir=dga_conf.nam.output_path_el, name='f_magn_loc')

    # Create f_dens:
    f1_dens = dga_conf.q_grid.irrk2fbz(mat=f1_dens)
    f2_dens = dga_conf.q_grid.irrk2fbz(mat=f2_dens)
    chi_dens_lambda_pp = pv.reshape_chi(chi=chi_lambda['dens'].mat, niv_pp=dga_conf.box.niv_pp)
    f_dens = f1_dens + (1 - dga_conf.sys.u * chi_dens_lambda_pp) * f2_dens
    del f1_dens, f2_dens, chi_dens_lambda_pp, chi_lambda
    gc.collect()
    plotting.plot_vertex_vvp(vertex=f_dens.mean(axis=(0, 1, 2)).real, pdir=dga_conf.nam.output_path_el, name='f_dens_loc')

    # Create f_sing and f_trip:
    f_sing = -1.5 * f_magn + 0.5 * f_dens
    f_trip = -0.5 * f_magn - 0.5 * f_dens
    plotting.plot_vertex_vvp(vertex=f_sing.mean(axis=(0, 1, 2)).real, pdir=dga_conf.nam.output_path_el, name='f_sing_loc')
    plotting.plot_vertex_vvp(vertex=f_trip.mean(axis=(0, 1, 2)).real, pdir=dga_conf.nam.output_path_el, name='f_trip_loc')

    pairing_vertices = {
        'f_sing': f_sing,
        'f_trip': f_trip
    }

    np.save(dga_conf.nam.output_path_el + 'pairing_vertices.npy', pairing_vertices, allow_pickle=True)


def perform_eliashberg_routine(dga_conf:conf.DgaConfig = None, sigma=None, el_conf:conf.EliashbergConfig = None):
        import EliashbergEquation as eq

        pairing_vertices = np.load(dga_conf.nam.output_path_el + 'pairing_vertices.npy', allow_pickle=True).item()
        gamma_sing = -pairing_vertices['f_sing']
        gamma_trip = -pairing_vertices['f_trip']
        #
        if (el_conf.sym_sing):
            gamma_sing = 0.5 * (gamma_sing + np.flip(gamma_sing, axis=(-1)))

        if (el_conf.sym_trip):
            gamma_trip = 0.5 * (gamma_trip - np.flip(gamma_trip, axis=(-1)))

        plotting.plot_vertex_vvp(vertex=gamma_trip.mean(axis=(0, 1, 2)).real, pdir=dga_conf.nam.output_path_el, name='gamma_trip_loc')

        g_generator = twop.GreensFunctionGenerator(beta=dga_conf.sys.beta, kgrid=dga_conf.q_grid, hr=dga_conf.sys.hr,
                                                   sigma=sigma)
        mu_dga = g_generator.adjust_mu(n=dga_conf.sys.n, mu0=dga_conf.sys.mu_dmft)
        gk_dga = g_generator.generate_gk(mu=mu_dga, qiw=[0, 0, 0, 0], niv=dga_conf.box.niv_pp).gk

        gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=el_conf.gap0_sing['k'], v_type=el_conf.gap0_sing['v'],
                                k_grid=dga_conf.q_grid.grid)
        norm = dga_conf.q_grid.nk_tot * dga_conf.sys.beta
        powiter_sing = eq.EliashberPowerIteration(gamma=gamma_sing, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
                                                  n_eig=el_conf.n_eig)

        gap0 = eq.get_gap_start(shape=np.shape(gk_dga), k_type=el_conf.gap0_trip['k'], v_type=el_conf.gap0_trip['v'],
                                k_grid=dga_conf.q_grid.grid)
        powiter_trip = eq.EliashberPowerIteration(gamma=gamma_trip, gk=gk_dga, gap0=gap0, norm=norm, shift_mat=True,
                                                  n_eig=el_conf.n_eig)

        eliashberg = {
            'lambda_sing': powiter_sing.lam,
            'lambda_trip': powiter_trip.lam,
            'delta_sing': powiter_sing.gap,
            'delta_trip': powiter_trip.gap,
        }
        np.save(dga_conf.nam.output_path_el + 'eliashberg.npy', eliashberg)
        np.savetxt(dga_conf.nam.output_path_el + 'eigenvalues.txt', [powiter_sing.lam.real, powiter_trip.lam.real], delimiter=',',
                   fmt='%.9f')
        np.savetxt(dga_conf.nam.output_path_el + 'eigenvalues_s.txt', [powiter_sing.lam_s.real, powiter_trip.lam_s.real], delimiter=',',
                   fmt='%.9f')

        for i in range(len(powiter_sing.gap)):
            plotting.plot_gap_function(delta=powiter_sing.gap[i].real, pdir=dga_conf.nam.output_path_el, name='sing_{}'.format(i),
                                       kgrid=dga_conf.q_grid,
                                       do_shift=True)
            plotting.plot_gap_function(delta=powiter_sing.gap[i].real, pdir=dga_conf.nam.output_path_el, name='trip_{}'.format(i),
                                       kgrid=dga_conf.q_grid,
                                       do_shift=True)

if __name__ == "__main__":
    path = 'C:/Users/pworm/Research/FiniteLayerNickelates/N=5/GGA+U/dx2-y2_modified/n0.82/Continuation'
    path_unique = uniquify(path)
    # os.mkdir(path_unique)
