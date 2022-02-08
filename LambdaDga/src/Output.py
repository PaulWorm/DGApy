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
    vrg_dens, vrg_magn = fp.load_spin_fermion(output_path=dga_conf.nam.output_path, name='Qiw', mpi_size=distributor.comm.size,
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

def fit_and_plot_oz(dga_conf: conf.DgaConfig = None):
    output_path = dga_conf.nam.output_path
    chi_lambda = np.load(output_path + 'chi_lambda.npy', allow_pickle=True).item()
    try:
        oz_coeff, _ = ozfunc.fit_oz_spin(dga_conf.q_grid, chi_lambda['magn'].mat[:, :, :, dga_conf.box.niw_core].real.flatten())
    except:
        oz_coeff = [-1,-1]

    np.savetxt(output_path + 'oz_coeff.txt', oz_coeff, delimiter=',', fmt='%.9f')
    plotting.plot_oz_fit(chi_w0=chi_lambda['magn'].mat[:, :, :, dga_conf.box.niw_core], oz_coeff=oz_coeff, qgrid=dga_conf.q_grid,
                         pdir=output_path, name='oz_fit')

def poly_fit(dga_conf: conf.DgaConfig = None, mat_data = None,name='poly_cont', n_extrap=4, order_extrap=3):
    v = mf.v_plus(beta=dga_conf.sys.beta, n=mat_data.shape[-1]//2)

    re_fs, im_fs, Z = a_cont.extract_coeff_on_ind(siwk=np.squeeze(mat_data.reshape(-1, mat_data.shape[-1])[:, mat_data.shape[-1]//2:]), indizes=dga_conf.k_grid.irrk_ind, v=v, N=n_extrap, order=order_extrap)
    re_fs = dga_conf.k_grid.irrk2fbz(mat=re_fs)
    im_fs = dga_conf.k_grid.irrk2fbz(mat=im_fs)
    Z = dga_conf.k_grid.irrk2fbz(mat=Z)

    extrap = {
        're_fs': re_fs,
        'im_fs': im_fs,
        'Z': Z
    }

    np.save(dga_conf.nam.output_path_pf + '{}.npy'.format(name), extrap, allow_pickle=True)
    plotting.plot_siwk_extrap(siwk_re_fs=re_fs, siwk_im_fs=im_fs, siwk_Z=Z, output_path=dga_conf.nam.output_path_pf, name=name, k_grid=dga_conf.k_grid)




if __name__ == "__main__":
    path = 'C:/Users/pworm/Research/FiniteLayerNickelates/N=5/GGA+U/dx2-y2_modified/n0.82/Continuation'
    path_unique = uniquify(path)
    # os.mkdir(path_unique)
