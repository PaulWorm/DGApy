# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Contains output related routines.


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import sys, os
import numpy as np
import Plotting as plotting
import Config as conf
import FourPoint as fp


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


def prepare_and_plot_vrg_dga(dga_conf: conf.DgaConfig = None, output_path=None, distributor=None):
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


if __name__ == "__main__":
    path = 'C:/Users/pworm/Research/FiniteLayerNickelates/N=5/GGA+U/dx2-y2_modified/n0.82/Continuation'
    path_unique = uniquify(path)
    # os.mkdir(path_unique)
