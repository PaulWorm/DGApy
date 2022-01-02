# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Contains the wrapper function for single-band Lambda-corrected DGA.


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import h5py
import matplotlib.pyplot as plt
import Hk as hamk
import ChemicalPotential as chempot
import w2dyn_aux
import MpiAux as mpiaux
import TwoPoint as twop
import FourPoint as fp
import SDE as sde
import RealTime as rt
import Indizes as ind
import LambdaCorrection as lc
import sys, os


# -------------------------------------- LAMBDA DGA FUNCTION WRAPPER ---------------------------------------------------

def lambda_dga(config=None, verbose=False, outpfunc=None):
    ''' Wrapper function for the \lambda-corrected one-band DGA routine. All relevant settings are contained in config'''
    # -------------------------------------------- UNRAVEL CONFIG ------------------------------------------------------
    # This is done to allow for more flexibility in the future

    comm = config['comm']
    wn_core = config['grids']['wn_core']
    niw_core = config['box_sizes']['niw_core']
    niv_core = config['box_sizes']['niv_core']
    niv_urange = config['box_sizes']['niv_urange']
    path = config['names']['input_path']
    fname_g2 = config['names']['fname_g2']
    beta = config['system']['beta']
    u = config['system']['u']
    hr = config['system']['hr']
    box_sizes = config['box_sizes']
    giw = config['dmft1p']['gloc']
    dmft1p = config['dmft1p']
    output_path = config['names']['output_path']
    do_pairing_vertex = config['options']['do_pairing_vertex']
    lambda_correction_type = config['options']['lambda_correction_type']

    k_grid = config['grids']['k_grid']
    q_grid = config['grids']['q_grid']

    nq_tot = q_grid.nk_tot()

    # ----------------------------------------------- MPI DISTRIBUTION -------------------------------------------------
    my_iw = wn_core
    realt = rt.real_time()

    # -------------------------------------------LOAD G2 FROM W2DYN ----------------------------------------------------
    g2_file = w2dyn_aux.g4iw_file(fname=path + fname_g2)

    g2_dens_loc = fp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='dens', iw=my_iw), giw=giw, channel='dens',
                                    beta=beta, iw=my_iw)
    g2_magn_loc = fp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='magn', iw=my_iw), giw=giw, channel='magn',
                                    beta=beta, iw=my_iw)

    g2_dens_loc.cut_iv(niv_cut=niv_core)
    g2_magn_loc.cut_iv(niv_cut=niv_core)

    dmft1p['g2_dens'] = g2_dens_loc
    dmft1p['g2_magn'] = g2_magn_loc

    if (verbose):
        outpfunc("Finished reading g2 from file.")

    # --------------------------------------------- LOCAL DMFT SDE ---------------------------------------------------------
    dmft_sde = sde.local_dmft_sde_from_g2(dmft_input=dmft1p, box_sizes=box_sizes)

    chi_dens_loc_mat = dmft_sde['chi_dens'].mat
    chi_magn_loc_mat = dmft_sde['chi_magn'].mat
    chi0_core = dmft_sde['chi0_core'].chi0
    chi0_urange = dmft_sde['chi0_urange'].chi0


    chi_dens_loc = fp.LocalSusceptibility(matrix=chi_dens_loc_mat, giw=dmft_sde['chi_dens'].giw,
                                          channel=dmft_sde['chi_dens'].channel,
                                          beta=dmft_sde['chi_dens'].beta, iw=wn_core)

    chi_magn_loc = fp.LocalSusceptibility(matrix=chi_magn_loc_mat, giw=dmft_sde['chi_magn'].giw,
                                          channel=dmft_sde['chi_magn'].channel,
                                          beta=dmft_sde['chi_magn'].beta, iw=wn_core)

    gamma_dens = fp.LocalFourPoint(matrix=dmft_sde['gamma_dens'].mat,
                                   giw=dmft_sde['gamma_dens'].giw,
                                   channel=dmft_sde['gamma_dens'].channel, beta=dmft_sde['gamma_dens'].beta, iw=wn_core)

    gamma_magn = fp.LocalFourPoint(matrix=dmft_sde['gamma_magn'].mat,
                                   giw=dmft_sde['gamma_magn'].giw,
                                   channel=dmft_sde['gamma_magn'].channel, beta=dmft_sde['gamma_magn'].beta, iw=wn_core)

    dmft_gamma = {
        'gamma_dens': gamma_dens,
        'gamma_magn': gamma_magn
    }

    dmft_sde['siw'] = dmft_sde['siw'] + dmft_sde['hartree']
    dmft_sde['chi0_core'] = chi0_core
    dmft_sde['chi0_urange'] = chi0_urange

    if (verbose):
        outpfunc("Finished local part.")
        outpfunc(realt.string_time('Local Part '))
    # ------------------------------------------------ NON-LOCAL PART  -----------------------------------------------------
    # ======================================================================================================================

    qiw_distributor = mpiaux.MpiDistributor(ntasks=wn_core.size * nq_tot, comm=comm, output_path=output_path,
                                            name='Qiw')
    index_grid_keys = ('qx', 'qy', 'qz', 'iw')
    qiw_grid = ind.IndexGrids(grid_arrays=q_grid.get_grid_as_tuple() + (wn_core,), keys=index_grid_keys,
                              my_slice=qiw_distributor.my_slice)

    # ----------------------------------------- NON-LOCAL LADDER SUCEPTIBILITY  --------------------------------------------

    dga_susc = fp.dga_susceptibility(dmft_input=dmft1p, local_sde=dmft_gamma, hr=hr, kgrid=k_grid.get_grid_as_tuple(),
                                     box_sizes=box_sizes,
                                     qiw_grid=qiw_grid.my_mesh, qiw_indizes=qiw_grid.my_indizes, niw=niw_core,
                                     file=qiw_distributor.file, do_pairing_vertex=do_pairing_vertex)

    if (verbose):
        outpfunc(realt.string_time('Non-local Susceptibility: '))

    if (do_pairing_vertex):
        f1_magn = np.zeros(np.shape(f_ladder['f1_magn']), dtype=complex)
        f2_magn = np.zeros(np.shape(f_ladder['f2_magn']), dtype=complex)
        f1_dens = np.zeros(np.shape(f_ladder['f1_dens']), dtype=complex)
        f2_dens = np.zeros(np.shape(f_ladder['f2_dens']), dtype=complex)
        comm.Allreduce(f_ladder['f1_magn'], f1_magn)
        comm.Allreduce(f_ladder['f2_magn'], f2_magn)
        comm.Allreduce(f_ladder['f1_dens'], f1_dens)
        comm.Allreduce(f_ladder['f2_dens'], f2_dens)

        f_ladder = {
            'f1_magn': f1_magn,
            'f2_magn': f2_magn,
            'f1_dens': f1_dens,
            'f2_dens': f2_dens
        }

    # ----------------------------------------------- LAMBDA-CORRECTION ------------------------------------------------

    chi_dens_ladder_mat = qiw_distributor.allgather(rank_result=dga_susc['chi_dens_urange'].mat)
    chi_magn_ladder_mat = qiw_distributor.allgather(rank_result=dga_susc['chi_magn_urange'].mat)

    chi0q_core = qiw_distributor.allgather(rank_result=dga_susc['chi0q_core'].mat)
    chi0q_urange = qiw_distributor.allgather(rank_result=dga_susc['chi0q_urange'].mat)

    chi_dens_ladder = fp.LadderSusceptibility(qiw=qiw_grid.meshgrid, channel='dens', u=dmft1p['u'], beta=dmft1p['beta'])
    chi_dens_ladder.mat = chi_dens_ladder_mat

    chi_magn_ladder = fp.LadderSusceptibility(qiw=qiw_grid.meshgrid, channel='magn', u=dmft1p['u'], beta=dmft1p['beta'])
    chi_magn_ladder.mat = chi_magn_ladder_mat

    if(lambda_correction_type=='both'):
        lambda_dens = lc.lambda_correction(lambda_start=lc.get_lambda_start(chi_dens_ladder), chir=chi_dens_ladder, chi_loc=chi_dens_loc, nq=np.prod(q_grid.nk))
        lambda_magn = lc.lambda_correction(lambda_start=lc.get_lambda_start(chi_magn_ladder), chir=chi_magn_ladder,
                                           chi_loc=chi_magn_loc, nq=np.prod(q_grid.nk))
    elif(lambda_correction_type=='totdens'):
        lambda_dens = 0.0
        lambda_magn = lc.lambda_correction_totdens(lambda_start=lc.get_lambda_start(chi_magn_ladder), chi_magn=chi_magn_ladder,
                                                   chi_magn_loc=chi_magn_loc,chi_dens_loc=chi_dens_loc,chi_dens=chi_dens_ladder,
                                                   nq=np.prod(q_grid.nk))
    elif (lambda_correction_type == 'none'):
        lambda_dens = 0.0
        lambda_else = 0.0
    else:
        raise ValueError('Unknown value for lambda_correction_type!')

    chi_dens_lambda = fp.LadderSusceptibility(qiw=qiw_grid.meshgrid, channel='dens', u=dmft1p['u'], beta=dmft1p['beta'])
    chi_dens_lambda.mat = 1. / (1. / chi_dens_ladder_mat + lambda_dens)

    chi_magn_lambda = fp.LadderSusceptibility(qiw=qiw_grid.meshgrid, channel='magn', u=dmft1p['u'], beta=dmft1p['beta'])
    chi_magn_lambda.mat = 1. / (1. / chi_magn_ladder_mat + lambda_magn)

    if (verbose):
        outpfunc(realt.string_time('Lambda correction: '))
    # ------------------------------------------- DGA SCHWINGER-DYSON EQUATION ---------------------------------------------

    chi_dens_lambda_my_qiw = fp.LadderSusceptibility(qiw=qiw_grid.my_mesh, channel='dens', u=dmft1p['u'],
                                                     beta=dmft1p['beta'])
    chi_dens_lambda_my_qiw.mat = chi_dens_lambda.mat[qiw_grid.my_slice]

    chi_magn_lambda_my_qiw = fp.LadderSusceptibility(qiw=qiw_grid.my_mesh, channel='magn', u=dmft1p['u'],
                                                     beta=dmft1p['beta'])
    chi_magn_lambda_my_qiw.mat = chi_magn_lambda.mat[qiw_grid.my_slice]

    g_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=k_grid.get_grid_as_tuple(), hr=hr,
                                               sigma=dmft1p['sloc'])

    sigma_dens_dga = sde.sde_dga(dga_susc['vrg_dens'], chir=chi_dens_lambda_my_qiw, g_generator=g_generator,
                                 mu=dmft1p['mu'], qiw=qiw_grid.my_mesh, nq=nq_tot, box_sizes=box_sizes)
    sigma_magn_dga = sde.sde_dga(dga_susc['vrg_magn'], chir=chi_magn_lambda_my_qiw, g_generator=g_generator,
                                 mu=dmft1p['mu'], qiw=qiw_grid.my_mesh, nq=nq_tot, box_sizes=box_sizes)

    sigma_dens_dga_reduce = np.zeros(np.shape(sigma_dens_dga), dtype=complex)
    comm.Allreduce(sigma_dens_dga, sigma_dens_dga_reduce)
    sigma_magn_dga_reduce = np.zeros(np.shape(sigma_magn_dga), dtype=complex)
    comm.Allreduce(sigma_magn_dga, sigma_magn_dga_reduce)

    sigma_dens_dga = sigma_dens_dga_reduce
    sigma_magn_dga = sigma_magn_dga_reduce
    sigma_dga_nc = -1*sigma_dens_dga_reduce + 3*sigma_magn_dga_reduce + dmft_sde['hartree'] - 2 * dmft_sde['siw_magn'] + 2 * dmft_sde['siw_dens'] \
                   - dmft_sde['siw'] + dmft1p['sloc'][dmft1p[ 'niv'] - niv_urange:dmft1p['niv'] + niv_urange]
    sigma_dga = sigma_dens_dga_reduce + 3 * sigma_magn_dga_reduce - 2 * dmft_sde['siw_magn'] + dmft_sde['hartree'] - \
                dmft_sde['siw'] + dmft1p['sloc'][dmft1p['niv'] - niv_urange:dmft1p['niv'] + niv_urange]

    if (verbose):
        outpfunc(realt.string_time('DGA Schwinger-Dyson equation: '))

    dga_sde = {
        'sigma_dens': sigma_dens_dga,
        'sigma_magn': sigma_magn_dga,
        'sigma': sigma_dga,
        'sigma_nc': sigma_dga_nc,
    }

    chi_lambda = {
        'chi_dens_lambda': chi_dens_lambda,
        'chi_magn_lambda': chi_magn_lambda,
        'lambda_dens': lambda_dens,
        'lambda_magn': lambda_magn
    }

    chi_ladder = {
        'chi_dens_ladder': chi_dens_ladder,
        'chi_magn_ladder': chi_magn_ladder,
        'chi0q_core': chi0q_core,
        'chi0q_urange': chi0q_urange
    }

    return dga_sde, dmft_sde, dmft_gamma, chi_lambda, chi_ladder
