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
    wn_rpa = config['grids']['wn_rpa']
    niw_core = config['box_sizes']['niw_core']
    niv_invbse = config['box_sizes']['niv_invbse']
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
    use_urange_for_lc = config['options']['use_urange_for_lc']
    lambda_correction_type = config['options']['lambda_correction_type']

    k_grid = config['grids']['k_grid']
    q_grid = config['grids']['q_grid']

    nq_tot = q_grid.nk_tot
    nq_irr = q_grid.nk_irr

    # ----------------------------------------------- MPI DISTRIBUTION -------------------------------------------------
    my_iw = wn_core
    realt = rt.real_time()

    # -------------------------------------------LOAD G2 FROM W2DYN ----------------------------------------------------
    g2_file = w2dyn_aux.g4iw_file(fname=path + fname_g2)

    g2_dens_loc = fp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='dens', iw=my_iw), giw=giw, channel='dens',
                                    beta=beta, iw=my_iw)
    g2_magn_loc = fp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='magn', iw=my_iw), giw=giw, channel='magn',
                                    beta=beta, iw=my_iw)

    g2_dens_loc.cut_iv(niv_cut=niv_invbse)
    g2_magn_loc.cut_iv(niv_cut=niv_invbse)

    dmft1p['g2_dens'] = g2_dens_loc
    dmft1p['g2_magn'] = g2_magn_loc

    if (verbose):
        outpfunc("Finished reading g2 from file.")

    # --------------------------------------------- LOCAL DMFT SDE ---------------------------------------------------------
    dmft_sde = sde.local_dmft_sde_from_g2(dmft_input=dmft1p, box_sizes=box_sizes)
    local_rpa_sde = sde.local_rpa_sde_correction(dmft_input=dmft1p, box_sizes=box_sizes, iw=wn_rpa)

    dmft_sde['siw_rpa_dens'] = local_rpa_sde['siw_rpa_dens']
    dmft_sde['siw_rpa_magn'] = local_rpa_sde['siw_rpa_magn']

    chi_dens_loc_mat = dmft_sde['chi_dens'].mat
    chi_magn_loc_mat = dmft_sde['chi_magn'].mat
    chi0_core = dmft_sde['chi0_core'].chi0
    chi0_urange = dmft_sde['chi0_urange'].chi0
    chi0_asympt = dmft_sde['chi0_asympt'].chi0

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

    if (np.size(wn_rpa) > 0):
        dmft_sde['siw_dens'] = dmft_sde['siw_dens'] + dmft_sde['siw_rpa_dens']
        dmft_sde['siw_magn'] = dmft_sde['siw_magn'] + dmft_sde['siw_rpa_magn']
        dmft_sde['siw'] = dmft_sde['siw_dens'] + dmft_sde['siw_magn'] + dmft_sde['hartree']

    else:
        dmft_sde['siw'] = dmft_sde['siw'] + dmft_sde['hartree']
    dmft_sde['chi0_core'] = chi0_core
    dmft_sde['chi0_urange'] = chi0_urange
    dmft_sde['chi0_asympt'] = chi0_asympt

    if (verbose):
        outpfunc("Finished local part.")
        outpfunc(realt.string_time('Local Part '))
    # ------------------------------------------------ NON-LOCAL PART  -----------------------------------------------------
    # ======================================================================================================================

    qiw_distributor = mpiaux.MpiDistributor(ntasks=wn_core.size * nq_irr, comm=comm, output_path=output_path,
                                            name='Qiw')
    index_grid_keys = ('irrq', 'iw')
    qiw_grid = ind.IndexGrids(grid_arrays=(q_grid.irrk_ind_lin,) + (wn_core,), keys=index_grid_keys,
                              my_slice=qiw_distributor.my_slice)

    index_grid_keys_fbz = ('qx', 'qy', 'qz', 'iw')
    qiw_grid_fbz = ind.IndexGrids(grid_arrays=q_grid.grid + (wn_core,), keys=index_grid_keys_fbz)

    qiw_distributor_rpa = mpiaux.MpiDistributor(ntasks=wn_rpa.size * nq_irr, comm=comm, output_path=output_path,
                                                name='Qiw')
    qiw_grid_rpa = ind.IndexGrids(grid_arrays=(q_grid.irrk_ind_lin,) + (wn_rpa,), keys=index_grid_keys,
                                  my_slice=qiw_distributor_rpa.my_slice)

    # ----------------------------------------- NON-LOCAL RPA SUCEPTIBILITY  -------------------------------------------

    chi_rpa = fp.rpa_susceptibility(dmft_input=dmft1p, box_sizes=box_sizes, hr=hr, kgrid=k_grid.grid,
                                    qiw_indizes=qiw_grid_rpa.my_mesh, q_grid=q_grid)

    # ----------------------------------------- NON-LOCAL LADDER SUCEPTIBILITY  ----------------------------------------

    qiw_distributor.open_file()
    dga_susc = fp.dga_susceptibility(dmft_input=dmft1p, local_sde=dmft_gamma, hr=hr, kgrid=k_grid.grid,
                                     box_sizes=box_sizes, qiw_grid=qiw_grid.my_mesh, niw=niw_core,
                                     file=qiw_distributor.file, do_pairing_vertex=do_pairing_vertex, q_grid=q_grid)
    qiw_distributor.close_file()

    if (verbose):
        outpfunc(realt.string_time('Non-local Susceptibility: '))

    # ----------------------------------------------- LAMBDA-CORRECTION ------------------------------------------------

    if (use_urange_for_lc):
        chi_dens_rpa_mat = qiw_distributor_rpa.allgather(rank_result=chi_rpa['chi_rpa_dens'].mat)
        chi_magn_rpa_mat = qiw_distributor_rpa.allgather(rank_result=chi_rpa['chi_rpa_magn'].mat)

        chi_dens_rpa = fp.LadderSusceptibility(qiw=qiw_grid_rpa.meshgrid, channel='dens', u=dmft1p['u'],
                                               beta=dmft1p['beta'])
        chi_dens_rpa.mat = chi_dens_rpa_mat

        chi_magn_rpa = fp.LadderSusceptibility(qiw=qiw_grid_rpa.meshgrid, channel='magn', u=dmft1p['u'],
                                               beta=dmft1p['beta'])
        chi_magn_rpa.mat = chi_magn_rpa_mat

        chi_dens_rpa.mat = q_grid.irrk2fbz(mat=qiw_grid_rpa.reshape_matrix(chi_dens_rpa.mat))
        chi_magn_rpa.mat = q_grid.irrk2fbz(mat=qiw_grid_rpa.reshape_matrix(chi_magn_rpa.mat))
    else:
        chi_dens_rpa = []
        chi_magn_rpa = []

    chi_dens_ladder_mat = qiw_distributor.allgather(rank_result=dga_susc['chi_dens_asympt'].mat)
    chi_magn_ladder_mat = qiw_distributor.allgather(rank_result=dga_susc['chi_magn_asympt'].mat)

    chi0q_core = qiw_distributor.allgather(rank_result=dga_susc['chi0q_core'].mat)
    chi0q_urange = qiw_distributor.allgather(rank_result=dga_susc['chi0q_urange'].mat)
    chi0q_asympt = qiw_distributor.allgather(rank_result=dga_susc['chi0q_asympt'].mat)

    chi_dens_ladder = fp.LadderSusceptibility(qiw=qiw_grid_fbz.meshgrid, channel='dens', u=dmft1p['u'],
                                              beta=dmft1p['beta'])

    chi_magn_ladder = fp.LadderSusceptibility(qiw=qiw_grid_fbz.meshgrid, channel='magn', u=dmft1p['u'],
                                              beta=dmft1p['beta'])

    # Rebuild the fbz for the lambda correction routine:
    chi_dens_ladder.mat = q_grid.irrk2fbz(mat=qiw_grid.reshape_matrix(chi_dens_ladder_mat))
    chi_magn_ladder.mat = q_grid.irrk2fbz(mat=qiw_grid.reshape_matrix(chi_magn_ladder_mat))

    lambda_ = lc.lambda_correction(chi_magn_ladder=chi_magn_ladder, chi_dens_ladder=chi_dens_ladder,
                                   chi_dens_rpa=chi_dens_rpa, chi_magn_rpa=chi_magn_rpa, chi_magn_dmft=chi_magn_loc,
                                   chi_dens_dmft=chi_dens_loc, chi_magn_rpa_loc=local_rpa_sde['chi_rpa_magn'],
                                   chi_dens_rpa_loc=local_rpa_sde['chi_rpa_magn'], nq=np.prod(q_grid.nk),
                                   use_rpa_for_lc=use_urange_for_lc)

    if (lambda_correction_type == 'spch'):
        lambda_dens = lambda_['lambda_dens_single']
        lambda_magn = lambda_['lambda_magn_single']
    elif (lambda_correction_type == 'sp'):
        lambda_dens = 0.0
        lambda_magn = lambda_['lambda_magn_totdens']
    elif (lambda_correction_type == 'sp_only'):
        lambda_dens = 0.0
        lambda_magn = lambda_['lambda_magn_single']
    elif (lambda_correction_type == 'none'):
        lambda_dens = 0.0
        lambda_magn = 0.0
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

    g_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=k_grid.grid, hr=hr,
                                               sigma=dmft1p['sloc'])

    sigma_dens_dga = sde.sde_dga(dga_susc['vrg_dens'], chir=chi_dens_lambda_my_qiw, g_generator=g_generator,
                                 mu=dmft1p['mu'], qiw_grid=qiw_grid.my_mesh, nq=nq_tot, box_sizes=box_sizes, q_grid=q_grid)
    sigma_magn_dga = sde.sde_dga(dga_susc['vrg_magn'], chir=chi_magn_lambda_my_qiw, g_generator=g_generator,
                                 mu=dmft1p['mu'], qiw_grid=qiw_grid.my_mesh, nq=nq_tot, box_sizes=box_sizes, q_grid=q_grid)

    sigma_dens_dga_reduce = np.zeros(np.shape(sigma_dens_dga), dtype=complex)
    comm.Allreduce(sigma_dens_dga, sigma_dens_dga_reduce)
    sigma_magn_dga_reduce = np.zeros(np.shape(sigma_magn_dga), dtype=complex)
    comm.Allreduce(sigma_magn_dga, sigma_magn_dga_reduce)

    sigma_dens_dga = k_grid.symmetrize_irrk(mat=sigma_dens_dga_reduce)
    sigma_magn_dga = k_grid.symmetrize_irrk(mat=sigma_magn_dga_reduce)

    sigma_dens_rpa = sde.rpa_sde(chir=chi_rpa['chi_rpa_dens'], g_generator=g_generator, niv_giw=niv_urange,
                                 mu=dmft1p['mu'], nq=nq_tot, u=u, qiw_grid=qiw_grid_rpa.my_mesh, q_grid=q_grid)
    sigma_magn_rpa = sde.rpa_sde(chir=chi_rpa['chi_rpa_magn'], g_generator=g_generator, niv_giw=niv_urange,
                                 mu=dmft1p['mu'], nq=nq_tot, u=u, qiw_grid=qiw_grid_rpa.my_mesh, q_grid=q_grid)

    sigma_dens_rpa_reduce = np.zeros(np.shape(sigma_dens_rpa), dtype=complex)
    comm.Allreduce(sigma_dens_rpa, sigma_dens_rpa_reduce)
    sigma_magn_rpa_reduce = np.zeros(np.shape(sigma_magn_rpa), dtype=complex)
    comm.Allreduce(sigma_magn_rpa, sigma_magn_rpa_reduce)

    # Sigma needs to be symmetrized within the corresponding BZ:
    sigma_dens_rpa = k_grid.symmetrize_irrk(mat=sigma_dens_rpa_reduce)
    sigma_magn_rpa = k_grid.symmetrize_irrk(mat=sigma_magn_rpa_reduce)

    if (wn_rpa.size > 0):
        sigma_dens_dga = sigma_dens_dga + sigma_dens_rpa
        sigma_magn_dga = sigma_magn_dga + sigma_magn_rpa

    sigma_dga = -1 * sigma_dens_dga + 3 * sigma_magn_dga + dmft_sde['hartree'] - 2 * dmft_sde['siw_magn'] + 2 * \
                dmft_sde['siw_dens'] \
                - dmft_sde['siw'] + dmft1p['sloc'][dmft1p['niv'] - niv_urange:dmft1p['niv'] + niv_urange]
    sigma_dga_nc = sigma_dens_dga + 3 * sigma_magn_dga - 2 * dmft_sde['siw_magn'] + dmft_sde['hartree'] - \
                   dmft_sde['siw'] + dmft1p['sloc'][dmft1p['niv'] - niv_urange:dmft1p['niv'] + niv_urange]


    if (verbose):
        outpfunc(realt.string_time('DGA Schwinger-Dyson equation: '))

    dga_sde = {
        'sigma_dens': sigma_dens_dga,
        'sigma_magn': sigma_magn_dga,
        'sigma_dens_rpa': sigma_dens_rpa,
        'sigma_magn_rpa': sigma_magn_rpa,
        'sigma': sigma_dga,
        'sigma_nc': sigma_dga_nc
    }

    chi_dens_lambda.mat = q_grid.irrk2fbz(mat=qiw_grid.reshape_matrix(chi_dens_lambda.mat))
    chi_magn_lambda.mat = q_grid.irrk2fbz(mat=qiw_grid.reshape_matrix(chi_magn_lambda.mat))

    chi_dens_lambda.qiw = qiw_grid_fbz.meshgrid
    chi_magn_lambda.qiw = qiw_grid_fbz.meshgrid

    chi_lambda = {
        'chi_dens_lambda': chi_dens_lambda,
        'chi_magn_lambda': chi_magn_lambda,
        'lambda_dens': lambda_dens,
        'lambda_magn': lambda_magn
    }

    chi0q_core = q_grid.irrk2fbz(mat=qiw_grid.reshape_matrix(chi0q_core))
    chi0q_urange = q_grid.irrk2fbz(mat=qiw_grid.reshape_matrix(chi0q_urange))
    chi0q_asympt = q_grid.irrk2fbz(mat=qiw_grid.reshape_matrix(chi0q_asympt))

    chi_ladder = {
        'chi_dens_ladder': chi_dens_ladder,
        'chi_magn_ladder': chi_magn_ladder,
        'chi0q_core': chi0q_core,
        'chi0q_urange': chi0q_urange,
        'chi0q_asympt': chi0q_asympt
    }

    return dga_sde, dmft_sde, dmft_gamma, chi_lambda, chi_ladder
