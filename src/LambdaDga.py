# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Contains the wrapper function for single-band Lambda-corrected DGA.


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import w2dyn_aux
import MpiAux as mpiaux
import TwoPoint as twop
import FourPoint as fp
import SDE as sde
import RealTime as rt
import Indizes as ind
import LambdaCorrection as lc
import MatsubaraFrequencies as mf
import Plotting as plotting
import Config as configs
import LocalRoutines as lr
import PairingVertex as pv


# --------------------------------------------- LAMBDA DGA CLASS -------------------------------------------------------


class LambdaDga():
    ''' Class to handle one step of lambda corrected DGA.'''

    sde_loc = None  # Output from the local Schwinger dyson equation
    chi_loc = None  # Local physical susceptibility
    f_loc = None # Local vertex
    chi_rpa_loc = None  # Local rpa physical susceptibility
    vrg = None # non-local spin-fermion vertex
    chi = None # non-local susceptibility
    chi_rpa = None # non-local rpa susceptibility
    lambda_ = None # Lambda for lambda-correction

    f_loc_chi0 = None # F_loc * chi0_q for lambda-dga local double counting correction


    sigma_dmft = None # DMFT self-energy
    sigma_dga = None # Self-energy from dga schwinger dyson equation
    sigma_rpa = None # Self-energy from rpa schinger dyson equation
    sigma = None # Self-energy as obtained by DGA with RPA and box-size corrections
    sigma_nc = None # Self-energy as obtained by DGA, but slightly different (yeah I know that comment is kinda useless)

    sigma_com = None
    sigma_com_loc = None # Components of the self-energy originating from different scattering channels

    def __init__(self, config: configs.DgaConfig = None, comm=None, sigma_dmft = None,gloc_dmft=None, sigma_start=None, gamma_magn=None, gamma_dens=None, adjust_mu=True,use_gloc_dmft=False):
        self.comm = comm  # MPI communicator
        self.sigma_dmft = sigma_dmft # Self-energy of DMFT
        self.sigma_start = sigma_start  # Input self-energy
        self.conf = config  # Config parameter settings. At a later stage this will possibly be integrated directly here.
        self.gamma_magn = gamma_magn  # DMFT local irreducible ph vertex in the magnetic channel
        self.gamma_dens = gamma_dens  # DMFT local irreducible ph vertex in the density channel

        # Set up the Green's function generator:
        self.g_gen = twop.GreensFunctionGenerator(beta=self.beta, kgrid=self.k_grid, hr=self.hr, sigma=sigma_start)

        # I am not yet sure if they should be set at the start, or rather computed on demand. But time will tell.
        if(adjust_mu):
            self.mu = self.g_gen.adjust_mu(n=self.n, mu0=self.mu_dmft)
        else:
            self.mu = self.mu_dmft
        self.gk = self.g_gen.generate_gk(mu=self.mu, niv=self.niv_urange + self.niw_urange)
        #%

        if(gloc_dmft is not None and config.opt.use_gloc_dmft is True):
            self.g_loc = mf.cut_v_1d(gloc_dmft,niv_cut=config.box.niv_padded)
        else:
            self.g_loc = self.gk.k_mean()

        # ----------------------------------------- CREATE MPI DISTRIBUTORS ----------------------------------------------------
        self.qiw_distributor = mpiaux.MpiDistributor(ntasks=self.conf.box.wn_core_plus.size * self.conf.q_grid.nk_irr,
                                                     comm=self.comm,
                                                     output_path=self.conf.nam.output_path,
                                                     name='Qiw')
        index_grid_keys = ('irrq', 'iw')
        self.qiw_grid = ind.IndexGrids(grid_arrays=(self.conf.q_grid.irrk_ind_lin,) + (self.conf.box.wn_core_plus,),
                                       keys=index_grid_keys,
                                       my_slice=self.qiw_distributor.my_slice)

        index_grid_keys_fbz = ('qx', 'qy', 'qz', 'iw')
        self.qiw_grid_fbz = ind.IndexGrids(grid_arrays=self.conf.q_grid.grid + (self.conf.box.wn_core,),
                                           keys=index_grid_keys_fbz)

        self.qiw_distributor_rpa = mpiaux.MpiDistributor(
            ntasks=self.conf.box.wn_rpa_plus.size * self.conf.q_grid.nk_irr,
            comm=self.comm)
        self.qiw_grid_rpa = ind.IndexGrids(grid_arrays=(self.conf.q_grid.irrk_ind_lin,) + (self.conf.box.wn_rpa_plus,),
                                           keys=index_grid_keys,
                                           my_slice=self.qiw_distributor_rpa.my_slice)


        index_grid_keys_fbz = ('qx', 'qy', 'qz', 'iw')
        self.qiw_grid_rpa_fbz = ind.IndexGrids(grid_arrays=self.conf.q_grid.grid + (self.conf.box.wn_rpa_plus,),
                                           keys=index_grid_keys_fbz)

    @property
    def is_root(self):
        return self.comm.rank == 0

    @property
    def beta(self):
        return self.conf.sys.beta

    @property
    def niv_urange(self):
        return self.conf.box.niv_urange

    @property
    def niw_core(self):
        return self.conf.box.niw_core

    @property
    def niw_urange(self):
        return self.conf.box.niw_urange

    @property
    def niv_core(self):
        return self.conf.box.niv_core

    @property
    def niv_pp(self):
        return self.conf.box.niv_pp

    @property
    def k_grid(self):
        return self.conf.k_grid

    @property
    def q_grid(self):
        return self.conf.q_grid

    @property
    def hr(self):
        return self.conf.sys.hr

    @property
    def mu_dmft(self):
        return self.conf.sys.mu_dmft

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    @property
    def n(self):
        return self.conf.sys.n

    @property
    def u(self):
        return self.conf.sys.u

    @property
    def gamma_dmft(self):
        return {'dens': self.gamma_dens,
                'magn': self.gamma_magn}

    @property
    def my_qiw_mesh(self):
        return self.qiw_grid.my_mesh

    @property
    def my_rpa_qiw_mesh(self):
        return self.qiw_grid_rpa.my_mesh

    @property
    def my_n_qiw_tasks(self):
        return self.qiw_grid.my_n_tasks

    @property
    def my_n_rpa_qiw_tasks(self):
        return self.qiw_grid_rpa.my_n_tasks

    def local_sde(self, safe_output=False, use_rpa_correction=True, interactive=False):
        self.sde_loc, self.chi_loc, vrg_loc, self.f_loc = lr.local_dmft_sde_from_gamma(dga_conf=self.conf,
                                                                                          giw=self.g_loc,
                                                                                          gamma_dmft=self.gamma_dmft)

        if (use_rpa_correction):
            rpa_sde_loc, self.chi_rpa_loc = sde.local_rpa_sde_correction(dga_conf=self.conf, giw=self.g_loc,
                                                                    box_sizes=self.conf.box,
                                                                    iw=self.conf.box.wn_rpa)
            self.sde_loc = lr.add_rpa_correction(dmft_sde=self.sde_loc, rpa_sde_loc=rpa_sde_loc,
                                                 wn_rpa=self.conf.box.wn_rpa)

        if (safe_output and self.is_root):
            np.save(self.conf.nam.output_path + 'dmft_sde.npy', self.sde_loc, allow_pickle=True)
            np.save(self.conf.nam.output_path + 'chi_dmft.npy', self.chi_loc, allow_pickle=True)
            np.save(self.conf.nam.output_path + 'vrg_dmft.npy', vrg_loc, allow_pickle=True)
            np.save(self.conf.nam.output_path + 'f_dmft.npy', self.f_loc, allow_pickle=True)

        if (interactive):
            if (self.is_root): plotting.plot_vrg_dmft(vrg_dmft=vrg_loc, beta=self.beta,
                                                      niv_plot=self.niv_urange,
                                                      output_path=self.conf.nam.output_path)

    def dga_bse_susceptibility(self, save_vrg=False):
        self.qiw_distributor.open_file()
        chi_dens = fp.LadderSusceptibility(channel='dens', beta=self.beta, u=self.u, qiw=self.my_qiw_mesh)
        chi_magn = fp.LadderSusceptibility(channel='magn', beta=self.beta, u=self.u, qiw=self.my_qiw_mesh)

        vrg_dens = fp.LadderObject(qiw=self.qiw_grid.my_mesh, channel='dens', beta=self.beta, u=self.u)
        vrg_magn = fp.LadderObject(qiw=self.qiw_grid.my_mesh, channel='magn', beta=self.beta, u=self.u)

        # floc_chi0_dens = fp.LadderObject(qiw=self.qiw_grid.my_mesh, channel='dens', beta=self.beta, u=self.u)
        # floc_chi0_magn = fp.LadderObject(qiw=self.qiw_grid.my_mesh, channel='magn', beta=self.beta, u=self.u)

        for iqw in range(self.my_n_qiw_tasks):
            qiw, wn = self.get_qiw(iqw=iqw,qiw_mesh=self.my_qiw_mesh)
            wn_lin = np.array(mf.cen2lin(wn, -self.conf.box.niw_core), dtype=int)

            gkpq_urange = self.g_gen.generate_gk(mu=self.mu, qiw=qiw, niv=self.niv_urange)

            chi0q_core = fp.Bubble(gk=self.gk.get_gk_cut_iv(niv_cut=self.niv_core),
                                   gkpq=gkpq_urange.get_gk_cut_iv(niv_cut=self.niv_core), beta=self.beta, wn=wn)
            chi0q_core.set_gchi0()

            chi0q_urange = fp.Bubble(gk=self.gk.get_gk_cut_iv(niv_cut=self.niv_urange), gkpq=gkpq_urange.gk, beta=self.beta, wn=wn)

            chi0q_urange.set_gchi0()

            gchi_aux_dens = fp.construct_gchi_aux(gammar=self.gamma_dens, gchi0=chi0q_core, u=self.u, wn_lin=wn_lin)
            gchi_aux_magn = fp.construct_gchi_aux(gammar=self.gamma_magn, gchi0=chi0q_core, u=self.u, wn_lin=wn_lin)

            chi_aux_dens = fp.susceptibility_from_four_point(four_point=gchi_aux_dens)
            chi_aux_magn = fp.susceptibility_from_four_point(four_point=gchi_aux_magn)

            chiq_dens = fp.chi_phys_from_chi_aux(chi_aux=chi_aux_dens, chi0_urange=chi0q_urange,
                                                 chi0_core=chi0q_core)

            chiq_magn = fp.chi_phys_from_chi_aux(chi_aux=chi_aux_magn, chi0_urange=chi0q_urange,
                                                 chi0_core=chi0q_core)

            vrgq_dens, vrgq_dens_core = fp.fermi_bose_from_chi_aux_urange(gchi_aux=gchi_aux_dens, gchi0=chi0q_core,
                                                                          niv_urange=self.niv_urange)
            vrgq_dens = fp.fermi_bose_asympt(vrg=vrgq_dens, chi_urange=chiq_dens)
            vrgq_magn, vrgq_magn_core = fp.fermi_bose_from_chi_aux_urange(gchi_aux=gchi_aux_magn, gchi0=chi0q_core,
                                                                          niv_urange=self.niv_urange)
            vrgq_magn = fp.fermi_bose_asympt(vrg=vrgq_magn, chi_urange=chiq_magn)

            # floc_chi0_dens.ladder[iqw] = np.sum(self.f_loc['dens'][wn_lin,:,:] * chi0q_urange.gchi0[None,:], axis=-1)
            # floc_chi0_magn.ladder[iqw] = np.sum(self.f_loc['magn'][wn_lin,:,:] * chi0q_urange.gchi0[None,:], axis=-1)

            chi_dens.mat[iqw] = chiq_dens.mat_asympt
            chi_magn.mat[iqw] = chiq_magn.mat_asympt

            vrg_dens.ladder[iqw] = vrgq_dens
            vrg_magn.ladder[iqw] = vrgq_magn

            if (self.conf.opt.do_pairing_vertex):
                self.save_ladder_vertex(file=self.qiw_distributor.file, iqw=iqw, wn=wn, gchi=gchi_aux_magn.mat,
                                        vrg=vrgq_magn_core.mat, gchi0=chi0q_core.gchi0, channel=vrgq_magn_core.channel,
                                        save_condition=True)
                self.save_ladder_vertex(file=self.qiw_distributor.file, iqw=iqw, wn=wn, gchi=gchi_aux_dens.mat,
                                        vrg=vrgq_dens_core.mat, gchi0=chi0q_core.gchi0,
                                        channel=vrgq_dens_core.channel,
                                        save_condition=False)

            # Save the lowest 5 frequencies for the spin-fermion vertex::
            if (np.abs(wn) < self.conf.box.niw_vrg_save and save_vrg == True):
                group = '/irrq{:03d}wn{:04d}/'.format(*self.my_qiw_mesh[iqw])
                self.qiw_distributor.file[group + 'vrg_magn/'] = self.beta * vrgq_magn.mat[self.niv_urange - self.conf.box.niv_vrg_save:self.niv_urange + self.conf.box.niv_vrg_save]
                self.qiw_distributor.file[group + 'vrg_dens/'] = self.beta * vrgq_dens.mat[self.niv_urange - self.conf.box.niv_vrg_save:self.niv_urange + self.conf.box.niv_vrg_save]

        chi_dens.mat_to_array()
        chi_magn.mat_to_array()

        vrg_dens.set_qiw_mat()
        vrg_magn.set_qiw_mat()

        # floc_chi0_dens._mat = np.array(floc_chi0_dens.ladder)
        # floc_chi0_magn._mat = np.array(floc_chi0_magn.ladder)

        self.chi = {
            'dens': chi_dens,
            'magn': chi_magn
        }
        self.vrg = {
            'dens': vrg_dens,
            'magn': vrg_magn
        }
        self.qiw_distributor.close_file()

        # self.f_loc_chi0 = {
        #     'dens': floc_chi0_dens,
        #     'magn': floc_chi0_magn
        # }

    def rpa_susceptibility(self):
        chi_rpa_dens = fp.LadderSusceptibility(channel='dens', beta=self.beta, u=self.u, qiw=self.my_rpa_qiw_mesh)
        chi_rpa_magn = fp.LadderSusceptibility(channel='magn', beta=self.beta, u=self.u, qiw=self.my_rpa_qiw_mesh)

        for iqw in range(self.my_n_rpa_qiw_tasks):
            qiw, wn = self.get_qiw(iqw=iqw,qiw_mesh=self.my_rpa_qiw_mesh)
            gkpq_urange = self.g_gen.generate_gk(mu=self.mu, qiw=qiw, niv=self.niv_urange)

            chi0q_urange = fp.Bubble(gk=self.gk.get_gk_cut_iv(niv_cut=self.niv_urange), gkpq=gkpq_urange.gk, beta=self.beta, wn=wn)

            chiq_dens = fp.chi_rpa(chi0_urange=chi0q_urange, channel='dens', u=self.u)
            chiq_magn = fp.chi_rpa(chi0_urange=chi0q_urange, channel='magn', u=self.u)

            chi_rpa_dens.mat[iqw] = chiq_dens.mat_asympt
            chi_rpa_magn.mat[iqw] = chiq_magn.mat_asympt

        chi_rpa_dens.mat_to_array()
        chi_rpa_magn.mat_to_array()

        self.chi_rpa = {
            'dens': chi_rpa_dens,
            'magn': chi_rpa_magn
        }

    def dga_ladder_susc_allgather_qiw_and_build_fbziw(self,channel=None):
        return fp.ladder_susc_allgather_qiw_and_build_fbziw(dga_conf=self.conf, distributor=self.qiw_distributor,
                                                            mat=self.chi[channel].mat, qiw_grid=self.qiw_grid,
                                                            qiw_grid_fbz=self.qiw_grid_fbz, channel=channel)

    def rpa_ladder_susc_allgather_qiw_and_build_fbziw(self,channel=None):
        return fp.ladder_susc_allgather_qiw_and_build_fbziw(dga_conf=self.conf, distributor=self.qiw_distributor_rpa,
                                                            mat=self.chi_rpa[channel].mat, qiw_grid=self.qiw_grid_rpa,
                                                            qiw_grid_fbz=self.qiw_grid_rpa_fbz, channel=channel)

    def lambda_correction(self, save_output=False):
        chi_dens = self.dga_ladder_susc_allgather_qiw_and_build_fbziw(channel='dens')
        chi_magn = self.dga_ladder_susc_allgather_qiw_and_build_fbziw(channel='magn')
        chi = {'dens':chi_dens,
               'magn':chi_magn}
        chi_rpa_dens = self.rpa_ladder_susc_allgather_qiw_and_build_fbziw(channel='magn')
        chi_rpa_magn = self.rpa_ladder_susc_allgather_qiw_and_build_fbziw(channel='dens')
        chi_rpa = {'dens': chi_rpa_dens,
               'magn': chi_rpa_magn}
        self.lambda_, n_lambda, chi_sum, lambda_start = lc.lambda_correction(dga_conf=self.conf, chi_ladder=chi, chi_rpa=chi_rpa,
                                                 chi_dmft=self.chi_loc,
                                                 chi_rpa_loc=self.chi_rpa_loc)

        if(self.is_root): fp.save_and_plot_chi_lambda(dga_conf=self.conf, chi_lambda=chi,name='ladder')
        lc.build_chi_lambda(dga_conf=self.conf, chi_ladder=self.chi, chi_rpa=self.chi_rpa, lambda_=self.lambda_)

        if(self.is_root and save_output):
            lc.build_chi_lambda(dga_conf=self.conf, chi_ladder=chi, chi_rpa=chi_rpa, lambda_=self.lambda_)
            np.save(self.conf.nam.output_path + 'chi_lambda.npy', chi, allow_pickle=True)
            fp.save_and_plot_chi_lambda(dga_conf=self.conf, chi_lambda=chi)
            string_temp = 'Number of positive frequencies used for {}: {}'
            np.savetxt(self.conf.nam.output_path + 'n_lambda_correction.txt',
                       [string_temp.format('magn', n_lambda['magn']), string_temp.format('dens', n_lambda['dens'])],
                       delimiter=' ', fmt='%s')
            string_temp = 'Lambda for {}: {}'
            np.savetxt(self.conf.nam.output_path + 'lambda.txt',
                       [string_temp.format('magn', self.lambda_['magn']), string_temp.format('dens', self.lambda_['dens'])],
                       delimiter=' ', fmt='%s')
            np.savetxt(self.conf.nam.output_path + 'lambda_start.txt',
                       [string_temp.format('magn', lambda_start['magn']),
                        string_temp.format('dens', lambda_start['dens'])],
                       delimiter=' ', fmt='%s')
            np.savetxt(self.conf.nam.output_path + 'chi_sum.txt',
                       [string_temp.format('magn', chi_sum['magn']),
                        string_temp.format('dens', chi_sum['dens']),
                        string_temp.format('dens_ladder', chi_sum['dens_ladder']),
                        string_temp.format('magn_ladder', chi_sum['magn_ladder'])],
                       delimiter=' ', fmt='%s')

    def dga_sde(self, interactive=False):
        self.sigma_dga = sde.sde_dga_wrapper(dga_conf=self.conf, vrg=self.vrg, f_loc_chi0=self.f_loc_chi0, chi=self.chi,
                                                              qiw_mesh=self.my_qiw_mesh,
                                                              sigma_input=self.sigma_start, mu_input=self.mu, distributor=self.qiw_distributor)

        self.sigma_rpa = sde.rpa_sde_wrapper(dga_conf=self.conf, sigma_input=self.sigma_start,mu_input=self.mu, chi=self.chi_rpa,
                                        qiw_grid=self.qiw_grid_rpa,
                                        distributor=self.qiw_distributor_rpa)



        if(interactive):
            if self.is_root: np.save(self.conf.nam.output_path + 'sigma_dga.npy', self.sigma_dga, allow_pickle=True)

    def build_dga_sigma(self, interactive=False):
        # Here I am not sure if one has to use sigma_dmft instead of sigma_start for self-consistency
        self.sigma, self.sigma_nc = sde.build_dga_sigma(dga_conf=self.conf, sigma_dga=self.sigma_dga, sigma_rpa=self.sigma_rpa,
                                              dmft_sde=self.sde_loc, sigma_dmft=self.sigma_dmft)
        if(interactive):
            if self.is_root: np.save(self.conf.nam.output_path + 'sigma.npy', self.sigma, allow_pickle=True)
            if self.is_root: np.save(self.conf.nam.output_path + 'sigma_nc.npy', self.sigma_nc, allow_pickle=True)
            if self.is_root: plotting.sigma_plots(dga_conf=self.conf, sigma_dga=self.sigma_dga, dmft_sde=self.sde_loc,
                                                    sigma_loc=self.sigma_dmft, sigma=self.sigma, sigma_nc=self.sigma_nc)
            if self.is_root: plotting.giwk_plots(dga_conf=self.conf, sigma=self.sigma, input_mu=self.mu,
                                                    output_path=self.conf.nam.output_path)


    def get_qiw(self, iqw=None, qiw_mesh=None):
        wn = qiw_mesh[iqw][-1]
        q_ind = qiw_mesh[iqw][0]
        q = self.q_grid.irr_kmesh[:, q_ind]
        qiw = np.append(-q, wn)  # WARNING: Here I am not sure if it should be +q or -q.
        return qiw, wn

    def save_ladder_vertex(self, file=None, iqw=None, wn=None, gchi=None, vrg=None, gchi0=None, channel=None,
                           save_condition=False):
        omega = pv.get_omega_condition(niv_pp=self.niv_pp)
        if np.abs(wn) < 2 * self.niv_pp:
            condition = omega == wn

            f1_slice, f2_slice = pv.ladder_vertex_from_chi_aux_components(gchi_aux=gchi,
                                                                          vrg=vrg,
                                                                          gchi0=gchi0,
                                                                          beta=self.beta,
                                                                          u_r=fp.get_ur(u=self.u,
                                                                                        channel=channel))

            group = '/irrq{:03d}wn{:04d}/'.format(*self.my_qiw_mesh[iqw])
            file[group + f'f1_{channel}/'] = pv.get_pp_slice_4pt(mat=f1_slice, condition=condition,
                                                                 niv_pp=self.niv_pp)
            file[group + f'f2_{channel}/'] = pv.get_pp_slice_4pt(mat=f2_slice, condition=condition,
                                                                 niv_pp=self.niv_pp)

            if save_condition: file[group + 'condition/'] = condition


# -------------------------------------- LAMBDA DGA FUNCTION WRAPPER ---------------------------------------------------

def lambda_dga(config=None, verbose=False, outpfunc=None):
    ''' Wrapper function for the \lambda-corrected one-band DGA routine. All relevant settings are contained in config'''
    # -------------------------------------------- UNRAVEL CONFIG ------------------------------------------------------
    # This is done to allow for more flexibility in the future

    comm = config['comm']
    wn_core = config['grids']['wn_core']
    wn_core_plus = config['grids']['wn_core_plus']
    wn_rpa = config['grids']['wn_rpa']
    wn_rpa_plus = config['grids']['wn_rpa_plus']
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
    lc_use_only_positive = config['options']['lc_use_only_positive']
    analyse_spin_fermion_contributions = config['options']['analyse_spin_fermion_contributions']

    k_grid = config['grids']['k_grid']
    q_grid = config['grids']['q_grid']

    nq_tot = q_grid.nk_tot
    nq_irr = q_grid.nk_irr

    # ----------------------------------------------- MPI DISTRIBUTION -------------------------------------------------
    my_iw = wn_core
    realt = rt.real_time()
    realt.create_file(fname=output_path + 'cpu_time_lambda_dga.txt')

    # -------------------------------------------LOAD G2 FROM W2DYN ----------------------------------------------------
    g2_file = w2dyn_aux.g4iw_file(fname=path + fname_g2)

    g2_dens_loc = fp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='dens', iw=my_iw), giw=giw, channel='dens',
                                    beta=beta, iw=my_iw)
    g2_magn_loc = fp.LocalFourPoint(matrix=g2_file.read_g2_iw(channel='magn', iw=my_iw), giw=giw, channel='magn',
                                    beta=beta, iw=my_iw)

    g2_dens_loc.cut_iv(niv_cut=niv_invbse)
    g2_magn_loc.cut_iv(niv_cut=niv_invbse)

    realt.write_time_to_file(string='Reading G2 from file:', rank=comm.rank)

    # --------------------------------------------- LOCAL DMFT SDE ---------------------------------------------------------
    dmft_sde = sde.local_dmft_sde_from_g2(dmft_input=dmft1p, box_sizes=box_sizes, g2_dens=g2_dens_loc,
                                          g2_magn=g2_magn_loc)
    local_rpa_sde = sde.local_rpa_sde_correction(dmft_input=dmft1p, box_sizes=box_sizes, iw=wn_rpa)

    dmft_sde['siw_rpa_dens'] = local_rpa_sde['siw_rpa_dens']
    dmft_sde['siw_rpa_magn'] = local_rpa_sde['siw_rpa_magn']

    chi_dens_loc_mat = dmft_sde['chi_dens'].mat
    chi_magn_loc_mat = dmft_sde['chi_magn'].mat

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

    realt.write_time_to_file(string='Local Part:', rank=comm.rank)
    # ------------------------------------------------ NON-LOCAL PART  -----------------------------------------------------
    # ======================================================================================================================

    qiw_distributor = mpiaux.MpiDistributor(ntasks=wn_core_plus.size * nq_irr, comm=comm, output_path=output_path,
                                            name='Qiw')
    index_grid_keys = ('irrq', 'iw')
    qiw_grid = ind.IndexGrids(grid_arrays=(q_grid.irrk_ind_lin,) + (wn_core_plus,), keys=index_grid_keys,
                              my_slice=qiw_distributor.my_slice)

    index_grid_keys_fbz = ('qx', 'qy', 'qz', 'iw')
    qiw_grid_fbz = ind.IndexGrids(grid_arrays=q_grid.grid + (wn_core,), keys=index_grid_keys_fbz)

    qiw_distributor_rpa = mpiaux.MpiDistributor(ntasks=wn_rpa_plus.size * nq_irr, comm=comm, output_path=output_path,
                                                name='Qiw')
    qiw_grid_rpa = ind.IndexGrids(grid_arrays=(q_grid.irrk_ind_lin,) + (wn_rpa_plus,), keys=index_grid_keys,
                                  my_slice=qiw_distributor_rpa.my_slice)

    # ----------------------------------------- NON-LOCAL RPA SUCEPTIBILITY  -------------------------------------------

    chi_rpa = fp.rpa_susceptibility(dmft_input=dmft1p, box_sizes=box_sizes, hr=hr, kgrid=k_grid.grid,
                                    qiw_indizes=qiw_grid_rpa.my_mesh, q_grid=q_grid)

    realt.write_time_to_file(string='Non-local RPA SDE:', rank=comm.rank)

    # ----------------------------------------- NON-LOCAL LADDER SUCEPTIBILITY  ----------------------------------------

    qiw_distributor.open_file()
    dga_susc = fp.dga_susceptibility(dmft_input=dmft1p, local_sde=dmft_gamma, hr=hr, kgrid=k_grid.grid,
                                     box_sizes=box_sizes, qiw_grid=qiw_grid.my_mesh, niw=niw_core,
                                     file=qiw_distributor.file, do_pairing_vertex=do_pairing_vertex, q_grid=q_grid)
    qiw_distributor.close_file()

    realt.write_time_to_file(string='Non-local Susceptibility:', rank=comm.rank)

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

        chi_dens_rpa.mat = mf.wplus2wfull(mat=chi_dens_rpa.mat)
        chi_magn_rpa.mat = mf.wplus2wfull(mat=chi_magn_rpa.mat)
    else:
        chi_dens_rpa = []
        chi_magn_rpa = []

    chi_dens_ladder_mat = qiw_distributor.allgather(rank_result=dga_susc['chi_dens_asympt'].mat)
    chi_magn_ladder_mat = qiw_distributor.allgather(rank_result=dga_susc['chi_magn_asympt'].mat)

    chi_dens_ladder = fp.LadderSusceptibility(qiw=qiw_grid_fbz.meshgrid, channel='dens', u=dmft1p['u'],
                                              beta=dmft1p['beta'])

    chi_magn_ladder = fp.LadderSusceptibility(qiw=qiw_grid_fbz.meshgrid, channel='magn', u=dmft1p['u'],
                                              beta=dmft1p['beta'])

    # Rebuild the fbz for the lambda correction routine:
    chi_dens_ladder.mat = q_grid.irrk2fbz(mat=qiw_grid.reshape_matrix(chi_dens_ladder_mat))
    chi_magn_ladder.mat = q_grid.irrk2fbz(mat=qiw_grid.reshape_matrix(chi_magn_ladder_mat))

    # Recreate the full omega dependency:
    chi_dens_ladder.mat = mf.wplus2wfull(mat=chi_dens_ladder.mat)
    chi_magn_ladder.mat = mf.wplus2wfull(mat=chi_magn_ladder.mat)

    if (qiw_distributor.my_rank == 0):
        chi_ladder = {
            'chi_dens_ladder': chi_dens_ladder,
            'chi_magn_ladder': chi_magn_ladder,
        }
        np.save(output_path + 'chi_ladder.npy', chi_ladder, allow_pickle=True)
        plotting.plot_chi_fs(chi=chi_ladder['chi_magn_ladder'].mat.real, output_path=output_path, kgrid=q_grid,
                             name='magn_ladder_w0')

    lambda_ = lc.lambda_correction(chi_magn_ladder=chi_magn_ladder, chi_dens_ladder=chi_dens_ladder,
                                   chi_dens_rpa=chi_dens_rpa, chi_magn_rpa=chi_magn_rpa, chi_magn_dmft=chi_magn_loc,
                                   chi_dens_dmft=chi_dens_loc, chi_magn_rpa_loc=local_rpa_sde['chi_rpa_magn'],
                                   chi_dens_rpa_loc=local_rpa_sde['chi_rpa_magn'], nq=np.prod(q_grid.nk),
                                   use_rpa_for_lc=use_urange_for_lc, lc_use_only_positive=lc_use_only_positive)

    if (qiw_distributor.my_rank == 0):
        np.savetxt(output_path + 'n_lambda_correction.txt', [lambda_['n_sum_dens'], lambda_['n_sum_magn']],
                   delimiter=',', fmt='%.9f')

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

    if (use_urange_for_lc):
        if (np.size(wn_rpa) > 0):
            chi_dens_rpa.mat = 1. / (1. / chi_dens_rpa_mat + lambda_dens)
            chi_magn_rpa.mat = 1. / (1. / chi_magn_rpa_mat + lambda_magn)

    realt.write_time_to_file(string='Lambda correction:', rank=comm.rank)
    # ------------------------------------------- DGA SCHWINGER-DYSON EQUATION ---------------------------------------------

    chi_dens_lambda_my_qiw = fp.LadderSusceptibility(qiw=qiw_grid.my_mesh, channel='dens', u=dmft1p['u'],
                                                     beta=dmft1p['beta'])

    chi_dens_lambda_my_qiw.mat = chi_dens_lambda.mat[qiw_grid.my_slice]

    chi_magn_lambda_my_qiw = fp.LadderSusceptibility(qiw=qiw_grid.my_mesh, channel='magn', u=dmft1p['u'],
                                                     beta=dmft1p['beta'])
    chi_magn_lambda_my_qiw.mat = chi_magn_lambda.mat[qiw_grid.my_slice]

    g_generator = twop.GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=k_grid.grid, hr=hr,
                                               sigma=dmft1p['sloc'])

    if (analyse_spin_fermion_contributions):
        sigma_dens_dga, sigma_dens_dga_re, sigma_dens_dga_im = sde.sde_dga_spin_fermion_contributions(
            vrg=dga_susc['vrg_dens'], chir=chi_dens_lambda_my_qiw, g_generator=g_generator,
            mu=dmft1p['mu'], qiw_grid=qiw_grid.my_mesh, nq=nq_tot, box_sizes=box_sizes,
            q_grid=q_grid)

        sigma_magn_dga, sigma_magn_dga_re, sigma_magn_dga_im = sde.sde_dga_spin_fermion_contributions(
            vrg=dga_susc['vrg_magn'], chir=chi_magn_lambda_my_qiw, g_generator=g_generator,
            mu=dmft1p['mu'], qiw_grid=qiw_grid.my_mesh, nq=nq_tot, box_sizes=box_sizes,
            q_grid=q_grid)

        sigma_magn_dga_re_reduce = np.zeros(np.shape(sigma_magn_dga_re), dtype=complex)
        comm.Allreduce(sigma_magn_dga_re, sigma_magn_dga_re_reduce)
        sigma_magn_dga_im_reduce = np.zeros(np.shape(sigma_magn_dga_im), dtype=complex)
        comm.Allreduce(sigma_magn_dga_im, sigma_magn_dga_im_reduce)

        sigma_dens_dga_re_reduce = np.zeros(np.shape(sigma_dens_dga_re), dtype=complex)
        comm.Allreduce(sigma_dens_dga_re, sigma_dens_dga_re_reduce)
        sigma_dens_dga_im_reduce = np.zeros(np.shape(sigma_dens_dga_im), dtype=complex)
        comm.Allreduce(sigma_dens_dga_im, sigma_dens_dga_im_reduce)

    else:
        sigma_dens_dga = sde.sde_dga(vrg=dga_susc['vrg_dens'], chir=chi_dens_lambda_my_qiw, g_generator=g_generator,
                                     mu=dmft1p['mu'], qiw_grid=qiw_grid.my_mesh, nq=nq_tot, box_sizes=box_sizes,
                                     q_grid=q_grid)

        sigma_magn_dga = sde.sde_dga(vrg=dga_susc['vrg_magn'], chir=chi_magn_lambda_my_qiw, g_generator=g_generator,
                                     mu=dmft1p['mu'], qiw_grid=qiw_grid.my_mesh, nq=nq_tot, box_sizes=box_sizes,
                                     q_grid=q_grid)

    sigma_dens_dga_reduce = np.zeros(np.shape(sigma_dens_dga), dtype=complex)
    comm.Allreduce(sigma_dens_dga, sigma_dens_dga_reduce)
    sigma_magn_dga_reduce = np.zeros(np.shape(sigma_magn_dga), dtype=complex)
    comm.Allreduce(sigma_magn_dga, sigma_magn_dga_reduce)

    sigma_dens_dga = mf.vplus2vfull(mat=sigma_dens_dga_reduce)
    sigma_magn_dga = mf.vplus2vfull(mat=sigma_magn_dga_reduce)

    sigma_dens_dga = k_grid.symmetrize_irrk(mat=sigma_dens_dga)
    sigma_magn_dga = k_grid.symmetrize_irrk(mat=sigma_magn_dga)

    realt.write_time_to_file(string='Non-local DGA SDE:', rank=comm.rank)

    sigma_dens_rpa = sde.rpa_sde(chir=chi_dens_rpa, g_generator=g_generator, niv_giw=niv_urange,
                                 mu=dmft1p['mu'], nq=nq_tot, u=u, qiw_grid=qiw_grid_rpa.my_mesh, q_grid=q_grid)
    sigma_magn_rpa = sde.rpa_sde(chir=chi_magn_rpa, g_generator=g_generator, niv_giw=niv_urange,
                                 mu=dmft1p['mu'], nq=nq_tot, u=u, qiw_grid=qiw_grid_rpa.my_mesh, q_grid=q_grid)

    sigma_dens_rpa_reduce = np.zeros(np.shape(sigma_dens_rpa), dtype=complex)
    comm.Allreduce(sigma_dens_rpa, sigma_dens_rpa_reduce)
    sigma_magn_rpa_reduce = np.zeros(np.shape(sigma_magn_rpa), dtype=complex)
    comm.Allreduce(sigma_magn_rpa, sigma_magn_rpa_reduce)

    sigma_dens_rpa = mf.vplus2vfull(mat=sigma_dens_rpa_reduce)
    sigma_magn_rpa = mf.vplus2vfull(mat=sigma_magn_rpa_reduce)

    # Sigma needs to be symmetrized within the corresponding BZ:
    sigma_dens_rpa = k_grid.symmetrize_irrk(mat=sigma_dens_rpa)
    sigma_magn_rpa = k_grid.symmetrize_irrk(mat=sigma_magn_rpa)

    realt.write_time_to_file(string='Non-local RPA SDE:', rank=comm.rank)

    if (wn_rpa.size > 0):
        sigma_dens_dga = sigma_dens_dga + sigma_dens_rpa
        sigma_magn_dga = sigma_magn_dga + sigma_magn_rpa

    sigma_dga = -1 * sigma_dens_dga + 3 * sigma_magn_dga + dmft_sde['hartree'] - 2 * dmft_sde['siw_magn'] + 2 * \
                dmft_sde['siw_dens'] \
                - dmft_sde['siw'] + dmft1p['sloc'][dmft1p['niv'] - niv_urange:dmft1p['niv'] + niv_urange]
    sigma_dga_nc = sigma_dens_dga + 3 * sigma_magn_dga - 2 * dmft_sde['siw_magn'] + dmft_sde['hartree'] - \
                   dmft_sde['siw'] + dmft1p['sloc'][dmft1p['niv'] - niv_urange:dmft1p['niv'] + niv_urange]

    dga_sde = {
        'sigma_dens': sigma_dens_dga,
        'sigma_magn': sigma_magn_dga,
        'sigma_dens_rpa': sigma_dens_rpa,
        'sigma_magn_rpa': sigma_magn_rpa,
        'sigma': sigma_dga,
        'sigma_nc': sigma_dga_nc
    }

    if (analyse_spin_fermion_contributions):
        sigma_dens_dga_re = mf.vplus2vfull(mat=sigma_dens_dga_re_reduce)
        sigma_dens_dga_im = mf.vplus2vfull(mat=sigma_dens_dga_im_reduce)
        sigma_magn_dga_re = mf.vplus2vfull(mat=sigma_magn_dga_re_reduce)
        sigma_magn_dga_im = mf.vplus2vfull(mat=sigma_magn_dga_im_reduce)

        sigma_dens_dga_re = k_grid.symmetrize_irrk(mat=sigma_dens_dga_re)
        sigma_dens_dga_im = k_grid.symmetrize_irrk(mat=sigma_dens_dga_im)
        sigma_magn_dga_re = k_grid.symmetrize_irrk(mat=sigma_magn_dga_re)
        sigma_magn_dga_im = k_grid.symmetrize_irrk(mat=sigma_magn_dga_im)

        dga_sde_sf_contrib = {
            'sigma_dens_re': sigma_dens_dga_re,
            'sigma_dens_im': sigma_dens_dga_im,
            'sigma_magn_re': sigma_magn_dga_re,
            'sigma_magn_im': sigma_magn_dga_im
        }
    else:
        dga_sde_sf_contrib = None

    if (qiw_distributor.my_rank == 0):
        chi_dens_lambda.mat = mf.wplus2wfull(q_grid.irrk2fbz(mat=qiw_grid.reshape_matrix(chi_dens_lambda.mat)))
        chi_magn_lambda.mat = mf.wplus2wfull(q_grid.irrk2fbz(mat=qiw_grid.reshape_matrix(chi_magn_lambda.mat)))

        chi_dens_lambda.qiw = qiw_grid_fbz.meshgrid
        chi_magn_lambda.qiw = qiw_grid_fbz.meshgrid

        chi_lambda = {
            'chi_dens_lambda': chi_dens_lambda,
            'chi_magn_lambda': chi_magn_lambda,
            'lambda_dens': lambda_dens,
            'lambda_magn': lambda_magn
        }
        np.save(output_path + 'chi_lambda.npy', chi_lambda, allow_pickle=True)
        # Safe chi_m at q = (0,0) and lowest Matsubara as proxy for the Knight shift:
        np.savetxt(output_path + 'Knight_shift.txt',
                   [chi_magn_lambda.mat[0, 0, 0, niw_core], chi_dens_lambda.mat[0, 0, 0, niw_core]], delimiter=',',
                   fmt='%.9f')

    realt.write_time_to_file(string='Building fbz and overhead:', rank=comm.rank)

    return dga_sde, dmft_sde, dmft_gamma, dga_sde_sf_contrib
