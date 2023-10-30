# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Module for analytic continuation routines. Includes wrapper routines for the Pade solver

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import mpi4py.MPI
import numpy as np
import ana_cont as cont
import dga.matsubara_frequencies as mf
import gc
import dga.config as config
import dga.plotting as plotting
import scipy.optimize as opt
import dga.mpi_aux as mpi_aux


# -------------------------------------------- Matsubara Fit ----------------------------------------------------------

def extract_coeff_on_ind(siwk=None, indizes=None, v=None, N=4, order=3):
    siwk_re_fs = np.zeros((len(indizes),))
    siwk_im_fs = np.zeros((len(indizes),))
    siwk_Z = np.zeros((len(indizes),))

    for i, ind in enumerate(indizes):
        siwk_re_fs[i], siwk_im_fs[i], siwk_Z[i] = extract_coefficient_imaxis(siwk=siwk[ind], iw=v, N=N, order=order)
    return siwk_re_fs, siwk_im_fs, siwk_Z


def poly_extrap_0(x, y, order, N):
    x_new = np.linspace(0, 1, 101)
    fit = np.polyfit(x[:N], y[:N], deg=order)
    return np.polyval(fit, x_new)[0]


def poly_der_extrap_0(x, y, order, N):
    x_new = np.linspace(0, 1, 101)
    fit = np.polyfit(x[:N], y[:N], deg=order)
    der = np.polyder(fit)
    return np.polyval(der, x_new)[0]


def get_gamma_bandshift_Z(wn, siw, order, N):
    gamma = -poly_extrap_0(wn, siw.imag, order, N)
    bandshift = poly_extrap_0(wn, siw.real, order, N)
    Z = 1 / (1 - poly_extrap_0(wn, siw.imag / wn, order, N))
    # Z = 1 / (1 - poly_der_extrap_0(wn, siw.imag, order, N))
    # Z = min(max(1 / (1 - siw.imag[0]/wn[0]),0),1)
    Z = min(max(Z, 0), 1)
    if (Z == 0): Z = 1
    return gamma, bandshift, Z


def extract_coefficient_imaxis(siwk=None, iw=None, N=4, order=3):
    ''' Exists for backward compatibility. '''
    gamma, bandshift, Z = get_gamma_bandshift_Z(iw, siwk, order, N)
    return bandshift, gamma, Z


# ---------------------------------------------- Real-frequency omega meshes -----------------------------------------------------

def lorentzian_w_mesh(wmin, wmax, n_points, cut):
    ''' Lorentzian mesh '''
    u = np.linspace(0, 1, n_points + 1)
    temp = np.tan(np.pi * (u * (1. - 2 * cut) + cut - 0.5))
    t = (temp - temp[0]) / (temp[-1] - temp[0])
    w = wmin + (wmax - wmin) * t
    w = (w[:-1] + w[1:]) / 2.0
    w = (w - w[0]) / (w[-1] - w[0]) * (wmax - wmin) + wmin
    return w


def hyperbolic_w_mesh(wmin, wmax, n_points):
    ''' Hyperbolic mesh '''
    u = np.linspace(-1, 1, n_points)
    w = np.sign(u) * (np.sqrt(1 + u ** 2) - 1)
    w = wmin + (wmax - wmin) * (w - w[0]) / (w[-1] - w[0])
    return w


def linear_w_mesh(wmin, wmax, n_points):
    ''' Linear mesh '''
    return np.linspace(wmin, wmax, n_points)


def tan_w_mesh(wmin, wmax, n_points):
    ''' Linear mesh '''
    w = np.tan(np.linspace(-np.pi / 2.5, np.pi / 2.5, num=n_points, endpoint=True)) * 1 / np.tan(np.pi / 2.5)
    w = wmin + (wmax - wmin) * (w - w[0]) / (w[-1] - w[0])
    return w

def get_w_mesh(mesh_type,wmin,wmax,nwr,cut=None):
    if (mesh_type == 'lorentzian'):
        assert cut is not None, 'For Lorentzian mesh cut has to be provided.'
        return lorentzian_w_mesh(wmin, wmax, nwr, cut)
    elif mesh_type == 'hyperbolic':
        return hyperbolic_w_mesh(wmin, wmax, nwr)
    elif mesh_type == 'linear':
        return linear_w_mesh(wmin, wmax, nwr)
    elif mesh_type == 'tan':
        return tan_w_mesh(wmin, wmax, nwr)
    else:
        raise ValueError('Unknown omega mesh type.')

# ---------------------------------------------- Class to handle MaxEnt ---------------------------------------------------------

MAX_ENT_DEFAULT_SETTINGS = {
    'cut': 0.04, # only relevant for the Lorentzian mesh,
    'n_fit': 20,
    'bw': 0.01,
    'nwr': 501,  # Number of frequencies on the real axis
    'wmax': 15,  # frequency range on the real axis
    'wmin': -15,  # frequency range on the real axis
    'err': 1e-3,  # Error for the analytical continuation
    'alpha_det_method': 'chi2kink',  # alpha determination method
    'optimizer': 'newton',  # alpha determination method
    'mesh_type': 'tan',  # mesh type
    'method': 'maxent_svd'
}

class MaxEnt():
    '''
        Class to handle analytic continuation
    '''

    def __init__(self, beta, kernel_mode,comm = None, config=None, **kwargs):

        # declare attributes:
        self.do_cont = True
        self.mesh_type = None
        self.wmin = None
        self.wmax = None
        self.err = None
        self.cut = None
        self.method = None
        self.optimizer = None
        self.alpha_det_method = None
        self.nwr = None
        self.bw = None
        self.n_fit = None


        if(comm == None):
            self.comm = mpi4py.MPI.COMM_WORLD
        else:
            self.comm = comm

        self.beta = beta  # Inverse Temperature
        self.kernel_mode = kernel_mode

        self.__dict__.update(MAX_ENT_DEFAULT_SETTINGS) # Initialize with Default settings.
        if(config is not None): self.__dict__.update(config)
        self.__dict__.update(kwargs)


        self.set_w_mesh()


    @property
    def use_preblur(self):
        if(self.bw == 0):
            return False
        else:
            return True

    def set_w_mesh(self):
        ''' Set the real-frequency mesh. '''
        self.w =  get_w_mesh(self.mesh_type,self.wmin,self.wmax,self.nwr,self.cut)

    def get_im_freq(self):
        if(self.kernel_mode == 'freq_fermionic'):
            return mf.vn(self.beta, self.n_fit, pos=True)
        elif(self.kernel_mode == 'freq_bosonic'):
            return mf.wn(self.beta, self.n_fit, pos=True)
        else:
            raise ValueError(f'Unknown kernel mode: {self.kernel_mode}.')

    def cut_matrix(self,mat):
        niv_mat = np.size(mat) // 2

        if(self.n_fit > niv_mat):
            self.n_fit = niv_mat

        if (self.kernel_mode == 'freq_fermionic'):
            return mat[niv_mat:niv_mat + self.n_fit]
        elif (self.kernel_mode == 'freq_bosonic'):
            return mat[niv_mat:niv_mat + self.n_fit + 1]
        else:
            raise ValueError(f'Unknown kernel mode: {self.kernel_mode}.')

    def get_model(self, mat):
        if (self.kernel_mode == 'freq_fermionic'):
            model = np.ones_like(self.w)
            model /= np.trapz(model, self.w)
        elif (self.kernel_mode == 'freq_bosonic'):
            model = np.ones_like(self.w)
            model *= mat[0].real/np.trapz(model, self.w)
        else:
            raise ValueError(f'Unknown kernel mode: {self.kernel_mode}.')

        return model

    def cont_single_ind(self, mat):
        mat_plus = self.cut_matrix(mat)
        im_axis = self.get_im_freq()
        model = self.get_model(mat)
        error_s = np.ones_like(im_axis) * self.err

        problem = cont.AnalyticContinuationProblem(im_axis=im_axis, re_axis=self.w, im_data=mat_plus,
                                                   kernel_mode=self.kernel_mode,
                                                   beta=self.beta)

        sol, _ = problem.solve(method=self.method, model=model, stdev=error_s, alpha_determination=self.alpha_det_method,
                               optimizer=self.optimizer, preblur=self.use_preblur, blur_width=self.bw, verbose=False)
        if(self.kernel_mode == 'freq_fermionic'):
            cont_mat = cont.GreensFunction(spectrum=sol.A_opt, wgrid=self.w, kind='fermionic').kkt()
        elif (self.kernel_mode == 'freq_bosonic'):
            cont_mat = sol.A_opt
        else:
            raise ValueError(f'Unknown kernel mode: {self.kernel_mode}.')
        return cont_mat


    def analytic_continuation(self, mat_list):
        cont_list = []
        for mat in mat_list:
            try:
                cont_list.append(self.cont_single_ind(mat))
            except:
                print('Error in analytic continuation. Setting result to 0.')
                cont_list.append(np.zeros_like(self.w)) # make sure the code does not hang up when ana_cont fails for one index
        return np.array(cont_list)

def mpi_ana_cont(mat, me_controller: MaxEnt, mpi_dist: mpi_aux.MpiDistributor, name,logger=None):
    if(logger is not None): logger.log_cpu_time(task=' MaxEnt controller for Siwk created ')
    my_mat = mpi_dist.scatter(mat)
    if(logger is not None): logger.log_cpu_time(task=f' {name} scattered among cores. ')
    cont = me_controller.analytic_continuation(my_mat)
    if(logger is not None): logger.log_cpu_time(task=f' {name} continuation performed. ')
    cont = mpi_dist.gather(cont)
    if(logger is not None): logger.log_cpu_time(task=f' {name} gather done. ')
    return cont

def save_and_plot_cont_fermionic(mat,w,k_grid, name,out_dir):
    np.save(out_dir + f'{name}_cont_fbz.npy', mat,
            allow_pickle=True)
    np.save(out_dir + 'w.npy', w, allow_pickle=True)

    plotting.plot_cont_fs(output_path=out_dir, name=name,mat=mat, v_real=w, k_grid=k_grid, w_plot=0)

    plotting.plot_cont_fs(output_path=out_dir, name=name,mat=mat, v_real=w, k_grid=k_grid, w_plot=0.1)

    plotting.plot_cont_fs(output_path=out_dir,name=name, mat=mat,v_real=w, k_grid=k_grid, w_plot=-0.1)

def save_and_plot_cont_bosonic(mat,w,k_grid, name,out_dir):
    np.save(out_dir + f'{name}_cont_fbz.npy', mat,
            allow_pickle=True)
    np.save(out_dir + 'w.npy', w, allow_pickle=True)

    plotting.plot_cont_fs_no_shift(output_path=out_dir, name=name,mat=mat, v_real=w, k_grid=k_grid, w_plot=0)

    plotting.plot_cont_fs_no_shift(output_path=out_dir, name=name,mat=mat, v_real=w, k_grid=k_grid, w_plot=0.1)

    plotting.plot_cont_fs_no_shift(output_path=out_dir,name=name, mat=mat,v_real=w, k_grid=k_grid, w_plot=-0.1)

# ---------------------------------------------- MaxEnt ----------------------------------------------------------------

def check_filling(v_real=None, gloc_cont=None):
    ind_w = v_real < 0
    n = np.trapz(-1. / np.pi * gloc_cont[ind_w].imag, v_real[ind_w]) * 2
    return n


def max_ent_loc(mat, me_conf: config.MaxEntConfig, bw):
    '''
        Perform the analytic continuation for a local quantity
    '''

    gloc_cont, chi2 = max_ent(mat=mat, v_real=me_conf.mesh, beta=me_conf.beta, n_fit=me_conf.n_fit,
                              alpha_det_method=me_conf.alpha_det_method, err=me_conf.err,
                              use_preblur=me_conf.use_preblur, bw=bw, return_chi2=True)
    return gloc_cont, chi2

def max_ent(mat=None, v_real=None, beta=None, n_fit=60, alpha_det_method='historic', err=2e-6, use_preblur=False, bw=None,
            optimizer='newton', return_chi2=False):
    niv_mat = np.size(mat) // 2
    mat_plus = mat[niv_mat:niv_mat + n_fit]
    iv_plus = 1j * mf.vn(beta, n_fit, pos=True)
    problem = cont.AnalyticContinuationProblem(im_axis=iv_plus.imag, re_axis=v_real, im_data=mat_plus,
                                               kernel_mode="freq_fermionic",
                                               beta=beta)
    model = np.ones_like(v_real)
    model /= np.trapz(model, v_real)
    error_s = np.ones((n_fit,), dtype=np.float64) * err
    sol, _ = problem.solve(method='maxent_svd', model=model, stdev=error_s, alpha_determination=alpha_det_method,
                           optimizer=optimizer, preblur=use_preblur, blur_width=bw, verbose=False)
    chi2 = sol.chi2
    cont_mat = cont.GreensFunction(spectrum=sol.A_opt, wgrid=v_real, kind='fermionic').kkt()
    del problem
    del sol
    del _
    gc.collect()
    if (return_chi2):
        return cont_mat, chi2
    else:
        return cont_mat





def max_ent_loc_bw_range(mat, me_conf: config.MaxEntConfig, name='', out_dir='./'):
    v_real = me_conf.mesh
    chi2 = []
    bw_range = me_conf.bw_range_loc
    for bw in bw_range:
        # mu-adjusted DGA Green's function:
        mat_cont, chi2_tmp = max_ent_loc(mat, me_conf, bw)
        chi2.append(chi2_tmp)
        plotting.plot_aw_loc(output_path=out_dir, v_real=v_real, gloc=mat_cont,
                             name=name + '-bw{}'.format(bw))
        np.save(out_dir + name + '_cont_bw{}.npy'.format(bw), mat_cont, allow_pickle=True)

    chi2 = np.array(chi2)

    def fitfun(x, a, b, c, d):
        return a + b / (1. + np.exp(-d * (x - c)))

    popt, pcov = opt.curve_fit(f=fitfun, xdata=np.log(bw_range), ydata=np.log(chi2), p0=(0., 5., 2., 0.))
    a, b, c, d = popt
    a_opt = c - me_conf.bw_fit_position / d
    bw_opt = np.exp(a_opt)
    np.savetxt(out_dir + 'bw_opt_' + name + '.txt', [bw_opt, ], delimiter=',', fmt='%.9f', header='bw_opt')
    plotting.plot_bw_fit(bw_opt=bw_opt, bw=bw_range, chi2=chi2, fits=[np.exp(fitfun(np.log(bw_range), a, b, c, d)), ],
                         output_path=out_dir, name='chi2_bw_{}'.format(name))
    np.save(out_dir + 'w.npy', me_conf.mesh, allow_pickle=True)
    return bw_opt
