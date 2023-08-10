# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Module for analytic continuation routines. Includes wrapper routines for the Pade solver

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import dga.pade_aux as pa
import dga.continuation as cont
import dga.matsubara_frequencies as mf
import gc
import dga.config as config
import dga.plotting as plotting
import dga.brillouin_zone as bz
import dga.two_point as twop
import dga.mpi_aux as mpi_aux
import scipy.optimize as opt

# --------------------------------------- Obtain the real frequency grid -----------------------------------------------

def v_real_tan(wmax=15, nw=501):
    return np.tan(np.linspace(-np.pi/2.5,np.pi/2.5,num=nw,endpoint=True))*wmax/np.tan(np.pi/2.5)


# --------------------------------------- Fit piecewise linear function ------------------------------------------------
def fit_piecewise(logx, logy, p2_deg=0):
    """ piecewise linear fit (e.g. for chi2)

    The curve is fitted by two linear curves; for the first few indices,
    the fit is done by choosing both the slope and the y-axis intercept.
    If ``p2_deg`` is ``0``, a constant curve is fitted (i.e., the only
    fit parameter is the y-axis intercept) for the last few indices,
    if it is ``1`` also the slope is fitted.

    The point (i.e., data point index) up to where the first linear curve
    is used (and from where the second linear curve is used) is also determined
    by minimizing the misfit. I.e., we search the minimum misfit with
    respect to the fit parameters of the linear curve and with respect
    to the index up to where the first curve is used.

    Parameters
    ----------
    logx : array
        the x-coordinates of the curve that should be fitted piecewise
    logy : array
        the y-corrdinates of the curve that should be fitted piecewise
    p2_deg : int
        ``0`` or ``1``, giving the polynomial degree of the second curve
    """
    chi2 = np.full(len(logx), np.nan)
    p1 = [0] * len(logx)
    p2 = [0] * len(logx)

    def denan(what, check=None):
        if check is None:
            check = what
        return what[np.logical_not(np.isnan(check))]
    for i in range(2, len(logx) - 2):
        chi2[i] = 0.0
        try:
            p1[i], residuals, rank, singular_values, rcond = np.polyfit(
                denan(logx[:i], logy[:i]), denan(logy[:i]), deg=1, full=True)
            if len(residuals) > 0:
                chi2[i] += residuals[0]
            p2[i], residuals, rank, singular_values, rcond = np.polyfit(
                denan(logx[i:], logy[i:]), denan(logy[i:]),
                deg=p2_deg, full=True)
            if len(residuals) > 0:
                chi2[i] += residuals[0]
        except TypeError:
            p1[i] = np.nan
            p2[i] = np.nan
            chi2[i] = np.nan
    i = np.nanargmin(chi2)
    if np.isnan(i):
        raise ValueError('chi2 is all NaN')

    X_x = ((p2[i][1] if p2_deg == 1 else p2[i][0]) - p1[i][1]) / \
        (p1[i][0] - (p2[i][0] if p2_deg == 1 else 0.0))
    idx = np.nanargmin(np.abs(logx - X_x))

    if np.isnan(idx):
        raise ValueError('abs(logx - X_x) is all NaN')

    return idx, (p1[i], p2[i])



# -------------------------------------------- Pade Approximation ------------------------------------------------------

def pade_kauf(mat=None,v_real=None, beta =None, n_fit =20):
    niv_mat = np.size(mat) // 2
    mat_plus = mat[niv_mat:niv_mat + n_fit]
    v_plus = mf.v_plus(beta=beta, n=n_fit) # Takes real argument for im-axis and 1j is added internally.
    pade_kauf = pa.PadeSolver(im_axis=v_plus,re_axis=v_real,im_data=mat_plus)
    sol = pade_kauf.solve(show_plot=True)
    return sol

def do_pade_on_ind(mat=None, ind_list=None, v_real=None, beta=None, method='thiele', n_fit=60, n_pade=20, delta=0.001):
    ''' ind are the indizes on which the pade approximation shall be performed'''
    n_ind = len(ind_list)
    nw = np.size(v_real)
    mat_cont = np.zeros((nw, n_ind), dtype=complex)
    for i, ind in enumerate(ind_list):
        mat_cont[:,i] = pade(mat=mat[ind], v_real=v_real, beta=beta, method=method, n_fit=n_fit, n_pade=n_pade, delta=delta)

    return mat_cont

def pade(mat=None, v_real=None, beta=None, method='thiele', n_fit=60, n_pade=20, delta=0.001):
    ''' mat may only depend on one Matsubara frequency and nothing else.'''
    niv_mat = np.size(mat) // 2
    mat_plus = mat[niv_mat:niv_mat + n_fit]
    iv_plus = mf.iv_plus(beta=beta, n=n_fit)
    v_real = v_real + 1j * delta
    if (method == 'thiele'):
        mat_cont = pa.padeThiele(iv_plus, np.atleast_2d(mat_plus).T, v_real)
    elif (method == 'non-linear'):
        coeff_pade = pa.padeNonlinear(iv_plus, mat_plus, n_pade)
        mat_cont = pa.epade(v_real, coeff_pade)
    elif(method == 'average'):
        mat_cont,_,_ = pa.pade(iv_plus, np.atleast_2d(mat_plus).T, v_real)
    else:
        raise ValueError('Method not recognized.')
    return mat_cont

# ---------------------------------------------- MaxEnt ----------------------------------------------------------------

def check_filling(v_real=None, gloc_cont=None):
    ind_w = v_real < 0
    n = np.trapz(-1. / np.pi * gloc_cont[ind_w].imag, v_real[ind_w]) * 2
    return n

def max_ent_loc(mat,me_conf: config.MaxEntConfig, bw):
    '''
        Perform the analytic continuation for a local quantity
    '''

    gloc_cont, chi2 = max_ent(mat=mat, v_real=me_conf.mesh, beta=me_conf.beta, n_fit=me_conf.n_fit,
                                    alpha_det_method=me_conf.alpha_det_method, err=me_conf.err,
                              use_preblur=me_conf.use_preblur, bw=bw, return_chi2=True)
    return gloc_cont, chi2

def do_max_ent_on_ind(mat=None, ind_list=None, v_real=None, beta=None, n_fit=60, alpha_det_method='historic', err=1e-4, use_preblur = False, bw=None):
    ''' ind are the indizes on which the pade approximation shall be performed'''
    n_ind = len(ind_list)
    nw = np.size(v_real)
    mat_cont = np.zeros((nw, n_ind), dtype=complex)
    for i, ind in enumerate(ind_list):
        mat_cont[:,i] = max_ent(mat=mat[ind], v_real=v_real, beta=beta, n_fit=n_fit, alpha_det_method=alpha_det_method, err=err, use_preblur = use_preblur, bw=bw)

    return mat_cont

def do_max_ent_on_ind_T(mat=None, ind_list=None, v_real=None, beta=None, n_fit=60, alpha_det_method='historic', err=1e-4, use_preblur = False, bw=None,optimizer='newton'):
    ''' ind are the indizes on which the pade approximation shall be performed'''
    n_ind = len(ind_list)
    nw = np.size(v_real)
    mat_cont = np.zeros((n_ind, nw), dtype=complex)
    if(bw==0):
        use_preblur = False
    for i, ind in enumerate(ind_list):
        mat_cont[i,:] = max_ent(mat=mat[ind], v_real=v_real, beta=beta, n_fit=n_fit, alpha_det_method=alpha_det_method, err=err, use_preblur = use_preblur, bw=bw, optimizer=optimizer)

    return mat_cont

def max_ent(mat=None, v_real=None, beta=None, n_fit=60, alpha_det_method='historic', err=2e-6, use_preblur = False, bw=None, optimizer='newton', return_chi2=False):
    niv_mat = np.size(mat) // 2
    mat_plus = mat[niv_mat:niv_mat + n_fit]
    iv_plus = mf.iv_plus(beta=beta, n=n_fit)
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
    if(return_chi2):
        return cont_mat, chi2
    else:
        return cont_mat

# -------------------------------------------- Matsubara Fit ----------------------------------------------------------
def extract_coeff_on_ind(siwk=None,indizes=None, v=None, N=4, order=3):

    siwk_re_fs = np.zeros((len(indizes),))
    siwk_im_fs = np.zeros((len(indizes),))
    siwk_Z = np.zeros((len(indizes),))

    for i, ind in enumerate(indizes):
        siwk_re_fs[i], siwk_im_fs[i], siwk_Z[i] = extract_coefficient_imaxis(siwk=siwk[ind], iw=v, N=N,order=order)
    return siwk_re_fs, siwk_im_fs, siwk_Z


def poly_extrap_0(x,y,order,N):
    x_new = np.linspace(0,1,101)
    fit = np.polyfit(x[:N],y[:N],deg=order)
    return np.polyval(fit,x_new)[0]

def poly_der_extrap_0(x,y,order,N):
    x_new = np.linspace(0,1,101)
    fit = np.polyfit(x[:N],y[:N],deg=order)
    der = np.polyder(fit)
    return np.polyval(der,x_new)[0]

def get_gamma_bandshift_Z(wn,siw,order,N):
    gamma = -poly_extrap_0(wn,siw.imag,order,N)
    bandshift = poly_extrap_0(wn,siw.real,order,N)
    Z = 1/(1-poly_der_extrap_0(wn,siw.imag,order,N))
    return gamma,bandshift,Z

def extract_coefficient_imaxis(siwk=None, iw=None, N=4, order=3):
    ''' Exists for backward compatibility. '''
    gamma, bandshift, Z = get_gamma_bandshift_Z(iw,siwk,order,N)
    return bandshift, gamma, Z


def max_ent_loc_bw_range(mat, me_conf: config.MaxEntConfig, name=''):
    v_real = me_conf.mesh
    chi2 = []
    bw_range = me_conf.bw_range_loc
    for bw in bw_range:
        # mu-adjusted DGA Green's function:
        mat_cont, chi2_tmp = max_ent_loc(mat, me_conf, bw)
        chi2.append(chi2_tmp)
        plotting.plot_aw_loc(output_path=me_conf.output_path_loc, v_real=v_real, gloc=mat_cont,
                             name=name + '-bw{}'.format(bw))
        np.save(me_conf.output_path_loc + name + '_cont_bw{}.npy'.format(bw), mat_cont, allow_pickle=True)

    chi2 = np.array(chi2)

    def fitfun(x, a, b, c, d):
        return a + b / (1. + np.exp(-d * (x - c)))

    popt, pcov = opt.curve_fit(f=fitfun, xdata=np.log(bw_range), ydata=np.log(chi2), p0=(0., 5., 2., 0.))
    a, b, c, d = popt
    a_opt = c - me_conf.bw_fit_position / d
    bw_opt = np.exp(a_opt)
    np.savetxt(me_conf.output_path_loc + 'bw_opt_' + name + '.txt', [bw_opt, ], delimiter=',', fmt='%.9f', header='bw_opt')
    plotting.plot_bw_fit(bw_opt=bw_opt, bw=bw_range, chi2=chi2, fits=[np.exp(fitfun(np.log(bw_range), a, b, c, d)), ],
                         output_path=me_conf.output_path_loc, name='chi2_bw_{}'.format(name))
    return bw_opt


def max_ent_irrk(mat, k_grid, me_conf: config.MaxEntConfig, comm, bw):
    mpi_distributor = mpi_aux.MpiDistributor(ntasks=k_grid.nk_irr, comm=comm)
    my_ind = k_grid.irrk_ind_lin[mpi_distributor.my_slice]
    mat_cont = do_max_ent_on_ind_T(mat=mat, ind_list=my_ind, v_real=me_conf.mesh,
                                          beta=me_conf.beta,
                                          n_fit=me_conf.n_fit, err=me_conf.err, alpha_det_method=me_conf.alpha_det_method,
                                          use_preblur=me_conf.use_preblur, bw=bw, optimizer=me_conf.optimizer)
    mat_cont = mpi_distributor.allgather(rank_result=mat_cont)
    return mat_cont


def max_ent_irrk_bw_range_sigma(sigma: twop.SelfEnergy, k_grid: bz.KGrid, me_conf: config.MaxEntConfig, comm, bw, logger=None,
                                name=''):
    hartree = sigma.smom0
    sigma = k_grid.map_fbz2irrk(sigma.get_siw(niv=me_conf.n_fit) - hartree)
    sigma_cont = max_ent_irrk(sigma, k_grid, me_conf, comm, bw)
    if (logger is not None): logger.log_cpu_time(task=' for {} MaxEnt done left are plots and gather '.format(name))

    if (logger is not None):  logger.log_cpu_time(task=' for {} Gather done left are plots '.format(name))
    if (comm.rank == 0):
        sigma_cont = k_grid.map_irrk2fbz(sigma_cont) + hartree  # re-add the hartree term

        plotting.plot_cont_fs(output_path=me_conf.output_path_nl_s,
                              name='swk_fermi_surface_' + name + '_cont_w0-bw{}'.format(bw),
                              gk=sigma_cont, v_real=me_conf.mesh, k_grid=k_grid, w_plot=0)

        plotting.plot_cont_fs(output_path=me_conf.output_path_nl_s,
                              name='swk_fermi_surface_' + name + '_cont_w0.1-bw{}'.format(bw),
                              gk=sigma_cont, v_real=me_conf.mesh, k_grid=k_grid, w_plot=0.1)

        plotting.plot_cont_fs(output_path=me_conf.output_path_nl_s,
                              name='swk_fermi_surface_' + name + '_cont_w-0.1-bw{}'.format(bw),
                              gk=sigma_cont, v_real=me_conf.mesh, k_grid=k_grid, w_plot=-0.1)

        np.save(me_conf.output_path_nl_s + 'swk_' + name + '_cont_fbz_bw{}.npy'.format(bw), sigma_cont,
                allow_pickle=True)
    return None
    # plotting.plot_cont_edc_maps(v_real=me_conf.mesh, gk_cont=sigma_cont, _k_grid=_k_grid,
    #                             output_path=me_conf.output_path_nl_s,
    #                             name='swk_fermi_surface_' + name + '_cont_edc_maps_bw{}'.format(bw))


def max_ent_irrk_bw_range_green(green: twop.GreensFunction, k_grid: bz.KGrid, me_conf: config.MaxEntConfig, comm, bw, logger=None,
                                name=''):
    green = k_grid.map_fbz2irrk(green.g_full())
    green_cont = max_ent_irrk(green, k_grid, me_conf, comm, bw)
    if (logger is not None): logger.log_cpu_time(task=' for {} MaxEnt done left are plots and gather '.format(name))

    if (logger is not None):  logger.log_cpu_time(task=' for {} Gather done left are plots '.format(name))
    if (comm.rank == 0):
        green_cont = k_grid.map_irrk2fbz(green_cont)

        plotting.plot_cont_fs(output_path=me_conf.output_path_nl_g,
                              name='gwk_fermi_surface_' + name + '_cont_w0-bw{}'.format(bw),
                              gk=green_cont, v_real=me_conf.mesh, k_grid=k_grid, w_plot=0)

        plotting.plot_cont_fs(output_path=me_conf.output_path_nl_g,
                              name='gwk_fermi_surface_' + name + '_cont_w0.1-bw{}'.format(bw),
                              gk=green_cont, v_real=me_conf.mesh, k_grid=k_grid, w_plot=0.1)

        plotting.plot_cont_fs(output_path=me_conf.output_path_nl_g,
                              name='gwk_fermi_surface_' + name + '_cont_w-0.1-bw{}'.format(bw),
                              gk=green_cont, v_real=me_conf.mesh, k_grid=k_grid, w_plot=-0.1)

        np.save(me_conf.output_path_nl_g + 'gwk_' + name + '_cont_fbz_bw{}.npy'.format(bw), green_cont,
                allow_pickle=True)
        # plotting.plot_cont_edc_maps(v_real=me_conf.mesh, gk_cont=green_cont, k_grid=k_grid,
        #                             output_path=me_conf.output_path_nl_g,
        #                             name='gwk_fermi_surface_' + name + '_cont_edc_maps_bw{}'.format(bw))
    return None