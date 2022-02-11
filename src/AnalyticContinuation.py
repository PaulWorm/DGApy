# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Module for analytic continuation routines. Includes wrapper routines for the Pade solver

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import PadeAux as pa
import continuation as cont
import MatsubaraFrequencies as mf
import TwoPoint as twop
import BrillouinZone as bz
import Hk as hamk
import gc

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

def max_ent_loc(me_conf = None, v_real=None, sigma=None,dga_conf=None,niv_cut=None, bw=0.0, nfit=60, adjust_mu=True, return_chi2=False):
    # Create Green's function:
    gk = twop.create_gk_dict(dga_conf=dga_conf, sigma=sigma, mu0=dga_conf.sys.mu_dmft, adjust_mu=adjust_mu, niv_cut=niv_cut)

    gloc = gk['gk'].mean(axis=(0, 1, 2))

    if(bw == 0):
        use_preblur = False
    else:
        use_preblur = me_conf.use_preblur
    gloc_cont, chi2 = max_ent(mat=gloc, v_real=v_real, beta=me_conf.beta, n_fit=nfit,
                                    alpha_det_method=me_conf.alpha_det_method, err=me_conf.err, use_preblur=use_preblur, bw=bw, return_chi2=return_chi2)
    if(return_chi2):
        return gloc_cont, gk, chi2
    else:
        return gloc_cont, gk

def max_ent_on_fs(v_real=None, sigma=None,config=None,k_grid=None,niv_cut=None, use_preblur=False, bw=0.0, err=1e-3, nfit=60, adjust_mu=True):
    dmft1p = config['dmft1p']
    gk = twop.create_gk_dict(sigma=sigma, kgrid=k_grid.grid, hr=config['system']['hr'], beta=dmft1p['beta'], n=dmft1p['n'],
                                  mu0=dmft1p['mu'], adjust_mu=adjust_mu, niv_cut=niv_cut)
    ind_gf0 = bz.find_qpd_zeros(qpd=(1./gk['gk'][:, :, :, niv_cut]).real, kgrid=k_grid)
    gk_cont = do_max_ent_on_ind(mat=gk['gk'], ind_list=ind_gf0, v_real=v_real,
                                               beta=dmft1p['beta'],
                                               n_fit=nfit, err=err, alpha_det_method='chi2kink', use_preblur = use_preblur, bw=bw)
    return gk_cont, ind_gf0, gk

def max_ent_irrk(v_real=None, sigma=None,config=None,k_grid=None,niv_cut=None, use_preblur=False, bw=0.0, err=1e-3, nfit=60, scal=1):
    dmft1p = config['dmft1p']
    gk = twop.create_gk_dict(sigma=sigma, kgrid=k_grid.grid, hr=config['system']['hr'], beta=dmft1p['beta'], n=dmft1p['n'],
                                  mu0=dmft1p['mu'], adjust_mu=True, niv_cut=niv_cut)
    nk_new = (k_grid.nk[0] / scal, k_grid.nk[1] / scal, 1)
    print(nk_new)
    nk_new = tuple([int(i) for i in nk_new])
    k_grid_small = bz.KGrid(nk=nk_new)
    ek = hamk.ek_3d(kgrid=k_grid_small.grid, hr=config['system']['hr'])
    k_grid_small.get_irrk_from_ek(ek=ek, dec=11)
    ind_irrk = [np.argmin(
        np.abs(np.array(k_grid.irr_kgrid) - np.atleast_2d(np.array(k_grid_small.irr_kgrid)[:, i]).T).sum(axis=0)) for i
                in k_grid_small.irrk_ind_lin]
    ind_irrk = np.squeeze(np.array(np.unravel_index(k_grid.irrk_ind[ind_irrk], shape=k_grid.nk))).T
    ind_irrk_fbz = [tuple(ind_irrk[i, :]) for i in np.arange(ind_irrk.shape[0])]

    gk_dga_max_ent_irrk = do_max_ent_on_ind(mat=gk['gk'], ind_list=ind_irrk_fbz, v_real=v_real,
                                                   beta=dmft1p['beta'],
                                                   n_fit=nfit, err=err, alpha_det_method='chi2kink', use_preblur = use_preblur, bw=bw)
    gk_dga_max_ent_fbz = k_grid_small.irrk2fbz(mat=gk_dga_max_ent_irrk.T)
    cont_dict = {
        'gk_cont': gk_dga_max_ent_fbz,
        'k_grid': k_grid_small,
        'ind_irrk': ind_irrk,
        'ind_irrk_fbz': ind_irrk_fbz
    }
    return cont_dict

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
        #print(i)
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


def extract_coefficient_imaxis(siwk=None, iw=None, N=4, order=3):
    xnew = np.linspace(-0.0, iw[N - 1] + 0.01, num=50)
    coeff_imag = np.polyfit(iw[0:N], siwk[0:N].imag, order)
    coeff_real = np.polyfit(iw[0:N], siwk[0:N].real, order)
    poly_imag = np.polyval(coeff_imag, xnew)
    poly_real = np.polyval(coeff_real, xnew)
    coeff_imag_der = np.polyder(poly_imag)
    poly_imag_der = np.polyval(coeff_imag_der, xnew)
    Z = 1. / (1. - poly_imag_der[0])

    return poly_real[0], poly_imag[0], Z


if __name__ == '__main__':
    input_path = '/mnt/c/users/pworm/Research/BEPS_Project/HoleDoping/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.85/LambdaDga_Nk6400_Nq6400_core32_urange32/'
