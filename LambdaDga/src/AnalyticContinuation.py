# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Module for analytic continuation routines. Includes wrapper routines for the Pade solver

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import PadeAux as pa
import continuation as cont
import MatsubaraFrequencies as mf

# --------------------------------------- Obtain the real frequency grid -----------------------------------------------

def v_real_tan(wmax=15, nw=501):
    return np.tan(np.linspace(-np.pi/2.5,np.pi/2.5,num=nw,endpoint=True))*wmax/np.tan(np.pi/2.5)


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

def do_max_ent_on_ind(mat=None, ind_list=None, v_real=None, beta=None, n_fit=60, alpha_det_method='historic', err=1e-4):
    ''' ind are the indizes on which the pade approximation shall be performed'''
    n_ind = len(ind_list)
    nw = np.size(v_real)
    mat_cont = np.zeros((nw, n_ind), dtype=complex)
    for i, ind in enumerate(ind_list):
        mat_cont[:,i] = max_ent(mat=mat[ind], v_real=v_real, beta=beta, n_fit=n_fit, alpha_det_method=alpha_det_method, err=err)

    return mat_cont

def max_ent(mat=None, v_real=None, beta=None, n_fit=60, alpha_det_method='historic', err=2e-6):
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
                                   optimizer="newton")
    cont_mat = cont.GreensFunction(spectrum=sol.A_opt, wgrid=v_real, kind='fermionic').kkt()
    return cont_mat

# -------------------------------------------- Matsubara Fit ----------------------------------------------------------
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
