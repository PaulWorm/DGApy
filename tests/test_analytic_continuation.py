import numpy as np
import matplotlib.pyplot as plt

from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import wannier
from dga import two_point as twop
from dga import analytic_continuation as ac
from test_util import test_data as td
from test_util import util_for_testing as t_util



def get_test_giw(nw=1001,niw=100,beta=10.,wmax=5, sigma=0.0001):
    wgrid = np.linspace(-wmax, wmax, nw)
    iwgrid = np.pi / beta * (2 * np.arange(niw) + 1)
    giw = np.zeros_like(iwgrid, dtype=complex)

    aw = ac.gauss_peak(-1., 0.2, 1., wgrid) + ac.gauss_peak(1., 0.2, 1., wgrid)
    norm = np.trapz(aw, wgrid)
    aw = aw / norm

    for i, iw in enumerate(iwgrid):
        giw[i] = -np.trapz(aw * wgrid / (iw ** 2 + wgrid ** 2), wgrid) - 1j * np.trapz(aw * iw / (iw ** 2 + wgrid ** 2), wgrid)

    giw += ac.noise(sigma, iwgrid)
    err = np.ones_like(iwgrid)*sigma
    return giw, aw, err, wgrid, iwgrid

def test_fermion_continuation_gauss(verbose=False):
    wmin, wmax = -5,5
    nwr = 1001

    beta = 10.
    niv = 100
    vn = mf.vn(beta, niv)

    np.random.seed(1)

    max_ent_obj = ac.MaxEnt(beta, kernel_mode='freq_fermionic', wmin=wmin, wmax=wmax, nwr=nwr, n_fit=10)

    w = max_ent_obj.w
    aw = ac.gauss_peak(-1., 0.2, 0.1, w) + ac.gauss_peak(1., 0.2, 0.1, w)
    gw = ac.kkt(aw,w,kind='general')
    giw = ac.transform_from_real_to_imag(gw,w,vn,kind='freq_fermionic')
    giw += ac.noise(0.0001, vn)

    gw_cont = max_ent_obj.cont_single_ind(giw)

    if verbose:
        plt.figure()
        plt.plot(w,-gw.imag/np.pi,'-',label='original')
        plt.plot(max_ent_obj.w,-gw_cont.imag/np.pi,'-',label='analytic continuation')
        plt.legend()
        plt.xlabel('w')
        plt.ylabel('A(w)')
        plt.show()

        plt.figure()
        plt.plot(w,gw.real,'-',label='original')
        plt.plot(max_ent_obj.w,gw_cont.real,'-',label='analytic continuation')
        plt.legend()
        plt.xlabel('w')
        plt.ylabel(r'$\Re G(w)$')
        plt.show()

    t_util.test_array(gw,gw_cont,'gauss_fermionic_analytic_continuation',rtol=0.05,atol=0.05)

def test_fermion_continuation_lorentzian(verbose=False):
    wmin, wmax = -5,5
    nwr = 1001

    beta = 10.
    niv = 100
    vn = mf.vn(beta, niv)

    np.random.seed(1)
    mesh_type = 'tan' # 'tan
    fit_position = 2.5
    max_ent_obj = ac.MaxEnt(beta, kernel_mode='freq_fermionic', wmin=wmin, wmax=wmax, nwr=nwr,
                            n_fit=60, bw=0.01, mesh_type=mesh_type, alpha_det_method='chi2kink',fit_position=fit_position)
    w = max_ent_obj.w
    aw = ac.lorentzian_peak(-1., 0.2, w) + ac.lorentzian_peak(1., 0.2, w)
    aw /= np.trapz(aw,w)
    gw = ac.kkt(aw,w,kind='general')
    giw = ac.transform_from_real_to_imag(gw,w,vn,kind='freq_fermionic')
    giw += ac.noise(0.0001, vn)
    gw_cont = max_ent_obj.cont_single_ind(giw)

    if verbose:
        plt.figure()
        plt.plot(w,-gw.imag/np.pi,'-',label='original')
        plt.plot(max_ent_obj.w,-gw_cont.imag/np.pi,'-',label='analytic continuation')
        plt.legend()
        plt.xlabel('w')
        plt.ylabel('A(w)')
        plt.show()

        plt.figure()
        plt.plot(w,gw.real,'-',label='original')
        plt.plot(max_ent_obj.w,gw_cont.real,'-',label='analytic continuation')
        plt.legend()
        plt.xlabel('w')
        plt.ylabel(r'$\Re G(w)$')
        plt.show()

    t_util.test_array(gw,gw_cont,'lorentzian_fermionic_analytic_continuation',rtol=0.5,atol=0.5)

def test_bosonic_continuation_gauss(verbose=False):
    wmin, wmax = 0,5
    nwr = 1001

    beta = 10.
    niv = 100
    wn = mf.wn(beta, niv)

    np.random.seed(1)
    mesh_type = 'tan' # 'tan
    max_ent_obj = ac.MaxEnt(beta, kernel_mode='freq_bosonic', wmin=wmin, wmax=wmax, nwr=nwr,
                            n_fit=100, bw=0.0, force_preblur=False, mesh_type=mesh_type,
                            alpha_det_method='chi2kink', err = 0.0001)
    w = max_ent_obj.w
    aw = -ac.gauss_peak(-1., 0.2, 0.1, w) + ac.gauss_peak(1., 0.2, 0.1, w) + ac.gauss_peak(0, 0.2, 0.5, w)
    gw = ac.kkt(aw,w,kind='general')
    giw = ac.transform_from_real_to_imag(gw,w,wn,kind='freq_bosonic')
    giw += ac.noise(0.00001, wn)
    gw_cont = max_ent_obj.cont_single_ind(giw.real) * 2/np.pi

    if verbose:
        plt.figure()
        plt.plot(w,-gw.imag/np.pi,'-',label='original')
        plt.plot(max_ent_obj.w,gw_cont,'-',label='analytic continuation')
        plt.legend()
        plt.xlabel('w')
        plt.ylabel('A(w)')
        plt.show()

    t_util.test_array(-gw.imag/np.pi,gw_cont,'gauss_bosonic_analytic_continuation',rtol=0.1,atol=0.1)

def test_bosonic_continuation_lorentzian(verbose=False):
    wmin, wmax = 0,5
    nwr = 1001

    beta = 10.
    niv = 100
    wn = mf.wn(beta, niv)

    np.random.seed(1)
    mesh_type = 'tan' # 'tan
    max_ent_obj = ac.MaxEnt(beta, kernel_mode='freq_bosonic', wmin=wmin, wmax=wmax, nwr=nwr,
                            n_fit=100, bw=0.0, mesh_type=mesh_type, alpha_det_method='chi2kink', force_preblur=True)
    w = max_ent_obj.w
    aw = ac.lorentzian_peak(0., 0.2, w)
    gw = ac.kkt(aw,w,kind='bosonic')
    giw = ac.transform_from_real_to_imag(gw,w,wn,kind='freq_bosonic')
    giw += ac.noise(0.0001, wn)
    gw_cont = max_ent_obj.cont_single_ind(giw.real) * 2/np.pi

    if verbose:
        plt.figure()
        plt.plot(w,-gw.imag/np.pi,'-',label='original')
        plt.plot(max_ent_obj.w,gw_cont,'-',label='analytic continuation')
        plt.legend()
        plt.xlabel('w')
        plt.ylabel('A(w)')
        plt.show()

    t_util.test_array(-gw.imag/np.pi,gw_cont,'lorentzian_bosonic_analytic_continuation',rtol=0.1,atol=0.1)

def test_analytic_continuation_giw(verbose=False, input_type='ed'):
    if input_type == 'ed':
        ddict, hr = td.load_minimal_dataset_ed()
    elif input_type == 'w2dyn':
        ddict, hr = td.load_minimal_dataset()
    else:
        raise ValueError(f'input_type = {input_type} not recognized.')

    # Set up the analytic continuation object:
    beta = ddict['beta']
    wmin, wmax = -10, 10
    nwr = 1001
    mesh_type = 'tan'
    max_ent_obj = ac.MaxEnt(ddict['beta'], kernel_mode='freq_fermionic', wmin=wmin, wmax=wmax, nwr=nwr, bw=0.1,
                            mesh_type=mesh_type, alpha_det_method='chi2kink')
    w = max_ent_obj.w
    nk = (80,80,1)
    sym = bz.two_dimensional_square_symmetries()
    k_grid = bz.KGrid(nk,sym)
    ek = hr.get_ek(k_grid)
    delta = 0.1
    sw = - 1j * delta * np.ones_like(w)

    niv = 100
    v = mf.vn(beta, niv)
    vn = mf.vn(niv)
    siw = ac.transform_from_real_to_imag(sw, w, v, kind='freq_fermionic')

    siwk_obj = twop.SelfEnergy(siw[None,None,None,:], beta)
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=ddict['n'])

    gwk_obj = twop.RealFrequencyGF(w,sw,ek, n=ddict['n'])

    giw_backtransform = ac.transform_from_real_to_imag(gwk_obj.gw, w, v, kind='freq_fermionic')
    niv_min = min(mf.niv_from_mat(giwk_obj.g_loc), mf.niw_from_mat(giw_backtransform))

    if verbose:
        plt.figure()
        plt.plot(giwk_obj.vn, giwk_obj.g_loc.imag, '-o', label='original')
        plt.plot(vn, giw_backtransform.imag, '-h', label='backtransform')
        plt.xlim(-5,20)
        plt.legend()
        plt.xlabel('w')
        plt.ylabel(r'$\Im G(i\nu_n)$')
        plt.show()

        plt.figure()
        plt.plot(giwk_obj.vn, giwk_obj.g_loc.real, '-o', label='original')
        plt.plot(vn, giw_backtransform.real, '-h', label='backtransform')
        plt.xlim(-5,20)
        plt.legend()
        plt.xlabel('w')
        plt.ylabel(r'$\Re G(i\nu_n)$')
        plt.show()
    t_util.test_array(mf.cut_v(giwk_obj.g_loc, niv_min), mf.cut_v(giw_backtransform, niv_min),
                      'giw_bracktransform', rtol=1e-2, atol=1e-3)

    # Analytic continuation:
    giw_loc = giwk_obj.g_loc
    giw_loc += ac.noise(0.00001, giwk_obj.vn)
    gw_cont = max_ent_obj.cont_single_ind(giw_loc)

    if verbose:
        plt.figure()
        plt.plot(gwk_obj.w, gwk_obj.aw, '-', label='original')
        plt.plot(max_ent_obj.w, -gw_cont.imag/np.pi, '-', label='max-ent')
        plt.xlim(None,None)
        plt.legend()
        plt.xlabel('w')
        plt.ylabel(r'$A(w)$')
        plt.show()

    t_util.test_array(gw_cont,gwk_obj.gw,
                      'gw_non_interacting_max_ent', rtol=1e-1, atol=1e-1)


if __name__ == '__main__':
    test_fermion_continuation_gauss(verbose=False)
    test_fermion_continuation_lorentzian(verbose=False)

    test_bosonic_continuation_gauss(verbose=False)
    test_bosonic_continuation_lorentzian(verbose=False)

    test_analytic_continuation_giw(verbose=False, input_type='ed')