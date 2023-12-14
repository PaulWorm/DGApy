import numpy as np
import matplotlib.pyplot as plt

from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import wannier
from dga import two_point as twop
from dga import analytic_continuation as ac
from dga import optics
from test_util import test_data as td
from test_util import util_for_testing as t_util


def test_drude_optical_conductivity(verbose=False):
    '''
        Test the drude optical conductivity.
    '''
    # Load test Hamiltonian:
    ddict, hr = td.load_minimal_dataset()
    beta, n = ddict['beta'], ddict['n']
    nk = (32, 32, 1)
    sym = bz.two_dimensional_square_symmetries()
    k_grid = bz.KGrid(nk, sym)
    ek = hr.get_ek(k_grid)
    wmin, wmax = -15, 15
    nwr = 1501

    w = np.linspace(wmin, wmax, nwr)
    # Set up the self-energy and corresponding Green's functions:
    niv = 200
    v = mf.vn(beta, niv)
    delta = 0.15
    sw = - 1j * delta * np.ones_like(w)
    gwk_obj = twop.RealFrequencyGF(w, sw, ek, n=n)

    # Transform to imaginary frequencies:
    siw = ac.transform_from_real_to_imag(sw, w, v, kind='freq_fermionic')
    siwk_obj = twop.SelfEnergy(siw[None, None, None, :], beta)
    giwk_obj = twop.GreensFunction(siwk_obj, ek, n=n)

    # Check the chemical potential:
    # print(f'Chemical potential Matsubara: {giwk_obj.mu}')
    # print(f'Chemical potential: {gwk_obj.mu}')
    t_util.test_array([giwk_obj.mu], [gwk_obj.mu], 'chemical_potential_consistency', rtol=0.1, atol=0.01)
    # Set up the analytical continuation object:
    bw = 0.0
    w_max_bub = 5.
    nwr = 250
    max_ent_obj = ac.MaxEnt(beta, kernel_mode='freq_bosonic', wmin=0, wmax=w_max_bub, nwr=nwr, bw=bw,
                            mesh_type='linear', alpha_det_method='chi2kink')

    # Compute the optical conductivity from Matsubara quantities:
    niw_cond = 20
    niv_sum = 400
    wn_cond = mf.wn(niw_cond)
    chijj_iw = optics.vec_get_chijj_bubble(giwk_obj, hr, k_grid, wn_cond, niv_sum)
    sigma_cont = max_ent_obj.cont_single_ind(chijj_iw.real)

    # compute the optical conductivity from real frequencies:
    sigma_bubble, w_bub = optics.vec_get_sigma_bub_realf(gwk_obj, hr, k_grid, beta, w_max_bub)

    if verbose:
        plt.figure()
        plt.plot(w_bub, sigma_bubble,'-', color='cornflowerblue', label='Realf')
        plt.plot(max_ent_obj.w, sigma_cont,'-', color='firebrick', label='Max-Ent')
        plt.legend()
        plt.xlabel(r'$\omega [t]$')
        plt.ylabel(r'$\sigma(\omega)$')
        plt.xlim(0,w_max_bub)
        plt.show()

    t_util.test_array(sigma_cont, sigma_bubble, 'drude_optical_conductivity', rtol=0.01, atol=0.05)



if __name__ == '__main__':
    test_drude_optical_conductivity(verbose=False)
