from dga import brillouin_zone as bz
from dga import two_point as twop
from dga import matsubara_frequencies as mf
from test_util import util_for_testing as t_util
from test_util import test_data as td


def test_self_energy():
    ddict, hr = td.load_minimal_dataset()

    siw_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    t_util.test_array([siw_obj.smom0, ], [twop.get_smom0(ddict['u'], ddict['n']), ], 'smom0_fitting', rtol=1e-3)
    t_util.test_array([siw_obj.smom1, ], [twop.get_smom1(ddict['u'], ddict['n']), ], 'smom1_fitting', rtol=1e-2)

    t_util.test_array(siw_obj.sigma_core, mf.cut_v(ddict['siw'], siw_obj.niv_core), 'sigma_core', rtol=1e-5, atol=1e-5)
    t_util.test_array(siw_obj.get_siw(niv=mf.niv_from_mat(ddict['siw'])), ddict['siw'], 'sigma_full', rtol=1e-2, atol=1e-5)


def test_greens_function():
    ddict, hr = td.load_minimal_dataset()

    siw_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    nk = (32, 32, 1)
    k_grid = bz.KGrid(nk=nk)
    ek = hr.get_ek(k_grid)
    giw_obj = twop.GreensFunction(sigma=siw_obj, ek=ek, n=ddict['n'])

    t_util.test_array([giw_obj.mu, ], [ddict['mu_dmft'], ], 'giw_mu_finding', rtol=1e-4, atol=1e-5)

    giw_obj_mu = twop.GreensFunction(sigma=siw_obj, ek=ek, mu=ddict['mu_dmft'])

    t_util.test_array([giw_obj_mu.n, ], [ddict['n'], ], 'giw_n', rtol=1e-4, atol=1e-5)

    t_util.test_array(mf.cut_v(giw_obj.g_loc, mf.niv_from_mat(ddict['giw'])), ddict['giw'], 'giw', rtol=1e-3, atol=1e-3)
    t_util.test_array(giw_obj.k_mean('core'), mf.cut_v(ddict['giw'], giw_obj.niv_core), 'giw_core', rtol=1e-3, atol=1e-3)

    t_util.test_array(mf.cut_v(giw_obj_mu.g_loc, mf.niv_from_mat(ddict['giw'])), ddict['giw'], 'giw_mu', rtol=1e-3, atol=1e-3)
    t_util.test_array(giw_obj_mu.k_mean('core'), mf.cut_v(ddict['giw'], giw_obj_mu.niv_core), 'giw_mu_core', rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    test_self_energy()
    test_greens_function()
