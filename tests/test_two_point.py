import numpy as np
import matplotlib.pyplot as plt

from dga import brillouin_zone as bz
from dga import two_point as twop
from dga import matsubara_frequencies as mf
from test_util import util_for_testing as t_util
from test_util import test_data as td


def test_self_energy(input_type='minimal', verbose=False):
    ddict, hr = td.load_testdataset(input_type)

    siw_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])

    if verbose:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        siw = mf.clip_pos_mf_vn(siw_obj.get_siw(4000)[0, 0, 0, :])
        vn = mf.vn(siw, pos=True)
        siw_dmft = mf.clip_pos_mf_vn(ddict['siw'])
        vn_dmft = mf.vn(siw_dmft, pos=True)

        axes[0].loglog(vn, siw.real, '-', color='cornflowerblue', label='core')
        axes[0].loglog(vn_dmft, siw_dmft.real, '--', color='firebrick', label='core')
        axes[1].loglog(vn, -siw.imag, '-', color='cornflowerblue', label='core')
        axes[1].loglog(vn_dmft, -siw_dmft.imag, '--', color='firebrick', label='core')
        plt.show()

    t_util.test_array([siw_obj.smom0, ], [twop.get_smom0(ddict['u'], ddict['n']), ], f'{input_type}_smom0_fitting', rtol=1e-4,
                      atol=1e-5)
    t_util.test_array([siw_obj.smom1, ], [twop.get_smom1(ddict['u'], ddict['n']), ], f'{input_type}_smom1_fitting', rtol=5e-3)

    t_util.test_array(siw_obj.sigma_core, mf.cut_v(ddict['siw'], siw_obj.niv_core), f'{input_type}_sigma_core',
                      rtol=1e-5, atol=1e-5)
    t_util.test_array(siw_obj.get_siw(niv=mf.niv_from_mat(ddict['siw'])), ddict['siw'], f'{input_type}_sigma_full',
                      rtol=1e-3, atol=1e-5)


def test_greens_function(input_type = 'minimal'):
    ddict, hr = td.load_testdataset(input_type)

    siw_obj = twop.SelfEnergy(ddict['siw'][None, None, None, :], ddict['beta'])
    nk = (32, 32, 1)
    k_grid = bz.KGrid(nk=nk)
    ek = hr.get_ek(k_grid)
    if input_type != 'ed_minimal':  # mu finding is known not to work well on ed data with a low number of bath sites
        giw_obj = twop.GreensFunction(sigma=siw_obj, ek=ek, n=ddict['n'])
    else:
        giw_obj = twop.GreensFunction(sigma=siw_obj, ek=ek, mu=ddict['mu_dmft'])

    giw_obj_mu = twop.GreensFunction(sigma=siw_obj, ek=ek, mu=ddict['mu_dmft'])

    t_util.test_array([giw_obj_mu.n, ], [ddict['n'], ], f'{input_type}_giw_n', rtol=1e-4, atol=1e-5)

    niv_cut = min([mf.niv_from_mat(ddict['giw']),mf.niv_from_mat(giw_obj.g_loc)])
    t_util.test_array(mf.cut_v(giw_obj.g_loc, niv_cut), mf.cut_v(ddict['giw'],niv_cut), f'{input_type}_giw',
                      rtol=1e-3, atol=1e-3)
    t_util.test_array(giw_obj.k_mean('core'), mf.cut_v(ddict['giw'], giw_obj.niv_core), f'{input_type}_giw_core',
                      rtol=1e-3, atol=1e-3)

    t_util.test_array(mf.cut_v(giw_obj_mu.g_loc, mf.niv_from_mat(ddict['giw'])), ddict['giw'], f'{input_type}_giw_mu',
                      rtol=1e-3, atol=1e-3)
    t_util.test_array(giw_obj_mu.k_mean('core'), mf.cut_v(ddict['giw'], giw_obj_mu.niv_core), f'{input_type}_giw_mu_core',
                      rtol=1e-3, atol=1e-3)

def main():
    input_types = ['minimal','quasi_1d']
    for it in input_types:
        test_self_energy(it)
        test_greens_function(it)

if __name__ == '__main__':
    main()

