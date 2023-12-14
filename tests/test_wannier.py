import os

from dga import brillouin_zone as bz
from dga import wannier
from test_util import util_for_testing as t_util

PATH_FOR_TEST_HR_HK = os.path.dirname(__file__) + '/TestHrAndHkFiles/'

def test_light_vertex():
    nk = (100,100,1)
    k_grid = bz.KGrid(nk,bz.two_dimensional_square_symmetries())
    hr = wannier.WannierHr(*wannier.wannier_one_band_2d_t_tp_tpp(1,-0.2,0.1))

    lv = hr.get_light_vertex(k_grid)
    lv_test = wannier.del_ek_del_kx_square(kx=k_grid.kx[:,None], ky=k_grid.ky[None,:],t=1,tp=-0.2,tpp=0.1)
    t_util.test_array(lv.real[:,:,0],lv_test[:,:],'light_vertex')

def test_convham():
    nk = (100,100,1)
    k_grid = bz.KGrid(nk,bz.two_dimensional_square_symmetries())
    hr = wannier.WannierHr(*wannier.wannier_one_band_2d_t_tp_tpp(1,-0.2,0.1))
    ek = hr.get_ek(k_grid)

    ek_test = wannier.ek_square(kx=k_grid.kx[:,None], ky=k_grid.ky[None,:],t=1,tp=-0.2,tpp=0.1)
    t_util.test_array(ek[:,:,0],ek_test[:,:], 'convham_square', rtol=1e-8, atol=1e-8)

    # test read from file:
    hr = wannier.create_wannier_hr_from_file(PATH_FOR_TEST_HR_HK + '1Band_t_tp_tpp_hr.dat')
    ek = hr.get_ek(k_grid)
    ek_test = wannier.ek_square(kx=k_grid.kx[:,None], ky=k_grid.ky[None,:],t=0.389093,tp=-0.097869,tpp=0.046592)

    t_util.test_array(ek[:,:,0],ek_test[:,:], 'convham_square_from_file', rtol=1e-8, atol=1e-8)
    ek_test = wannier.ek_3d(k_grid.grid, wannier.one_band_2d_t_tp_tpp(*[0.389093,-0.097869,0.046592]))
    t_util.test_array(ek[:,:,0],ek_test[:,:], 'convham_square_from_file_2', rtol=1e-8, atol=1e-8)

def test_read_write_hr_hk():
    path = PATH_FOR_TEST_HR_HK
    fname = '1onSTO-2orb_hr'
    hr = wannier.create_wannier_hr_from_file(path + fname + '.dat')
    hr.save_hr(path, fname + '_test.dat')
    hr_test = wannier.create_wannier_hr_from_file(path + fname + '_test.dat')

    t_util.test_array(hr.hr, hr_test.hr, 'read_hr')
    t_util.test_array(hr.r_grid, hr_test.r_grid, 'read_r_grid')
    t_util.test_array(hr.r_weights, hr_test.r_weights, 'read_r_weights')
    t_util.test_array(hr.orbs, hr_test.orbs, 'read_orbs')

    nk = (32,32,1)
    k_grid = bz.KGrid(nk)
    hr.save_hk(k_grid,path,fname + '_hk_test.dat')
    ek = hr.get_ek(k_grid,one_band=False)

    ek_test, kmesh_list_test = wannier.read_hk_w2k(path + fname + '_hk_test.dat')

    ek_test = k_grid.map_fbz_list2mesh(ek_test)
    t_util.test_array(ek, ek_test, 'write_hk')
    t_util.test_array(k_grid.kmesh_list, kmesh_list_test, 'write_kmesh')


def test_emery_model():
    nk = (32,32,1)
    k_grid = bz.KGrid(nk,symmetries=bz.two_dimensional_square_symmetries())

    # create the emery model hr:
    ed, ep, tpd, tpp, tpp_p = 6.5, 0.0, 2.1, 1.0, 0.2
    hr = wannier.WannierHr(*wannier.wannier_emery_model(ed,ep,tpd,tpp,tpp_p))
    ek_emery = hr.get_ek(k_grid, one_band=False)

    # create the emery hk directly from the momentum space representation:
    ek_emery_test = wannier.emery_model_ek(k_grid,ed,ep,tpd,tpp,tpp_p)
    t_util.test_array(ek_emery,ek_emery_test,'emery_model_consistency',rtol=1e-8,atol=1e-8)


if __name__ == '__main__':
    test_convham()
    test_light_vertex()
    test_read_write_hr_hk()
    test_emery_model()

