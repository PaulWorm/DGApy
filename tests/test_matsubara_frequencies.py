import numpy as np

# import util_for_testing as t_util
from test_util import util_for_testing as t_util
from dga import matsubara_frequencies as mf


def test_vn():
    niv_odd = 9
    niv_even = 10
    beta = 2.
    vn_odd = np.arange(-niv_odd, niv_odd)
    v_odd = (np.arange(-niv_odd, niv_odd)*2+1)*np.pi/beta
    t_util.test_function(mf.vn, np.arange(-niv_even, niv_even), 'vn_even', niv_even)
    t_util.test_function(mf.vn, np.arange(-niv_odd, niv_odd), 'vn_odd', niv_odd)
    t_util.test_function(mf.vn, np.arange(0, niv_even), 'vn_even_pos', niv_even, pos=True)
    t_util.test_function(mf.vn, np.arange(0, niv_odd), 'vn_odd_pos', niv_odd, pos=True)
    t_util.test_function(mf.vn, v_odd, 'v_odd', beta, niv_odd)
    t_util.test_function(mf.vn, v_odd, 'v_odd_from_mat', beta, v_odd)
    t_util.test_function(mf.vn, vn_odd, 'vn_odd_from_mat', vn_odd)
    t_util.test_function(mf.vn, vn_odd+10, 'vn_shift', vn_odd,shift=10)

def test_wn():
    niw_odd = 9
    niw_even = 10
    beta = 2.
    wn_odd = np.arange(-niw_odd, niw_odd + 1)
    w_odd = (np.arange(-niw_odd, niw_odd + 1) * 2) * np.pi / beta
    t_util.test_function(mf.wn, np.arange(-niw_even, niw_even + 1), 'wn_even', niw_even)
    t_util.test_function(mf.wn, wn_odd, 'wn_odd', niw_odd)
    t_util.test_function(mf.wn, np.arange(0, niw_even + 1), 'wn_even_pos', niw_even, pos=True)
    t_util.test_function(mf.wn, np.arange(0, niw_odd + 1), 'wn_odd_pos', niw_odd, pos=True)
    t_util.test_function(mf.wn, w_odd, 'w_odd', beta, niw_odd)
    t_util.test_function(mf.wn, w_odd, 'w_odd_from_mat', beta, w_odd)
    t_util.test_function(mf.wn, wn_odd, 'wn_odd_from_mat', wn_odd)
    t_util.test_function(mf.wn, wn_odd+10, 'wn_shift', wn_odd,shift=10)

def test_mat_at_w0():
    wn = mf.wn(10)
    t_util.test_function(mf.mat_at_w0, 0, 'mat_at_w0', wn)
    wn = mf.wn(10,pos=True)
    t_util.test_function(mf.mat_at_w0, 0, 'mat_at_w0_pos', wn,pos=True)

def test_niw_from_mat():
    niw_arr = [9,10]
    for niw in niw_arr:
        mat = mf.wn(niw)[:,None] + mf.wn(niw)[None,:]
        t_util.test_function(mf.niw_from_mat, niw, f'niw_from_mat_axis_0_niw_{niw}', mat,axis=0)
        t_util.test_function(mf.niw_from_mat, niw, f'niw_from_mat_axis_1_niw_{niw}', mat,axis=1)
        mat = mf.wn(niw,pos=True)[:,None] + mf.wn(niw,pos=True)[None,:]
        t_util.test_function(mf.niw_from_mat, niw, f'niw_from_mat_axis_0_pos_niw_{niw}', mat,axis=0,pos=True)
        t_util.test_function(mf.niw_from_mat, niw, f'niw_from_mat_axis_1_pos_niw_{niw}', mat,axis=1,pos=True)

def test_niv_from_mat():
    niv_arr = [9,10]
    for niv in niv_arr:
        mat = mf.vn(niv)[:, None] + mf.vn(niv)[None, :]
        t_util.test_function(mf.niv_from_mat, niv, f'niv_from_mat_axis_0_niv_{niv}', mat, axis=0)
        t_util.test_function(mf.niv_from_mat, niv, f'niv_from_mat_axis_1_niv_{niv}', mat, axis=1)
        mat = mf.vn(niv, pos=True)[:, None] + mf.vn(niv, pos=True)[None, :]
        t_util.test_function(mf.niv_from_mat, niv, f'niv_from_mat_axis_0_pos_niv_{niv}', mat, axis=0, pos=True)
        t_util.test_function(mf.niv_from_mat, niv, f'niv_from_mat_axis_1_pos_niv_{niv}', mat, axis=1, pos=True)

def test_fermionic_full_nu_range():
    niv_core = 5
    niv_shell = 5
    niv_full = niv_core + niv_shell
    beta = 10.
    vn_full = 1 + 1j * mf.vn(beta, niv_full)

    vn_shell = mf.inv_cut_v_1d(vn_full, niv_core, niv_shell)
    vn_core = mf.cut_v_1d(vn_full, niv_core)
    vn_full_concat = mf.concatenate_core_asmypt(vn_core, vn_shell)
    t_util.test_array(vn_full_concat,vn_full, 'concatenate_core_asympt')

    vn_shell_pos = vn_shell[niv_shell:]
    vn_shell_restored = mf.fermionic_full_nu_range(vn_shell_pos)
    t_util.test_array(vn_shell_restored, vn_shell, 'fermionic_full_nu_range')

    niw_core = 0
    wn_core = mf.wn(niw_core)
    shell_slices = mf.wn_slices_shell(vn_full, niv_shell,niv_core, w= wn_core)
    t_util.test_array(shell_slices[0], vn_shell, 'wn_slices_shell')

    # t_util.test_statement(False,'test_failed')

if __name__ == '__main__':
    test_vn()
    test_wn()
    test_mat_at_w0()
    test_niw_from_mat()
    test_niv_from_mat()
    test_fermionic_full_nu_range()
