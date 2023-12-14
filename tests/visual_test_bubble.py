import numpy as np
import scipy.optimize as opt
import sys

sys.path.append('../src')
import dga.two_point as tp
import dga.hk as hamk
import dga.brillouin_zone as bz
import dga.matsubara_frequencies as mf
import dga.bubble as bub
from test_util import util_for_testing as ut, test_data as td

PLOT_PATH = './TestPlots/TestChi0/'

def test_bubble_convergence(siw,ek,beta,u,n,count=1):
    siwk = tp.SelfEnergy(siw[None,None,None,:],beta,pos=False,smom0=tp.get_smom0(u,n),smom1=tp.get_smom1(u,n))
    giwk = tp.GreensFunction(siwk,ek,n=n)
    niv_asympt = 10000
    giwk.set_g_asympt(niv_asympt)

    niw = 400
    wn = mf.wn(niw)
    bubble_gen = bub.BubbleGenerator(wn=wn, giwk_obj=giwk, freq_notation='minus')
    niv_sum = [100,200,400,600,800,1600,2500]#,6000,8000]#,6400]
    chi0_list = [bubble_gen.get_chi0(niv) for niv in niv_sum]
    chi0_asympt = bubble_gen.get_asymptotic_correction(600)
    # chi0_asympt = bubble_gen.get_chi0_shell(100,400)
    chi0_asympt2 = bubble_gen.get_asymptotic_correction(2500) #bubble_gen.get_chi0_shell(2500,3000)
    chi0_asympt_pure = bubble_gen.get_exact_asymptotics()
    chi0_full = chi0_list[3] + chi0_asympt
    chi0_full2 = chi0_list[-1] + chi0_asympt2

    import matplotlib.pyplot as plt
    colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(chi0_list)))
    plt.figure()
    for i,chi0 in enumerate(chi0_list):
        plt.plot(wn,chi0.real,'o',color = colors[i],label=f'niv = {niv_sum[i]}')
    plt.plot(wn,chi0_full.real,'o',color = 'k',label=f'niv = asympt')
    plt.legend()
    plt.ylabel(r'$\chi^0(i\omega_n)$')
    plt.xlabel(r'n')
    plt.tight_layout()
    plt.savefig(PLOT_PATH+f'BubbleConvergence_full_{count}.png')
    plt.show()

    colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(chi0_list)))
    plt.figure()
    for i,chi0 in enumerate(chi0_list):
        plt.loglog(wn,chi0.real,'o',color = colors[i],label=f'niv = {niv_sum[i]}')
    plt.loglog(wn,chi0_full.real,'o',color = 'k',label=f'niv = asympt')
    plt.loglog(wn,chi0_full2.real,'h',color = 'firebrick',label=f'niv = asympt2',ms=2)
    plt.loglog(wn,chi0_asympt_pure.real,'p',color = 'cornflowerblue',label=f'niv = pure-asympt',ms=1)
    plt.legend()
    plt.ylabel(r'$\chi^0(i\omega_n)$')
    plt.xlabel(r'n')
    plt.tight_layout()
    plt.savefig(PLOT_PATH+f'BubbleConvergence_full_loglog_{count}.png')
    plt.show()

    colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(chi0_list)))
    plt.figure()
    for i,chi0 in enumerate(chi0_list):
        plt.loglog(wn,np.abs(chi0.real-chi0_full2.real),'o',color = colors[i],label=f'niv = {niv_sum[i]}')
    plt.loglog(wn,np.abs(chi0_full.real-chi0_full2.real),'o',color = 'k',label=f'niv = asympt')
    plt.legend()
    plt.ylabel(r'$\chi^0(i\omega_n) - \chi^0_{exact}(i\omega_n)$')
    plt.xlabel(r'n')
    plt.tight_layout()
    plt.savefig(PLOT_PATH+f'BubbleConvergence_abs_diff_full_loglog_{count}.png')
    plt.show()

    plt.figure()
    for i,chi0 in enumerate(chi0_list):
        plt.plot(1/niv_sum[i],chi0[niw].real,'o',color = colors[i],label=f'niv = {niv_sum[i]}')
    plt.plot(0,chi0_full[niw].real,'o',color = 'k',label=f'niv = asympt')
    plt.plot(0,chi0_full2[niw].real,'h',color = 'firebrick',label=f'niv = asympt',ms=4)
    plt.legend()
    plt.xlabel('1/niv')
    plt.ylabel(r'$\chi^0(i\omega_n =1)$')
    plt.tight_layout()
    plt.savefig(PLOT_PATH + f'BubbleConvergence_w0_{count}.png')
    plt.show()

    iwnp = niw
    plt.figure()
    for i,chi0 in enumerate(chi0_list):
        plt.plot(1/niv_sum[i],chi0[niw+iwnp].real,'o',color = colors[i],label=f'niv = {niv_sum[i]}')
    plt.plot(0,chi0_full[niw+iwnp].real,'o',color = 'k',label=f'niv = asympt')
    plt.plot(0,chi0_full2[niw+iwnp].real,'h',color = 'firebrick',label=f'niv = asympt2',ms=4)
    x,y = 1/np.array(niv_sum)[3:], np.array([chi0[niw+iwnp].real for chi0 in chi0_list[3:]])
    f = lambda x, a, b: a + x * b
    fit = opt.curve_fit(f,x,y,p0=(0,-1))
    x_new = np.linspace(0,1/np.array(niv_sum)[2],100)[::-1]
    plt.plot(x_new,f(x_new,*fit[0]), label='fit')
    plt.legend()
    plt.xlabel('1/niv')
    plt.ylabel(r'$\chi^0(i\omega_n =1)$')
    plt.tight_layout()
    plt.savefig(PLOT_PATH + f'BubbleConvergence_w{iwnp}_{count}.png')
    plt.show()

    plt.figure()
    for i,chi0 in enumerate(chi0_list):
        plt.plot(1/niv_sum[i],1/beta*np.sum(chi0).real,'o',color = colors[i],label=f'niv = {niv_sum[i]}')
    plt.plot(0,1/beta*np.sum(chi0_full).real,'o',color = 'k',label=f'niv = asympt')
    plt.plot(0,1/beta*np.sum(chi0_full2).real,'h',color = 'firebrick',label=f'niv = asympt2',ms=4)
    plt.hlines((1-n/2)*n/2,0,1/niv_sum[0],ls='--',color = 'k')
    plt.legend()
    plt.xlabel('1/niv')
    plt.ylabel(r'$sum_w \chi^0(i\omega_n)$')
    plt.tight_layout()
    plt.savefig(PLOT_PATH + f'BubbleConvergence_sum_w{iwnp}_{count}.png')
    plt.show()

    ut.test_array(chi0_full2[niw+iwnp].real, chi0_full[niw+iwnp].real,count,rtol=1e-2)
    # assert np.isclose(chi0_full2[niw+iwnp].real, chi0_full[niw+iwnp].real)


def test_bubble_convergence_1():
    ddict = td.get_data_set_1()
    nk = (42,42,1)
    k_grid = bz.KGrid(nk=nk,symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid,hr=ddict['hr'])
    test_bubble_convergence(ddict['siw'],ek,ddict['beta'],ddict['u'],ddict['n'],count=1)

def test_bubble_convergence_2():
    ddict = td.get_data_set_2()
    nk = (42,42,1)
    k_grid = bz.KGrid(nk=nk,symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid,hr=ddict['hr'])
    test_bubble_convergence(ddict['siw'],ek,ddict['beta'],ddict['u'],ddict['n'],count=2)

def test_bubble_convergence_3():
    ddict = td.get_data_set_3()
    nk = (42,42,1)
    k_grid = bz.KGrid(nk=nk,symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid,hr=ddict['hr'])
    test_bubble_convergence(ddict['siw'],ek,ddict['beta'],ddict['u'],ddict['n'],count=3)

def test_bubble_convergence_4():
    ddict = td.get_data_set_4()
    nk = (42,42,1)
    k_grid = bz.KGrid(nk=nk,symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid,hr=ddict['hr'])
    test_bubble_convergence(ddict['siw'],ek,ddict['beta'],ddict['u'],ddict['n'],count=4)

def test_bubble_convergence_9():
    ddict = td.get_data_set_9()
    nk = (42,42,1)
    k_grid = bz.KGrid(nk=nk,symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid,hr=ddict['hr'])
    test_bubble_convergence(ddict['siw'],ek,ddict['beta'],ddict['u'],ddict['n'],count=9)

def test_bubble_convergence_6():
    ddict = td.get_data_set_6()
    nk = (42,42,1)
    k_grid = bz.KGrid(nk=nk,symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid,hr=ddict['hr'])
    test_bubble_convergence(ddict['siw'],ek,ddict['beta'],ddict['u'],ddict['n'],count=6)

if __name__ == '__main__':
    # test_bubble_convergence_1()
    # test_bubble_convergence_9()
    test_bubble_convergence_6()
    # test_bubble_convergence_2()
    # test_bubble_convergence_4()
    # test_bubble_convergence_3()
    pass














