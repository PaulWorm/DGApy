import numpy as np
import sys, os
sys.path.append('../src')
import TestData as td
import TwoPoint as tp
import Hk as hamk
import BrillouinZone as bz
import MatsubaraFrequencies as mf
import Bubble as bub

PLOT_PATH = './TestPlots/'

def test_bubble_convergence(siw,ek,beta,u,n,count=1):
    siwk = tp.SelfEnergy(siw[None,None,None,:],beta,pos=False,smom0=tp.get_smom0(u,n),smom1=tp.get_smom1(u,n))
    giwk = tp.GreensFunction(siwk,ek,n=n)
    niv_asympt = 6000
    giwk.set_g_asympt(niv_asympt)
    print(giwk.mem)

    niw = 400
    wn = mf.wn(niw)
    bubble_gen = bub.LocalBubble(wn=wn,giw=giwk)
    niv_sum = [100,200,400,800,1600,2500]#,6000,8000]#,6400]
    chi0_list = [bubble_gen.get_chi0(niv) for niv in niv_sum]
    chi0_asympt = bubble_gen.get_chi0_shell(100,400)
    chi0_asympt2 = bubble_gen.get_chi0_shell(2500,3000)
    chi0_full = chi0_list[0] + chi0_asympt
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
    plt.legend()
    plt.ylabel(r'$\chi^0(i\omega_n)$')
    plt.xlabel(r'n')
    plt.tight_layout()
    plt.savefig(PLOT_PATH+f'BubbleConvergence_full_loglog_{count}.png')
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
    # print(chi0_full[niw].real - chi0_full2[niw].real)
    # print(chi0_full[niw].real - chi0_list[-1][niw].real)

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

if __name__ == '__main__':
    # test_bubble_convergence_1()
    test_bubble_convergence_2()
    # test_bubble_convergence_3()
    pass














