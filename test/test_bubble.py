import numpy as np
import sys, os
sys.path.append('../src')
import TestData as td
import TwoPoint as tp
import Hk as hamk
import BrillouinZone as bz
import MatsubaraFrequencies as mf
import LocalFourPoint as lfp

PLOT_PATH = './TestPlots/'

def test_bubble_convergence(siw,ek,beta,u,n,count=1):
    siwk = tp.SelfEnergy(siw[None,None,None,:],beta,pos=False,smom0=tp.get_smom0(u,n),smom1=tp.get_smom1(u,n))
    giwk = tp.GreensFunction(siwk,ek,n=n)
    niv_asympt = 4000
    giwk.set_g_asympt(niv_asympt)
    g_loc = giwk.k_mean(range='full')
    print(giwk.mem)

    niw = 100
    wn = mf.wn(niw)
    bubble_gen = lfp.LocalBubble(wn=wn,giw=g_loc,beta=beta)
    niv_sum = [100,200,400,800,1600,3200]#,6400]
    chi0_list = [bubble_gen.get_chi0(niv) for niv in niv_sum]

    import matplotlib.pyplot as plt
    colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(chi0_list)))
    plt.figure()
    for i,chi0 in enumerate(chi0_list):
        plt.plot(wn,chi0.real,'o',color = colors[i],label=f'niv = {niv_sum[i]}')
    plt.legend()
    plt.ylabel(r'$\chi^0(i\omega_n)$')
    plt.xlabel(r'n')
    plt.tight_layout()
    plt.savefig(PLOT_PATH+f'BubbleConvergence_full_{count}.png')
    plt.show()

    plt.figure()
    for i,chi0 in enumerate(chi0_list):
        plt.plot(1/niv_sum[i],chi0[niw].real,'o',color = colors[i],label=f'niv = {niv_sum[i]}')
    plt.legend()
    plt.xlabel('1/niv')
    plt.ylabel(r'$\chi^0(i\omega_n =1)$')
    plt.tight_layout()
    plt.savefig(PLOT_PATH + f'BubbleConvergence_w0_{count}.png')
    plt.show()

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
    test_bubble_convergence_1()
    test_bubble_convergence_2()
    test_bubble_convergence_3()