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
    niv_asympt = 6000
    giwk.set_g_asympt(niv_asympt)
    g_loc = giwk.k_mean(range='full')
    print(giwk.mem)

    niw = 400
    wn = mf.wn(niw)
    bubble_gen = lfp.LocalBubble(wn=wn,giw=g_loc,beta=beta)
    niv_sum = [100,200,400,800,1600,2500]#,6000,8000]#,6400]
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

    colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(chi0_list)))
    plt.figure()
    for i,chi0 in enumerate(chi0_list):
        plt.loglog(wn,chi0.real,'o',color = colors[i],label=f'niv = {niv_sum[i]}')
    plt.legend()
    plt.ylabel(r'$\chi^0(i\omega_n)$')
    plt.xlabel(r'n')
    plt.tight_layout()
    plt.savefig(PLOT_PATH+f'BubbleConvergence_full_loglog_{count}.png')
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
    # test_bubble_convergence_1()
    # test_bubble_convergence_2()
    # test_bubble_convergence_3()

    ddict = td.get_data_set_2()
    nk = (42,42,1)
    k_grid = bz.KGrid(nk=nk,symmetries=bz.two_dimensional_square_symmetries())
    ek = hamk.ek_3d(k_grid.grid,hr=ddict['hr'])
    ek_mom0 = np.mean(ek*ek)
    niw_chi0 = 500
    wn = mf.w(ddict['beta'],niw_chi0)
    chi0_asympt = np.zeros((wn.size),dtype=complex)
    ind = wn != 0
    beta,u,n,mu = ddict['beta'],ddict['u'],ddict['n'],ddict['mu']
    fac1 = (mu-u*n/2)**2 + ek_mom0 + u**2 * n/2*(1-n/2)
    fac2 = (mu-u*n/2)**2
    chi0_asympt[ind] = beta/2* 1/wn[ind]**2*(fac1-fac2)
    ind = wn == 0
    chi0_asympt[ind] = beta/4 - beta**3/24 * fac1 - beta**3/48 * fac2

    def get_chi0_asympt_sum(wn,niv_sum):
        chi0_asympt_sum = np.zeros((wn.size),dtype=complex)
        v = 1/(1j*mf.v(beta,n=niv_sum+wn.size//2))
        niv = v.size//2
        for i,iwn in enumerate(wn):
            print(iwn)
            v_sum = v[niv-niv_sum:niv+niv_sum]
            vpw_sum = v[niv-niv_sum+iwn:niv+niv_sum+iwn]
            chi0_asympt_sum[i] += np.sum(v_sum*vpw_sum)
            chi0_asympt_sum[i] -= np.sum(v_sum**2*vpw_sum+v_sum*vpw_sum**2)*(mu - u*n/2)
            chi0_asympt_sum[i] += np.sum(v_sum**3*vpw_sum+v_sum*vpw_sum**3)*fac1
            chi0_asympt_sum[i] += np.sum(v_sum**2*vpw_sum**2)*fac2
        return chi0_asympt_sum*(-1/beta)



    siwk = tp.SelfEnergy(ddict['siw'][None,None,None,:],beta,pos=False,smom0=tp.get_smom0(u,n),smom1=tp.get_smom1(u,n))
    giwk = tp.GreensFunction(siwk,ek,n=n)
    niv_asympt = 10000
    giwk.set_g_asympt(niv_asympt)
    g_loc = giwk.k_mean(range='full')
    print(giwk.mem)


    wn_chi0 = mf.wn(niw_chi0)
    w_chi0 = mf.w(beta,niw_chi0)
    bubble_gen = lfp.LocalBubble(wn=wn_chi0,giw=g_loc,beta=beta)
    niv_chi0_sum = 2000
    chi0_list = bubble_gen.get_chi0(niv_chi0_sum)

    chi0_asympt_sum = get_chi0_asympt_sum(wn_chi0,niv_sum=niv_chi0_sum)
    correction = chi0_asympt.real-chi0_asympt_sum.real
    import matplotlib.pyplot as plt
    plt.figure()
    ind = wn > 0
    ind_chi0 = w_chi0 > 0
    plt.loglog(wn[ind],chi0_asympt.real[ind],'-h',color='cornflowerblue',markeredgecolor='cornflowerblue',alpha=0.8)
    plt.loglog(w_chi0[ind_chi0],(chi0_list.real[ind_chi0]+correction[ind_chi0].real),'-h',color='firebrick',markeredgecolor='firebrick',alpha=0.8)
    plt.loglog(w_chi0[ind_chi0],chi0_asympt_sum.real[ind_chi0],'-h',color='seagreen',markeredgecolor='seagreen',alpha=0.8)
    plt.loglog(w_chi0,correction,'-h',color='navy',markeredgecolor='navy',alpha=0.8)
    plt.show()

    plt.plot(w_chi0,correction)
    plt.show()














