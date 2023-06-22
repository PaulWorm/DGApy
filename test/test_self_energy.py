import numpy as np
import sys,os
sys.path.append('../src')
sys.path.append('./src')
import TwoPoint as tp
import TestData as td
import MatsubaraFrequencies as mf
import matplotlib.pyplot as plt
import w2dyn_aux_dga

PLOT_PATH = './TestPlots/'

def test_mom_fit(siw,beta,u,n,count=1):
    sigma_dmft = tp.SelfEnergy(siw[None,None,None,:],beta,pos=False)
    smom0 = tp.get_smom0(u, n)
    smom1 = tp.get_smom1(u, n)
    print('---------------')
    print(f'Hartree: {smom0}')
    print(f'Hartree fit: {sigma_dmft.smom0}')
    print(f'First moment: {smom1}')
    print(f'First moment fit: {sigma_dmft.smom1}')
    print(f'Niv core: {sigma_dmft.estimate_niv_core()}')
    print('---------------')

    niv_full = 2000 + sigma_dmft.niv_core
    niv_core = sigma_dmft.niv_core
    min_plot = niv_full + niv_core - 50
    max_plot = niv_full + niv_core + 50
    vn_asympt = mf.vn(niv_full)
    v_asympt = mf.v(beta=sigma_dmft.beta,n=niv_full)
    sigma_asympt = sigma_dmft.get_siw(niv_core=niv_core, niv_full=niv_full)

    fig, ax = plt.subplots(1,2,figsize=(7,4))
    ax[0].plot(vn_asympt[min_plot:max_plot], sigma_asympt[0, 0, 0, min_plot:max_plot].real, '-o', color='cornflowerblue')
    ax[0].plot(vn_asympt[min_plot:max_plot], np.ones_like(vn_asympt[min_plot:max_plot])*smom0, '-', color='k')
    ax[1].plot(vn_asympt[min_plot:max_plot], sigma_asympt[0, 0, 0, min_plot:max_plot].imag, '-o', color='cornflowerblue')
    ax[1].plot(vn_asympt[min_plot:max_plot], 1/v_asympt[min_plot:max_plot]*smom1, '-', color='k')
    ax[0].set_xlabel(r'$\nu_n$')
    ax[1].set_xlabel(r'$\nu_n$')
    ax[0].set_ylabel(r'$\Re \Sigma$')
    ax[1].set_ylabel(r'$\Im \Sigma$')
    plt.tight_layout()
    plt.savefig(PLOT_PATH+f'SelfEnergyMomentExtraction_Test_{count}.png')
    plt.show()

def test_mom_fit_1():
    ddict = td.get_data_set_1()
    test_mom_fit(ddict['siw'],ddict['beta'],ddict['u'],ddict['n'],count=1)

def test_mom_fit_2():
    ddict = td.get_data_set_2()
    # print(ddict['siw'].shape)
    test_mom_fit(ddict['siw'],ddict['beta'],ddict['u'],ddict['n'],count=2)

def test_mom_fit_3():
    ddict = td.get_data_set_3()
    # print(ddict['siw'].shape)
    test_mom_fit(ddict['siw'],ddict['beta'],ddict['u'],ddict['n'],count=3)

if __name__ == '__main__':
    test_mom_fit_1()
    test_mom_fit_2()
    test_mom_fit_3()
