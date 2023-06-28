import numpy as np

import dga.two_point as tp
import TestData as td
import dga.matsubara_frequencies as mf
import matplotlib.pyplot as plt


PLOT_PATH = './TestPlots/'

def test_mom_fit(siw,beta,u,n,count=1):
    sigma_dmft = tp.SelfEnergy(siw[None,None,None,:],beta,pos=False, err=1e-4)
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
    niv_dmft = np.size(siw)//2
    min_plot_dmft = niv_dmft + niv_core - 50
    max_plot_dmft = niv_dmft + niv_core + 50
    vn_asympt = mf.vn(niv_full)
    v_asympt = mf.v(beta=sigma_dmft.beta,n=niv_full)
    sigma_asympt = sigma_dmft.get_siw(niv=niv_full)

    fig, ax = plt.subplots(1,2,figsize=(7,4))
    ax[0].plot(vn_asympt[min_plot:max_plot], sigma_asympt[0, 0, 0, min_plot:max_plot].real, '-o', color='cornflowerblue')
    ax[0].plot(vn_asympt[min_plot:max_plot], siw[min_plot_dmft:max_plot_dmft].real, '-x', color='firebrick',label='Input')
    ax[0].plot(vn_asympt[min_plot:max_plot], np.ones_like(vn_asympt[min_plot:max_plot])*smom0, '-', color='k')
    ax[1].plot(vn_asympt[min_plot:max_plot], sigma_asympt[0, 0, 0, min_plot:max_plot].imag, '-o', color='cornflowerblue')
    ax[1].plot(vn_asympt[min_plot:max_plot], siw[min_plot_dmft:max_plot_dmft].imag, '-x', color='firebrick', label='Input')
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

def test_mom_fit_4():
    ddict = td.get_data_set_4()
    test_mom_fit(ddict['siw'],ddict['beta'],ddict['u'],ddict['n'],count=2)

def test_mom_fit_3():
    ddict = td.get_data_set_3()
    test_mom_fit(ddict['siw'],ddict['beta'],ddict['u'],ddict['n'],count=3)

def test_mom_fit_7():
    ddict = td.get_data_set_7()
    test_mom_fit(ddict['siw'],ddict['beta'],ddict['u'],ddict['n'],count=3)

def test_mom_fit_8():
    ddict = td.get_data_set_8()
    test_mom_fit(ddict['siw'],ddict['beta'],ddict['u'],ddict['n'],count=3)

if __name__ == '__main__':
    # test_mom_fit_1()
    # test_mom_fit_4()
    # test_mom_fit_3()
    # test_mom_fit_7()
    test_mom_fit_8()

# #%%
#     import h5py
#     file = h5py.File('./2DSquare_U8_tp-0.25_tpp0.12_beta75_n0.85/1p-data.hdf5','a')
#     tmp = file['/dmft-last/ineq-001/giw/value'][()]
#     niv_cut = 30000
#     niv = np.shape(tmp)[-1]//2
#     del file['dmft-last/ineq-001/giw/value']
#     dset = file.create_dataset('dmft-last/ineq-001/giw/value', (1, 2,2*niv_cut),dtype=complex)
#     dset[:,:,:] = tmp[:,:,niv-niv_cut:niv+niv_cut]
#     file.close()



    # #%%
    # ddict = td.get_data_set_8()
    # sigma_dmft = tp.SelfEnergy(ddict['siw'][None,None,None,:],ddict['beta'],pos=False)
    # niv = np.shape(ddict['siw'])[-1]//2
    # tmp = ddict['siw'][niv:]
    # smom0 = np.mean(tmp[niv-500:].real)
    # iv_fit = mf.iv(ddict['beta'],niv)
    # fit_smom0, fit_smom1 = tp.fit_smom(iv_fit, ddict['siw'][None,None,None,:])
    # print(smom0)
    # print(fit_smom0)

