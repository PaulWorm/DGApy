import numpy as np
import matplotlib.pyplot as plt
import BrillouinZone as bz
import MatsubaraFrequencies as mf


def get_smom0(u, n):
    '''Return Hartree for the single-band SU(2) symmetric case'''
    return u * n / 2


def get_smom1(u, n):
    ''' return 1/ivu asymptotic prefactor of Im(Sigma) for the single-band SU(2) symmetric case'''
    return u ** 2 * n / 2 * (1 - n / 2)


def fit_smom(iv=None, siwk=None, only_positive=True):
    """Read or calculate self-energy moments"""
    niv = siwk.shape[-1] // 2
    if (not only_positive):
        siwk = siwk[..., niv:]

    n_freq_fit = int(0.2 * niv)
    if n_freq_fit < 4:
        n_freq_fit = 4
    s_loc = np.mean(siwk, axis=(0, 1, 2))  # moments should not depend on momentum

    iwfit = iv[niv - n_freq_fit:]
    fitdata = s_loc[niv - n_freq_fit:]
    mom0 = np.mean(fitdata.real)
    mom1 = np.mean(fitdata.imag * iwfit.imag)  # There is a minus sign in Josef's corresponding code, but this complies with the output from w2dyn.

    return mom0, mom1


class SelfEnergy():
    ''' class to handle self-energies'''

    def __init__(self, sigma, beta, only_positive=True, smom0=None, smom1=None):
        assert len(np.shape(sigma)) == 4, 'Currently only single-band SU(2) supported with [kx,ky,kz,v]'
        if (not only_positive):
            niv = sigma.shape[-1] // 2
            sigma = sigma[..., niv:]

        self.sigma = sigma
        self.beta = beta
        iv_plus = mf.iv_plus(beta, self.niv)
        fit_mom0, fit_mom1 = fit_smom(iv_plus, sigma)

        # Set the moments for the symptotic behaviour:
        if (smom0 is None):
            self.smom0 = fit_mom0
        else:
            self.smom0 = smom0

        if (smom1 is None):
            self.smom1 = fit_mom1
        else:
            self.smom1 = smom1

    @property
    def niv(self):
        return self.sigma.shape[-1]

    @property
    def nk(self):
        return self.sigma.shape[:-1]

    def get_siw(self, niv_core=None,niv_full = None):
        if(niv_core is None):
            niv_core = self.niv
        if (niv_core <= self.niv and niv_full is None):
            return np.concatenate((np.conj(np.flip(self.sigma[..., :niv_core], -1)), self.sigma[..., :niv_core]), axis=-1)
        else:
            iv_asympt = mf.iv_plus(self.beta, n=niv_full, n_min=niv_core)
            asympt = (self.smom0 - 1 / iv_asympt * self.smom1)[None, None, None,:] * np.ones(self.nk)[:, :, :,None]
            sigma_asympt = np.concatenate((self.sigma[...,:niv_core], asympt), axis=-1)
            return np.concatenate((np.conj(np.flip(sigma_asympt, -1)), sigma_asympt), axis=-1)


if __name__ == '__main__':
    import w2dyn_aux

    path = '../test/2DSquare_U8_tp-0.2_tpp0.1_beta17.5_n0.90/'
    file = '1p-data.hdf5'

    dmft_file = w2dyn_aux.w2dyn_file(fname=path + file)
    siw = dmft_file.get_siw()[0, 0, :][None, None, None, :]
    beta = dmft_file.get_beta()
    u = dmft_file.get_udd()
    n = dmft_file.get_totdens()
    sigma_dmft = SelfEnergy(sigma=siw, beta=beta, only_positive=False)

    print('---------------')
    print(f'Hartree: {get_smom0(u, n)}')
    print(f'Hartree fit: {sigma_dmft.smom0}')
    print(f'First moment: {get_smom1(u, n)}')
    print(f'First moment fit: {sigma_dmft.smom1}')
    print('---------------')
    niv_full  = 2000
    niv_core = 200
    min_plot = niv_full+niv_core-50
    max_plot = niv_full+niv_core+50
    vn_asympt = mf.vn(niv_full)
    sigma_asympt = sigma_dmft.get_siw(niv_core=niv_core,niv_full=niv_full)
    plt.figure()
    plt.plot(vn_asympt[min_plot:max_plot],sigma_asympt[0,0,0,min_plot:max_plot].imag,'-o',color='cornflowerblue')
    plt.show()

    print(mf.vn_plus(10,5))