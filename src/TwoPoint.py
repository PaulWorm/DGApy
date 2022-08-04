# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import Hk as hk
import ChemicalPotential as chempot
import BrillouinZone as bz

# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

# ------------------------------------------------ OBJECTS -------------------------------------------------------------


# ======================================================================================================================
# ---------------------------------------------- TWO-POINT PARENT CLASS ------------------------------------------------
# ======================================================================================================================

class two_point():
    ''' Parent function for two-point correlation function and derived quantities like self-energy'''

    def __init__(self, beta=1.0, niv=0):
        self.set_beta(beta)
        self.set_niv(niv)
        self.set_vn()
        self.set_iv()

    def set_beta(self, beta):
        self._beta = beta

    def set_niv(self, niv):
        self._niv = niv

    def set_vn(self):
        self._vn = np.arange(-self._niv, self._niv)

    def set_iv(self):
        try:
            self._vn
        except:
            self.set_vn()
        self._iv = (self._vn * 2 + 1) * 1j * np.pi / self._beta


def create_gk_dict(dga_conf=None, sigma=None, mu0=None, adjust_mu=True, niv_cut=None):
    if (niv_cut is not None):
        niv = sigma.shape[-1] // 2
        sigma = sigma[..., niv - niv_cut:niv + niv_cut]

    gk_dga_generator = GreensFunctionGenerator(beta=dga_conf.sys.beta, kgrid=dga_conf.k_grid, hr=dga_conf.sys.hr,
                                               sigma=sigma)
    if (adjust_mu):
        mu_dga = gk_dga_generator.adjust_mu(n=dga_conf.sys.n, mu0=mu0)
    else:
        mu_dga = mu0
    n_dga = gk_dga_generator.get_fill(mu=mu_dga)
    gk_dga = gk_dga_generator.generate_gk(mu=mu_dga)

    gf_dict = {
        'gk': gk_dga._gk,
        'mu': gk_dga._mu,
        'n': n_dga,
        'iv': gk_dga._iv,
        'beta': gk_dga._beta,
        'niv': gk_dga.niv
    }
    return gf_dict


# ======================================================================================================================
# ----------------------------------------------------- GKIW CLASS -----------------------------------------------------
# ======================================================================================================================

class GreensFunction(object):
    ''' Contains routines for the Matsubara one-particle Green's function'''

    def __init__(self, iv=None, beta=1.0, mu=0, ek=None, sigma=None):
        self.iv = iv
        self.beta = beta
        self.mu = mu
        self._ek = ek
        self.sigma = sigma
        self.gk = self.get_gk()

    @property
    def iv(self):
        return self._iv

    @iv.setter
    def iv(self, iv):
        self._iv = iv

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self,value):
        self._mu = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    @property
    def niv(self) -> int:
        return self.iv.shape[-1] // 2

    @property
    def gk(self):
        return self._gk

    @gk.setter
    def gk(self, value):
        self._gk = value

    def get_gk(self):
        return 1. / (self._iv + self._mu - self._ek - self._sigma)

    def get_giw(self):
        return self._gk.mean(axis=(0, 1, 2))

    def cut_self_iv(self, niv_cut=0):
        self.iv = self.cut_iv(arr=self.iv, niv_cut=niv_cut)
        self.sigma = self.cut_iv(arr=self.sigma, niv_cut=niv_cut)
        self.gk = self.cut_iv(arr=self.gk, niv_cut=niv_cut)

    def get_gk_cut_iv(self, niv_cut=0):
        return self.cut_iv(arr=self.gk, niv_cut=niv_cut)

    def cut_iv(self, arr=None, niv_cut=0):
        niv = arr.shape[-1] // 2
        return arr[..., niv - niv_cut:niv + niv_cut]

    def k_mean(self):
        return self.gk.mean(axis=(0,1,2))


class GreensFunctionGenerator():
    '''Class that takes the ingredients for a Green's function and return GreensFunction objects'''

    def __init__(self, beta=1.0, kgrid: bz.KGrid =None, hr=None, sigma=None,set_smom=True):
        self._beta = beta
        self.k_grid = kgrid
        self._hr = hr

        # Add k-dimensions for local self-energy
        if (len(sigma.shape) == 1):
            self._sigma = sigma[None, None, None, :]
            self._sigma_type = 'loc'
        else:
            self._sigma = sigma
            self._sigma_type = 'nloc'

        if(set_smom):
            self.set_smom()

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def k_grid(self):
        return self._k_grid

    @k_grid.setter
    def k_grid(self, value):
        self._k_grid = value

    @property
    def hr(self):
        return self._hr

    @property
    def sigma(self):
        return self._sigma

    @property
    def smom(self):
        return self._smom

    @property
    def niv_sigma(self):
        return self._sigma.shape[-1] // 2

    @property
    def nkx(self):
        return self.k_grid.nk[0]

    @property
    def nky(self):
        return self.k_grid.nk[1]

    @property
    def nkz(self):
        return self.k_grid.nk[2]

    def generate_gk(self, mu=0, qiw=(0, 0, 0, 0), niv=-1, v_range='full'):
        q = qiw[:3]
        wn = int(qiw[-1])
        niv = self.check_niv(niv=niv,wn=wn)
        v_slice = self.get_v_slice(niv=niv, v_range=v_range)
        iv = self.get_iv(niv=niv, wn=wn)[v_slice]
        kgrid = self.add_q_to_kgrid(q=q)
        ek = hk.ek_3d(kgrid=kgrid, hr=self.hr)
        sigma = self.cut_sigma(niv_cut=niv, wn=wn)[...,v_slice]
        if(self._sigma_type == 'nloc'):
            sigma = self.shift_mat_by_q(mat=sigma,q=q)
        return GreensFunction(iv=iv[None, None, None, :], beta=self.beta, mu=mu, ek=ek[:, :, :, None], sigma=sigma)

    def get_ekpq(self,qiw=(0, 0, 0, 0)):
        q = qiw[:3]
        kgrid = self.add_q_to_kgrid(q=q)
        return hk.ek_3d(kgrid=kgrid, hr=self.hr)

    def get_v_slice(self,niv=None,v_range='full'):
        if v_range in {'+','p','plus'}:
            return slice(niv, None)
        elif v_range in {'-','m','minus'}:
            return slice(0,niv)
        else:
            return slice(0,None)

    def check_niv(self, niv=None, wn=None):
        if (niv == -1):
            niv = self.niv_sigma - int(np.abs(wn))
        return niv

    def generate_gk_plus(self, mu=0, qiw=[0, 0, 0, 0], niv=-1):
        return self.generate_gk(mu=mu, qiw=qiw, niv=niv, v_range='+')

    def generate_gk_minus(self, mu=0, qiw=[0, 0, 0, 0], niv=-1):
        return self.generate_gk(mu=mu, qiw=qiw, niv=niv, v_range='-')

    def get_iv(self, niv=0, wn=0):
        niv = self.check_niv(niv=niv,wn=wn)
        return 1j * ((np.arange(-niv, niv) - wn) * 2 + 1) * np.pi / self.beta

    def cut_sigma(self, niv_cut=-1, wn=0):
        niv = self.niv_sigma
        if (niv_cut == -1):
            niv_cut = niv - int(np.abs(wn))
        return self.sigma[..., niv - niv_cut - wn:niv + niv_cut - wn]

    def add_q_to_kgrid(self, q=(0,0,0)):
        return self.k_grid.add_q_to_kgrid(q=q)

    def shift_mat_by_q(self,mat=None,q=(0,0,0)):
        return self.k_grid.shift_mat_by_q(mat=mat,q=q)

    def set_smom(self):
        iv = self.get_iv(niv=-1, wn=0)
        smom = chempot.fit_smom(iv=iv, siwk=self.sigma)
        self._smom = smom

    def adjust_mu(self, n=None, mu0=0, verbose=False):
        iv = self.get_iv(niv=-1, wn=0)
        ek = hk.ek_3d(kgrid=self.k_grid.grid, hr=self.hr)
        mu = chempot.update_mu(mu0=mu0, target_filling=n, iv=iv, hk=ek, siwk=self.sigma,
                               beta=self.beta, smom0=self.smom[0], verbose=verbose)
        return mu

    def get_fill(self, mu=None, verbose=False):
        iv = self.get_iv(niv=-1, wn=0)
        ek = hk.ek_3d(kgrid=self.k_grid.grid, hr=self.hr)
        hloc = np.mean(ek)
        n, _ = chempot.get_fill(iv=iv, hk=ek, siwk=self.sigma, beta=self.beta, smom0=self.smom[0], hloc=hloc, mu=mu,
                                verbose=verbose)
        return n


# ======================================================================================================================
# ------------------------------------------ MultiOrbitalGreensFunctionModule ------------------------------------------
# ======================================================================================================================


# class MultiOrbGreensFunctionGenerator():
#     ''' Multi-orbital Green's function generator'''

# def get_gkw_multi_orbital(v=None, hk=None, sw=None, dc=None, mu=None):
#     return jnp.linalg.inverse()


if __name__ == '__main__':
    #input_path = '/mnt/c/users/pworm/Research/U2BenchmarkData/2DSquare_U2_tp-0.0_tpp0.0_beta15_mu1/'
    input_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta10_n0.85/KonvergenceAnalysis/'
    path_dga = input_path + 'LambdaDga_lc_sp_Nk1024_Nq1024_core27_invbse27_vurange80_wurange27/'
    fname_dmft = '1p-data.hdf5'

    import w2dyn_aux

    # load contents from w2dynamics DMFT file:
    f1p = w2dyn_aux.w2dyn_file(fname=input_path + fname_dmft)
    dmft1p = f1p.load_dmft1p_w2dyn()
    f1p.close()

    # Set up real-space Wannier Hamiltonian:
    import Hr as hr

    t = 1.0
    tp = -0.20 * t
    tpp = 0.10 * t
    hr = hr.one_band_2d_t_tp_tpp(t=t, tp=tp, tpp=tpp)

    # Define k-grid
    nkf = 32
    nqf = 32
    nk = (nkf, nkf, 1)
    nq = (nqf, nqf, 1)

    # Generate k-meshes:
    import BrillouinZone as bz

    k_grid = bz.KGrid(nk=nk)
    q_grid = bz.KGrid(nk=nq)

    g_generator = GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=k_grid, hr=hr,
                                          sigma=dmft1p['sloc'])
    print('check')
    mu = g_generator.adjust_mu(n=dmft1p['n'],mu0=dmft1p['mu'])
    print('check')
    gk = g_generator.generate_gk(mu=mu)
    print('check')

    import matplotlib.pyplot as plt

    ek = hk.ek_3d(kgrid=g_generator.k_grid.grid, hr=g_generator.hr)

    plt.imshow(ek[:,:,0], cmap='RdBu', origin='lower')
    plt.colorbar()
    plt.show()

    plt.imshow(-gk.gk.imag[:,:,0,gk.niv], cmap='RdBu', origin='lower')
    plt.colorbar()
    plt.show()

    sigma_nl = np.load(path_dga + 'sigma.npy', allow_pickle=True)

    g_generator_nl = GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=k_grid, hr=hr,
                                          sigma=sigma_nl)
    mu_snl = g_generator_nl.adjust_mu(n=dmft1p['n'],mu0=dmft1p['mu'])
    gk_snl = g_generator_nl.generate_gk(mu=mu_snl)
    gk_snl_pi = g_generator_nl.generate_gk(mu=mu_snl,qiw=[np.pi,np.pi,0,0])
    gk_p = g_generator_nl.generate_gk_plus(mu=mu_snl,qiw=[0,0,0,0])
    gk_p_loc = gk_p.k_mean()

    plt.imshow(-gk_snl.gk.imag[:,:,0,gk_snl.niv], cmap='RdBu', origin='lower')
    plt.colorbar()
    plt.show()

    plt.imshow(-gk_snl_pi.gk.imag[:,:,0,gk_snl_pi.niv], cmap='RdBu', origin='lower')
    plt.colorbar()
    plt.show()

    plt.plot(gk_p_loc)
    plt.show()
