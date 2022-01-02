# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import Hk as hk
import ChemicalPotential as chempot


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


# ======================================================================================================================
# ----------------------------------------------------- GKIW CLASS -----------------------------------------------------
# ======================================================================================================================

class GreensFunction(object):
    ''' Contains routines for the Matsubara one-particle Green's function'''

    def __init__(self, iv=None, beta=1.0, mu=0, ek=None, sigma=None):
        self._iv = iv
        self._beta = beta
        self._mu = mu
        self._ek = ek
        self._sigma = sigma
        self._gk = self.get_gk()

    @property
    def iv(self):
        return self._iv

    @iv.setter
    def iv(self, iv):
        self._iv = iv

    @property
    def niv(self):
        return self.iv.size // 2

    @property
    def beta(self):
        return self._beta

    @property
    def mu(self):
        return self._mu

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
    def gk(self, gk):
        self._gk = gk

    def get_gk(self):
        return 1. / (self._iv + self._mu - self._ek - self._sigma)

    def get_giw(self):
        return self._gk.mean(axis=(0, 1, 2))

    def cut_self_iv(self, niv_cut=0):
        self.iv = self.cut_iv(arr=self.iv, niv_cut=niv_cut)
        self.sigma = self.cut_iv(arr=self.sigma, niv_cut=niv_cut)
        self.gk = self.cut_iv(arr=self.gk, niv_cut=niv_cut)

    def cut_iv(self, arr=None, niv_cut=0):
        niv = arr.shape[-1] // 2
        return arr[..., niv - niv_cut:niv + niv_cut]


class GreensFunctionGenerator():
    '''Class that takes the ingredients for a Green's function and return GreensFunction objects'''

    def __init__(self, beta=1.0, kgrid=None, hr=None, sigma=None):
        self._beta = beta
        self._kgrid = kgrid
        self._hr = hr

        # Add k-dimensions for local self-energy
        if (len(sigma.shape) == 1):
            self._sigma = sigma[None, None, None, :]
        else:
            self._sigma = sigma

        self.set_smom()

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def kgrid(self):
        return self._kgrid

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

    def nkx(self):
        return self.kgrid[0].size

    def nky(self):
        return self.kgrid[1].size

    def nkz(self):
        return self.kgrid[2].size

    def generate_gk(self, mu=0, qiw=[0, 0, 0, 0], niv=-1):
        q = qiw[:3]
        wn = int(qiw[-1])
        kgrid = self.add_q_to_kgrid(q=q)
        ek = hk.ek_3d(kgrid=kgrid, hr=self.hr)
        sigma = self.cut_sigma(niv_cut=niv, wn=wn)
        iv = self.get_iv(niv=niv, wn=wn)
        return GreensFunction(iv=iv[None, None, None, :], beta=self.beta, mu=mu, ek=ek[:, :, :, None], sigma=sigma)

    def get_iv(self, niv=0, wn=0):
        if (niv == -1):
            niv = self.niv_sigma - int(np.abs(wn))
        return 1j * ((np.arange(-niv, niv) - wn) * 2 + 1) * np.pi / self.beta

    def cut_sigma(self, niv_cut=-1, wn=0):
        niv = self.niv_sigma
        if (niv_cut == -1):
            niv_cut = niv - int(np.abs(wn))
        return self.sigma[..., niv - niv_cut - wn:niv + niv_cut - wn]

    def add_q_to_kgrid(self, q=None):
        assert (np.size(self.kgrid) == np.size(q)), 'Kgrid and q have different dimensions.'
        kgrid = []
        for i in range(np.size(q)):
            kgrid.append(self.kgrid[i] + q[i])
        return kgrid

    def set_smom(self):
        iv = self.get_iv(niv=-1, wn=0)
        smom = chempot.fit_smom(iv=iv, siwk=self.sigma)
        self._smom = smom

    def adjust_mu(self, n=None, mu0=0, verbose=False):
        iv = self.get_iv(niv=-1, wn=0)
        ek = hk.ek_3d(kgrid=self.kgrid, hr=self.hr)
        mu = chempot.update_mu(mu0=mu0, target_filling=n, iv=iv, hk=ek, siwk=self.sigma,
                               beta=self.beta, smom0=self.smom[0], verbose=verbose)
        return mu


# ======================================================================================================================
# ------------------------------------------ MultiOrbitalGreensFunctionModule ------------------------------------------
# ======================================================================================================================


# class MultiOrbGreensFunctionGenerator():
#     ''' Multi-orbital Green's function generator'''

# def get_gkw_multi_orbital(v=None, hk=None, sw=None, dc=None, mu=None):
#     return jnp.linalg.inverse()


if __name__ == '__main__':
    input_path = '/mnt/c/users/pworm/Research/U2BenchmarkData/2DSquare_U2_tp-0.0_tpp0.0_beta15_mu1/'
    fname_dmft = '1p-data.hdf5'

    import w2dyn_aux

    # load contents from w2dynamics DMFT file:
    f1p = w2dyn_aux.w2dyn_file(fname=input_path + fname_dmft)
    dmft1p = f1p.load_dmft1p_w2dyn()
    f1p.close()

    # Set up real-space Wannier Hamiltonian:
    import Hr as hr

    t = 1.00
    tp = -0.20 * t * 0
    tpp = 0.10 * t * 0
    hr = hr.one_band_2d_t_tp_tpp(t=t, tp=tp, tpp=tpp)

    # Define k-grid
    nkf = 200
    nqf = 200
    nk = (nkf, nkf, 1)
    nq = (nqf, nqf, 1)

    # Generate k-meshes:
    import BrillouinZone as bz

    k_grid = bz.KGrid(nk=nk, name='k')
    q_grid = bz.KGrid(nk=nq, name='q')

    g_generator = GreensFunctionGenerator(beta=dmft1p['beta'], kgrid=k_grid.get_grid_as_tuple(), hr=hr,
                                          sigma=dmft1p['sloc'])
    niv_urange = 100
    gk = g_generator.generate_gk(mu=dmft1p['mu'], qiw=[0, 0, 0, 0], niv=niv_urange)

    import Indizes as ind

    index_grid_keys = ('qx', 'qy', 'qz', 'iw')
    qiw_grid = ind.IndexGrids(grid_arrays=q_grid.get_grid_as_tuple() + (0,), keys=index_grid_keys)
    qx = q_grid.grid['qx'][0]
    qy = q_grid.grid['qy'][nqf//2]
    qz = q_grid.grid['qz'][0]
    q = [qx,qy,qz,0]
    gkpq = g_generator.generate_gk(mu=dmft1p['mu'], qiw=q + [0,], niv=niv_urange)


    # Plot the Fermi-surface:
    import Plotting as plotting

    plotting.plot_giwk_fs(giwk=gk.gk, plot_dir=input_path, kgrid=k_grid, do_shift=False, kz=0, niv_plot=niv_urange, name='gk')
    plotting.plot_giwk_fs(giwk=gkpq.gk, plot_dir=input_path, kgrid=k_grid, do_shift=False, kz=0, niv_plot=niv_urange, name='gkpq')

#
