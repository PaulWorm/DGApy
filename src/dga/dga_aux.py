import numpy as np
import matplotlib.pyplot as plt
import TwoPoint_old as twop
import matsubara_frequencies as mf
import brillouin_zone as bz
import analytic_continuation as anacont


class DgaDataManager():
    ''' Class to manage the output files of a dga run.'''

    def __init__(self, path='./', load_siwk=True, load_giwk=True):
        self.path = path  # location of the output folder
        self.config = np.load(path + 'config.npy', allow_pickle=True).item()
        self.siwk = None
        self.giwk = None

        if (load_siwk): self.load_siwk()
        if (load_giwk): self.load_giwk()

    @property
    def beta(self):
        return self.config.sys.beta

    @property
    def n(self):
        return self.config.sys.n

    @property
    def hr(self):
        return self.config.sys.hr

    @property
    def kgrid(self):
        return self.config._k_grid

    @property
    def mu_dmft(self):
        return self.config.sys.mu_dmft

    @property
    def kx(self):
        return self.kgrid.kx

    @property
    def mu(self):
        return self.giwk.mu

    def get_giwk(self):
        if (self.siwk is None):
            self.load_siwk()
        g_fac = twop.GreensFunctionGenerator(beta=self.beta, kgrid=self.kgrid, hr=self.hr, sigma=self.siwk)
        mu = g_fac.adjust_mu(n=self.n, mu0=self.mu_dmft)
        giwk = g_fac.generate_gk(mu=mu)
        return giwk

    def load_siwk(self):
        self.siwk = np.load(self.path + 'sigma.npy', allow_pickle=True)

    def load_giwk(self):
        try:
            giwk = np.load(self.path + 'giwk.npy', allow_pickle=True).item()
        except:
            giwk = self.get_giwk()
            np.save(self.path + 'giwk.npy', giwk, allow_pickle=True)
        self.giwk = giwk

    def get_fs_ind(self):
        if self.giwk is None:
            self.load_giwk()

        qpd_fs = self.giwk.gk[:, :, 0, self.giwk.niv]
        ind = bz.get_fermi_surface_ind(qpd_fs)
        return ind

    def get_fs_ind_1q(self):
        if self.giwk is None:
            self.load_giwk()

        nk = self.giwk.gk.shape[0]
        qpd_fs = self.giwk.gk[:nk // 2, :nk // 2, 0, self.giwk.niv]
        ind = bz.get_fermi_surface_ind(qpd_fs)
        return ind

    def get_k_val_fermi_surface(self, ind=0, is_1q=True):
        if (is_1q):
            ind_fs = self.get_fs_ind_1q()
        else:
            ind_fs = self.get_fs_ind()
        return np.stack((self.kx[ind_fs[ind][:, 0]], self.kx[ind_fs[ind][:, 1]]), axis=1)

    def get_A_fermi_surface(self, ind=0, N=2, order=1, is_1q=True):
        if (is_1q):
            ind_fs = self.get_fs_ind_1q()
        else:
            ind_fs = self.get_fs_ind()
        giwk_fs = self.giwk.gk[ind_fs[ind][:, 0], ind_fs[ind][:, 1], 0, self.giwk.niv:]
        indizes = np.arange(len(giwk_fs))
        v = mf.v_plus(beta=self.beta, n=self.giwk.niv)
        re_fs, im_fs, Z_fs = anacont.extract_coeff_on_ind(siwk=giwk_fs, indizes=indizes, v=v, N=N, order=order)
        return -1 / np.pi * im_fs


if __name__ == '__main__':
    base = 'D:/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta22.5_n0.90/'
    path = base + 'LambdaDga_lc_sp_Nk19600_Nq19600_core60_invbse60_vurange200_wurange60/'

    dga_data = DgaDataManager(path=path)
    dga_data.load_giwk()
    dga_data.load_siwk()
    fs_loc = dga_data.get_fs_ind()

    # plt.figure()
    # plt.pcolormesh(dga_data.kx,dga_data.kx,-1/np.pi*dga_data.giwk.gk[:,:,0,dga_data.giwk.niv].imag, cmap = 'magma')
    # plt.plot(dga_data.kx[fs_loc[0][:,0]],dga_data.kx[fs_loc[0][:,1]])
    # plt.show()

    # Perform simple analytic continuation to the FS on the Fermi-surface:
    a_fs = dga_data.get_A_fermi_surface()
    plt.figure()
    plt.plot(a_fs)
    plt.show()
