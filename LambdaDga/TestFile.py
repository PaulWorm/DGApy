# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import sys
import importlib
import time
import numpy as np
import w2dyn_aux
import FourPoint as fp
import TwoPoint as twop
import SDE as sde
import Hk as hk
import h5py
import matplotlib.pyplot as plt

importlib.reload(twop)

# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

class real_time():
    ''' simple class to keep track of real time '''

    def __init__(self):
        self._ts = time.time()
        self._tm = []
        self._tm.append(0)

    def measure_time(self):
        self._tm.append(time.time() - self._ts - self._tm[-1])

    def print_time(self,string=''):
        self.measure_time()
        print(string + 'took {} seconds'.format(self._tm[-1]))

# ----------------------------------------------- PARAMETERS -----------------------------------------------------------
path = './'
dataset = ''
#path = 'C:/Users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/LambdaDga_Python/'
do_pairing_vertex = True
t0 = time.time()
Nk = 8
t_x = 1./4.
t_y = t_x
t_xy = -0.25 * t_x * 0
t_yx = t_xy
t2_x = 0.12 * t_x * 0
t2_y = t2_x
nivp = 5
t_mat  = np.array([[t_x, t_y],[t_xy, t_yx],[t2_x, t2_y]])

niw_core = 19
niv_core = 20
niv_urange = 100
niv_asympt = 5000
iw = np.arange(-niw_core, niw_core + 1)

Nkx = Nk
Nky = Nk
Nkz = 1
Nqx = Nk
Nqy = Nk
Nqz = 1

box_sizes = {
    "niw_core": niw_core,
    "niv_core": niv_core,
    "niv_urange": niv_urange,
    "niv_asympt": niv_asympt,
    "Nkx": Nkx,
    "Nky": Nky,
    "Nkz": Nkz,
    "Nqx": Nqx,
    "Nqy": Nqy,
    "Nqz": Nqz
}


kx = np.arange(0, 2 * Nkx)
ky = np.arange(0, 2 * Nky)

tb = hk.tight_binding()
ek = tb.ek_2d(kx=kx[:,None],ky=ky[None,:],t_mat=t_mat)

f1p = w2dyn_aux.w2dyn_file(fname=path+'1p-data.hdf5')
dmft1p = f1p.load_dmft1p_w2dyn()
f1p.close()

gkiw = twop.matsubara_greens_function(beta=dmft1p['beta'], u=dmft1p['u'], mu=dmft1p['mu'], niv=dmft1p['sloc'].size//2,
                                      ek=ek[:, :, None], sigma=dmft1p['sloc'])
giw = gkiw.get_giw()

# --------------------------------------------- LOCAL BUBBLE -----------------------------------------------------------

realt = real_time()
chi0 = fp.local_bubble(giw=giw, beta=dmft1p['beta'], box_sizes=box_sizes, iw=iw)
chi0.set_chi0(range='niv_core')
chi0.set_chi0(range='niv_urange')
chi0.set_chi0_asympt()
chi0.set_gchi0(range='niv_core')
realt.print_time(string='Chi0 calculation ')

gcon_dens = fp.local_connected_four_point(loc_bub=chi0, u=dmft1p['u'], channel='dens')
gcon_dens.set_g2_from_w2dyn_file(fname=path+'g4iw_sym.hdf5')
gcon_dens._g2 = gcon_dens.cut_iv(four_point=gcon_dens._g2,niv_cut=gcon_dens._lb._box_sizes['niv_core'])
gcon_dens._niv_g2 = gcon_dens._g2.shape[-1] // 2
gcon_dens.set_gchir_from_g2()
gcon_dens.set_gammar_from_gchir()
realt.print_time(string='Gamma extraction')
gcon_dens.set_gchi_aux_from_gammar()
realt.print_time(string='Chi-aux calculation')
gcon_dens.set_chir_from_chi_aux()
realt.print_time(string='Chi_r calculation ')
gcon_dens.set_fermi_bose()
realt.print_time(string='Fermi-bose calculation ')


plt.plot(gkiw._vn, giw.imag)
plt.show()
