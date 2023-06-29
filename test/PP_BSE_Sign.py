import numpy as np
import matplotlib.pyplot as plt
import FourPoint as fp
import w2dyn_aux_dga
import MatsubaraFrequencies as mf
import PairingVertex as pv
from FourPoint import  LocalFourPoint
import matplotlib.colors as colors

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def get_gchi0_pp(giw, niv, wn=0):
    niv_giw = giw.size // 2
    return -beta * giw[niv_giw - niv:niv_giw + niv] *giw[niv_giw - niv - wn:niv_giw + niv - wn]
def vec_get_gchi0_pp(giw,niv,iw):
    return np.array([get_gchi0_pp(giw,niv,wn) for wn in iw])

def bse_pp(gamma,chi0_pp,sign=1):
    niv = gamma.shape[-1]//2
    return np.matmul( np.linalg.inv(np.eye(2 * niv, dtype=complex) + sign * 0.5 * gamma * chi0_pp[:, None]),gamma)

def inv_bse_pp(F,chi0_pp,sign=1):
    niv = F.shape[-1]//2
    return np.matmul(F, np.linalg.inv(np.eye(2 * niv, dtype=complex) - sign * 0.5 * F * chi0_pp[:, None]))

def inv_bse_ph(F,chi0,sign=1):
    niv = F.shape[-1]//2
    return np.matmul(F, np.linalg.inv(np.eye(2 * niv, dtype=complex) - sign * chi0[:,None] * F))

def f_from_gchi(gchi,gchi0):
    return np.diag(1./gchi0) - 1/gchi0[:,None] * gchi * 1/gchi0[None,:]

def gammar_from_gchir_wn(gchir=None, gchi0=None):
    return -(np.diag(1. / gchi0) - np.linalg.inv(gchir))

def gammar_from_gchir(gchir = None, gchi0 = None):
    gammar = np.array(
        [gammar_from_gchir_wn(gchir=gchir.mat[wn], gchi0=gchi0.gchi0[wn]) for wn in gchir.wn_lin])
    return LocalFourPoint(matrix=gammar, channel=gchir.channel, beta=gchir.beta, wn=gchir.wn)

path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta20_n0.85/'
path = '/mnt/d/Research/HoleDopedNickelates/2DSquare_U8_tp-0.25_tpp0.12_beta25_n0.85/from_Motoharu/'
# path = 'D:Research/U2BenchmarkData/2DSquare_U2_tp-0.0_tpp0.0_beta20_mu1/'
fname1p = '1p-data.hdf5'
fname2p = 'g4iw_sym.hdf5'


# Load 1-particle stuff
f1p = w2dyn_aux.w2dyn_file(fname=path + fname1p)
dmft1p = f1p.load_dmft1p_w2dyn()
giw_dmft = dmft1p['gloc']
beta = dmft1p['beta']
u = dmft1p['u']
f1p.close()

# Load 2-particle stuff
f2p = w2dyn_aux_dga.g4iw_file(fname=path + fname2p)
niw_core = f2p.get_niw(channel='dens')
niw_core = 50
niv_invbse = niw_core
wn_core = mf.wn(n=niw_core)
g2_dens = fp.LocalFourPoint(matrix=f2p.read_g2_iw(channel='dens', iw=wn_core), channel='dens',
                       beta=dmft1p['beta'], wn=wn_core)
g2_magn = fp.LocalFourPoint(matrix=f2p.read_g2_iw(channel='magn', iw=wn_core), channel='magn',
                       beta=dmft1p['beta'], wn=wn_core)
f2p.close()

g2_dens.cut_iv(niv_cut=niv_invbse)
g2_magn.cut_iv(niv_cut=niv_invbse)


niv_core = g2_dens.mat.shape[-1]
niv_urange = niv_core

# Create generalized susceptibility:
gchi_dens = fp.chir_from_g2(g2_dens,giw_dmft)
gchi_magn = fp.chir_from_g2(g2_magn,giw_dmft)

# Extract gamma:
chi0_urange = fp.BubbleGenerator(giw=giw_dmft, beta=beta, niv=niv_urange, wn=wn_core)
chi0_core = fp.BubbleGenerator(giw=giw_dmft, beta=beta, niv=niv_invbse, wn=wn_core)

gamma_dens =  gammar_from_gchir(gchir=gchi_dens, gchi0=chi0_core)
gamma_magn = gammar_from_gchir(gchir=gchi_magn, gchi0=chi0_core)

# gamma_dens =  fp.gammar_from_gchir(gchir=gchi_dens, gchi0_urange=chi0_core, u=u)
# gamma_magn = fp.gammar_from_gchir(gchir=gchi_magn, gchi0_urange=chi0_core, u=u)


# Build vertex from BSE:

F_dens = fp.local_vertex_inverse_bse(gamma=gamma_dens, chi0=chi0_core, u=u)
F_magn = fp.local_vertex_inverse_bse(gamma=gamma_magn, chi0=chi0_core, u=u)
F_dens_from_gchi = f_from_gchi(gchi_dens.mat[niw_core],chi0_core.gchi0[niw_core])
F_magn_from_gchi = f_from_gchi(gchi_magn.mat[niw_core],chi0_core.gchi0[niw_core])
F_updo = 0.5 * (F_dens - F_magn)

F_dens_pp = pv.ph_to_pp_notation(mat_ph=F_dens,wn=0)
F_magn_pp = pv.ph_to_pp_notation(mat_ph=F_magn,wn=0)
# F_magn_ph = pv.ph_to_pp_notation(mat_ph=F_magn_pp,wn=0)
F_updo_pp = 0.5 * (F_dens_pp - F_magn_pp)
F_trip_pp = F_updo_pp - F_magn_pp
F_sing_pp = F_updo_pp + F_magn_pp

gamma_dens_bse = inv_bse_ph(F_dens_from_gchi,chi0_core.gchi0[niw_core],1)
gamma_magn_bse = inv_bse_ph(F_magn_from_gchi,chi0_core.gchi0[niw_core],1)

gamma_dens_from_gchi = gammar_from_gchir_wn(gchi_dens.mat[niw_core],chi0_core.gchi0[niw_core])
gamma_magn_from_gchi = gammar_from_gchir_wn(gchi_magn.mat[niw_core],chi0_core.gchi0[niw_core])
# gamma_magn_bse = inv_bse_ph(F_magn[niw_core],chi0_core.gchi0[niw_core],1)
# pp-notation shift

niv_pp = F_sing_pp.shape[-1] // 2
chi0_pp = get_gchi0_pp(giw_dmft,niv_pp,wn=0)

sign_sing = -1
sign_trip = -sign_sing
gamma_pp_sing_pp = inv_bse_pp(F_sing_pp,chi0_pp,sign=sign_sing)
gamma_pp_trip_pp = inv_bse_pp(F_trip_pp,chi0_pp,sign=sign_trip)


# gamma_updo_pp = 0.5 * (gamma_dens_pp-gamma_magn_pp)
# gamma_sing_pp = gamma_updo_pp - gamma_magn_pp
# gamma_trip_pp = gamma_updo_pp + gamma_magn_pp

F_sing_bse_pp = bse_pp(gamma_pp_sing_pp,chi0_pp,sign_sing)
F_trip_bse_pp = bse_pp(gamma_pp_trip_pp,chi0_pp,sign_trip)




F_dens_bse_pp = 0.5 * (3*F_trip_bse_pp + F_sing_bse_pp)
F_magn_bse_pp = 0.5 * (F_trip_bse_pp - F_sing_bse_pp)

# Figures

def draw_gamma(ax,gamma):
    # norm=colors.TwoSlopeNorm(vmin=gamma.real.min(), vcenter=0., vmax=gamma.real.max())
    # norm=colors.TwoSlopeNorm(vmin=-u, vcenter=0., vmax=+u)
    # im = ax.imshow(gamma.real, cmap='RdBu',norm=norm)
    im = ax.imshow(gamma.real, cmap='RdBu')
    fig.colorbar(im, ax=(ax), aspect=20, fraction=0.1, location='right',
                 pad=0.13)

fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(7,5))
axes = axes.flatten()

draw_gamma(axes[0],gamma_pp_sing_pp)
draw_gamma(axes[1],gamma_pp_trip_pp)
draw_gamma(axes[2],gamma_pp_sing_pp)
draw_gamma(axes[3],gamma_pp_trip_pp)
plt.tight_layout()
plt.show()


# fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(7,5))
# axes = axes.flatten()
#
# # draw_gamma(axes[0],gamma_dens_from_gchi*beta**2-u)
# draw_gamma(axes[0],gamma_dens_bse*beta**2-u)
# draw_gamma(axes[1],gamma_magn_from_gchi*beta**2+u)
# draw_gamma(axes[2],gchi_dens.mat[niw_core])
# draw_gamma(axes[3],gchi_magn.mat[niw_core])
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(7,5))
# axes = axes.flatten()
#
# # draw_gamma(axes[0],gamma_dens_from_gchi*beta**2-u)
# draw_gamma(axes[0],g2_dens.mat[niw_core])
# draw_gamma(axes[1],g2_magn.mat[niw_core])
# draw_gamma(axes[2],gchi_dens.mat[niw_core])
# draw_gamma(axes[3],gchi_magn.mat[niw_core])
# plt.tight_layout()
# plt.show()

# fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(7,5))
# axes = axes.flatten()
#
# draw_gamma(axes[0],F_sing_pp)
# draw_gamma(axes[1],F_trip_pp)
# draw_gamma(axes[2],F_sing_bse_pp)
# draw_gamma(axes[3],F_trip_bse_pp)
# plt.tight_layout()
# plt.show()
# #
# fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(7,5))
# axes = axes.flatten()
#
# draw_gamma(axes[0],F_dens_pp)
# draw_gamma(axes[1],F_magn_pp)
# draw_gamma(axes[2],F_dens_bse_pp)
# draw_gamma(axes[3],F_magn_bse_pp)
# plt.tight_layout()
# plt.show()

fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(7,5))
axes = axes.flatten()

draw_gamma(axes[0],gamma_dens.mat[niw_core])
draw_gamma(axes[1],gamma_magn.mat[niw_core])
draw_gamma(axes[2],gamma_dens_bse)
draw_gamma(axes[3],gamma_magn_bse)
draw_gamma(axes[4],gamma_dens_from_gchi)
draw_gamma(axes[5],gamma_magn_from_gchi)
plt.tight_layout()
plt.show()

# fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(7,7))
# axes = axes.flatten()
#
# draw_gamma(axes[0],F_dens[niw_core])
# draw_gamma(axes[1],F_magn[niw_core])
# draw_gamma(axes[2],F_dens_from_gchi)
# draw_gamma(axes[3],F_magn_from_gchi)
# draw_gamma(axes[4],F_dens[niw_core]-F_dens_from_gchi)
# draw_gamma(axes[5],F_magn[niw_core]-F_magn_from_gchi)
# plt.tight_layout()
# plt.show()







