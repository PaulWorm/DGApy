import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd

import BrillouinZone as bz
import Hk as hamk
import RealFrequencyTools as rft

eV2K = 11600
t_cuprates = 0.45


def fermi_func_der(w, beta):
    ind = w*beta < 20
    fder = np.zeros((np.size(w),))
    fder[ind] = - beta * np.exp(w[ind] * beta) / (np.exp(w[ind] * beta) + 1) ** 2
    return fder


def get_dos_w0(gwk):
    nw0 = np.shape(gwk)[-1] // 2
    return np.mean(-1 / np.pi * gwk[..., nw0].imag)


def get_dos_kbT(gwk, w, beta):
    w_eV = w * t_cuprates
    nb = np.shape(gwk)[1]
    T = 1 / beta
    ind = np.where(np.abs(w_eV) < T)[0]
    av_dos = np.mean(-1 / np.pi * gwk[..., ind].imag)
    return av_dos


def get_dos_fermi(gwk, w, beta):
    aw = np.mean(-1 / np.pi * gwk[:, :, :].imag, axis=(0, 1, 2))
    nf_der = fermi_func_der(w, beta)
    return np.trapz(-aw * nf_der, w)


def get_T(beta):
    return 1 / float(beta) * t_cuprates * eV2K


def load_dos_fermi(path, beta):
    gwk = np.load(path + 'gwk.npy')[:, :, 0, :]  # remove kz-axis
    w = np.load(path + 'w.npy')  # remove kz-axis
    dos_w0 = get_dos_w0(gwk)
    T = 1 / beta
    av_dos = get_dos_kbT(gwk, w, T)
    return dos_w0, av_dos


def load_swk():
    swk = np.load(base + dir_name + cont_dir + f'swk_dga_sigma_cont_fbz_bw{bw}.npy', allow_pickle=True)
    swk[swk.imag < sigma_background] = swk[swk.imag < sigma_background] - sigma_background * 1j
    w = np.load(base + dir_name + cont_dir + f'me_config_dga_sigma_bw{bw}.npy'.format(bw), allow_pickle=True).item().mesh
    return swk, w


ncore = 120
nk = 140
nurange = 800
bs = '90'
beta = float(bs)
ns = '0.90'

dir_name = f'LambdaDga_lc_sp_Nk{nk ** 2}_Nq{nk ** 2}_core{ncore}_invbse{ncore}_vurange{nurange}_wurange{ncore}/'
base = f'D:/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta{bs}_n{ns}/'
cont_dir = 'AnaContSigmaFBZ/'
bw = '0.1'
sigma_background = 0.02
swk, w = load_swk()
nw0 = np.argmin(np.abs(w))

config = np.load(base.format(bs, ns) + dir_name + 'config.npy', allow_pickle=True).item()

k_grid = bz.KGrid(nk=config.box.nk)
ek = hamk.ek_3d(k_grid.grid, config.sys.hr)
# nw0 = me_config.mesh.size // 2

mu = rft.adjust_mu(mu0=1.91945865, n_target=float(ns), swk=swk, w=w, ek=ek)
gwk = rft.get_gwk(mu, swk, w, ek)

# ----------------------------------------------------------------------
# Get Density of states close to the Fermi-level:
dos_w0 = get_dos_w0(gwk)
av_dos = get_dos_kbT(gwk, w, beta)
dos_fermi = get_dos_fermi(gwk, w, beta)

print('----------------')
print(r'$DOS(\omega=0) =$'+ f'{dos_w0}')
print(r'$DOS(kbT) =$'+ f'{av_dos}')
print(r'$DOS(Fermi) =$'+ f'{dos_fermi}')
print('----------------')
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Get Self-energy and effective mass at specific locations along the BZ:
gwk_cut = gwk[nk // 2:, nk // 2:, 0, nw0]
indx, indy = rft.get_zero_contour((1 / gwk_cut).real)
anode = [indx[0], indy[0]]
node = [indx[len(indx) // 2], indy[len(indy) // 2]]

print('----------------')
print(f'K-point Node: ({k_grid.kx[node[0]]}, {k_grid.ky[node[1]]})')
print(f'K-point Antinode: ({k_grid.kx[anode[0]]}, {k_grid.ky[anode[1]]})')
print('----------------')

sigma_node = swk[node[0], node[1], 0, nw0]
sigma_anode = swk[anode[0], anode[1], 0, nw0]
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Create Pandas dataframe and safe it:

names = ['beta', 'n', 'nkx', 'nky', 'ncore', 'ninvbse', 'nurange', 'bw', 'wmax', 'nwr', 'dos_w0', 'dos_av', 'dos_fermi', 's_node', 's_anode']
values = [beta, float(ns), nk, nk, ncore, ncore, nurange, float(bw), np.max(w), np.size(w), dos_w0, av_dos, dos_fermi, sigma_node, sigma_anode]

ddict = {'beta': beta,
         'n': float(ns),
         'nkx': nk,
         'nky': nk,
         'ncore': ncore,
         'ninvbse': ncore,
         'nurange': nurange,
         'bw': float(bw),
         'wmax': np.max(w),
         'nwr': np.size(w),
         'dos_w0': dos_w0,
         'dos_av': av_dos,
         'dos_fermi': dos_fermi,
         's_node': sigma_node,
         's_anode': sigma_anode, }
data = pd.DataFrame(ddict, index=[0])

data.to_csv(base+dir_name+cont_dir+'ContinuationData.csv')
np.savetxt(base + dir_name + cont_dir + 'dos_w0.txt', np.atleast_1d(dos_w0))
np.savetxt(base + dir_name + cont_dir + 'dos_av.txt', np.atleast_1d(av_dos))
np.savetxt(base + dir_name + cont_dir + 'dos_fermi.txt', np.atleast_1d(dos_fermi))
np.savetxt(base + dir_name + cont_dir + 's_node.txt', np.atleast_1d(sigma_node))
np.savetxt(base + dir_name + cont_dir + 's_anode.txt', np.atleast_1d(sigma_anode))
np.savetxt(base + dir_name + cont_dir + 'Analysis.txt',np.array([[dos_w0,av_dos,dos_fermi,sigma_node.real,sigma_node.imag,sigma_anode.real,sigma_anode.imag],]),fmt='%.9f')
# np.savetxt(base + dir_name + cont_dir + 'Analysis.txt',np.array([[dos_w0,av_dos,dos_fermi,sigma_node.real,sigma_node.imag,sigma_anode.real,sigma_anode.imag],]))
