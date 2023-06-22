import numpy as np
import matplotlib.pyplot as plt
import MatsubaraFrequencies as mf
import BrillouinZone as bz
import TwoPoint_old as twop

def load_dga(base, fname):
    config = np.load(base + fname + 'config.npy', allow_pickle=True).item()
    sigma = np.load(base + fname + 'sigma.npy', allow_pickle=True)
    beta = config.sys.beta
    mu_dmft = config.sys.mu_dmft
    hr = config.sys.hr
    n = config.sys.n
    kgrid = bz.KGrid(nk=config.box.nk)
    gk_gen = twop.GreensFunctionGenerator(beta=beta, kgrid=kgrid, hr=hr, sigma=sigma)
    mu = gk_gen.adjust_mu(n=n, mu0=mu_dmft)
    print('----------------------------------')
    print(f'Sanity check filling:')
    print(f'n-ist = {gk_gen.get_fill(mu=mu)}')
    print(f'n-soll = {n}')
    print('----------------------------------')

    #shift the Green's function:
    gk_obj = gk_gen.generate_gk(mu=mu)
    sigma = kgrid.shift_mat_by_q(sigma, q=(np.pi, np.pi, 0))
    gk = kgrid.shift_mat_by_q(gk_obj.gk, q=(np.pi, np.pi, 0))
    ddict = {
        'niv': gk_obj.niv,
        'sigma': sigma,
        'gk': gk,
        'mu':mu,
        'kgrid': kgrid,
        'kx': np.linspace(-np.pi, np.pi, kgrid.nk[0], endpoint=False),
        'masks': bz.get_bz_masks(kgrid.nk[0])
    }
    return ddict

def get_fermi_surface_location(kx, gk):
    nk = np.size(kx)
    cs1 = plt.contour(kx[nk // 2:], kx[nk // 2:], (1 / gk[nk // 2:, nk // 2:]).real, cmap='RdBu', levels=[0, ])
    paths_1 = cs1.collections[0].get_paths()
    plt.close()
    return paths_1

def get_siwk_arc(fs_path, siwk, kx):
    siwk_arc = np.zeros((np.size(fs_path[0].vertices[:, 0]), np.shape(siwk)[-1]), dtype=complex)
    for i in range(np.size(fs_path[0].vertices[:, 0])):
        x = np.argmin(np.abs(fs_path[0].vertices[i, 0] - kx))
        y = np.argmin(np.abs(fs_path[0].vertices[i, 1] - kx))
        siwk_arc[i, :] = siwk[x, y, 0, :]
    return siwk_arc

def check_pg_criterion(siwk_arc):
    is_arc = np.ones((siwk_arc[:,0].size,), dtype=bool)
    niv = np.shape(siwk_arc)[-1]//2
    nk_arc = np.shape(siwk_arc)[0]
    for i in range(nk_arc):
        if(siwk_arc[i,niv].imag < siwk_arc[i,niv+1].imag):
            is_arc[i] = False
    return is_arc


ncore = 100
nk = 140
nurange = 600
bs = '60'
beta = float(bs)
ns = '0.875'
tp = '-0.2'
tpp = '0.1'

# Input directories:
dir_name = f'LambdaDga_lc_sp_Nk{nk ** 2}_Nq{nk ** 2}_core{ncore}_invbse{ncore}_vurange{nurange}_wurange{ncore}/'
base = f'D:/Research/HoleDopedCuprates/2DSquare_U8_tp{tp}_tpp{tpp}_beta{bs}_n{ns}/'

# Output Directories:
dir_name_out = 'G:/My Drive/Research/HubbardModelDgaAnalysis/ArcLength/'
fname_out = f'2DSquare_ArcLength_tp{tp}_tpp{tpp}_n{ns}.txt'

data = load_dga(base, dir_name)
fsk = get_fermi_surface_location(data['kx'], data['gk'][:, :, 0, data['niv']])
siwk_arc = get_siwk_arc(fsk, data['sigma'], data['kx'])
vn = mf.vn(siwk_arc.shape[-1] // 2)
arc = check_pg_criterion(siwk_arc)
arc_length = np.mean(arc)

# Write data to files:
np.savetxt(base + dir_name + 'mu_dga.txt', np.atleast_1d(data['mu']) , fmt='%.9f')
np.savetxt(base + dir_name + 'arc_length.txt', np.atleast_1d(arc_length), fmt='%.9f')

with open(dir_name_out + fname_out,'a') as f:
    # np.savetxt(f, [float(bs), arc_length])
    np.savetxt(f, np.array([[float(bs),arc_length],]), fmt='%.9f')

print('Finished!')