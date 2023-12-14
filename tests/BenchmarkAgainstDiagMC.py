import numpy as np
import matplotlib.pyplot as plt
import dga.matsubara_frequencies as mf

# Load DiagMC data:

T = ['0.33','0.1','0.065','0.063']
fname = './2DSquare_U2_tp-0.0_tpp0.0_beta15_mu1/DiagMcBm/{}_T_{}.csv'
diag_mc_node = []
diag_mc_anode = []
for iT in T:
    data = np.genfromtxt(fname.format('N',iT), delimiter=',')
    diag_mc_node.append(data)
    data = np.genfromtxt(fname.format('AN',iT), delimiter=',')
    diag_mc_anode.append(data)

# Load the DGA data:

# beta = ['3','10','15','16']
beta = ['3','10','15','16','20']

niv_plot = 30
fname = './2DSquare_U2_tp-0.0_tpp0.0_beta{}_mu1/LambdaDga_lc_spch_Nk19600_Nq19600_wcore30_vcore30_vshell200/siwk_dga.npy'
dga_node = []
dga_anode = []

for ib in beta:
    data = np.load(fname.format(ib),allow_pickle=True)

    nk = np.shape(data)[:-1]
    nkx = nk[0]
    niv = np.shape(data)[-1]//2

    iv = mf.v_plus(float(ib),niv_plot)
    dga_node.append([iv, data[nkx//4,nkx//4,0,niv:niv+niv_plot].imag])
    dga_anode.append([iv, data[nkx//2,0,0,niv:niv+niv_plot].imag])

    # plt.imshow(data[:, :, 0, niv].imag,cmap='RdBu')
    # plt.scatter([nkx//4,nkx//2],[nkx//4,0],c='k',marker='o')
    # plt.show()

#%%

fig, axes = plt.subplots(2,2,figsize=(12,10))
axes = axes.flatten()

line_colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, len(T)))
for i,iT in enumerate(T):
    axes[0].plot(diag_mc_node[i][:,0],diag_mc_node[i][:,1],'-o',color=line_colors[i],markeredgecolor='k',alpha=0.8,
                 label=f'DiagMC: T = {iT}')
    axes[1].plot(diag_mc_anode[i][:,0],diag_mc_anode[i][:,1],'-o',color=line_colors[i],markeredgecolor='k',alpha=0.8,
                 label=f'DiagMC: T = {iT}')

line_colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, len(beta)))
for i, ib in enumerate(beta):
    axes[2].plot(dga_node[i][0],dga_node[i][1],'-h',color=line_colors[i],markeredgecolor='k',alpha=0.8,
                 label=f'DGA: T = {np.round(1/float(ib),3)}')
    axes[3].plot(dga_anode[i][0],dga_anode[i][1],'-h',color=line_colors[i],markeredgecolor='k',alpha=0.8,
                 label=f'DGA: T = {np.round(1/float(ib),3)}')

for ax in axes:
    ax.grid()
    ax.set_xlabel(r'$\nu_n$')
    ax.legend()
    ax.set_xlim(0,6)
    ax.set_ylim(-0.13,-0.04)

axes[0].set_ylabel(r'$\Im \Sigma_{Node}$')
axes[1].set_ylabel(r'$\Im \Sigma_{Antinode}$')

axes[0].set_title('Node')
axes[1].set_title('Antinode')
plt.tight_layout()
plt.show()

