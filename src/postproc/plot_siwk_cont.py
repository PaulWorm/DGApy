import sys, os
import numpy as np
from ruamel.yaml import YAML
import dga.config as config
import real_frequency_two_point as rtp
import dga.wannier as wannier
import dga.brillouin_zone as bz
import util
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors

# Load data:

base_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta17.5_n0.925' \
       '/LDGA_spch_Nk10000_Nq10000_wc40_vc40_vs400/'
base_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta17.5_n0.85' \
       '/LDGA_spch_Nk10000_Nq10000_wc40_vc40_vs10/'
# base_path = '/mnt/d/Research/HoleDopedCuprates/2DSquare_U8_tp-0.2_tpp0.1_beta22.5_n0.70' \
#        '/LDGA_spch_Nk10000_Nq10000_wc70_vc70_vs100/'
max_ent_dir = base_path + 'MaxEntSiwk/'
pdir = max_ent_dir
save_fig = True 

siwk_cont_file = 'swk_dga_cont_fbz.npy'

swk = np.load(max_ent_dir+siwk_cont_file,allow_pickle=True)
cut_off = 0.04
swk[np.abs(swk.imag) < cut_off] = swk[np.abs(swk.imag) < cut_off].real - 1j * cut_off
fname_conf = 'dga_config.yaml'
conf_file = YAML().load(open(base_path + fname_conf))

dga_config = config.DgaConfig(conf_file)
me_config = config.MaxEntConfig(1,12.5,conf_file)

# Build the Green's function on the Real-frequency axis:
mu0 = np.loadtxt(base_path+'mu.txt')[0]
dmft_input = np.load(base_path+'dmft_input.npy',allow_pickle=True).item()
n_target = dmft_input['n']
ek = dga_config.lattice.get_ek()
ek_shift = dga_config.lattice.k_grid.shift_mat_by_pi(ek)
mu = rtp.adjust_mu(mu0,n_target,swk,me_config.mesh,ek)
mu_lda = rtp.adjust_mu(mu0,n_target,0*swk,me_config.mesh,ek)


gwk = rtp.get_giwk(mu,swk,me_config.mesh,ek)
gwk0 = rtp.get_giwk(mu,swk*0,me_config.mesh,ek)

np.save(pdir+'gwk_cont.npy',gwk,allow_pickle=True)
np.savetxt(pdir+'mu_dga.txt',[mu,mu_lda],header='mu_dga mu_lda')
gwk_shift = dga_config.lattice.k_grid.shift_mat_by_pi(gwk)
swk_shift = dga_config.lattice.k_grid.shift_mat_by_pi(swk)
#%%
nk = dga_config.lattice.nk
nw0 = np.argmin(np.abs(me_config.mesh))
kx_shift = np.linspace(-np.pi,np.pi,nk[0], endpoint=True)
# Create k-path cuts:
bz_kpath1 = '0.5 0.5 0.|0.99 0.5 0.|0.99 0.99 0.|0.5 0.5 0.'
k_path1 = bz.KPath(nk=nk,path=bz_kpath1,path_deliminator='|',kx=kx_shift,ky=kx_shift)


bz_kpath2 = '0.81 0.0 0.0|0.81 1 0.0'
k_path2 = bz.KPath(nk=nk,path=bz_kpath2,path_deliminator='|',kx=kx_shift,ky=kx_shift)
# k_path2.k_val[1] = fold_kval(k_path2.k_val[1])

bz_kpath3 = '0.88 0.0 0.0|0.88 1 0.0'
k_path3 = bz.KPath(nk=nk,path=bz_kpath3,path_deliminator='|',kx=kx_shift,ky=kx_shift)
# k_path3.k_val[1] = fold_kval(k_path3.k_val[1])

bz_kpath4 = '0.95 0.0 0.0|0.95 1 0.0'
k_path4 = bz.KPath(nk=nk,path=bz_kpath4,path_deliminator='|',kx=kx_shift,ky=kx_shift)
# k_path4.k_val[1] = fold_kval(k_path4.k_val[1])


masks = bz.get_bz_masks(nk=nk[0])
qpd = rtp.get_dqp_bw((1/gwk_shift).real)

fs_loc = bz.find_zeros(gwk_shift[:,:,0,nw0])
#%%
k_points_fermi = np.array([kx_shift[fs_loc[:,0]],kx_shift[fs_loc[:,1]]]).T

# Find intersection of bz-kpaths with the "Fermi-surface"

fermi_crossing_1 = util.find_fermi_crossing(k_points_fermi,k_path1)
fermi_crossing_2 = util.find_fermi_crossing(k_points_fermi,k_path2)
fermi_crossing_3 = util.find_fermi_crossing(k_points_fermi,k_path3)
fermi_crossing_4 = util.find_fermi_crossing(k_points_fermi,k_path4)


# Plots:
x_ticks = ['$\Gamma$', 'X', 'M', '$\Gamma$']
cm2in = 1 / 2.54
bgc = (240 / 255, 240 / 255, 240 / 255)
# Figure:
font = {'size': 11}
matplotlib.rc('font', **font)
dpi = 500
alpha = 0.8
ms = 4
vmax = 1
lw = 1

# Figure 1:
fig = plt.figure(facecolor=bgc, figsize=(14.6 * cm2in, 7 * cm2in), dpi=dpi)
gs = fig.add_gridspec(2, 4,
                      left=0.08, right=0.98, bottom=0.1, top=0.95,
                      wspace=0.3)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[0, 3])
ax4 = fig.add_subplot(gs[1, :2])
ax5 = fig.add_subplot(gs[1, 2])
ax6 = fig.add_subplot(gs[1, 3])
axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax6]

im1 = axes[0].pcolormesh(kx_shift,kx_shift,-1/np.pi * gwk_shift[:,:,0,nw0].imag, cmap='magma')


divnorm = colors.TwoSlopeNorm(vmin=gwk_shift[:,:,0,nw0].real.min(), vcenter=0, vmax=gwk_shift[:,:,0,nw0].real.max())
im2 = axes[0].pcolormesh(kx_shift,kx_shift, np.ma.masked_array(gwk_shift[:,:,0,nw0].real, masks[0]), cmap='RdBu',norm=divnorm)

axes[0].pcolormesh(kx_shift,kx_shift, np.ma.masked_array(qpd[:,:,0,nw0].real, masks[2]), cmap='Greys')
# axes[0].plot(fs_loc[0].vertices[:,0],fs_loc[0].vertices[:,1], '-w', alpha=alpha)
ind = np.logical_and(k_points_fermi[:,0] > 0, k_points_fermi[:,1] > 0)
axes[0].plot(k_points_fermi[ind,0],k_points_fermi[ind,1], '-w', alpha=alpha)
axes[0].set_aspect('equal')
axes[0].plot(k_path1.k_val[0],k_path1.k_val[1],'-',color='tab:orange', alpha=alpha,lw=lw)
axes[0].plot(k_path2.k_val[0],k_path2.k_val[1],'-',color='cornflowerblue', alpha=alpha,lw=lw)
axes[0].plot(k_path3.k_val[0],k_path3.k_val[1],'-',color='firebrick', alpha=alpha,lw=lw)
axes[0].plot(k_path4.k_val[0],k_path4.k_val[1],'-',color='seagreen', alpha=alpha,lw=lw)

def plot_fermi_crossings(axes1,axes2,fermi_crossings,k_path,colors):
    for i,fc in enumerate(fermi_crossings):
        axes1.plot(k_path.k_val[0][fc],k_path.k_val[1][fc],'o',markeredgecolor='k',color=colors[i])
        y_lim = axes2.get_ylim()
        axes2.vlines(k_path.k_axis[fc],y_lim[0],y_lim[1],ls='--',lw=lw,color=colors[i],alpha=alpha)
        axes2.set_ylim(*y_lim)


# axes[0].set_xlim(0,np.pi)
# axes[0].set_ylim(0,np.pi)

def plot_kpath(ax,k_path):
    ax.pcolormesh(k_path.k_axis,me_config.mesh,-1/np.pi * gwk_shift[k_path.ikx,k_path.iky,0,:].imag.T, cmap = 'magma', vmax=vmax)
    ax.plot(k_path.k_axis,ek_shift[k_path.ikx,k_path.iky,0]-mu_lda,lw=lw,color='w',alpha=alpha,ms=0)
    # ax.plot(k_path.k_axis,np.min(np.abs((1/gwk)[k_path.ikx,k_path.iky,0].real),axis=1),lw=lw,color='w',alpha=alpha)
    # ax.plot(np.min(np.abs((1/gwk)[k_path.ikx,k_path.iky,0].real),axis=0),me_config.mesh,lw=lw,color='w',alpha=alpha)
    ax.set_ylim(-2,2)
    ax.get_yaxis().set_visible(False)
    ax.hlines(0,0,1,ls='--',colors='w',lw=lw)

plot_kpath(axes[1],k_path2)
plot_kpath(axes[2],k_path3)
plot_kpath(axes[3],k_path4)
plot_kpath(axes[4],k_path1)

plot_fermi_crossings(axes[0],axes[1],fermi_crossing_2,k_path2,colors=['cornflowerblue',])
plot_fermi_crossings(axes[0],axes[2],fermi_crossing_3,k_path3,colors=['firebrick',])
plot_fermi_crossings(axes[0],axes[3],fermi_crossing_4,k_path4,colors=['seagreen',])
plot_fermi_crossings(axes[0],axes[4],fermi_crossing_1[1:],k_path1,colors=['tab:orange','goldenrod'])

axes[5].set_xlim(-1,1)
axes[6].set_xlim(-1,1)
def plot_fc_2(axes1,axes2,k_path,fermi_crossing,colors):
    ind = np.logical_and(me_config.mesh > -1, me_config.mesh < 1)
    for i, fc in enumerate(fermi_crossing):
        axes1.plot(me_config.mesh[ind],-1/np.pi * gwk_shift[k_path.ikx[fc],k_path.iky[fc],0,ind].imag,'-',color=colors[i],alpha=alpha,ms=ms)
        axes2.plot(me_config.mesh[ind],swk_shift[k_path.ikx[fc],k_path.iky[fc],0,ind].imag,'-',color=colors[i],alpha=alpha,ms=ms)

plot_fc_2(axes[5],axes[6],k_path1,fermi_crossing_1[1:],colors=['tab:orange','goldenrod'])
plot_fc_2(axes[5],axes[6],k_path2,fermi_crossing_2,colors=['cornflowerblue',])
plot_fc_2(axes[5],axes[6],k_path3,fermi_crossing_3,colors=['firebrick',])
plot_fc_2(axes[5],axes[6],k_path4,fermi_crossing_4,colors=['seagreen',])


y_lim = axes[5].get_ylim()
axes[5].set_ylim(0,y_lim[1])

axes[1].get_xaxis().set_visible(False)
axes[2].get_xaxis().set_visible(False)
axes[3].get_xaxis().set_visible(False)

axes[4].get_yaxis().set_visible(True)
axes[4].set_xticks(k_path1.k_axis[k_path1.cind],x_ticks)

axes[4].set_ylabel('Energy [t]')
axes[0].set_ylabel('$k_x$')
axes[0].set_yticks([-np.pi,0,np.pi],['$\pi$','$0$','$\pi$'])
axes[0].set_xticks([-np.pi,0,np.pi],['$\pi$','$0$','$\pi$'])
axes[5].set_yticks([0,0.5])
axes[6].set_yticks([0,-1,-2])

box1 = axes[1].get_position()
cbar_ax = fig.add_axes([box1.x1+0.005, box1.y0, 0.015, box1.y1-box1.y0])
cbar1 = plt.colorbar(im1, cax=cbar_ax)
cbar1.ax.tick_params(size=0)
cbar1.set_ticks([])
fig.text(box1.x1+0.022,  box1.y0+0.012, 'min', rotation='vertical')
fig.text(box1.x1+0.022,  box1.y1-0.085, 'max', rotation='vertical')

box0 = axes[0].get_position()
cbar_ax = fig.add_axes([box0.x1+0.005, box0.y0, 0.015, box0.y1-box0.y0])
cbar2 = plt.colorbar(im2, cax=cbar_ax)
cbar2.ax.tick_params(size=0)
cbar2.set_ticks([])

fig.text(box0.x1+0.022, box0.y0+0.012, 'min', rotation='vertical')
fig.text(box0.x1+0.022, box0.y0+0.19, '0', rotation='vertical')
fig.text(box0.x1+0.022, box0.y1-0.085, 'max', rotation='vertical')

# axes[1].text(0.45,-2.45,'$k_x$')
# axes[2].text(0.45,-2.45,'$k_x$')
# axes[3].text(0.45,-2.45,'$k_x$')

axes[0].text(-0.25,-0.7,'$\Gamma$',color='w')
axes[0].text(np.pi-0.7,-0.7,'$X$',color='w')
axes[0].text(np.pi-0.7,np.pi+0.1,'$M$',color='k')

def draw_axes(axes):
    axes.text(0.3,-1.85,'$k_x$',color='w')
    axes.text(0.01,-0.8,'$\omega$',color='w')
    axes.arrow(0.05,-1.8,0.16,0.0,head_width = 0.1,head_length = 0.05,color='w')
    axes.arrow(0.05,-1.8,0.00,0.16*4,head_width = 0.1/4,head_length = 0.05*4,color='w')
    # axes.annotate("", xy=(0.5, -1.8), xytext=(0, 0),
    #         arrowprops=dict(arrowstyle="->"),color='white')
draw_axes(axes[1])
draw_axes(axes[2])
draw_axes(axes[3])
# axes[3].text(-0.2,-0.8,'Energy [t]',rotation='vertical')

def shift_plot(ax,shift):
    box = ax.get_position()
    box.x0 = box.x0 + shift[0]
    box.y0 = box.y0 + shift[1]
    box.x1 = box.x1 + shift[2]
    box.y1 = box.y1 + shift[3]
    ax.set_position(box)

# shift_plot(axes[0],(-0.05,0,-0.05,0))

# y_lim = axes[6].get_ylim()
# axes[6].set_ylim(0,y_lim[1])
# axes[5].hlines(0,0,1,ls='--',colors='w',lw=lw)

box = axes[5].get_position()
fig.text(box.x0+0.005,box.y1-0.05,r'$\bf{(f)}$',fontweight='bold')
fig.text(box.x0+0.005,box.y1+0.02,r'$A(\omega)$',fontweight='bold')
box = axes[6].get_position()
fig.text(box.x0+0.005,box.y1-0.05,r'$\bf{(g)}$',fontweight='bold')
fig.text(box.x0+0.005,box.y1+0.02,r'$\Im \Sigma(\omega)$',fontweight='bold')

axes[0].text(-np.pi+0.2,np.pi-0.8,r'$\bf{(a)}$')

axes[1].text(0.02,1.5,r'$\bf{(b)}$',color='w')
axes[2].text(0.02,1.5,r'$\bf{(c)}$',color='w')
axes[3].text(0.02,1.5,r'$\bf{(d)}$',color='w')

axes[4].text(0.005,1.5,r'$\bf{(e)}$',color='w')

box = axes[5].get_position()
fig.text((box.x0+box.x1)*0.5-0.01,(box.y0)-0.09,'$\omega$')

box = axes[6].get_position()
fig.text((box.x0+box.x1)*0.5-0.01,(box.y0)-0.09,'$\omega$')


if(save_fig):
    plt.savefig(pdir + 'gwk_cont_analysis.png')
    # plt.savefig('./Plots/' + figure_name_base.format(bs,ns) + '_analysis.pdf')
plt.show()
print('Finished!')


