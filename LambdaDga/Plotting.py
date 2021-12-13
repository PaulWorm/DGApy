# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import itertools
import matplotlib.pyplot as plt
import FourPoint as fp

# -------------------------------------- DEFINE MODULE WIDE VARIABLES --------------------------------------------------

# __markers__ = itertools.cycle(('o','s','v','8','v','^','<','>','p','*','h','H','+','x','D','d','1','2','3','4'))
__markers__ = ('o', 's', 'v', '8', 'v', '^', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '1', '2', '3', '4')


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def plot_susc(susc=None):
    plt.plot(susc.iw_core, susc.mat.real, 'o')
    plt.xlim([-2, susc.beta])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'\chi')
    plt.show()


def plot_fp(fp: fp.LocalFourPoint = None, w_plot=0, name='', niv_plot=-1):
    if (niv_plot == -1):
        niv_plot = fp.niv
    A = fp
    plt.imshow(
        A.mat[A.iw == w_plot, A.niv - niv_plot:A.niv + niv_plot, A.niv - niv_plot:A.niv + niv_plot].squeeze().real,
        cmap='RdBu', extent=[-niv_plot, niv_plot, -niv_plot, niv_plot])
    plt.colorbar()
    plt.title(name + '-' + A.channel + '(' + r'$ \omega=$' + '{})'.format(w_plot))
    plt.xlabel(r'$\nu\prime$')
    plt.ylabel(r'$\nu$')
    plt.show()


def plot_tp(tp: fp.LocalThreePoint = None, niv_cut=-1, name=''):
    if (niv_cut == -1):
        niv_cut = tp.niv
    A = tp
    plt.imshow(A.mat.real[:, tp.niv - niv_cut:tp.niv + niv_cut], cmap='RdBu',
               extent=[-niv_cut, niv_cut, A.iw[0], A.iw[-1]])
    plt.colorbar()
    plt.title(r'$\Re$' + name + '-' + A.channel)
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$\omega$')
    plt.show()

    plt.imshow(A.mat.imag[:, tp.niv - niv_cut:tp.niv + niv_cut], cmap='RdBu',
               extent=[A.iw[0], A.iw[-1], -niv_cut, niv_cut])
    plt.colorbar()
    plt.title(r'$\Im$' + name + '-' + A.channel)
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$\omega$')
    plt.show()


def plot_chiw(wn_list=None, chiw_list=None, labels_list=None, channel=None, plot_dir=None, niw_plot=20):
    markers = __markers__
    np = len(wn_list)
    assert np < len(markers), 'More plot-lines requires, than markers available.'

    size = 2 * np + 1

    for i in range(len(wn_list)):
        plt.plot(wn_list[i], chiw_list[i].real, markers[i], ms=size - 2 * i, label=labels_list[i])
    plt.legend()
    plt.xlim(-2,niw_plot)
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\chi_{}$'.format(channel))
    if (plot_dir is not None):
        plt.savefig(plot_dir + 'chiw_{}.png'.format(channel))
    try:
        plt.show()
    except:
        plt.close()

def plot_siw(vn_list=None, siw_list=None, labels_list=None, plot_dir=None, niv_plot=200):
    markers = __markers__
    np = len(vn_list)
    assert np < len(markers), 'More plots-lines requires, than markers avaiable.'

    size = 2 * np + 1

    plt.subplot(211)
    for i in range(len(vn_list)):
        plt.plot(vn_list[i], siw_list[i].real, markers[i], ms=size - 2 * i, label=labels_list[i])
    plt.legend()
    plt.xlim([0, niv_plot])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Re \Sigma$')
    plt.subplot(212)
    for i in range(len(vn_list)):
        plt.plot(vn_list[i], siw_list[i].imag, markers[i], ms=size - 2 * i, label=labels_list[i])
    plt.xlim([0, niv_plot])
    plt.legend()
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Im \Sigma$')
    if (plot_dir is not None):
        plt.savefig(plot_dir + 'siw_check.png')
    try:
        plt.show()
    except:
        plt.close()


def plot_siwk_fs(siwk=None, plot_dir=None, kgrid=None, do_shift=False, kz=0,niv_plot=None):
    if(niv_plot==None):
        niv_plot = np.shape(siwk)[-1] // 2

    siwk_plot = np.copy(siwk)
    kx = kgrid._grid['kx']
    ky = kgrid._grid['ky']
    if(do_shift):
        siwk_plot = np.roll(siwk_plot,kgrid.nk[0]//2,0)
        siwk_plot = np.roll(siwk_plot,kgrid.nk[1]//2,1)
        kx = kx - np.pi
        ky = ky - np.pi



    fig = plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.imshow(siwk_plot[:,:,kz,niv_plot].real,cmap='RdBu', extent=[kx[0],kx[-1],ky[0],ky[-1]], origin='lower')
    plt.colorbar()
    plt.xlabel(r'$k_x$')
    plt.ylabel(r'$k_y$')
    plt.title(r'$\Re \Sigma$')
    plt.subplot(122)
    plt.imshow(siwk_plot[:,:,kz,niv_plot].imag,cmap='RdBu', extent=[kx[0],kx[-1],ky[0],ky[-1]], origin='lower')
    plt.colorbar()
    plt.xlabel(r'$k_x$')
    plt.ylabel(r'$k_y$')
    plt.title(r'$\Im \Sigma$')

    plt.tight_layout()

    if (plot_dir is not None):
        plt.savefig(plot_dir + 'siwk_kz{}_Niv{}.png'.format(kz,niv_plot))
    try:
        plt.show()
    except:
        plt.close()