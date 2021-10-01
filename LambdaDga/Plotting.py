# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import FourPoint as fp


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------

def plot_susc(susc=None):
    plt.plot(susc.iw, susc.mat.real, 'o')
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
               extent=[A.iw[0], A.iw[-1], -niv_cut, niv_cut])
    plt.colorbar()
    plt.title(r'$\Re$' + name + '-' + A.channel)
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$\omega$')
    plt.show()

    plt.imshow(A.mat.imag[:, tp.niv - niv_cut:tp.niv + niv_cut], cmap='RdBu',
               extent=[A.iw[0], A.iw[-1], -niv_cut, niv_cut])
    plt.colorbar()
    plt.title(r'$\Im$' +name + '-' + A.channel)
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$\omega$')
    plt.show()
