import numpy as np
import scipy.optimize as opt

def get_dqp_bw(data):
    qdp = np.ones_like(data)
    qdp[data < 0] = 0
    return qdp

def get_giwk(mu, siwk, w, ek):
    return 1 / (w[None,None,None,:] + mu - ek[...,None] - siwk)

def get_filling(w,giwk):
    ind = w <= 0
    return np.trapz(np.mean(-1/np.pi * giwk.imag,axis=(0,1,2))[ind],w[ind])

def opt_func(mu,n,siwk,w,ek):
    giwk = get_giwk(mu,siwk,w,ek)
    fill_current = get_filling(w,giwk)
    print(f'{mu=}')
    return (fill_current*2 - n)**2

def adjust_mu(mu0,n_target,siwk,w,ek):
    fit = opt.minimize(opt_func,x0=mu0,args=(n_target,siwk,w,ek))
    return fit.x