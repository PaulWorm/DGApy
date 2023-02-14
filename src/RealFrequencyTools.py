import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def get_gwk(mu, swk, w, ek):
    return 1 / (w[None,None,None,:] + mu - ek[...,None] - swk)

def get_filling(w,gwk):
    ind = w <= 0
    return np.trapz(np.mean(-1/np.pi * gwk.imag,axis=(0,1,2))[ind],w[ind])

def opt_func(mu,n,swk,w,ek):
    gwk = get_gwk(mu,swk,w,ek)
    fill_current = get_filling(w,gwk)
    print(f'{mu=}')
    return (fill_current*2 - n)**2

def adjust_mu(mu0,n_target,swk,w,ek):
    fit = opt.minimize(opt_func,x0=mu0,args=(n_target,swk,w,ek))
    print('----------------------')
    print(f'Target filling: {n_target}')
    print(f'Adjusted filling: {2*get_filling(w,get_gwk(fit.x,swk,w,ek))}')
    print('----------------------')
    return fit.x

def get_zero_contour(data):
    indx = np.arange(0,np.shape(data)[1])
    indy = np.arange(0,np.shape(data)[0])
    fig1 = plt.figure()
    cs1 = plt.contour(indx,indy, data, cmap='RdBu', levels=[0, ])
    path = cs1.collections[0].get_paths()
    plt.close(fig1)
    for i in range(len(path)):
        if (i == 0):
            indices_x = np.array(np.round(path[i].vertices[:, 0], 0).astype(int))
            indices_y = np.array(np.round(path[i].vertices[:, 1], 0).astype(int))
        else:
            indices_x = np.concatenate((indices_x,np.array(np.round(path[i].vertices[:, 0], 0).astype(int))),axis=0)
            indices_y = np.concatenate((indices_y,np.array(np.round(path[i].vertices[:, 1], 0).astype(int))),axis=0)
    return indices_x,indices_y

def extract_Z_realf(sigma,w,wmin=-0.1,wmax=0.1,order=2):
    ind = np.logical_and(w < wmax,w > wmin)
    poly_fit = np.polyfit(w[ind],sigma[ind].real,order)
    poly_der = np.polyder(poly_fit)
    print(poly_der)
    return (1 - poly_der[0])**(-1)

