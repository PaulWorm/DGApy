import numpy as np
import matplotlib.pyplot as plt
import Hr as hr_mod
import Hk as hk
import TwoPoint as twop
import BrillouinZone as bz

def fermi_liquid_sigma(w=None,T=None, gamma=None, u=1.0):
    cutoff = u/2 * 1./(np.sqrt(gamma))
    beta = 3./4. * np.pi/2. * cutoff * u**2/4
    alpha =3./4. * np.pi/(2.*cutoff) * gamma
    ind = np.abs(w) < cutoff
    ind2 = np.abs(w) > cutoff
    sigma = np.zeros(np.size(w), dtype = complex)
    sigma[ind] = -1j * w[ind]**2*alpha - gamma * w[ind] - 1j * T**2*np.pi**2
    sigma[ind2] = -1j * beta/(w[ind2]**2) - u**2./(4*w[ind2])
    return sigma

def get_g_sigma_loc(v=None,ek=None,mu=None,sigma=None):
    g = 1./(v[None,None,None,:] - ek[:,:,:,None] + mu - sigma[None,None,None,:])
    return g



if __name__ == '__main__':

    u = 2.0
    Z = 0.6
    gamma = Z**(-1) - 1
    beta = 5
    T = 1./10

    nw = 1001
    w0_ind = nw//2
    w = np.linspace(-10,10,nw)
    sigma = fermi_liquid_sigma(w=w,T=T,gamma=gamma,u=u)

    plt.plot(w,sigma.imag)
    plt.show()

    nkf = 120
    nk = (nkf,nkf,1)
    q_grid = bz.NamedKGrid(nk=nk, name='k')

    t = 1.0
    tp = -0.2
    tpp = 0.1
    n = 0.90
    mu0 = -1.0
    hr = hr_mod.one_band_2d_t_tp_tpp(t=t, tp=tp, tpp=tpp)
    ek = hk.ek_3d(kgrid=q_grid.get_grid_as_tuple(),hr=hr)

    gk = get_g_sigma_loc(v=w,ek=ek,mu=mu0,sigma=sigma)
    gk = np.roll(gk, q_grid.nk[0] // 2, 0)
    gk = np.roll(gk, q_grid.nk[1] // 2, 1)
    gk_loc = np.mean(gk,axis=(0,1,2))
    awk = -1/np.pi * gk[:,:,0,:].imag
    aw = -1/np.pi *gk_loc.imag

    grad_fs = np.gradient(-1/np.pi * gk[:,:,0,w0_ind].imag)
    curv_x = np.gradient(grad_fs[0])[1]
    curv_y = np.gradient(grad_fs[1])[0]

    ind = w <= 0
    occ = np.trapz(aw[ind],w[ind])
    print(f'{occ=}')

    plt.plot(w,aw)
    plt.show()

    plt.imshow(-1/np.pi * gk[:,:,0,w0_ind].imag, cmap='RdBu')
    plt.colorbar()
    plt.show()

    plt.imshow(grad_fs[0], cmap='RdBu')
    plt.show()

#%%
    plt.imshow(curv_x, cmap='RdBu')
    plt.colorbar()
    arc_x = nkf//4 + 6
    arc_y = nkf//4 + 7
    plt.plot(arc_x * np.ones(nkf),np.linspace(0,nkf,nkf), 'k')
    plt.plot(np.linspace(0,nkf,nkf),arc_y * np.ones(nkf), 'k')
    plt.show()

    plt.plot(w,awk[arc_x,arc_y,:])
    plt.show()

#%%
    plt.imshow(curv_x, cmap='RdBu')
    plt.colorbar()
    pg_x = 0
    pg_y = nkf//2 - 7
    plt.plot(pg_x * np.ones(nkf),np.linspace(0,nkf,nkf), 'k')
    plt.plot(np.linspace(0,nkf,nkf),pg_y * np.ones(nkf), 'k')
    plt.show()

    plt.plot(w,awk[pg_x,pg_y,:])
    plt.show()




