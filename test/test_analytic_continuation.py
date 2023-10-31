import numpy as np
import matplotlib.pyplot as plt
import ana_cont as cont
import dga.analytic_continuation as dga_cont

def gauss_peak(maxpos,width,weight,wgrid):
    a =weight/(np.sqrt(2.*np.pi)*width)*np.exp(-0.5*(wgrid-maxpos)**2/width**2)
    return a

def noise(sigma,iwgrid):
    return np.random.randn(iwgrid.shape[0])*sigma

def get_test_giw(nw=1001,niw=100,beta=10.,wmax=5, sigma=0.0001):
    wgrid = np.linspace(-wmax, wmax, nw)
    iwgrid = np.pi / beta * (2 * np.arange(niw) + 1)
    giw = np.zeros_like(iwgrid, dtype=complex)

    aw = gauss_peak(-1., 0.2, 1., wgrid) + gauss_peak(1., 0.2, 1., wgrid)
    norm = np.trapz(aw, wgrid)
    aw = aw / norm

    for i, iw in enumerate(iwgrid):
        giw[i] = -np.trapz(aw * wgrid / (iw ** 2 + wgrid ** 2), wgrid) - 1j * np.trapz(aw * iw / (iw ** 2 + wgrid ** 2), wgrid)

    giw += noise(sigma, iwgrid)
    err = np.ones_like(iwgrid)*sigma
    return giw, aw, err, wgrid, iwgrid


# create a grid that is denser in the center and more sparse outside.
wmax=5.
nw=250
wgrid = dga_cont.tan_w_mesh(-wmax,wmax,nw)

model=np.exp(-(wgrid)**2)
model=model/np.trapz(model,wgrid) # normalize the model

giw, aw, err, w, iwgrid = get_test_giw()

probl=cont.AnalyticContinuationProblem(im_axis=iwgrid,re_axis=wgrid,im_data=giw,kernel_mode='freq_fermionic')

sol=probl.solve(method='maxent_svd',model=model,stdev=err,alpha_determination='chi2kink')

f1=plt.figure()
plt.plot(wgrid,sol[0].A_opt)
plt.plot(w,aw)
f1.show()


#%%
plt.plot(giw.imag)
plt.show()
#%%
import dga.matsubara_frequencies as mf
me_controller = dga_cont.MaxEnt(50,10.,'freq_fermionic', bw=0,wmin=-5,wmax=5)

giw_cont = me_controller.analytic_continuation([mf.fermionic_full_nu_range(giw)])

f1=plt.figure()
plt.plot(me_controller.w,-1/np.pi * giw_cont[0].imag)
plt.plot(w,aw)
f1.show()
