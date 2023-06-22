import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import FourPoint as fp
import TwoPoint_old as tp
import MatsubaraFrequencies as mf
import w2dyn_aux_dga
import jax.scipy as jcp

path = './2DSquare_U8_tp-0.25_tpp0.12_beta25_n0.85/'

giw_file = w2dyn_aux.w2dyn_file(fname=path + '1p-data.hdf5')
beta = giw_file.get_beta()
u = giw_file.get_udd()
totdens = giw_file.get_totdens()
mu = giw_file.get_mu()
giw = giw_file.get_giw()[0, 0, :]
giw_obj = tp.LocalGreensFunction(mat=giw,beta=beta,hf_denom=mu-u*totdens/2)


def chi0_asympt_iwn(niv,iwn):
    iwnd2 = iwn // 2
    iwnd2mod2 = iwn // 2 + iwn % 2
    iv = mf.iv(beta,niv,shift=iwnd2)
    ivmw = mf.iv(beta,niv,shift=-iwnd2mod2)
    g_tail = 1/(iv + mu - u*totdens/2)
    gshift_tail = 1/(ivmw + mu - u*totdens/2)
    chi0_tail = - 1/beta * np.sum(g_tail*gshift_tail)
    return chi0_tail

def chi0_asympt(niv_asympt,niv,wn):
    full = jnp.array([chi0_asympt_iwn(niv_asympt, iwn) for iwn in wn])
    inner = jnp.array([chi0_asympt_iwn(niv, iwn) for iwn in wn])
    return full - inner

# def chi_asympt_2(wn):
#     a = mu - u*totdens/2
#     inner = jnp.array([chi0_asympt_iwn(niv, iwn) for iwn in wn])
#     for iwn in wn:
#         if(iwn == 0):
#             0.25 * beta * np.sech(a*beta/2)**2
#         else:
#             beta/(4*iwn*np.pi) * np.sech(a*beta/2)*np.sech(a*beta/2)




niw = 100
niv = niw
wn = mf.wn(niw)


niv_range = [30,100,120, 160, 320, 640, 800, 6000]
niv_asmpt = 2000
chi0_conv_check = []
asympt = []
for n in niv_range:
    chi0_conv_check.append(fp.vec_get_chi0_sum(giw, beta, n, wn))
    asympt.append(chi0_asympt(niv_asmpt,n,wn))

colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(niv_range)))
fig = plt.figure()
for i, chi0 in enumerate(chi0_conv_check):
    plt.plot(wn, chi0.real+asympt[i], '-o', color=colors[i], markeredgecolor='k', alpha=0.7)
plt.show()
#
fig = plt.figure()
for i, chi0 in enumerate(chi0_conv_check):
    plt.plot(i, chi0[niw+10].real+asympt[i][niw], '-o', color=colors[i], markeredgecolor='k', alpha=0.7)
    plt.plot(i, chi0[niw+10].real, '-h', color=colors[i], markeredgecolor='k', alpha=0.7)
plt.show()

fig = plt.figure()
for i, a in enumerate(asympt):
    plt.plot(wn, a.real, '-o', color=colors[i], markeredgecolor='k', alpha=0.7)
plt.show()

#%%
gchi0 = fp.LocalBubble(wn=wn,giw=giw_obj,niv=100)
plt.figure(dpi=300)
plt.plot(wn,chi0_conv_check[1],'-o', color='cornflowerblue', markeredgecolor='k', alpha=0.7)
plt.plot(wn,gchi0.chi0,'-o', color='firebrick', markeredgecolor='k', alpha=0.7)
plt.show()

#%%
niv_giw = giw.size // 2
iv = mf.iv(beta,niv_giw)
g_tail = 1/(iv + mu - u*totdens/2)
plt.figure()
plt.loglog(iv.imag,giw.imag-g_tail.imag,'-o', color='cornflowerblue', markeredgecolor='k', alpha=0.7)
# plt.xlim(500,800)
plt.show()

#%%

gchi0_2 = fp.LocalBubble(wn=wn,giw=giw_obj,niv=100)
gchi0_3 = fp.LocalBubble(wn=wn,giw=giw_obj,niv=500)

plt.figure()
plt.plot(wn,gchi0_2.chi0_tilde,'-o', color='cornflowerblue', markeredgecolor='k', alpha=0.7)
plt.plot(wn,gchi0_2.chi0,'-o', color='firebrick', markeredgecolor='k', alpha=0.7)
plt.plot(wn,gchi0_3.chi0,'-o', color='forestgreen', markeredgecolor='k', alpha=0.7)
plt.plot(wn,gchi0_3.chi0_tilde,'-o', color='goldenrod', markeredgecolor='k', alpha=0.7)
# plt.xlim(500,800)
plt.show()

#%%

gchi0 = fp.LocalBubble(wn=wn,giw=giw_obj,niv=100)
gchi0_inv = fp.LocalBubble(wn=wn,giw=giw_obj,niv=100,is_inv=True)

print(1./gchi0.mat - gchi0_inv.mat)

plt.figure()
plt.imshow((1./gchi0.mat - gchi0_inv.mat).imag, cmap = 'RdBu')
plt.colorbar()
plt.show()


#%%
niv_asmpt_range = [100,120, 160, 320, 640, 800, 6000, 20000]
niv = 500
chi0_conv_check = []
asympt = []

for n in niv_asmpt_range:
    chi0_conv_check.append(fp.vec_get_chi0_sum(giw, beta, niv, wn))
    asympt.append(chi0_asympt(n,niv,wn))

colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(niv_range)))
fig = plt.figure()
for i, chi0 in enumerate(chi0_conv_check):
    plt.plot(wn, chi0.real+asympt[i], '-o', color=colors[i], markeredgecolor='k', alpha=0.7)
plt.show()
#
fig = plt.figure()
for i, chi0 in enumerate(chi0_conv_check):
    plt.plot(i, chi0[niw+10].real+asympt[i][niw], '-o', color=colors[i], markeredgecolor='k', alpha=0.7)
    plt.plot(i, chi0[niw+10].real, '-h', color=colors[i], markeredgecolor='k', alpha=0.7)
plt.show()

fig = plt.figure()
for i, a in enumerate(asympt):
    plt.plot(wn, a.real, '-o', color=colors[i], markeredgecolor='k', alpha=0.7)
plt.show()
