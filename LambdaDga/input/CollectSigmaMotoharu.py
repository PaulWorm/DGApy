import numpy as np
import h5py
import matplotlib.pyplot as plt



#path = 'C:/Users/pworm/Research/Superconductivity/Nicklates/testdata_from_Motoharu/test_set_from_Motoharu/tpri-0.25-tpppri0.12-U2.0-beta300-n0.85-LQ61-nw100/eliashberg-new-converged/INPUT-for-eliashberg/klist/'
#path = 'C:/Users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta80_t0.5_tp0_tpp0_n0.85/RawDataMotoharu/klist/'
#path = 'C:/Users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.95/RawDataMotoharu/klist/'
#path = 'C:/Users/pworm/Research/HubbardModel/2DSquare_U8_tp-0.25_tpp0.12_beta100_n0.85/from_Motoharu/klist/'
#path = 'mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta16_t0.5_tp0_tpp0_n0.85/RawDataMotoharu/klist/'
path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/U1.0_beta80_t0.5_tp0_tpp0_n0.85/RawDataMotoharu/klist/'
#path = '/mnt/c/users/pworm/Research/Superconductivity/2DHubbard_Testsets/Testset1/RawDataMotoharu/klist/'

Nk   = 120
Niv = 1024

siwk = np.zeros((Niv,Nk,Nk), dtype = complex)
siwk_d = np.zeros((Niv,Nk,Nk), dtype = complex)
siwk_m = np.zeros((Niv,Nk,Nk), dtype = complex)
siwk_rest = np.zeros((Niv,Nk,Nk), dtype = complex)

iter = 1
for ikx in range(Nk//2-1,Nk):
    for iky in range(Nk//2-1,ikx+1):
        print(iter)
        tmp = np.loadtxt(path + 'SELF_Q_{0:06d}.dat'.format(iter))
        siwk[:,ikx,iky] = tmp[:,1] + 1j * tmp[:,2]
        siwk_d[:,ikx,iky] = tmp[:,3] + 1j * tmp[:,4]
        siwk_m[:,ikx,iky] = tmp[:,5] + 1j * tmp[:,6]
        siwk_rest[:,ikx,iky] = tmp[:,7] + 1j * tmp[:,8]
        iter = iter + 1



for ikx in range(Nk//2-1,Nk):
    for iky in range(ikx,Nk):
        siwk[:,ikx,iky] = siwk[:,iky,ikx]
        siwk_d[:,ikx,iky] = siwk_d[:,iky,ikx]
        siwk_m[:,ikx,iky] = siwk_m[:,iky,ikx]
        siwk_rest[:,ikx,iky] = siwk_rest[:,iky,ikx]

for ikx in range(Nk//2-1):
    for iky in range(Nk//2-1,Nk):
        siwk[:,ikx,iky] = siwk[:,Nk-1-ikx,iky]
        siwk_d[:,ikx,iky] = siwk_d[:,Nk-1-ikx,iky]
        siwk_m[:,ikx,iky] = siwk_m[:,Nk-1-ikx,iky]
        siwk_rest[:,ikx,iky] = siwk_rest[:,Nk-1-ikx,iky]

for ikx in range(Nk):
    for iky in range(Nk//2-1):
        siwk[:,ikx,iky] = siwk[:,ikx,Nk-1-iky]
        siwk_d[:,ikx,iky] = siwk_d[:,ikx,Nk-1-iky]
        siwk_m[:,ikx,iky] = siwk_m[:,ikx,Nk-1-iky]
        siwk_rest[:,ikx,iky] = siwk_rest[:,ikx,Nk-1-iky]



plt.imshow(siwk[0,:,:].imag, cmap = 'RdBu', origin='lower')
plt.colorbar()
plt.savefig(path + 'im_sigma.jpg')
plt.show()

plt.imshow(siwk[0,:,:].real, cmap = 'RdBu', origin='lower')
plt.colorbar()
plt.savefig(path + 're_sigma.jpg')
plt.show()


f = h5py.File(path+'Siwk.hdf5','w')

f['siwk'] = siwk
f['siwk_d'] = siwk_d
f['siwk_m'] = siwk_m
f['siwk_rest'] = siwk_rest

f.close()

dict = {
    'siwk': siwk,
    'siwk_d': siwk_d,
    'siwk_m': siwk_m,
    'siwk_rest': siwk_rest,
    'Nk': Nk,
    'Niv': Niv
}

np.save(path+'siwk.npy', dict)