import numpy as np
import matplotlib.pyplot as plt
import re

KNOWN_K_POINTS = {
    'Gamma': np.array([0,0,0]),
    'X': np.array([0.5,0,0]),#np.array([np.pi,0,0]),
    'Y': np.array([0,0.5,0]),#np.array([0,np.pi,0]),
    'M': np.array([0.5,0.5,0]),#np.array([np.pi,np.pi,0]),
    'M2':np.array([0.25,0.25,0]) #np.array([np.pi/2,np.pi/2,0])
}

class KPath():
    '''
        Object to generate paths in the Brillouin zone.
    '''

    path_deliminator = '-'
    def __init__(self,nk,path,kx=None,ky=None,kz=None):
        '''
            nk: number of points in each dimension (tuple)
            path: desired path in the Brillouin zone (string)
        '''

        self.path = path
        self.nk = nk

        # Set k-grids:
        self.kx = self.set_kgrid(kx)
        self.ky = self.set_kgrid(ky)
        self.kz = self.set_kgrid(kz)

        # Set the k-path:
        self.ckp = self.corner_k_points()
        self.kpts, self.nkp = self.build_k_path()

    def set_kgrid(self,k_in):
        if(k_in is None):
            k = np.linspace(0,np.pi*2,nk[0],endpoint=False)
        else:
            k = k_in
        return k

    @property
    def ckps(self):
        ''' Corner k-point strings'''
        return self.path.split(self.path_deliminator)

    @property
    def cind(self):
        return np.concatenate(([0],np.cumsum(self.nkp)-1))

    def corner_k_points(self):
        ckps = self.ckps
        ckp = np.zeros((np.size(ckps),3))
        for i,kps in enumerate(ckps):
            if(kps in KNOWN_K_POINTS.keys()):
                ckp[i,:] = KNOWN_K_POINTS[kps]
            else:
                ckp[i,:] = get_k_point_from_string(kps)

        return ckp

    def build_k_path(self):
        k_path = []
        nkp = []
        nckp = np.shape(self.ckp)[0]
        for i in range(nckp-1):
            print(i)
            segment, nkps = kpath_segment(self.ckp[i],self.ckp[i+1],self.nk)
            nkp.append(nkps)
            if(i == 0):
                k_path = segment
            else:
                k_path = np.concatenate((k_path,segment))
        return k_path, nkp

    def plot(self,fname=None):
        fig = plt.figure()
        plt.plot(self.kpts[:,0],color='cornflowerblue',label='$k_x$')
        plt.plot(self.kpts[:,1],color='firebrick',label='$k_y$')
        plt.plot(self.kpts[:,2],color='seagreen',label='$k_z$')
        plt.legend()
        plt.xlabel('Path-index')
        plt.ylabel('k-index')
        if(fname is not None):
            plt.savefig(fname + '_q_path.png', dpi=300)
        plt.show()

def kpath_segment(k_start,k_end,nk):
    nkp = int(np.round(np.linalg.norm(k_start*nk-k_end*nk)))
    print(nkp)
    k_segment = k_start[None,:]*nk + np.linspace(0,1,nkp,endpoint=True)[:,None] * ((k_end-k_start)*nk)[None,:]
    k_segment = np.round(k_segment).astype(int)
    return k_segment, nkp

def get_k_point_from_string(string):
    scoords = string.split(' ')
    coords = np.array([eval(sc) for sc in scoords])
    return coords


# Test regular expression matching:
nk = (100,100,1)
path = 'Gamma-X-M2-Gamma'
match = path.split('-')
# kp = get_k_point_from_string(match[-1])
k_start = KNOWN_K_POINTS['Gamma']
k_end = KNOWN_K_POINTS['X']
# segment = kpath_segment(k_start,k_end,nk)
k_path = KPath(nk=nk,path=path)
k_path.plot()
print(k_path.kpts.shape)

