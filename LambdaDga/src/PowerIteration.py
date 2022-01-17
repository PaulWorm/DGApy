import numpy as np
import matplotlib.pyplot as  plt
from scipy.spatial.transform import Rotation as rot

class PowerIteration():
    ''' Class to handle power iteration routines'''

    def __init__(self, mat=None, v0=None, eps=10 ** -6, max_count = 10000, shift_mat=0, n_eig = 1):
        self.eps = eps
        self.max_count = max_count
        self.mat = mat
        self.v0=v0
        self.shift_mat = shift_mat
        self.n_eig = n_eig
        self.v = []
        self.lam = []
        self.lam_s = None

        self.get_eig()

    def get_eig(self):
        if(self.shift_mat):
            self.lam_s,_ = self.power_iteration()
        for i in range(self.n_eig):
            lam, v = self.power_iteration()
            self.lam.append(lam)
            self.v.append(v)
        self.lam = np.array(self.lam)
        self.v = np.array(self.v)
        if(self.shift_mat):
            self.lam += self.lam_s

    def power_iteration(self):
        v_old = self.v0
        lambda_old = 1
        converged = False
        count = 0
        while (not converged):
            count += 1
            v_old = gram_schmidt(v=v_old, basis=self.v)
            v_new = self.mat @ v_old
            if(self.lam_s is not None):
                v_new = v_new  - v_old * self.lam_s
            lambda_new = np.sum(np.conj(v_old)*v_new)/np.sum(np.conj(v_old) * v_old)
            v_old = v_new/lambda_new #/np.sum(np.abs(v_new))
            if (np.abs(lambda_new - lambda_old) < self.eps or count > self.max_count):
                converged = True
            lambda_old = lambda_new
        return lambda_new, v_old

def gram_schmidt(v=None, basis=None):
    v_new = np.copy(v)
    for ui in basis:
        v_new -= proj(v=v,u=ui)
    return v_new

def proj(v=None,u=None):
    '''Projection of v onto u.'''
    return np.sum(u*v)/np.sum(u*u) * u



if __name__ == '__main__':

    rot_mat = rot.from_rotvec(np.pi/2 * np.array([0, 0, 1]))
    mat_eig = np.diag((-4,2,3))
    mat = rot_mat.as_matrix() @ mat_eig @ rot_mat.inv().as_matrix()

    powiter = PowerIteration(mat=mat,v0=np.random.rand(3),shift_mat=True,n_eig=2)
    powiter.power_iteration()
    lam_s = powiter.lam_s
    lam = powiter.lam
    print(f'{lam_s=}')
    print(f'{lam=}')
