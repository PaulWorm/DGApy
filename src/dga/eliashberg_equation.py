# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Module for the linear Eliashberg equation as for DGA and similar ladder routines.

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np

# ------------------------------------------------ CLASSES -------------------------------------------------------------

class EliashberPowerIteration():
    ''' Class to handle the linearized Eliashberg equaiton via the means of power iteration.
        gamma: [nkx,nky,nkz,2*niv,2*niv] (we only look at omega = 0 pairing)
        gk: [nkx,nky,nkz,2*niv]
        gap0: [nkx,nky,nkz,2*niv]
        shift_mat: compute the largest eigenvalue (lam_s) and subsequently compute eigenvalues of A - I lam_s
    '''

    def __init__(self, gamma=None,gk=None, gap0=None, norm=None, eps=1e-6, max_count = 10000, shift_mat=0, n_eig = 1):
        self.eps = eps
        self.max_count = max_count
        self.gammax = np.fft.fftn(gamma, axes=(0, 1, 2))
        self.gk = gk
        self.gap0=gap0
        self.shift_mat = shift_mat
        self.n_eig = n_eig
        self.gap = []
        self.lam = []
        self.lam_s = None
        self.gap_s = None
        self.norm = norm
        self.gap0_s = np.random.random_sample(np.shape(gap0))

        self.get_eig()

    def get_eig(self):
        if (self.shift_mat):
            self.lam_s, self.gap_s = self.power_iteration()
        for i in range(self.n_eig):
            lam, gap = self.power_iteration()
            self.lam.append(lam)
            self.gap.append(gap)
        self.lam = np.array(self.lam)
        self.gap = np.array(self.gap)
        if (self.shift_mat):
            self.lam += self.lam_s

    def power_iteration(self):
        if(self.shift_mat and self.lam_s is None):
            gap_old = self.gap0_s
        else:
            gap_old = self.gap0 # always initialize with the same starting point
        lambda_old = 1          # Randomly start with
        converged = False
        count = 0
        while (not converged):
            count += 1
            gap_old = gram_schmidt_eliash(v=gap_old, basis=self.gap, gk=self.gk)
            gap_gg = np.fft.ifftn(gap_old * np.abs(self.gk) ** 2, axes=(0, 1, 2))
            gap_new = 1. / self.norm * np.sum(self.gammax * gap_gg[..., None, :], axis=-1)
            gap_new = np.fft.fftn(gap_new, axes=(0, 1, 2))
            # gep_new = remove_prev_eigenvals(v=gap_new, basis=self.gap, lams=self.lam)
            if (self.lam_s is not None):
                gap_new = gap_new - gap_old * self.lam_s
            lambda_new = np.sum(np.conj(gap_old) * gap_new) / np.sum(np.conj(gap_old) * gap_old)
            gap_old = gap_new/lambda_new

            if (np.abs(lambda_new - lambda_old) < self.eps or count > self.max_count):
                converged = True
            lambda_old = lambda_new

        return lambda_new, gap_old

# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------


def gram_schmidt(v=None, basis=None):
    v_new = np.copy(v)
    for ui in basis:
        v_new -= proj(v=v,u=ui)
    return v_new

def gram_schmidt_eliash(v=None, basis=None, gk=None):
    v_new = np.copy(v)
    for ui in basis:
        v_new = v_new - proj_eliash(v=v,u=ui,gk=gk)
    return v_new

def remove_prev_eigenvals(v,basis,lams):
    v_new = np.copy(v)
    for i,ui in enumerate(basis):
        v_new = v_new - lams[i] * ui * np.sum(np.conj(ui)*v)#/np.sum(ui*ui)
    return v_new

def proj(v=None,u=None):
    '''Projection of v onto u.'''
    return  np.sum(np.conj(u)*v)/np.sum(np.conj(u)*u) * u

def proj_eliash(v=None,u=None,gk=None):
    '''Projection of v onto u.'''
    # return np.sum(u*np.abs(gk)**2*v)/np.sum(u*np.abs(gk)**2*u) * u
    return np.sum(np.conj(u)*v)/np.sum(np.conj(u)*u) * u

def get_gap_start(shape=None, k_type=None, v_type=None, k_grid=None):
    gap0 = np.zeros(shape, dtype=complex)
    niv = shape[-1] // 2
    if (k_type == 'd-wave'):
        gap0[..., niv:] = np.repeat(d_wave(k_grid)[:, :, :, None], niv, axis=3)
    elif (k_type == 'p-wave-y'):
        gap0[..., niv:] = np.repeat(p_wave_y(k_grid)[:, :, :, None], niv, axis=3)
    elif (k_type == 'p-wave-x'):
        gap0[..., niv:] = np.repeat(p_wave_x(k_grid)[:, :, :, None], niv, axis=3)
    else:
        gap0 = np.random.random_sample(shape)

    if (v_type == 'even'):
        gap0[..., :niv] = gap0[..., niv:]
    elif (v_type == 'odd'):
        gap0[..., :niv] = -gap0[..., niv:]
    else:
        gap0 = np.random.random_sample(shape)

    return gap0


def d_wave(k_grid=None):
    return (-np.cos(k_grid[0])[:, None, None] + np.cos(k_grid[1])[None, :, None])


def p_wave_y(k_grid=None):
    return (np.sin(k_grid[0])[:, None, None] * 0 + np.sin(k_grid[1])[None, :, None])

def p_wave_x(k_grid=None):
    return (np.sin(k_grid[0])[:, None, None] + 0*np.sin(k_grid[1])[None, :, None])

def linear_eliashberg(gamma=None, gk=None, eps=10 ** -6, max_count=10000, norm=1.0, gap0=None, shift_mat=True, n=1):
    # Gamma has shape [nkx,nky,nkz,niv,niv]
    powiter = EliashberPowerIteration(gamma=gamma,gk=gk,gap0=gap0,norm=norm,eps=eps,max_count=max_count,shift_mat=shift_mat,n_eig=n)

    return powiter.lam_s,powiter.lam_s, powiter.lam, powiter.gap


def symmetrize_gamma(gamma,channel):
    ''' Symmetrize Gamma. '''
    if(channel == 'sing'):
        gamma_sym = 0.5 * (gamma + np.flip(gamma, axis=(-1)))
    elif(channel == 'trip'):
        gamma_sym = 0.5 * (gamma - np.flip(gamma, axis=(-1)))
    else:
        raise ValueError('Channel must be sing or trip')
    return gamma_sym

if __name__ == '__main__':
    pass
