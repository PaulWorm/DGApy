import numpy as np
import Hk as hk

def get_extent(kgrid=None):
    return [kgrid.kx[0], kgrid.kx[-1], kgrid.ky[0], kgrid.ky[-1]]

def get_extent_pi_shift(kgrid=None):
    return [kgrid.kx[0]-np.pi, kgrid.kx[-1]-np.pi, kgrid.ky[0]-np.pi, kgrid.ky[-1]-np.pi]

def find_arc_node(ak_fs=None,kgrid=None):
    mask = kgrid.kmesh[0] == kgrid.kmesh[1]
    ind = tuple(np.argwhere(mask)[np.argmax(ak_fs[mask])])
    return ind

def find_arc_anti_node(ak_fs=None,kgrid=None):
    mask = kgrid.kmesh[0] == 0
    ind = tuple(np.argwhere(mask)[np.argmax(ak_fs[mask])])
    return ind

def shift_mat_by_pi(mat,nk):
    mat_shift = np.copy(mat)
    mat_shift = np.roll(mat_shift,nk[0]//2,0)
    mat_shift = np.roll(mat_shift,nk[1]//2,1)
    return mat_shift

def find_fermi_surface_peak(ak_fs=None,kgrid=None):
    eps = 0.000001

    ind = []
    ind2 = []
    kmesh = kgrid.kmesh

    mask = np.logical_and(np.pi/2-eps <= kgrid.kx, kgrid.kx <= np.pi + eps)
    kx = kgrid.kx[mask]
    for ikx in kx:
        mask = np.logical_and(kmesh[1] == ikx, kmesh[0] <= np.pi + eps)
        ind.append(tuple(np.argwhere(mask)[np.argmax(ak_fs[mask])]))
    for ikx in kx:
        mask = np.logical_and(kmesh[0] == ikx, kmesh[1] <= np.pi + eps)
        ind2.append(tuple(np.argwhere(mask)[np.argmax(ak_fs[mask])]))


    ind = ind[::-1] + ind2
    return ind

def find_arc_peaks(ak_fs=None,kgrid=None):
    eps = 0.000001

    ind = []
    kmesh = kgrid.kmesh

    mask = np.logical_and(np.pi/2-eps <= kgrid.kx, kgrid.kx <= np.pi + eps)
    kx = kgrid.kx[mask]
    for ikx in kx:
        mask = np.logical_and(kmesh[1] == ikx, kmesh[0] <= np.pi + eps)
        ind.append(tuple(np.argwhere(mask)[np.argmax(ak_fs[mask])]))

    ind = ind[::-1]
    return ind

def find_qpd_zeros(qpd=None,kgrid=None):
    eps = 0.000001
    ind = []
    kmesh = kgrid.kmesh
    #mask = np.logical_and(0 <= kgrid.kx, kgrid.kx <= np.pi / 2 - eps)
    mask = np.logical_and(np.pi / 2 - eps <= kgrid.kx, kgrid.kx <= np.pi + eps)
    kx = kgrid.kx[mask]
    for ikx in kx:
        mask = np.logical_and(kmesh[0] == ikx, kmesh[1] <= np.pi + eps)
        ind.append(tuple(np.argwhere(mask)[np.argmin(np.abs(qpd[mask]))]))
        # asign = np.sign(qpd[mask])
        # signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        # signchange[0] = 0
        # ind.append(tuple(np.squeeze(np.argwhere(mask)[np.argwhere(signchange)])))

    ind = ind[::-1]
    return ind



class KGrid():
    ''' Class to build the k-grid for the Brillouin zone.'''

    def __init__(self, nk=None, ek=None):
        #self.dec = 10
        self.nk = nk
        self.set_k_axes()
        self.set_kmesh()
        self.set_ind_lin()
        self.set_ind_tuple()

        if(ek is not None):
            self.get_irrk_from_ek(ek=ek)

    @property
    def grid(self):
        return (self.kx, self.ky, self.kz)

    @property
    def nk_tot(self):
        return np.prod(self.nk)

    @property
    def nk_irr(self):
        return np.size(self.irrk_ind)

    @property
    def irr_kgrid(self):
        return tuple([self.irr_kmesh[ax] for ax in range(len(self.nk))])

    def set_kmesh(self):
        self.kmesh = np.array(np.meshgrid(self.kx, self.ky, self.kz))

    def set_ind_lin(self):
        self.ind_lin = np.arange(0,self.nk_tot)

    def set_ind_tuple(self):
        self.ind = np.squeeze(np.array(np.unravel_index(self.ind_lin,self.nk))).T


    def get_irrk_from_ek(self, ek=None, dec=10):
        _, self.irrk_ind, self.irrk_inv, self.irrk_count = np.unique(np.round(ek, decimals=dec),
                                                                             return_index=True, return_inverse=True,
                                                                             return_counts=True)
        self.irr_kmesh  = np.array([self.kmesh[ax].flatten()[self.irrk_ind] for ax in range(len(self.nk))])
        self.irrk_ind_lin = np.arange(0,self.nk_irr)
        self.fbz2irrk = self.irrk_ind[self.irrk_inv]

    def set_irrk2fbz(self):
        mat = np.reshape(self.ind_lin,self.nk)
        _, self.irrk_ind, self.irrk_inv, self.irrk_count = np.unique(np.round(mat),
                                                                             return_index=True, return_inverse=True,
                                                                             return_counts=True)
        self.irr_kmesh  = np.array([self.kmesh[ax].flatten()[self.irrk_ind] for ax in range(len(self.nk))])
        self.irrk_ind_lin = np.arange(0,self.nk_irr)
        self.fbz2irrk = self.irrk_ind[self.irrk_inv]

    def set_k_axes(self):
        self.kx = np.linspace(0, 2 * np.pi, self.nk[0], endpoint=False)
        self.ky = np.linspace(0, 2 * np.pi, self.nk[1], endpoint=False)
        self.kz = np.linspace(0, 2 * np.pi, self.nk[2], endpoint=False)

    def irrk2fbz(self,mat):
        ''' First dimenstion has to be irrk'''
        old_shape = np.shape(mat)
        mat_fbz = mat[self.irrk_inv,...].reshape(self.nk + old_shape[1:])
        return mat_fbz

    def symmetrize_irrk(self,mat):
        '''Shape of mat has to be [kx,ky,kz,...]'''
        mat_reshape = np.reshape(mat,(self.nk_tot,-1))
        shp = np.shape(mat_reshape)
        reduced = np.zeros((self.nk_irr,)+shp[1:], dtype=mat_reshape.dtype)
        for i in range(self.nk_irr):
            reduced[i,...] = np.mean(mat_reshape[self.fbz2irrk == self.irrk_ind[i],...], axis=0)
        symmetrized = self.irrk2fbz(mat=reduced)
        return symmetrized

    def shift_mat_by_pi(self,mat):
        mat_shift = np.roll(mat,self.nk[0]//2,0)
        mat_shift = np.roll(mat_shift,self.nk[1]//2,1)
        return mat_shift




def grid_2d(nk=16, name='k'):
    kx = np.arange(0, nk) * 2 * np.pi / nk
    ky = np.arange(0, nk) * 2 * np.pi / nk
    kz = np.arange(0, 1)
    grid = {
        '{}x'.format(name): kx,
        '{}y'.format(name): ky,
        '{}z'.format(name): kz
    }
    return grid


def named_grid(nk=None, name='k'):
    kx = np.linspace(0, 2 * np.pi, nk[0], endpoint=False)
    ky = np.linspace(0, 2 * np.pi, nk[1], endpoint=False)
    kz = np.linspace(0, 2 * np.pi, nk[2], endpoint=False)
    grid = {
        '{}x'.format(name): kx,
        '{}y'.format(name): ky,
        '{}z'.format(name): kz
    }
    return grid


def grid(nk=None):
    kx = np.linspace(0, 2 * np.pi, nk[0], endpoint=False)
    ky = np.linspace(0, 2 * np.pi, nk[1], endpoint=False)
    kz = np.linspace(0, 2 * np.pi, nk[2], endpoint=False)
    return kx, ky, kz


class NamedKGrid():
    ''' Class that contains information about the k-grid'''

    def __init__(self, nk=None, name='k', type='fbz'):
        self.nk = nk
        self.name = name
        self.type = type
        if (type == 'fbz'):
            self.grid = named_grid(nk=self.nk, name=self.name)
        else:
            raise NotImplementedError

        self.axes = ('x', 'y', 'z')

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def nk(self):
        return self._nk

    @nk.setter
    def nk(self, value):
        self._nk = value

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    def axis_float_to_string(self, ax=0):
        if (ax == 0):
            return self.axes[0]
        if (ax == 1):
            return self.axes[1]
        if (ax == 2):
            return self.axes[2]

    def nk_tot(self):
        return np.prod(self.nk)

    def get_k(self, ax='x'):
        if (type(ax) is not str):
            ax = self.axis_float_to_string(ax=ax)
        return self.grid['{}{}'.format(self.name, ax)]

    def get_grid_as_tuple(self):
        grid = tuple([self.get_k(ax=ax) for ax in self.axes])
        return grid


def get_irr_grid(ek=None, dec=15):
    ek = np.round(ek, decimals=dec)
    unique, unique_indizes, unique_inverse, unique_counts = np.unique(ek, return_index=True, return_inverse=True,
                                                                      return_counts=True)
    return unique, unique_indizes, unique_inverse, unique_counts


if __name__ == '__main__':
    nk = 8
    grid_k = grid_2d(nk=nk)
    grid_q = grid_2d(nk=nk, name='q')

    qgrid = NamedKGrid(nk=(nk, nk, 1), name='q')
    qx = qgrid.get_k(ax=0.)
    print(f'{qx=}')

    nk_tot = qgrid.nk_tot()

    Grid = NamedKGrid(nk=(nk, nk, 1), name='k')
    kgrid = Grid.get_grid_as_tuple()

    t = 1.0
    tp = -0.2
    tpp = 0.1
    hr = np.array([[t, t, 0], [tp, tp, 0.], [tpp, tpp, 0]])
    ek = hk.ek_3d(Grid.get_grid_as_tuple(), hr)

    k_grid = KGrid(nk=(nk, nk, 1),ek=ek)
    mask = k_grid.irrk_ind[k_grid.irrk_inv]



    ind_equal = np.equal(k_grid.irrk_inv[:,None],k_grid.irrk_ind_lin[None,:])
    n_equal_max = np.max(k_grid.irrk_count)

    mat = ek
    mat_reshape = np.reshape(mat, (k_grid.nk_tot, -1))
    shp = np.shape(mat_reshape)
    reduced = np.zeros((k_grid.nk_irr,) + shp[1:], dtype=mat_reshape.dtype)
    for i in range(k_grid.nk_irr):
        reduced[i, ...] = np.mean(mat_reshape[k_grid.fbz2irrk == k_grid.irrk_ind[i], ...], axis=0)
    symmetrized = k_grid.irrk2fbz(mat=reduced)
