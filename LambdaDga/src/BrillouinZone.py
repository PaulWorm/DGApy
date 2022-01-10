import numpy as np
import Hk as hk


class KGrid():
    ''' Class to build the k-grid for the Brillouin zone.'''

    def __init__(self, nk=None, ek=None):
        self.dec = 10
        self.nk = nk
        self.set_k_axes()
        self.set_kmesh()

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
        self.kmesh = np.meshgrid(self.kx, self.ky, self.kz)


    def get_irrk_from_ek(self, ek=None):
        _, self.irrk_ind, self.irrk_inv, self.irrk_count = np.unique(np.round(ek, decimals=self.dec),
                                                                             return_index=True, return_inverse=True,
                                                                             return_counts=True)
        self.irr_kmesh  = np.array([self.kmesh[ax].flatten()[self.irrk_ind] for ax in range(len(self.nk))])
        self.irrk_ind_lin = np.arange(0,self.nk_irr)

    def set_k_axes(self):
        self.kx = np.linspace(0, 2 * np.pi, self.nk[0], endpoint=False)
        self.ky = np.linspace(0, 2 * np.pi, self.nk[1], endpoint=False)
        self.kz = np.linspace(0, 2 * np.pi, self.nk[2], endpoint=False)

    def irrk2fbz(self,mat):
        ''' First dimenstion has to be irrk'''
        old_shape = np.shape(mat)
        mat_fbz = mat[self.irrk_inv,...].reshape(self.nk + old_shape[1:])
        return mat_fbz




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
    nk = 200
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

    dec = 10
    ek = np.round(ek, decimals=dec)
    unique, unique_indizes, unique_inverse, unique_counts = np.unique(ek, return_index=True, return_inverse=True,
                                                                      return_counts=True)
    kx = kgrid[0]
    ky = kgrid[1]
    kz = kgrid[2]
    KX, KY, KZ = np.meshgrid(kx, ky, kz)

    nk_irr = np.size(unique)
    nk_irr_min = nk ** 2 / 8
    print(f'{nk_irr=}')
    print(f'{nk_irr_min=}')
    print(np.sum(unique[unique_inverse].reshape(nk,nk,1)-ek))
