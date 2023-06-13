# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------


def cen2lin(val=None, start=0):
    return val - start


# ------------------------------------------------ OBJECTS -------------------------------------------------------------

class MeshIndexGrids():
    ''' Class to handle index distribution. Accepts meshes and arrays to build a "supermesh" whose elements can then
        be distributed among cores with mpi distributors. Contrary to the class IndexGrids this can also handle
        irreducible k-meshes.
    '''

    def __init__(self, meshes=[], grid_arrays=[], my_slice=slice(0, None, None)):

        self.my_slice = my_slice
        self.grid_arrays = grid_arrays
        self.meshes = meshes
        self.set_meshgrid()

    @property
    def n_grid_arrays(self):
        return len(self.grid_arrays)

    @property
    def n_meshes(self):
        return len(self.meshes)

    @property
    def size_meshes(self):
        return np.array([np.shape(self.meshes[i])[0] for i in self.n_meshes])

    @property
    def n_indizes(self):
        return self.n_grid_arrays + np.sum(self.size_meshes)

    def set_meshgrid(self):
        # unpack tuple with star
        # reshape to be a long array, where each column represents one index
        self._meshgrid = np.array(np.meshgrid(*self.grid_arrays, )).reshape(self.n_indizes, -1).T
        indizes = [np.arange(0, grid_size) for grid_size in self.grid_sizes()]
        self._indizes = np.array(np.meshgrid(*indizes, )).reshape(self.n_indizes, -1).T

class IndexGrids():
    ''' Class for handling index distribution among mpi cores. Does not do the distribution itself, but rather takes
        a slice (my_slice) as input, which determines the tasks of the current process.
        The default for my_slice is a slice over the whole array.
    '''

    def __init__(self, grid_arrays=None, keys=None, my_slice=slice(0, None, None)):

        if keys is None:
            keys = tuple([ind for ind in range(len(grid_arrays))])
        assert len(keys) == len(np.unique(keys)), "Keys are ambiguous"
        assert len(keys) == len(grid_arrays), "Number of keys does not match number of passed grids"

        self.my_slice = my_slice
        self.grid_arrays = grid_arrays
        self.keys = keys  # np.array(keys)
        self.set_meshgrid()

    @property
    def my_slice(self):
        return self._my_slice

    @my_slice.setter
    def my_slice(self, value):
        self._my_slice = value

    @property
    def grid_arrays(self):
        return self._grid_arrays

    @grid_arrays.setter
    def grid_arrays(self, value):
        self._grid_arrays = value

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, value):
        self._keys = value

    @property
    def n_indizes(self):
        return len(self.grid_arrays)

    @property
    def indizes(self):
        return self._indizes

    @property
    def meshgrid(self):
        return self._meshgrid

    @property
    def my_mesh(self):
        return self.meshgrid[self.my_slice]

    @property
    def my_indizes(self):
        return self.indizes[self.my_slice]

    @property
    def my_n_tasks(self):
        return self.my_mesh.shape[0]


    def grid_sizes(self):
        return tuple([self.grid_size(ind=ind) for ind in range(self.n_indizes)])

    def grid_size(self, ind=None, key=None):
        assert key is not ind, 'Only key OR ind as input accepted.'
        if (key is not None):
            ind = self.key2ind(key=key)
        return np.size(self.grid_arrays[ind])

    def mesh_size(self, indizes=None):
        if indizes is None:
            return np.prod([self.grid_size(ind) for ind in range(self.n_indizes)])
        else:
            return np.prod([self.grid_size(ind) for ind in np.atleast_1d(indizes)])

    def key2ind(self, key=None):
        ind = np.argwhere(self.keys == key)[0, 0]
        return ind

    def ind2key(self, ind=0):
        return self.keys[ind]

    def grid_size_by_key(self, key=None):
        ind = self.key2ind(key=key)
        assert np.size(ind) == 1, 'Key is ambiguous'
        return np.size(self.grid_arrays[ind])

    def get_array_by_key(self, key=None):
        ind = self.key2ind(key=key)
        return self.grid_arrays[ind]

    def get_nth_array(self, ind=0):
        return self.grid_arrays[ind]

    def get_nth_key(self, ind=0):
        return self.keys[ind]

    def get_my_array(self, ind=0):
        return np.unique(self.my_mesh[:,ind])

    def set_meshgrid(self):
        # unpack tuple with star
        # reshape to be a long array, where each column represents one index
        self._meshgrid = np.array(np.meshgrid(*self.grid_arrays,indexing='ij')).reshape(self.n_indizes, -1).T
        indizes = [np.arange(0,grid_size) for grid_size in self.grid_sizes()]
        self._indizes = np.array(np.meshgrid(*indizes, )).reshape(self.n_indizes, -1).T

    def reshape_matrix(self, mat=None):
        ''' The first dimension of the input matrix, must match the size of the meshgrid.'''
        old_shape = mat.shape[1:]
        new_shape = self.grid_sizes()
        return np.reshape(mat, newshape=new_shape + old_shape)

    def mean(self, mat=None, axes=None):
        mat = self.reshape_matrix(mat=mat)
        return np.mean(mat, axis=axes)

class qiw():
    ''' Class for handling {q,iw_core} four-indizes
        Indizes must be of increasing value. q's must start at 0, iw_core can start in the negative
    '''

    def __init__(self, qgrid=None, iw=None, my_slice=None):
        self._qx = qgrid[0]
        self._qy = qgrid[1]
        self._qz = qgrid[2]
        self._iw = iw
        self.set_qiw()
        self.my_slice = my_slice

    @property
    def qx(self):
        return self._qx

    @property
    def qy(self):
        return self._qy

    @property
    def qz(self):
        return self._qz

    @property
    def iw(self):
        return self._iw

    @property
    def nqx(self) -> int:
        return self.qx.size

    @property
    def nqy(self) -> int:
        return self.qy.size

    @property
    def nqz(self) -> int:
        return self.qz.size

    @property
    def niw(self) -> int:
        return self.iw.size

    @property
    def size(self):
        return self.qiw.shape[0]

    @property
    def qiw(self):
        return self._qiw

    @property
    def my_size(self):
        return self.my_qiw.shape[0]

    @property
    def my_slice(self):
        return self._my_slice

    @my_slice.setter
    def my_slice(self, value):
        self._my_slice = value

    @property
    def my_qiw(self):
        return self.qiw[self.my_slice]

    @property
    def my_qx(self):
        return self.my_qiw[:, 0]

    @property
    def my_qy(self):
        return self.my_qiw[:, 1]

    @property
    def my_qz(self):
        return self.my_qiw[:, 2]

    @property
    def my_iw(self):
        return self.my_qiw[:, 3]

    @property
    def my_wn(self):
        return cen2lin(self.my_iw, self.iw[0])

    # @property
    # def my_wn_indizes(self):
    #     return np.arange(0,np.size(self.my_iw))

    @property
    def nq(self):
        return self.nqx * self.nqy * self.nqz

    def wn(self, ind):
        return int(cen2lin(val=self.my_qiw[ind, -1], start=self.iw[0]))

    def set_qiw(self):
        self._qiw = np.array(np.meshgrid(self.qx, self.qy, self.qz, self.iw)).reshape(4, -1).T

    def reshape_full(self, mat=None):
        ''' qiw-axis has to be first. Also array must be known on the whole {q,iw} space'''
        old_shape = mat.shape[1:]
        return np.reshape(mat, newshape=(self.nqx, self.nqy, self.nqz, self.niw) + old_shape)

    def q_mean_full(self, mat=None):
        mat_reshape = self.reshape_full(mat=mat)
        return mat_reshape.mean(axis=(0, 1, 2))


class TestClass():

    def __init__(self, arr=None):
        self.arr = arr

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, value):
        self._arr = value


if __name__ == '__main__':
    qx = np.arange(0, 9)
    qy = np.arange(0, 7)
    qz = np.arange(0, 1)
    iw = np.arange(-10, 11)
    qgrid = [qx, qy, qz]
    indizes = qiw(qgrid=qgrid, iw=iw)
    my_slice = slice(1, 5, None)

    #my_slice = slice(-1)
    mpi_indizes = IndexGrids(grid_arrays=(qx, qy, qz, iw), keys=('qx', 'qy', 'qz', 'iw'), my_slice=my_slice)
    print(mpi_indizes.grid_size(ind=0))

    matrix = np.random.rand(mpi_indizes.mesh_size(),10)

    #matrix = mpi_indizes.reshape_matrix(mat=matrix)
    matrix = mpi_indizes.mean(mat=matrix,axes=(0,1))

    import brillouin_zone as bz
    import hr as hamr
    import hk as hamk
    nk = [8,8,1]
    qgrid = bz.KGrid(nk=nk)
    hr = hamr.standard_cuprates()
    ek = hamk.ek_3d(kgrid=qgrid.grid, hr=hr)
    qgrid.get_irrk_from_ek(ek=ek)

    irr_kmesh = [qgrid.kmesh[ax].flatten()[qgrid.irrk_ind] for ax in np.arange(3)]

    print(f'{np.shape(qgrid.irrk_ind)=}')
    print(f'{np.shape(qgrid.kmesh)=}')
    print(f'{qgrid.irr_kmesh=}')
    print(f'{qgrid.irr_kgrid=}')

    #indizes = IndexGrids(grid_arrays=(qgrid.irr_kgrid + (iw,)), keys=('irrqx','irrqy','irrqz','iw'),my_slice=my_slice)
