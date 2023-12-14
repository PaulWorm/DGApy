'''
 Module to handle operations within the (irreduzible) Brilloun zone.
'''
import warnings

import numpy as np
import matplotlib.pyplot as plt

KNOWN_SYMMETRIES = ['x-inv', 'y-inv', 'z-inv', 'x-y-sym', 'x-y-inv']

KNOWN_K_POINTS = {
    'Gamma': np.array([0, 0, 0]),
    'X': np.array([0.5, 0, 0]),  # np.array([np.pi,0,0]),
    'Y': np.array([0, 0.5, 0]),  # np.array([0,np.pi,0]),
    'M': np.array([0.5, 0.5, 0]),  # np.array([np.pi,np.pi,0]),
    'M2': np.array([0.25, 0.25, 0]),  # np.array([np.pi/2,np.pi/2,0])
    'Z': np.array([0.0, 0.0, 0.5]),  # np.array([np.pi/2,np.pi/2,0])
    'R': np.array([0.5, 0.0, 0.5]),  # np.array([np.pi/2,np.pi/2,0])
    'A': np.array([0.5, 0.5, 0.5])  # np.array([np.pi/2,np.pi/2,0])
}

LABELS = {
    'Gamma': r'$\Gamma$',
    'X': 'X',
    'Y': 'Y',
    'M': 'M',
    'M2': 'M/2',
    'Z': 'Z',
    'R': 'R',
    'A': 'A'
}


def get_extent(kgrid=None):
    return [kgrid.kx[0], kgrid.kx[-1], kgrid.ky[0], kgrid.ky[-1]]


def get_extent_pi_shift(kgrid=None):
    return [kgrid.kx[0] - np.pi, kgrid.kx[-1] - np.pi, kgrid.ky[0] - np.pi, kgrid.ky[-1] - np.pi]


def shift_mat_by_pi(mat, nk=None):
    if nk is None:
        nk = [mat.shape[0], mat.shape[1]]
    mat_shift = np.copy(mat)
    mat_shift = np.roll(mat_shift, nk[0] // 2, 0)
    mat_shift = np.roll(mat_shift, nk[1] // 2, 1)
    return mat_shift


def find_zeros(mat):
    ''' Finds the zero crossings of a 2D matrix.
        Not sure if this should be transfered to the Plotting module.
    '''
    ind_x = np.arange(mat.shape[0])
    ind_y = np.arange(mat.shape[1])

    cs1 = plt.contour(ind_x, ind_y, mat.T.real, cmap='RdBu', levels=[0, ])
    paths = cs1.collections[0].get_paths()
    plt.close()
    paths = np.atleast_1d(paths)
    vertices = []
    for path in paths:
        vertices.extend(path.vertices)
    vertices = np.array(vertices, dtype=int)
    return vertices


def two_dimensional_square_symmetries():
    return ['x-inv', 'y-inv', 'x-y-sym']


def two_dimensional_nematic_symmetries():
    return ['x-inv', 'y-inv']


def quasi_two_dimensional_square_symmetries():
    return ['x-inv', 'y-inv', 'z-inv', 'x-y-sym']


def quasi_one_dimensional_square_symmetries():
    return ['x-inv', 'y-inv']


def simultaneous_x_y_inversion():
    return ['x-y-inv', ]


def inv_sym(mat, axis):
    '''in-place inversion symmetry applied to mat along dimension axis
        assumes that the grid is from [0,2pi), hence 0 does not map.
    '''
    assert axis in [0, 1, 2], f'axix = {axis} but must be in [0,1,2]'
    assert len(np.shape(mat)) >= 3, f'dim(mat) = {len(np.shape(mat))} but must be at least 3 dimensional'
    len_ax = np.shape(mat)[axis] // 2
    mod_2 = np.shape(mat)[axis] % 2
    if axis == 0: mat[len_ax + 1:, :, :, ...] = mat[1:len_ax + mod_2, :, :, ...][::-1]
    if axis == 1: mat[:, len_ax + 1:, :, ...] = mat[:, 1:len_ax + mod_2, :, ...][:, ::-1]
    if axis == 2: mat[:, :, len_ax + 1:, ...] = mat[:, :, 1:len_ax + mod_2, ...][:, :, ::-1]


def x_y_sym(mat):
    '''in-place x-y symmetry applied to mat'''
    assert len(np.shape(mat)) >= 3, f'dim(mat) = {len(np.shape(mat))} but must be at least 3 dimensional'
    if mat.shape[0] == mat.shape[1]:
        mat[:, :, :, ...] = np.minimum(mat, np.transpose(mat, axes=(1, 0, 2, *range(3, mat.ndim))))
    else:
        warnings.warn('Matrix not square. Doing nothing.')


def x_y_inv(mat):
    ''' simultaneous inversion in x and y direction'''
    assert len(np.shape(mat)) >= 3, f'dim(mat) = {len(np.shape(mat))} but must be at least 3 dimensional'
    len_ax_x = np.shape(mat)[0] // 2
    mod_2_x = np.shape(mat)[0] % 2
    mat[len_ax_x + 1:, 1:, :, ...] = mat[1:len_ax_x + mod_2_x, 1:, :][::-1, ::-1, :, ...]


def apply_symmetry(mat, sym):
    '''apply a single symmetry to matrix'''
    assert sym in KNOWN_SYMMETRIES, f'sym = {sym} not in known symmetries {KNOWN_SYMMETRIES}.'
    if sym == 'x-inv': inv_sym(mat, 0)
    if sym == 'y-inv': inv_sym(mat, 1)
    if sym == 'z-inv': inv_sym(mat, 2)
    if sym == 'x-y-sym': x_y_sym(mat)
    if sym == 'x-y-inv': x_y_inv(mat)


def apply_symmetries(mat, symmetries):
    '''apply symmetries to matrix inplace'''
    assert len(np.shape(mat)) >= 3, f'dim(mat) = {len(np.shape(mat))} but must at least 3 dimensional'
    if not symmetries:
        return
    for sym in symmetries:
        apply_symmetry(mat, sym)


class KGrid():
    ''' Class to build the k-grid for the Brillouin zone.
    '''

    def __init__(self, nk=None, symmetries=()):
        # Attributes:
        self.kx = None  # kx-grid
        self.ky = None  # ky-grid
        self.kz = None  # kz-grid
        self.irrk_ind = None  # Index of the irreducible BZ points
        self.irrk_inv = None  # Index map back to the full BZ from the irreducible one
        self.irrk_count = None  # duplicity of each k-points in the irreducible BZ
        self.irr_kmesh = None  # k-meshgrid of the irreduzible BZ
        self.fbz2irrk = None  # index map from the full BZ to the irreduzible one
        self.symmetries = symmetries

        self.nk = nk
        self.set_k_axes()
        self.set_ind_tuple()

        self.set_fbz2irrk()
        self.set_irrk_maps()
        self.set_irrk_mesh()

    def set_fbz2irrk(self):
        self.fbz2irrk = np.reshape(np.arange(0, np.prod(self.nk)), self.nk)
        apply_symmetries(self.fbz2irrk, self.symmetries)

    def set_irrk_maps(self):
        _, self.irrk_ind, self.irrk_inv, self.irrk_count = np.unique(self.fbz2irrk,
                                                                     return_index=True, return_inverse=True,
                                                                     return_counts=True)

    def set_irrk_mesh(self):
        self.irr_kmesh = np.array([self.kmesh[ax].flatten()[self.irrk_ind] for ax in range(len(self.nk))])

    @property
    def kx_shift(self):
        return self.kx - np.pi

    @property
    def ky_shift(self):
        return self.ky - np.pi

    @property
    def kz_shift(self):
        return self.kz - np.pi

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

    @property
    def ind_lin(self):
        return np.arange(0, self.nk_tot)

    @property
    def irrk_ind_lin(self):
        return np.arange(0, self.nk_irr)

    @property
    def kmesh(self):
        ''' meshgrid of {kx,ky,kz}'''
        return np.array(np.meshgrid(self.kx, self.ky, self.kz, indexing='ij'))

    @property
    def kmesh_list(self):
        '''
            list of {kx,ky,kz}
        '''
        return self.kmesh.reshape((3, -1))

    @property
    def kmesh_ind(self):
        '''
            indizes of {kx,ky,kz}
            Only works for meshes that go from 0 to 2pi
        '''
        ind_x = np.arange(0, self.nk[0])
        ind_y = np.arange(0, self.nk[1])
        ind_z = np.arange(0, self.nk[2])
        return np.array(np.meshgrid(ind_x, ind_y, ind_z, indexing='ij'))

    @property
    def irrk_mesh_ind(self):
        '''
            indizes of {kx,ky,kz} in the irreducible BZ
        '''
        return np.array([self.kmesh_ind[i].flatten()[self.irrk_ind] for i in range(3)])

    def set_ind_tuple(self):
        self.ind = np.squeeze(np.array(np.unravel_index(self.ind_lin, self.nk))).T

    def set_k_axes(self):
        self.kx = np.linspace(0, 2 * np.pi, self.nk[0], endpoint=False)
        self.ky = np.linspace(0, 2 * np.pi, self.nk[1], endpoint=False)
        self.kz = np.linspace(0, 2 * np.pi, self.nk[2], endpoint=False)

    def map_irrk2fbz(self, mat, shape='mesh'):
        ''' First dimenstion has to be irrk'''
        old_shape = np.shape(mat)
        if shape == 'mesh':
            return mat[self.irrk_inv, ...].reshape(self.nk + old_shape[1:])
        elif shape == 'list':
            return mat[self.irrk_inv, ...].reshape((self.nk_tot, *old_shape[1:]))
        else:
            raise ValueError('shape has to be "mesh" or "list"')

    def get_q_list(self):
        ''' Return list of all q-point indizes in the BZ'''
        return np.array([self.kmesh_ind[i].flatten() for i in range(3)]).T

    def get_irrq_list(self):
        ''' Return list of all q-point indizes in the irreduzible BZ '''
        return np.array([self.kmesh_ind[i].flatten()[self.irrk_ind] for i in range(3)]).T

    def k_mean(self, mat, shape='irrk'):
        ''' Performs summation over the k-mesh'''
        if shape == 'fbz-mesh' or shape == 'mesh':
            return np.mean(mat, axis=(0, 1, 2))
        elif shape == 'fbz-list' or shape == 'list':
            return np.mean(mat, axis=(0,))
        elif shape == 'irrk':
            return 1 / (self.nk_tot) * np.dot(np.moveaxis(mat, 0, -1), self.irrk_count)
        else:
            raise ValueError('type has to be "fbz-mesh", "fbz-list" or "irrk"')

    def map_fbz2irrk(self, mat, shape='mesh'):
        '''
            mesh: [kx,ky,kz,...]
            list: [k,...]
        '''
        if shape == 'mesh':
            return mat.reshape((-1, *np.shape(mat)[3:]))[self.irrk_ind, ...]
        elif shape == 'list':
            return mat.reshape((self.nk_tot, *np.shape(mat)[1:]))[self.irrk_ind, ...]
        else:
            raise ValueError('type has to be "mesh" or "list"')

    def map_fbz_mesh2list(self, mat):
        ''' [kx,ky,kz,...] -> [k,...] '''
        return mat.reshape((self.nk_tot, *np.shape(mat)[3:]))

    def map_fbz_list2mesh(self, mat):
        '''[k,...] -> [kx,ky,kz,...]'''
        return mat.reshape((*self.nk, *np.shape(mat)[1:]))

    def symmetrize_irrk(self, mat):
        '''Shape of mat has to be [kx,ky,kz,...]'''
        mat_reshape = np.reshape(mat, (self.nk_tot, -1))
        shp = np.shape(mat_reshape)
        reduced = np.zeros((self.nk_irr,) + shp[1:], dtype=mat_reshape.dtype)
        for i in range(self.nk_irr):
            reduced[i, ...] = np.mean(mat_reshape[self.fbz2irrk == self.irrk_ind[i], ...], axis=0)
        return self.map_irrk2fbz(mat=reduced)

    def shift_mat_by_pi(self, mat, axes=(0, 1, 2)):
        mat_shift = mat
        for ax in axes:
            shift = np.shape(mat_shift)[ax] // 2
            mat_shift = np.roll(mat_shift, shift, ax)
        return mat_shift

    def shift_mat_by_q(self, mat, q=(0, 0, 0)):
        ''' Structure of mat has to be {kx,ky,kz,...} '''
        ind = self.find_q_ind(q=q)
        return np.roll(mat, ind, axis=(0, 1, 2))

    def shift_mat_by_ind(self, mat, ind=(0, 0, 0)):
        ''' Structure of mat has to be {kx,ky,kz,...} '''
        return np.roll(mat, ind, axis=(0, 1, 2))

    def find_q_ind(self, q=(0, 0, 0)):
        ind = []
        ind.append(np.argmin(np.abs(self.kx - q[0])))
        ind.append(np.argmin(np.abs(self.ky - q[1])))
        ind.append(np.argmin(np.abs(self.kz - q[2])))
        return ind

    def add_q_to_kgrid(self, q=(0, 0, 0)):
        assert len(self.grid) == np.size(q), 'Kgrid and q have different dimensions.'
        kgrid = []
        for i in range(np.size(q)):
            kgrid.append(self.grid[i] - q[i])
        return kgrid

    def get_k_slice(self, mat, k_slice_dim=0, k_slice_val=0):
        '''
            Return the slice along k_slice_dim at k_slice_val
        '''
        assert k_slice_dim >= 0 and k_slice_dim < 3, 'k_slice_dim has to be in [0,1,2]'
        assert mat.shape[:3] == self.nk, 'mat has to be of shape [nkx,nky,nkz,...]'

        if k_slice_dim == 0:
            k_ind = np.argmin(np.abs(self.kx - k_slice_val))
            return mat[k_ind, :, :, ...]
        elif k_slice_dim == 1:
            k_ind = np.argmin(np.abs(self.ky - k_slice_val))
            return mat[:, k_ind, :, ...]
        else:
            k_ind = np.argmin(np.abs(self.kz - k_slice_val))
            return mat[:, :, k_ind, ...]


class KPath():
    '''
        Object to generate paths in the Brillouin zone.
        Currently assumed that the BZ grid is from (0,2*pi)
    '''

    def __init__(self, nk, path, kx=None, ky=None, kz=None, path_deliminator='-'):
        '''
            nk: number of points in each dimension (tuple)
            path: desired path in the Brillouin zone (string)
        '''
        self.path_deliminator = path_deliminator
        self.path = path
        self.nk = nk

        # Set k-grids:
        self.kx = self.set_kgrid(kx, nk[0])
        self.ky = self.set_kgrid(ky, nk[1])
        self.kz = self.set_kgrid(kz, nk[2])

        # Set the k-path:
        self.ckp = self.corner_k_points()
        self.kpts, self.nkp = self.build_k_path()
        self.k_val = self.get_kpath_val()
        self.k_points = self.get_kpoints()

    def get_kpath_val(self):
        k = [self.kx[self.kpts[:, 0]], self.kx[self.kpts[:, 1]], self.kx[self.kpts[:, 2]]]
        return k

    def set_kgrid(self, k_in, nk):
        if k_in is None:
            k = np.linspace(0, np.pi * 2, nk, endpoint=False)
        else:
            k = k_in
        return k

    @property
    def ckps(self):
        ''' Corner k-point strings'''
        return self.path.split(self.path_deliminator)

    @property
    def labels(self):
        ''' Labels of the k-points for plotting'''
        count = 0
        ckps = self.ckps
        l = []
        for k_p in ckps:
            if k_p in KNOWN_K_POINTS:
                l.append(LABELS[k_p])
            else:
                l.append(f'K{count}')
            count += 1
        return l

    @property
    def x_ticks(self):
        ''' Return ticks values for plotting'''
        return self.k_axis[self.cind]

    @property
    def cind(self):
        return np.concatenate(([0], np.cumsum(self.nkp) - 1))

    @property
    def ikx(self):
        return self.kpts[:, 0]

    @property
    def iky(self):
        return self.kpts[:, 1]

    @property
    def ikz(self):
        return self.kpts[:, 2]

    @property
    def k_axis(self):
        return np.linspace(0, 1, np.sum(self.nkp), endpoint=True)

    @property
    def nk_tot(self):
        return np.sum(self.nkp)

    @property
    def nk_seg(self):
        return np.diff(self.cind)

    def get_kpoints(self):
        return np.array(self.k_val).T

    def corner_k_points(self):
        ckps = self.ckps
        ckp = np.zeros((np.size(ckps), 3))
        for i, kps in enumerate(ckps):
            if kps in KNOWN_K_POINTS:
                ckp[i, :] = KNOWN_K_POINTS[kps]
            else:
                ckp[i, :] = get_k_point_from_string(kps)

        return ckp

    def map_to_kpath(self, mat):
        ''' Map mat [kx,ky,kz,...] onto the k-path'''
        return mat[self.ikx, self.iky, self.ikz, ...]

    def build_k_path(self):
        k_path = []
        nkp = []
        nckp = np.shape(self.ckp)[0]
        for i in range(nckp - 1):
            segment, nkps = kpath_segment(self.ckp[i], self.ckp[i + 1], self.nk)
            nkp.append(nkps)
            if i == 0:
                k_path = segment
            else:
                k_path = np.concatenate((k_path, segment))
        return k_path, nkp

    def plot_kpoints(self, fname=None):
        plt.figure()
        plt.plot(self.kpts[:, 0], color='cornflowerblue', label='$k_x$')
        plt.plot(self.kpts[:, 1], color='firebrick', label='$k_y$')
        plt.plot(self.kpts[:, 2], color='seagreen', label='$k_z$')
        plt.legend()
        plt.xlabel('Path-index')
        plt.ylabel('k-index')
        if fname is not None:
            plt.savefig(fname + '_q_path.png', dpi=300)
        plt.show()

    def plot_kpath(self, mat, verbose=False, do_save=True, pdir='./', name='k_path', ylabel='Energy [t]'):
        '''
            mat: [kx,ky,kz]
        '''
        plt.figure()
        plt.xticks(self.x_ticks, self.labels)
        plt.vlines(self.x_ticks, np.min(mat), np.max(mat), ls='-', color='grey', alpha=0.8)
        plt.hlines(0, self.k_axis[0], self.k_axis[-1], ls='--', color='grey', alpha=0.8)
        plt.plot(self.k_axis, self.map_to_kpath(mat), '-k')
        plt.ylabel(ylabel)
        plt.xlim(self.k_axis[0], self.k_axis[-1])
        plt.ylim(np.min(mat), np.max(mat))
        if do_save:
            plt.savefig(pdir + '/' + name + '.png', dpi=300)
        if verbose:
            plt.show()
        plt.close()

    def get_bands(self, ek):
        ''' Return the bands along the k-path '''
        ek_kpath = self.map_to_kpath(ek)
        bands = np.zeros((ek_kpath.shape[:-1]))
        for i, eki in enumerate(ek_kpath):
            val, _ = np.linalg.eig(eki)
            bands[i, :] = np.sort(val)
        return bands


def kpath_segment(k_start, k_end, nk):
    nkp = int(np.round(np.linalg.norm(k_start * nk - k_end * nk)))
    k_segment = k_start[None, :] * nk + np.linspace(0, 1, nkp, endpoint=False)[:, None] * ((k_end - k_start) * nk)[None, :]
    k_segment = np.round(k_segment).astype(int)
    for i, nki in enumerate(nk):
        ind = np.where(k_segment[:, i] >= nki)
        k_segment[ind, i] = k_segment[ind, i] - nki
    return k_segment, nkp


def get_k_point_from_string(string):
    scoords = string.split(' ')
    coords = np.array([float(sc) for sc in scoords])
    return coords


def get_bz_masks(nk):
    mask_1q = np.ones((nk, nk), dtype=int)
    mask_2q = np.ones((nk, nk), dtype=int)
    mask_3q = np.ones((nk, nk), dtype=int)
    mask_4q = np.ones((nk, nk), dtype=int)
    mask_3q[:nk // 2, :nk // 2] = 0
    mask_1q[nk // 2:, :nk // 2] = 0
    mask_2q[nk // 2:, nk // 2:] = 0
    mask_4q[:nk // 2, nk // 2:] = 0
    return [mask_1q, mask_2q, mask_3q, mask_4q]


def shift_mat_by_ind(mat, ind=(0, 0, 0)):
    ''' Structure of mat has to be {kx,ky,kz,...} '''
    return np.roll(mat, ind, axis=(0, 1, 2))


if __name__ == '__main__':
    pass
