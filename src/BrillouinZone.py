'''
 Module to handle operations within the (irreduzible) Brilloun zone.
'''
import numpy as np
import matplotlib.pyplot as plt

KNOWN_SYMMETRIES = ['x-inv', 'y-inv', 'z-inv', 'x-y-sym']


def get_extent(kgrid=None):
    return [kgrid.kx[0], kgrid.kx[-1], kgrid.ky[0], kgrid.ky[-1]]


def get_extent_pi_shift(kgrid=None):
    return [kgrid.kx[0] - np.pi, kgrid.kx[-1] - np.pi, kgrid.ky[0] - np.pi, kgrid.ky[-1] - np.pi]


def find_arc_node(ak_fs=None, kgrid=None):
    mask = kgrid.kmesh[0] == kgrid.kmesh[1]
    ind = tuple(np.argwhere(mask)[np.argmax(ak_fs[mask])])
    return ind


def find_arc_anti_node(ak_fs=None, kgrid=None):
    mask = kgrid.kmesh[0] == 0
    ind = tuple(np.argwhere(mask)[np.argmax(ak_fs[mask])])
    return ind


def shift_mat_by_pi(mat, nk):
    mat_shift = np.copy(mat)
    mat_shift = np.roll(mat_shift, nk[0] // 2, 0)
    mat_shift = np.roll(mat_shift, nk[1] // 2, 1)
    return mat_shift


def find_fermi_surface_peak(ak_fs=None, kgrid=None):
    eps = 0.000001

    ind = []
    ind2 = []
    kmesh = kgrid.kmesh

    mask = np.logical_and(np.pi / 2 - eps <= kgrid.kx, kgrid.kx <= np.pi + eps)
    kx = kgrid.kx[mask]
    for ikx in kx:
        mask = np.logical_and(kmesh[1] == ikx, kmesh[0] <= np.pi + eps)
        ind.append(tuple(np.argwhere(mask)[np.argmax(ak_fs[mask])]))
    for ikx in kx:
        mask = np.logical_and(kmesh[0] == ikx, kmesh[1] <= np.pi + eps)
        ind2.append(tuple(np.argwhere(mask)[np.argmax(ak_fs[mask])]))

    ind = ind[::-1] + ind2
    return ind


def find_arc_peaks(ak_fs=None, kgrid=None):
    eps = 0.000001

    ind = []
    kmesh = kgrid.kmesh

    mask = np.logical_and(np.pi / 2 - eps <= kgrid.kx, kgrid.kx <= np.pi + eps)
    kx = kgrid.kx[mask]
    for ikx in kx:
        mask = np.logical_and(kmesh[1] == ikx, kmesh[0] <= np.pi + eps)
        ind.append(tuple(np.argwhere(mask)[np.argmax(ak_fs[mask])]))

    ind = ind[::-1]
    return ind


def find_qpd_zeros(qpd=None, kgrid=None):
    eps = 0.000001
    ind = []
    kmesh = kgrid.kmesh
    # mask = np.logical_and(0 <= kgrid.kx, kgrid.kx <= np.pi / 2 - eps)
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


def get_fermi_surface_ind(qpd_fs):
    nk = np.shape(qpd_fs)[0]
    ind_kx = np.arange(nk)
    cs1 = plt.contour(ind_kx, ind_kx, qpd_fs, cmap='RdBu', levels=[0, ])
    paths = cs1.collections[0].get_paths()
    plt.close()
    ind = []
    for path in paths:
        ind_kx = np.array(np.round(path.vertices[:, 0], 0).astype(int))
        ind_ky = np.array(np.round(path.vertices[:, 1], 0).astype(int))
        ind.append(np.stack((ind_kx, ind_ky), axis=1))
    return ind


def two_dimensional_square_symmetries():
    return ['x-inv', 'y-inv', 'x-y-sym']


def two_dimensional_nematic_symmetries():
    return ['x-inv', 'y-inv']


def quasi_two_dimensional_square_symmetries():
    return ['x-inv', 'y-inv', 'z-inv', 'x-y-sym']


def inv_sym(mat, axis):
    '''in-place inversion symmetry applied to mat along dimension axis'''
    assert axis in [0, 1, 2], f'axix = {axis} but must be in [1,2,3]'
    assert len(np.shape(mat)) == 3, f'dim(mat) = {len(np.shape(mat))} but must be 3 dimensional'
    len_ax = np.shape(mat)[axis] // 2
    mod_2 = np.shape(mat)[axis] % 2
    if (axis == 0): mat[len_ax + mod_2:, :, :] = mat[:len_ax, :, :][::-1]
    if (axis == 1): mat[:, len_ax + mod_2:, :] = mat[:, :len_ax, :][:, ::-1]
    if (axis == 2): mat[:, :, len_ax + mod_2:] = mat[:, :, :len_ax][:, :, :-1]


def x_y_sym(mat):
    '''in-place x-y symmetry applied to mat'''
    assert len(np.shape(mat)) == 3, f'dim(mat) = {len(np.shape(mat))} but must be 3 dimensional'
    if (mat.shape[0] == mat.shape[1]):
        mat[:, :, :] = np.minimum(mat, np.transpose(mat, axes=(1, 0, 2)))
    else:
        Warning('Matrix not square. Doing nothing.')


def apply_symmetry(mat, sym):
    '''apply a single symmetry to matrix'''
    assert sym in KNOWN_SYMMETRIES, f'sym = {sym} not in known symmetries {KNOWN_SYMMETRIES}.'
    if (sym == 'x-inv'): inv_sym(mat, 0)
    if (sym == 'y-inv'): inv_sym(mat, 1)
    if (sym == 'z-inv'): inv_sym(mat, 2)
    if (sym == 'x-y-sym'): x_y_sym(mat)


def apply_symmetries(mat, symmetries):
    '''apply symmetries to matrix'''
    assert len(np.shape(mat)) == 3, f'dim(mat) = {len(np.shape(mat))} but must be 3 dimensional'
    if (not symmetries):
        return
    for sym in symmetries:
        apply_symmetry(mat, sym)


class KGrid():
    ''' Class to build the k-grid for the Brillouin zone.'''

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
        return np.array(np.meshgrid(self.kx, self.ky, self.kz))

    @property
    def kmesh_ind(self):
        ''' indizes of {kx,ky,kz}
            Only works for meshes that go from 0 to 2pi
        '''
        return np.array(np.meshgrid(np.arange(0, self.nk[0]), np.arange(0, self.nk[1]), np.arange(0, self.nk[2])))

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

    def map_irrk2fbz(self, mat):
        ''' First dimenstion has to be irrk'''
        old_shape = np.shape(mat)
        return  mat[self.irrk_inv, ...].reshape(self.nk + old_shape[1:])

    def map_fbz2irrk(self, mat):
        '''[kx,ky,kz,...]'''
        return  mat.reshape((-1,*np.shape(mat)[3:]))[self.irrk_ind, ...]


    def symmetrize_irrk(self, mat):
        '''Shape of mat has to be [kx,ky,kz,...]'''
        mat_reshape = np.reshape(mat, (self.nk_tot, -1))
        shp = np.shape(mat_reshape)
        reduced = np.zeros((self.nk_irr,) + shp[1:], dtype=mat_reshape.dtype)
        for i in range(self.nk_irr):
            reduced[i, ...] = np.mean(mat_reshape[self.fbz2irrk == self.irrk_ind[i], ...], axis=0)
        return self.map_irrk2fbz(mat=reduced)

    def shift_mat_by_pi(self, mat):
        mat_shift = np.roll(mat, self.nk[0] // 2, 0)
        mat_shift = np.roll(mat_shift, self.nk[1] // 2, 1)
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
        assert (len(self.grid) == np.size(q)), 'Kgrid and q have different dimensions.'
        kgrid = []
        for i in range(np.size(q)):
            kgrid.append(self.grid[i] - q[i])
        return kgrid




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


def grid(nk=None):
    kx = np.linspace(0, 2 * np.pi, nk[0], endpoint=False)
    ky = np.linspace(0, 2 * np.pi, nk[1], endpoint=False)
    kz = np.linspace(0, 2 * np.pi, nk[2], endpoint=False)
    return kx, ky, kz


def get_irr_grid(ek=None, dec=15):
    ek = np.round(ek, decimals=dec)
    unique, unique_indizes, unique_inverse, unique_counts = np.unique(ek, return_index=True, return_inverse=True,
                                                                      return_counts=True)
    return unique, unique_indizes, unique_inverse, unique_counts


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
        if (k_in is None):
            k = np.linspace(0, np.pi * 2, nk, endpoint=False)
        else:
            k = k_in
        return k

    @property
    def ckps(self):
        ''' Corner k-point strings'''
        return self.path.split(self.path_deliminator)

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
            if (kps in KNOWN_K_POINTS.keys()):
                ckp[i, :] = KNOWN_K_POINTS[kps]
            else:
                ckp[i, :] = get_k_point_from_string(kps)

        return ckp

    def build_k_path(self):
        k_path = []
        nkp = []
        nckp = np.shape(self.ckp)[0]
        for i in range(nckp - 1):
            segment, nkps = kpath_segment(self.ckp[i], self.ckp[i + 1], self.nk)
            nkp.append(nkps)
            if (i == 0):
                k_path = segment
            else:
                k_path = np.concatenate((k_path, segment))
        return k_path, nkp

    def plot(self, fname=None):
        fig = plt.figure()
        plt.plot(self.kpts[:, 0], color='cornflowerblue', label='$k_x$')
        plt.plot(self.kpts[:, 1], color='firebrick', label='$k_y$')
        plt.plot(self.kpts[:, 2], color='seagreen', label='$k_z$')
        plt.legend()
        plt.xlabel('Path-index')
        plt.ylabel('k-index')
        if (fname is not None):
            plt.savefig(fname + '_q_path.png', dpi=300)
        plt.show()


def kpath_segment(k_start, k_end, nk):
    nkp = int(np.round(np.linalg.norm(k_start * nk - k_end * nk)))
    k_segment = k_start[None, :] * nk + np.linspace(0, 1, nkp, endpoint=False)[:, None] * ((k_end - k_start) * nk)[None, :]
    # k_segment = k_start[None,:]*nk + np.linspace(0,1,nkp,endpoint=True)[:,None] * ((k_end-k_start)*nk)[None,:]
    # print(k_segment)
    k_segment = np.round(k_segment).astype(int)
    for i, nki in enumerate(nk):
        ind = np.where(k_segment[:, i] >= nki)
        k_segment[ind, i] = k_segment[ind, i] - nki
    return k_segment, nkp


def get_k_point_from_string(string):
    scoords = string.split(' ')
    coords = np.array([eval(sc) for sc in scoords])
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
    nk = (16, 16, 1)
    kx = np.linspace(0, 2 * np.pi, nk[0])
    ky = np.linspace(0, 2 * np.pi, nk[1])
    kz = np.linspace(0, 2 * np.pi, nk[2])

    fbz2irrk = np.reshape(np.arange(0, np.prod(nk)), nk)


    def x_inv_sym(mat):
        len_x = np.shape(mat)[0] // 2
        mod_2 = np.shape(mat)[0] % 2
        mat[len_x + mod_2:, ...] = mat[:len_x, ...][::-1]


    def y_inv_sym(mat):
        len_y = np.shape(mat)[1] // 2
        mod_2 = np.shape(mat)[1] % 2
        mat[:, len_y + mod_2:, ...] = mat[:, :len_y, ...][:, ::-1]


    def z_inv_sym(mat):
        len_z = np.shape(mat)[1] // 2
        mod_2 = np.shape(mat)[1] % 2
        mat[:, :, len_z + mod_2:, ...] = mat[:, :, :len_z, ...][:, :, :-1]


    # inv_sym(fbz2irrk,0)
    # inv_sym(fbz2irrk,1)
    ind_upper = np.triu_indices(nk[0], 1, nk[1])
    ind_lower = np.tril_indices(nk[0], 0, nk[1])
    tmp = fbz2irrk[:, :, 0]
    print(tmp[ind_upper].shape)
    print(tmp[ind_lower].shape)

    inv_sym(fbz2irrk, 0)
    inv_sym(fbz2irrk, 1)
    x_y_sym(fbz2irrk)

    print(fbz2irrk[:, :, 0])

    plt.pcolormesh(fbz2irrk[..., 0], cmap='terrain')
    plt.show()

    k_grid = KGrid(nk, symmetries=two_dimensional_square_symmetries())

    plt.pcolormesh(k_grid.fbz2irrk[..., 0], cmap='terrain')
    plt.show()
    # nk = 8
    # k_grid = KGrid(nk=(nk,nk,1))
    #
    # ek = np.cos(k_grid.kx[:,None,None]) + np.cos(k_grid.ky[None,:,None])
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(ek[:,:,0], cmap='RdBu', origin='lower')
    # plt.show()
    #
    # ek_shift = k_grid.shift_mat_by_q(mat=ek,q=(np.pi,np.pi,0))
    # ek_shift_2 = k_grid.shift_mat_by_pi(mat=ek)
    #
    # plt.imshow(ek_shift[:,:,0], cmap='RdBu', origin='lower')
    # plt.show()
    #
    # plt.imshow(ek_shift_2[:,:,0], cmap='RdBu', origin='lower')
    # plt.show()
    # mat = k_grid.add_q_to_kgrid(q=(0,0,0))
