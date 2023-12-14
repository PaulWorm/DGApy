'''
    This module contains routines for wannier and tight-binding Hamiltonians.
    It builds upon the brilloun_zone.py module.
    Simple t-tp-tpp models as well as reading wannier90 Hr files is supported.
'''
import numpy as np
import pandas as pd
from warnings import warn

from dga import brillouin_zone as bz


class WannierHr():
    '''
        class to handle wannier Hamiltonians
    '''

    def __init__(self, hr, r_grid, r_weights, orbs):
        self.hr = hr
        self.r_grid = r_grid
        self.r_weights = r_weights
        self.orbs = orbs

    def get_ek(self, k_grid: bz.KGrid, one_band=True):
        ek = convham(self.hr, self.r_grid, self.r_weights, k_grid.kmesh.reshape(3, -1))
        n_orbs = ek.shape[-1]
        if one_band:
            return ek.reshape(*k_grid.nk, n_orbs, n_orbs)[:, :, :, 0, 0]
        else:
            return ek.reshape(*k_grid.nk, n_orbs, n_orbs)

    def get_light_vertex(self, k_grid: bz.KGrid, der=0, one_band=True):
        lv = light_vertex(self.hr, self.r_grid, self.r_weights, k_grid.kmesh.reshape(3, -1), der=der)
        n_orbs = lv.shape[-1]
        if one_band:
            return lv.reshape(*k_grid.nk, n_orbs, n_orbs)[:, :, :, 0, 0]
        else:
            return lv.reshape(*k_grid.nk, n_orbs, n_orbs)

    def save_hr(self, path, name='wannier_hr.dat'):
        ''' save the Hamiltonian to file.'''
        write_hr_w2k(path + name, self.hr, self.r_grid, self.r_weights, self.orbs)

    def save_hk(self, k_grid: bz.KGrid, path, name='wannier.hk'):
        ''' save the Hamiltonian to file.'''
        ek = self.get_ek(k_grid, one_band=False)
        kmesh = k_grid.kmesh_list
        ek = k_grid.map_fbz_mesh2list(ek)
        write_hk_wannier90(ek, path + name, kmesh, k_grid.nk)


def create_wannier_hr_from_file(fname):
    ''' Reads a wannier90 file and creates an instance of the Wannier90 class.'''
    return WannierHr(*read_hr_w2k(fname))


# --------------------------------------- CONSTRUCT REAL SPACE HAMILTONIANS --------------------------------------------

def create_r_grid(r2ind, n_orbs):
    ''' Create the real-space grid from a dictionary r2ind.'''
    n_rp = len(r2ind)
    r_grid = np.zeros((n_rp, n_orbs, n_orbs, 3))
    for r_vec in r2ind.keys():
        r_grid[r2ind[r_vec], :, :, :] = r_vec
    return r_grid


def create_orbs(n_rp, n_orbs):
    ''' Create the orbital matricies for the real-space Hamiltonian.'''
    orbs = np.zeros((n_rp, n_orbs, n_orbs, 2))
    for io1 in range(n_orbs):
        for io2 in range(n_orbs):
            orbs[io1, io2, :] = np.array([io1 + 1, io2 + 1])
    return orbs


def insert_hr_elemend(hr_mat, r2ind, r_vec, orb1, orb2, hr_elem):
    orb1 = orb1 - 1
    orb2 = orb2 - 1
    r_ind = r2ind[r_vec]
    hr_mat[r_ind, orb1, orb2] = hr_elem


def emery_model_ek(k_grid: bz.KGrid, ed, ep, tpd, tpp, tpp_p):
    ''' Create the momentum-space Hamiltonian for the Emery model.
        This is primarily inteded for testing purposes.
    '''

    def hk_single_k(kx, ky, ed, ep, tpp, tpd, tpp_p):
        h = np.array([[ed, tpd * (1 - np.exp(-1j * kx)), tpd * (1 - np.exp(-1j * ky))]
                         , [tpd * (1 - np.exp(1j * kx)), ep + 2 * tpp_p * np.cos(kx),
                            tpp * (1 - np.exp(1j * kx)) * (1 - np.exp(-1j * ky))],
                      [tpd * (1 - np.exp(1j * ky)), tpp * (1 - np.exp(-1j * kx)) * (1 - np.exp(1j * ky)),
                       ep + 2 * tpp_p * np.cos(ky)]])
        return h

    hk_kspace = []
    for kx, ky, _ in k_grid.kmesh_list.T:
        hk_kspace.append(hk_single_k(kx, ky, ed, ep, tpp, tpd, tpp_p))
    n_orbs = 3
    hk_kspace = np.array(hk_kspace).reshape(k_grid.nk + (n_orbs, n_orbs))
    return hk_kspace


def wannier_emery_model(ed, ep, tpd, tpp, tpp_p):
    ''' Create the real-space Hamiltonian for the Emery model.'''
    n_orbs = 3
    r2ind = {(0, 0, 0): 0, (1, 0, 0): 1, (0, 1, 0): 2, (-1, 0, 0): 3, (0, -1, 0): 4, (1, -1, 0): 5, (-1, 1, 0): 6}
    n_rp = len(r2ind)
    r_grid = create_r_grid(r2ind, n_orbs)
    hr = np.zeros((n_rp, n_orbs, n_orbs))
    orbs = create_orbs(n_rp, n_orbs)
    r_weights = np.ones(n_rp)[:, None]
    hr = np.zeros((n_rp, n_orbs, n_orbs))

    # Build the real-space Hamiltonian:

    # on-site energies:
    r_vec = (0, 0, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 1, 1, ed)
    insert_hr_elemend(hr, r2ind, r_vec, 2, 2, ep)
    insert_hr_elemend(hr, r2ind, r_vec, 3, 3, ep)

    # tpd hopping:
    r_vec = (0, 0, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 1, 2, tpd)
    insert_hr_elemend(hr, r2ind, r_vec, 2, 1, tpd)
    insert_hr_elemend(hr, r2ind, r_vec, 1, 3, tpd)
    insert_hr_elemend(hr, r2ind, r_vec, 3, 1, tpd)

    r_vec = (-1, 0, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 1, 2, -tpd)

    r_vec = (0, -1, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 1, 3, -tpd)

    r_vec = (1, 0, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 2, 1, -tpd)

    r_vec = (0, 1, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 3, 1, -tpd)

    # tpp hopping:
    r_vec = (0, 0, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 2, 3, tpp)
    insert_hr_elemend(hr, r2ind, r_vec, 3, 2, tpp)

    r_vec = (1, 0, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 2, 3, -tpp)

    r_vec = (0, 1, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 3, 2, -tpp)

    r_vec = (-1, 0, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 3, 2, -tpp)

    r_vec = (0, -1, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 2, 3, -tpp)

    r_vec = (-1, 1, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 3, 2, tpp)

    r_vec = (1, -1, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 2, 3, tpp)

    # tpp' hopping:
    r_vec = (1, 0, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 2, 2, tpp_p)

    r_vec = (0, 1, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 3, 3, tpp_p)

    r_vec = (-1, 0, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 2, 2, tpp_p)

    r_vec = (0, -1, 0)
    insert_hr_elemend(hr, r2ind, r_vec, 3, 3, tpp_p)

    return hr, r_grid, r_weights, orbs


def wannier_one_band_2d_t_tp_tpp(t, tp, tpp):
    hr = -np.array([t, t, t, t, tp, tp, tp, tp, tpp, tpp, tpp, tpp])[:, None, None]
    r_grid = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
                       [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
                       [2, 0, 0], [0, 2, 0], [-2, 0, 0], [0, -2, 0]])[:, None, None, :]
    r_weights = np.ones((12, 1))
    orbs = np.ones((12, 1, 1, 2))
    return hr, r_grid, r_weights, orbs


def one_band_2d_t_tp_tpp(t=1.0, tp=0., tpp=0.):
    return np.array([[t, t, 0], [tp, tp, 0.], [tpp, tpp, 0]])


def one_band_2d_quasi1d(tx=1.0, ty=0, tppx=0, tppy=0, tpxy=0):
    return np.array([[tx, ty, 0], [tpxy, tpxy, 0.], [tppx, tppy, 0]])


def one_band_2d_nematic(tx=1.0, ty=0, tppx=0, tppy=0, tpxy=0):
    return np.array([[tx, ty, 0], [tpxy, tpxy, 0.], [tppx, tppy, 0]])


def wannier_one_band_2d_nematic(tx, ty, tppx, tppy, tpxy):
    hr = -np.array([tx, ty, tx, ty, tpxy, tpxy, tpxy, tpxy, tppx, tppy, tppx, tppy])[:, None, None]
    r_grid = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
                       [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
                       [2, 0, 0], [0, 2, 0], [-2, 0, 0], [0, -2, 0]])[:, None, None, :]
    r_weights = np.ones((12, 1))
    orbs = np.ones((12, 1, 1, 2))
    return hr, r_grid, r_weights, orbs


def one_band_2d_triangular_t_tp_tpp(t=1.0, tp=0, tpp=0):
    return np.array([[t, t, 0], [tp, 0, 0.], [tpp, tpp, 0]])


def standard_cuprates(t=1.0):
    tp = -0.2 * t
    tpp = 0.1 * t
    return np.array([[t, t, 0], [tp, tp, 0.], [tpp, tpp, 0]])


def motoharu_nickelates(t=0.25):
    tp = -0.25 * t
    tpp = 0.12 * t
    return np.array([[t, t, 0], [tp, tp, 0.], [tpp, tpp, 0]])


def unfrustrated_square(t=1.00):
    tp = 0
    tpp = 0
    return np.array([[t, t, 0], [tp, tp, 0.], [tpp, tpp, 0]])


# pylint: disable=invalid-name
def Ba2CuO4_plane():
    # Ba2CuO3.25 parameters
    # tx = 0.018545
    # ty = 0.470181
    # tpxy = 0.006765
    # tppx = 0.001255
    # tppy = 0.084597
    tx = 0.0185
    ty = 0.47
    tpxy = 0.0068
    tppx = 0.0013
    tppy = 0.085

    return one_band_2d_quasi1d(tx=tx, ty=ty, tppx=tppx, tppy=tppy, tpxy=tpxy)


def Ba2CuO4_plane_2d_projection():
    # Ba2CuO3.25 2D-projection parameters
    tx = 0.0258
    ty = 0.5181
    tpxy = 0.0119
    tppx = -0.0014
    tppy = 0.0894
    return one_band_2d_quasi1d(tx=tx, ty=ty, tppx=tppx, tppy=tppy, tpxy=tpxy)


# pylint: enable=invalid-name

# ==================================================================================================================


def read_hr_w2k(fname):
    '''
        Load the H(R) LDA-Hamiltonian from a wien2k hr file.
    '''
    hr_file = pd.read_csv(fname, skiprows=1, names=np.arange(15), sep=r'\s+', dtype=float, engine='python')
    n_bands = hr_file.values[0][0].astype(int)
    nr = hr_file.values[1][0].astype(int)

    tmp = np.reshape(hr_file.values, (np.size(hr_file.values), 1))
    tmp = tmp[~np.isnan(tmp)]

    r_weights = tmp[2:2 + nr].astype(int)
    r_weights = np.reshape(r_weights, (np.size(r_weights), 1))
    ns = 7
    n_tmp = np.size(tmp[2 + nr:]) // ns
    tmp = np.reshape(tmp[2 + nr:], (n_tmp, ns))

    r_grid = np.reshape(tmp[:, 0:3], (nr, n_bands, n_bands, 3))
    orbs = np.reshape(tmp[:, 3:5], (nr, n_bands, n_bands, 2))
    hr = np.reshape(tmp[:, 5] + 1j * tmp[:, 6], (nr, n_bands, n_bands))
    # hr_dict = {
    #     'hr': hr,
    #     'r_grid': r_grid,
    #     'r_weights': r_weights,
    #     'orbs': orbs
    # }
    # return hr_dict
    return hr, r_grid, r_weights, orbs


# ==================================================================================================================
# pylint: disable=C0209
def write_hr_w2k(fname, hr, r_grid, r_weights, orbs):
    '''
        Write a real-space Hamiltonian in the format of w2k to a file.
    '''
    n_columns = 15
    n_r = hr.shape[0]
    n_bands = hr.shape[-1]
    file = open(fname, 'w', encoding='utf-8')
    file.write('# Written using the wannier module of the dga code\n')
    file.write(f'{n_bands} \n')
    file.write(f'{n_r} \n')

    for i in range(0, len(r_weights), n_columns):
        line = '    '.join(map(str, r_weights[i:i + n_columns, 0]))
        file.write('    ' + line + '\n')
    hr = hr.reshape(n_r * n_bands ** 2, 1)
    r_grid = r_grid.reshape(n_r * n_bands ** 2, 3).astype(int)
    orbs = orbs.reshape(n_r * n_bands ** 2, 2).astype(int)

    for i in range(0, n_r * n_bands ** 2):
        line = '{: 5d}{: 5d}{: 5d}{: 5d}{: 5d}{: 12.6f}{: 12.6f}'.format(*r_grid[i, :], *orbs[i, :], hr[i, 0].real, hr[i, 0].imag)
        file.write(line + '\n')


# pylint: enable=C0209

# ------------------------------------------------ OBJECTS -------------------------------------------------------------

def ek_square(kx=None, ky=None, t=1.0, tp=0.0, tpp=0.0):
    return - 2. * t * (np.cos(kx) + np.cos(ky)) - 4. * tp * np.cos(kx) * np.cos(ky) \
           - 2. * tpp * (np.cos(2. * kx) + np.cos(2. * ky))


def del_ek_del_kx_square(kx=None, ky=None, t=1.0, tp=0.0, tpp=0.0):
    return - 2. * t * (-np.sin(kx)) + 4. * tp * np.sin(kx) * np.cos(ky) \
           - 2. * tpp * (-2 * np.sin(2. * kx))


def ekpq_3d(kx=None, ky=None, kz=None, qx=0, qy=0, qz=0, t_mat=None):
    kx = kx + qx
    ky = ky + qy
    kz = kz + qz
    ek = - 2.0 * (t_mat[0, 0] * np.cos(kx) + t_mat[0, 1] * np.cos(ky) + t_mat[0, 2] * np.cos(kz))
    ek = ek - 2.0 * (t_mat[1, 0] * np.cos(kx + ky) + t_mat[1, 1] * np.cos(kx - ky))
    ek = ek - 2.0 * (t_mat[2, 0] * np.cos(2. * kx) + t_mat[2, 1] * np.cos(2. * ky) + t_mat[2, 2] * np.cos(kz))
    return ek


def ek_3d(kgrid=None, hr=None):
    kx = kgrid[0][:, None, None]
    ky = kgrid[1][None, :, None]
    kz = kgrid[2][None, None, :]
    ek = - 2.0 * (hr[0, 0] * np.cos(kx) + hr[0, 1] * np.cos(ky) + hr[0, 2] * np.cos(kz))
    ek = ek - 2.0 * (hr[1, 0] * np.cos(kx + ky) + hr[1, 1] * np.cos(kx - ky))
    ek = ek - 2.0 * (hr[2, 0] * np.cos(2. * kx) + hr[2, 1] * np.cos(2. * ky) + hr[2, 2] * np.cos(kz))
    return ek


def ek_3d_klist(kgrid=None, hr=None):
    kx = kgrid[:, 0]
    ky = kgrid[:, 1]
    kz = kgrid[:, 2]
    ek = - 2.0 * (hr[0, 0] * np.cos(kx) + hr[0, 1] * np.cos(ky) + hr[0, 2] * np.cos(kz))
    ek = ek - 2.0 * (hr[1, 0] * np.cos(kx + ky) + hr[1, 1] * np.cos(kx - ky))
    ek = ek - 2.0 * (hr[2, 0] * np.cos(2. * kx) + hr[2, 1] * np.cos(2. * ky) + hr[2, 2] * np.cos(kz))
    return ek


# ==================================================================================================================
def read_hk_w2k(fname, spin_sym=True):
    """ Reads a Hamiltonian f$ H_{bb'}(k) f$ from a text file.

    Expects a text file with white-space separated values in the syntax as
    generated by wannier90:  the first line is a header with three integers,
    optionally followed by '#'-prefixed comment:

        <no of k-points> <no of wannier functions> <no of bands (ignored)>

    For each k-point, there is a header line with the x, y, z coordinates of the
    k-point, followed by <nbands> rows as lines of 2*<nbands> values each, which
    are the real and imaginary part of each column.

    Returns: A pair (hk, kpoints):
     - hk       three-dimensional complex array of the Hamiltonian H(k),
                where the first index corresponds to the k-point and the other
                dimensions mark band indices.
     - kpoints  two-dimensional real array, which contains the
                components (x,y,z) of the different kpoints.
    """
    hk_file = open(fname, 'r', encoding='utf-8')
    spin_orbit = not spin_sym

    def nextline():
        line = hk_file.readline()
        return line[:line.find('#')].split()

    # parse header
    header = nextline()
    if header[0] == 'VERSION':
        warn('Version 2 headers are obsolete (specify in input file!)')
        nkpoints, natoms = map(int, nextline())
        lines = np.array([nextline() for _ in range(natoms)], np.int)
        nbands = np.sum(lines[:, :2])
        del lines, natoms
    elif len(header) != 3:
        warn('Version 1 headers are obsolete (specify in input file!)')
        header = list(map(int, header))
        nkpoints = header[0]
        nbands = header[1] * (header[2] + header[3])
    else:
        nkpoints, nbands, _ = map(int, header)
    del header

    # nspins is the spin dimension for H(k); G(iw), Sigma(iw) etc. will always
    # be spin-dependent
    if spin_orbit:
        if nbands % 2: raise RuntimeError('Spin-structure of Hamiltonian!')
        nbands //= 2
        nspins = 2
    else:
        nspins = 1
    # GS: inside read_hamiltonian nspins is therefore equal to 1 if spin_orbit=0
    # GS: outside nspins is however always set to 2. Right?
    # GS: this also means that we have to set nspins internally in read_ImHyb too

    hk_file.flush()

    # parse data
    hk = np.fromfile(hk_file, sep=' ')
    hk = hk.reshape(-1, 3 + 2 * nbands ** 2 * nspins ** 2)
    kpoints_file = hk.shape[0]
    if kpoints_file > nkpoints:
        warn('truncating Hk points')
    elif kpoints_file < nkpoints:
        raise ValueError(f'problem! {kpoints_file} < {nkpoints}')
    kpoints = hk[:nkpoints, :3]

    hk = hk[:nkpoints, 3:].reshape(nkpoints, nspins, nbands, nspins, nbands, 2)
    hk = hk[..., 0] + 1j * hk[..., 1]
    if not np.allclose(hk, hk.transpose(0, 3, 4, 1, 2).conj()):
        warn('Hermiticity violation detected in Hk file')

    # go from wannier90/convert_Hamiltonian structure to our Green's function
    # convention
    hk = hk.transpose(0, 2, 1, 4, 3)
    hk_file.close()

    hk = np.squeeze(hk)

    return hk, kpoints.T


# ==================================================================================================================


# ==================================================================================================================
def convham2(hr=None, r_grid=None, r_weights=None, kmesh=None):
    '''
        Builds the k-space LDA-Hamiltonian from the real-space one.
        H(k) is constructed within the ir-BZ
        hk (nk,Nbands,Nbands)
    '''

    nk = np.size(kmesh, 1)
    n_band = np.size(hr, 1)
    hk = np.zeros((nk, n_band, n_band), dtype=complex)
    r_weights = np.squeeze(r_weights)
    for ib2 in range(n_band):
        for ib1 in range(n_band):
            for ik in range(nk):
                fft_grid = np.exp(1j * np.dot(np.squeeze(r_grid[:, ib1, ib2, :]), kmesh[0:3, ik])) / r_weights
                hk[ik, ib1, ib2] = np.sum(fft_grid * hr[:, ib1, ib2])
    return hk


# ==================================================================================================================

# ==================================================================================================================
def convham(hr=None, r_grid=None, r_weights=None, kmesh=None):
    '''
        Builds the k-space LDA-Hamiltonian from the real-space one.
        hk (Nk,Nbands,Nbands)
    '''

    fft_grid = np.exp(1j * np.matmul(r_grid, kmesh)) / r_weights[:, None, None]
    hk = np.transpose(np.sum(fft_grid * hr[..., None], axis=0), axes=(2, 0, 1))
    return hk


# ==================================================================================================================

def light_vertex(hr=None, r_grid=None, r_weights=None, kmesh=None, der=0):
    '''
        Builds the light vertex del e_k/del k from the real-space Hamiltonian.
    '''

    fft_grid = np.exp(1j * np.matmul(r_grid, kmesh)) / r_weights[:, None, None] * 1j * r_grid[:, :, :, der][:, :, :, None]
    hk = np.transpose(np.sum(fft_grid * hr[..., None], axis=0), axes=(2, 0, 1))
    return hk


# ==================================================================================================================

def write_hk_wannier90(hk, fname, kmesh, nk):
    '''
        Writes a Hamiltonian to a text file in wannier90 style.
    '''

    # .reshape(nk)
    # write hamiltonian in the wannier90 format to file
    f = open(fname, 'w', encoding='utf-8')
    n_orb = np.shape(hk)[-1]
    # header: no. of k-points, no. of wannier functions(bands), no. of bands (ignored)
    print(np.prod(nk), n_orb, n_orb, file=f)
    for ik in range(np.prod(nk)):
        print(kmesh[0, ik], kmesh[1, ik], kmesh[2, ik], file=f)
        hk_slice = np.copy(hk[ik, ...])
        np.savetxt(f, hk_slice.view(float), fmt='%.12f', delimiter=' ', newline='\n', header='', footer='', comments='#')

    f.close()


if __name__ == '__main__':
    pass
