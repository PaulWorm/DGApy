'''
    This module contains routines for wannier and tight-binding Hamiltonians.
    It builds upon the brilloun_zone.py module.
    Simple t-tp-tpp models as well as reading wannier90 Hr files is supported.
'''
import numpy as np
import pandas as pd
from warnings import warn

import dga.brillouin_zone as bz


class WannierHr():
    '''
        class to handle wannier Hamiltonians
    '''
    def __init__(self, hr, r_grid, r_weights, orbs):
        self.hr = hr
        self.r_grid = r_grid
        self.r_weights = r_weights
        self.orbs = orbs

    def get_ek(self,k_grid: bz.KGrid):
        ek = convham2(self.hr,self.r_grid,self.r_weights,k_grid.kmesh.reshape(3,-1))
        n_orbs = ek.shape[-1]
        return ek.reshape(*k_grid.nk,n_orbs,n_orbs)

    def get_ek_one_band(self, k_grid: bz.KGrid):
        return self.get_ek(k_grid)[:,:,:,0,0]

    def save_hr(self,path,name='wannier_hr.dat'):
        ''' save the Hamiltonian to file.'''
        write_hr_w2k(path+name,self.hr,self.r_grid,self.r_weights,self.orbs)

def create_wannier_hr_from_file(fname):
    ''' Reads a wannier90 file and creates an instance of the Wannier90 class.'''
    return WannierHr(*read_hr_w2k(fname))

# --------------------------------------- CONSTRUCT REAL SPACE HAMILTONIANS --------------------------------------------

def wannier_one_band_2d_t_tp_tpp(t,tp,tpp):
    hr = -np.array([t,t,t,t,tp,tp,tp,tp,tpp,tpp,tpp,tpp])[:,None,None]
    r_grid= np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],
                     [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],
                     [2,0,0],[0,2,0],[-2,0,0],[0,-2,0]])[:,None,None,:]
    r_weights = np.ones((12,1))
    orbs = np.ones((12,1,1,2))
    return hr, r_grid, r_weights, orbs

def one_band_2d_t_tp_tpp(t=1.0, tp=0., tpp=0.):
    return np.array([[t, t, 0], [tp, tp, 0.], [tpp, tpp, 0]])


def one_band_2d_quasi1D(tx=1.0, ty=0, tppx=0, tppy=0, tpxy=0):
    return np.array([[tx, ty, 0], [tpxy, tpxy, 0.], [tppx, tppy, 0]])


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

    return one_band_2d_quasi1D(tx=tx, ty=ty, tppx=tppx, tppy=tppy, tpxy=tpxy)


def Ba2CuO4_plane_2D_projection():
    # Ba2CuO3.25 2D-projection parameters
    tx = 0.0258
    ty = 0.5181
    tpxy = 0.0119
    tppx = -0.0014
    tppy = 0.0894
    return one_band_2d_quasi1D(tx=tx, ty=ty, tppx=tppx, tppy=tppy, tpxy=tpxy)


# ==================================================================================================================



def read_hr_w2k(fname):
    '''
        Load the H(R) LDA-Hamiltonian from a wien2k hr file.
    '''
    Hr_file = pd.read_csv(fname, skiprows=1, names=np.arange(15), sep='\s+', dtype=float, engine='python')
    Nbands = Hr_file.values[0].astype(int)[0]
    Nr = Hr_file.values[1].astype(int)[0]

    tmp = np.reshape(Hr_file.values, (np.size(Hr_file.values), 1))
    tmp = tmp[~np.isnan(tmp)]

    r_weights = tmp[2:2 + Nr].astype(int)
    r_weights = np.reshape(r_weights, (np.size(r_weights), 1))
    Ns = 7
    Ntmp = np.size(tmp[2 + Nr:]) // Ns
    tmp = np.reshape(tmp[2 + Nr:], (Ntmp, Ns))

    r_grid = np.reshape(tmp[:, 0:3], (Nr, Nbands, Nbands, 3))
    orbs = np.reshape(tmp[:, 3:5], (Nr, Nbands, Nbands, 2))
    hr = np.reshape(tmp[:, 5] + 1j * tmp[:, 6], (Nr, Nbands, Nbands))
    # hr_dict = {
    #     'hr': hr,
    #     'r_grid': r_grid,
    #     'r_weights': r_weights,
    #     'orbs': orbs
    # }
    # return hr_dict
    return hr, r_grid, r_weights, orbs


# ==================================================================================================================

def write_hr_w2k(fname, hr, r_grid, r_weights, orbs):
    '''
        Write a real-space Hamiltonian in the format of w2k to a file.
    '''
    n_columns = 15
    n_r = hr.shape[0]
    n_bands = hr.shape[-1]
    file = open(fname, 'w')
    file.write(f'# Written using the wannier module of the dga code\n')
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

# ------------------------------------------------ OBJECTS -------------------------------------------------------------

def ek_square(kx=None, ky=None, t=1.0, tp=0.0, tpp=0.0):
    return - 2. * t * (np.cos(kx) + np.cos(ky)) - 4. * tp * np.cos(kx) * np.cos(ky) \
           - 2. * tpp * (np.cos(2. * kx) + np.cos(2. * ky))


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
    kx = kgrid[:,0]
    ky = kgrid[:,1]
    kz = kgrid[:,2]
    ek = - 2.0 * (hr[0, 0] * np.cos(kx) + hr[0, 1] * np.cos(ky) + hr[0, 2] * np.cos(kz))
    ek = ek - 2.0 * (hr[1, 0] * np.cos(kx + ky) + hr[1, 1] * np.cos(kx - ky))
    ek = ek - 2.0 * (hr[2, 0] * np.cos(2. * kx) + hr[2, 1] * np.cos(2. * ky) + hr[2, 2] * np.cos(kz))
    return ek


# ==================================================================================================================
def read_Hk_w2k(fname, spin_sym = True):
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
    hk_file = open(fname,'r')
    spin_orbit = not spin_sym
    def nextline():
        line = hk_file.readline()
        return line[:line.find('#')].split()

    # parse header
    header = nextline()
    if header[0] == 'VERSION':
        warn("Version 2 headers are obsolete (specify in input file!)")
        nkpoints, natoms = map(int, nextline())
        lines = np.array([nextline() for _ in range(natoms)], np.int)
        nbands = np.sum(lines[:, :2])
        del lines, natoms
    elif len(header) != 3:
        warn("Version 1 headers are obsolete (specify in input file!)")
        header = list(map(int, header))
        nkpoints = header[0]
        nbands = header[1] * (header[2] + header[3])
    else:
        nkpoints, nbands, _ = map(int, header)
    del header

    # nspins is the spin dimension for H(k); G(iw), Sigma(iw) etc. will always
    # be spin-dependent
    if spin_orbit:
        if nbands % 2: raise RuntimeError("Spin-structure of Hamiltonian!")
        nbands //= 2
        nspins = 2
    else:
        nspins = 1
    # GS: inside read_hamiltonian nspins is therefore equal to 1 if spin_orbit=0
    # GS: outside nspins is however always set to 2. Right?
    # GS: this also means that we have to set nspins internally in read_ImHyb too

    hk_file.flush()

    # parse data
    hk = np.fromfile(hk_file, sep=" ")
    hk = hk.reshape(-1, 3 + 2 * nbands ** 2 * nspins ** 2)
    kpoints_file = hk.shape[0]
    if kpoints_file > nkpoints:
        warn("truncating Hk points")
    elif kpoints_file < nkpoints:
        raise ValueError("problem! %d < %d" % (kpoints_file, nkpoints))
    kpoints = hk[:nkpoints, :3]

    # TODO: check with Martin if this the actual spin structure ...
    hk = hk[:nkpoints, 3:].reshape(nkpoints, nspins, nbands, nspins, nbands, 2)
    hk = hk[..., 0] + 1j * hk[..., 1]
    if not np.allclose(hk, hk.transpose(0, 3, 4, 1, 2).conj()):
        warn("Hermiticity violation detected in Hk file")

    # go from wannier90/convert_Hamiltonian structure to our Green's function
    # convention
    hk = hk.transpose(0, 2, 1, 4, 3)
    hk_file.close()

    Hk = np.squeeze(hk)
    kmesh = np.concatenate((kpoints,np.ones((np.shape(kpoints)[0],1))),1)

    return Hk, kmesh
# ==================================================================================================================





# ==================================================================================================================
def convham(Hr = None, Rgrid = None, Rweights = None, kmesh = None):
    '''
        Builds the k-space LDA-Hamiltonian from the real-space one.
        H(k) is constructed within the ir-BZ
        Hk (Nk,Nbands,Nbands)
    '''

    Nk = np.size(kmesh,1)
    Nband = np.size(Hr,1)
    Hk = np.zeros((Nk, Nband, Nband), dtype = complex)
    Rweights = np.squeeze(Rweights)
    for ib2 in range(Nband):
        for ib1 in range(Nband):
            for ik in range(Nk):
                FFTGrid = np.exp(1j * np.dot(np.squeeze(Rgrid[:,ib1,ib2,:]),kmesh[0:3,ik])) / Rweights
                Hk[ik,ib1,ib2] = np.sum(FFTGrid * Hr[:,ib1,ib2])
    return Hk
# ==================================================================================================================

# ==================================================================================================================
def convham2(Hr = None, Rgrid = None, Rweights = None, kmesh = None):
    '''
        Builds the k-space LDA-Hamiltonian from the real-space one.
        Hk (Nk,Nbands,Nbands)
    '''

    FFTGrid = np.exp(1j * np.matmul(Rgrid,kmesh)) / Rweights[:,None,None]
    Hk = np.transpose(np.sum(FFTGrid * Hr[...,None], axis=0),axes=(2,0,1))
    return Hk
# ==================================================================================================================

# ==================================================================================================================
def write_hk_wannier90(Hk, fname, kmesh, nk):
    '''
        Writes a Hamiltonian to a text file in wannier90 style.
    '''

    # .reshape(nk)
    # write hamiltonian in the wannier90 format to file
    f = open(fname, 'w')
    n_orb = np.shape(Hk)[-1]
    # header: no. of k-points, no. of wannier functions(bands), no. of bands (ignored)
    print(np.prod(nk), n_orb, n_orb, file=f)
    for ik in range(np.prod(nk)):
            print(kmesh[0, ik], kmesh[1, ik], kmesh[2, ik], file=f)
            np.savetxt(f, Hk[ik,...].view(float), fmt='%.12f',delimiter=' ', newline='\n', header='', footer='', comments='#')

    f.close()

if __name__ == '__main__':
    path = '../../test/TestHrAndHkFiles/'
    # fname = '1Band_t_tp_tpp_hr.dat'
    fname = '1onSTO-2orb_hr.dat'

    # Hr_file = pd.read_csv(path+fname, skiprows=1, names=np.arange(15), sep='\s+', dtype=float, engine='python')
    hr, r_grid, r_weights, orbs = read_hr_w2k(path+fname)

    fname_write = fname.split(sep='.')[0] + '_rewrite.dat'
    write_hr_w2k(path+fname_write,hr, r_grid, r_weights, orbs)

    hr_2, r_grid_2, r_weights_2, orbs_2 = read_hr_w2k(path + fname)

    assert np.allclose(hr,hr_2)
    assert np.allclose(r_grid,r_grid_2)
    assert np.allclose(r_weights,r_weights_2)
    assert np.allclose(orbs,orbs_2)

    path = '../../test/TestHrAndHkFiles/'
    fname = '1Band_t_tp_tpp_hr.dat'

    t,tp,tpp = 0.389093, -0.097869, 0.046592
    e0 = 0.267672


    nk = (16,16,1)
    k_grid = bz.KGrid(nk=nk)
    ham_r = create_wannier_hr_from_file(path+fname)
    hk_from_hr = ham_r.get_ek_one_band(k_grid)

    ham_r_tb = WannierHr(*wannier_one_band_2d_t_tp_tpp(t,tp,tpp))
    hk_direct = ham_r_tb.get_ek_one_band(k_grid)

    import matplotlib.pyplot as plt
    import dga.plotting as plotting

    fig, axes = plt.subplots(1, 3, dpi=251, figsize=(13, 5))
    im = axes[0].imshow(hk_from_hr[:,:,0].real,cmap='RdBu')
    plotting.insert_colorbar(axes[0],im)
    im = axes[1].imshow(hk_direct[:,:,0].real,cmap='RdBu')
    plotting.insert_colorbar(axes[1],im)
    im = axes[2].imshow(hk_from_hr[:,:,0].real-hk_direct[:,:,0].real,cmap='RdBu')
    plotting.insert_colorbar(axes[2],im)
    plt.tight_layout()
    plt.show()