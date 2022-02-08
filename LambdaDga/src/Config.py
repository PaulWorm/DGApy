# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Config object that contains configuration options and handles the task distribution.



# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import MatsubaraFrequencies as mf
import BrillouinZone as bz

# ----------------------------------------------- CLASSES --------------------------------------------------------------
class ConfigBase():
    '''
        Base config class.
    '''

    def set(self, **kwargs):
        self.__dict__.update(kwargs)

    def as_dict(self):
        return self.__dict__

class BoxSizes(ConfigBase):
    '''
        Contains the grid size parameters for a DGA run.
    '''

    def __init__(self):
        self._niw_core = None # Number of bosonic Matsubaras for Gamma
        self._niv_core = None # Number of fermionic Matsubaras for Gamma
        self._niv_urange = None # Number of fermionic Matsubaras for Gamma = U
        self.niw_urange = None # Number of bosonic Matsubaras for Gamma = U. For this range DGA reduces to RPA
        self.niv_invbse = None # Number of frequencies used for the inversion to obtain Gamma-local
        self._niv_asympt = None # Number of frequencies for the asymptotic chi = chi_0 range.
        self._niv_dmft = None # Number of fermionic Matsubaras for the DMFT self-energy
        self.niw_vrg_save = None # Number of bosonic Matsubaras for saving the spin-fermion vertex
        self.niv_vrg_save = None # Number of fermionic Matsubaras for saving the spin-fermion vertex
        self.nk = None # (nkx,nky,nkz) tuple of linear momenta. Used for fermionic quantities
        self.nq = None # (nkx,nky,nkz) tuple of linear momenta. Used for bosonic quantities

        # Corresponding grids:
        self.vn_dmft = None
        self.vn_core = None
        self.vn_urange = None
        self.vn_asympt = None
        self.wn_core_plus = None

    def set(self):
     raise NotImplementedError('This method is not implemented for the BoxSizes object.')

    @property
    def niw_core(self):
        return self._niw_core

    @niw_core.setter
    def niw_core(self, value):
        self._niw_core = value
        self.wn_core = mf.wn(n=self.niw_core)
        self.wn_core_plus = mf.wn_plus(n=self.niw_core)

    @property
    def niv_core(self):
        return self._niw_core

    @niw_core.setter
    def niv_core(self, value):
        self._niv_core = value
        self.vn_core = mf.vn(n=self.niv_core)

    @property
    def niv_dmft(self):
        return self._niv_dmft

    @niv_dmft.setter
    def niv_dmft(self,value):
        self._niv_dmft = value
        self.vn_dmft = mf.vn(n=self.niv_dmft)

    @property
    def niv_urange(self):
        return self._niv_urange

    @niv_urange.setter
    def niv_urange(self, value):
        self._niv_urange = value
        self.vn_urange = mf.vn(n=self.niv_urange)

    @property
    def niv_asympt(self):
        return self._niv_asympt

    @niv_urange.setter
    def niv_asympt(self, value):
        self._niv_asympt = value
        self.vn_asympt = mf.vn(n=self.niv_asympt)

    @property
    def wn_rpa(self):
        assert self.niw_urange is not None, "niw_core is None, but must have a value >= 0"
        assert self.niw_core is not None, "niw_core is None, but must have a value >= 0"
        return mf.wn_outer(n_core=self.niw_core, n_outer=self.niw_urange)

    @property
    def wn_rpa_plus(self):
        assert self.niw_urange is not None, "niw_core is None, but must have a value >= 0"
        assert self.niw_core is not None, "niw_core is None, but must have a value >= 0"
        return mf.wn_outer_plus(n_core=self.niw_core, n_outer=self.niw_urange)

    @property
    def niv_pp(self):
        '''Number of fermionic Matsubaras for the singlet/triplet vertex'''
        return np.min((self.niw_core // 2, self.niv_core // 2))

    #@property

    # def set_niw_core(self, value):
    #     self.niw_core = value
    #     self.wn_core = mf.wn(n=self.box.niw_core)
    #     # Set Matsubara grids:
    #     self.vn_dmft = mf.vn(n=self.box.niv_dmft)
    #     self.vn_core = mf.vn(n=self.v.niv_core)
    #     self.vn_urange = mf.vn(n=self.box.niv_urange)
    #     self.vn_asympt = mf.vn(n=self.box.niv_asympt)
    #
    #     self.wn_core_plus = mf.wn_plus(n=self.box.niw_core)
    #     self.wn_rpa = mf.wn_outer(n_core=self.box.niw_core, n_outer=self.box.niw_urange)
    #     self.wn_rpa_plus = mf.wn_outer_plus(n_core=self.box.niw_core, n_outer=self.box.niw_urange)


class Names(ConfigBase):
    '''
        Contains the name strings for several input and output files and paths.
    '''

    def __init__(self):
        self.input_path = None # Input path where the input files are located
        self.output_path = None # Path where the output is written to
        self.output_path_sp = None # Path where the spin-fermion vertex and it's plots are written to
        self.output_path_pf = None # Path where the poly-fit output is written to
        self.output_path_ac = None # Path where the ana-cont output is written to
        self.fname_g2 = 'g4iw_sym.hdf5' # File containing the DMFT two-particle vertex
        self.fname_dmft = '1p-data.hdf5' # File containing the DMFT one-particle run
        self.fname_ladder_vertex = None # Name for the ladder vertex

class Options(ConfigBase):
    '''
        Contains flags and options.
    '''

    def __init__(self):
        self.do_pairing_vertex = False # Calculate the superconducting pairing vertex
        self.lambda_correction_type = 'sp' # Type of lambe correction to be used
        self.lc_use_only_positive = True # Restrict lambda correction to the frequencies where the DMFT susceptibility is positive
        self.use_urange_for_lc = False # Use the Gamma = U range for the lambda correction. WARNING: This should most certainly not be used
        self.sym_sing = True # symmetrize the singlet vertex
        self.sym_trip = True # symmetrize the triplet vertex
        self.analyse_spin_fermion_contributions = False # Compute the contributions from Re(vrg) and Im(vrg) seperately
        self.use_fbz = False # Use the full Brillouin zone instead of the irreduzible one
        self.g2_input_type = 'w2dyn' # Type of loading the g2 function as input. Available is ['w2dyn']
        self.g1_input_type = 'w2dyn' # Type of loading the g2 function as input. Available is ['w2dyn']

class SystemParamter(ConfigBase):
    '''
        System specific parameters
    '''

    def __init__(self):
        self.n = None # Average number of particles per spin-orbital
        self.beta = None # Inverse temperature
        self.hr = None # Real-space kinetic hamiltonian
        self.u = None # Hubbard interaction 'U'
        self.t = None # next-nearest neighbour hopping 't'. Sets the unit of energy

class DgaConfig(ConfigBase):
    '''
        Contains the configuration parameters and flags for a DGA run.
    '''

    def __init__(self, BoxSizes: BoxSizes=None, Names: Names=None, Options: Options=None, SystemParameter: SystemParamter=None, ek_funk = None):
        self.box = BoxSizes
        self.nam = Names
        self.opt = Options
        self.sys = SystemParameter
        self.k_grid = None # Kgrid object for handling operations in the Brillouin zone
        self.q_grid = None # same as k_grid but for bosonic objects
        self.ek_func = ek_funk # Function to use for obtaining e(k)from H(r)

        self.dec = 11 # Number of decimals for ek check

        # Set the k and q grid:
        self.k_grid = self.set_kgrid()
        self.q_grid = self.set_kgrid()

    # Forwarding attributes (kinda polluted the name space, but there should be no reason that it creates errors):
    def __getattr__(self, attr):
        if hasattr(self.box, attr):
            return getattr(self.box, attr)
        elif(hasattr(self.nam, attr)):
            return getattr(self.nam, attr)
        elif(hasattr(self.opt, attr)):
            return getattr(self.opt, attr)
        else:
            return getattr(self.sys, attr)

    def set_kgrid(self):
        k_grid = bz.KGrid(nk=self.box.nk)
        if(self.opt.use_fbz):
            ek = self.ek_func(kgrid=k_grid.grid, hr=self.sys.hr)
            k_grid.get_irrk_from_ek(ek=ek, dec=self.dec)
        else:
            k_grid.set_irrk2fbz()
        return k_grid



        # self.vn_dmft = None # Fermionic Matsubara frequencies for DMFT input
        # self.vn_core = None # Fermionic Matsubara frequencies for

    # def set_grids(self):
    #     self.vn_dmft = mf.vn(n=self.box_sizes.niv_dmft)
    #     self.vn_core = mf.vn(n=self.box_sizes.niv_core)
    #     self.vn_urange = mf.vn(n=self.box_sizes.niv_urange)
    #     self.vn_asympt = mf.vn(n=self.box_sizes.niv_asympt)
    #     self.wn_core = mf.wn(n=self.box_sizes.niw_core)
    #     self.wn_core_plus = mf.wn_plus(n=self.box_sizes.niw_core)
    #     self.wn_rpa = mf.wn_outer(n_core=self.box_sizes.niw_core, n_outer=self.box_sizes.niw_urange)
    #     self.wn_rpa_plus = mf.wn_outer_plus(n_core=self.box_sizes.niw_core, n_outer=self.box_sizes.niw_urange)






if __name__=='__main__':

    box_sizes = BoxSizes()
    box_sizes.set(niv_urange = 10, niw_urange = 20, _niw_core = 10)
    bs_dict = box_sizes.as_dict()

    #box_sizes.niw_core =

