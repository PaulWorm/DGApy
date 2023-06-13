# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Config object that contains configuration options and handles the task distribution.


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import numpy as np
import MatsubaraFrequencies as mf
import BrillouinZone as bz
import OmegaMeshes as omesh
import argparse
from typing import List, Tuple
import Hr as hamr


# ----------------------------------------------- ARGUMENT PARSER ------------------------------------------------------

def create_dga_argparser():
    ''' Set up an argument parser for the DGA code. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', default='./dga_config.yaml', type=str, help=' File and location of the config '
                                                                                         'file. ')
    return parser


# ----------------------------------------------- CLASSES --------------------------------------------------------------
class ConfigBase():
    '''
        Base config class.
    '''

    def update_dict(self, **kwargs):
        self.__dict__.update(kwargs)

    def set_from_obj(self,obj):
        for key in self.__dict__.keys():
            if hasattr(obj, key):
                setattr(self,key,obj.__dict__[key])
        return None

    def as_dict(self):
        return self.__dict__

    def set(self,obj):
        if(type(obj) == dict):
            self.update_dict(**obj)
        else:
            self.set_from_obj(obj)
        return None


class BoxSizes(ConfigBase):
    '''
        Contains the grid size parameters for a DGA run.
    '''

    def __init__(self, config_dict=None):
        self.niw_core: int = -1  # Number of bosonic Matsubaras for Gamma
        self.niv_core: int = -1  # Number of fermionic Matsubaras for Gamma
        self.niv_shell: int = 0  # Number of fermionic Matsubaras for Gamma = U

        if(config_dict is not None):
            self.update_dict(**config_dict) # forwards parameters from config

    @property
    def niv_full(self):
        return self.niv_shell + self.niw_core

    @property
    def vn_full(self):
        return mf.vn(n=self.niv_full)

    @property
    def niv_asympt(self):
        return self.niv_full * 2 + self.niw_core * 2

    @property
    def niv_pp(self):
        '''Number of fermionic Matsubaras for the singlet/triplet vertex'''
        return np.min((self.niw_core // 2, self.niv_core // 2))

class LatticeConfig(ConfigBase):
    '''
        Contains the information about the Lattice and Brillouin zone
    '''

    def __init__(self, config_dict):
        self.nk: Tuple[int,int,int] = (16,16,1)  # (nkx,nky,nkz) tuple of linear momenta. Used for fermionic quantities
        self.nq: Tuple[int,int,int] = (16,16,1)  # (nkx,nky,nkz) tuple of linear momenta. Used for bosonic quantities
        self.symmetries = []                   # Lattice symmetries. Either string or tuple of strings
        self.tb_params = None                  # Tight binding parameter. Loading Hr will maybe be implemented later.
        self.type = None

        self.update_dict(**config_dict) # forwards parameters from config
        if('nq' not in config_dict):
            print('Notification: nq not set in config. Setting nq = nk')
            self.nq = self.nk

        self.check_symmetries()

        # set k-grids:
        self.nk = tuple(self.nk)
        self.nq = tuple(self.nq)

        self.k_grid = bz.KGrid(self.nk,self.symmetries)
        self.q_grid = bz.KGrid(self.nq,self.symmetries)


        if(self.tb_params is None):
            raise ValueError('tb_params connot be none. Tight-bindign parameters must be supplied.')


    def check_symmetries(self):
        ''' Set symmetries if known: '''
        if(self.symmetries == "two_dimensional_square"):
            self.symmetries = bz.two_dimensional_square_symmetries()

    @property
    def nk_tot(self):
        return np.prod(self.nk)

    @property
    def nq_tot(self):
        return np.prod(self.nq)

    def set_hr(self):
        ''' Return the tight-binding hamiltonian.'''
        if(self.tb_params is not None):
            if(self.type == 't_tp_tpp'):
                return hamr.one_band_2d_t_tp_tpp(*self.tb_params)
        else:
            raise  NotImplementedError('Currently only t_tp_tpp tight-binding model implemented.')


class Names(ConfigBase):
    '''
        Contains the name strings for several input and output files and paths.
    '''

    def __init__(self):
        self.input_path = None  # Input path where the input files are located
        self.output_path = None  # Path where the output is written to
        self.output_path_sp = None  # Path where the spin-fermion vertex and it's plots are written to
        self.output_path_pf = None  # Path where the poly-fit output is written to
        self.output_path_ac = None  # Path where the ana-cont output is written to
        self.output_path_el = None  # Path where the Eliashberg output is written to
        self.fname_g2 = 'g4iw_sym.hdf5'  # File containing the DMFT two-particle vertex
        self.fname_dmft = '1p-data.hdf5'  # File containing the DMFT one-particle run
        self.fname_ladder_vertex = None  # Name for the ladder vertex


class Options(ConfigBase):
    '''
        Contains flags and options.
    '''

    def __init__(self):
        self.do_pairing_vertex = False  # Calculate the superconducting pairing vertex
        self.do_max_ent_loc = True  # Perform the analytic continuation for the local Green's function
        self.do_max_ent_irrk = True  # Perform the analytic continuation for the Green's function in the irreduzible BZ
        self.lambda_correction_type = 'sp'  # Type of lambe correction to be used
        self.lc_use_only_positive = True  # Restrict lambda correction to the frequencies where the DMFT susceptibility is positive
        self.use_urange_for_lc = False  # Use the Gamma = U range for the lambda correction. WARNING: This should most certainly not be used
        self.sym_sing = True  # symmetrize the singlet vertex
        self.sym_trip = True  # symmetrize the triplet vertex
        self.analyse_spin_fermion_contributions = False  # Compute the contributions from Re(vrg) and Im(vrg) seperately
        self.analyse_w0_contribution = False  # Analyze how much stems from the w0 component.
        self.use_fbz = False  # Use the full Brillouin zone instead of the irreduzible one
        self.g2_input_type = 'w2dyn'  # Type of loading the g2 function as input. Available is ['w2dyn']
        self.g1_input_type = 'w2dyn'  # Type of loading the g2 function as input. Available is ['w2dyn']
        self.use_gloc_dmft = False  # Use DMFT output local Greens function


class SystemParamter(ConfigBase):
    '''
        System specific parameters
    '''

    def __init__(self):
        self.n = None  # Average number of particles per spin-orbital
        self.beta = None  # Inverse temperature
        self.hr = None  # Real-space kinetic hamiltonian
        self.u = None  # Hubbard interaction 'U'
        self.t = None  # next-nearest neighbour hopping 't'. Sets the unit of energy
        self.mu_dmft = None  # DMFT chemical potential


class MaxEntConfig(ConfigBase):
    '''
        Settings for the analytical continuation.
    '''

    def __init__(self, t=1.0, beta=None, mesh_type='lorentzian', cut=0.04, nwr=501, wmax=15, err=1e-2):
        self.cut = cut  # Relevant for the lorentzian mesh
        self.t = t  # Unit of energy. Everything is scaled by this. We use t for this here.
        self.nwr = nwr  # Number of frequencies on the real axis
        self.wmax = wmax * self.t  # frequency range on the real axis
        self.use_preblur = True  # whether o not to use the pre-blur feature
        self.err = err  # Error for the analytical continuation
        self.beta = beta  # Inverse Temperature
        self.alpha_det_method = 'chi2kink'  # alpha determination method
        self.optimizer = 'newton'  # alpha determination method
        self.mesh_type = mesh_type  # mesh type

    @property
    def mesh_type(self):
        return self._mesh_type

    @mesh_type.setter
    def mesh_type(self, value):
        self._mesh_type = value
        self.mesh = self.get_omega_mesh()

    def get_omega_mesh(self):
        if (self.mesh_type == 'lorentzian'):
            return omesh.LorentzianOmegaMesh(omega_min=-self.wmax, omega_max=self.wmax, n_points=self.nwr, cut=self.cut)
        elif self.mesh_type == 'hyperbolic':
            return omesh.HyperbolicOmegaMesh(omega_min=-self.wmax, omega_max=self.wmax, n_points=self.nwr)
        elif self.mesh_type == 'linear':
            return np.linspace(-self.wmax, self.wmax, self.nwr)
        elif self.mesh_type == 'tan':
            return np.tan(np.linspace(-np.pi / 2.5, np.pi / 2.5, num=self.nwr, endpoint=True)) * self.wmax / np.tan(np.pi / 2.5)
        else:
            raise ValueError('Unknown omega mesh type.')

    def get_n_fit_opt(self, n_fit_min=None, n_fit_max=None):
        ''' Returns an estimate for the optimal value for n_fit  '''
        return np.min((np.max((n_fit_min, int(self.beta * 4 * self.t))), n_fit_max))

    def get_bw_opt(self):
        ''' Return an estimate for the optimal blur width value '''
        return np.pi / (self.beta * self.t)

    def dump_settings_to_txt(self, fname='AnaContSettings', nfit=-1):
        text_file = open(fname + '.txt', 'w')
        text_file.write(f'nfit={nfit} \n')
        text_file.write(f'err={self.err} \n')
        text_file.write(f'wmax={self.wmax} \n')
        text_file.write(f'nwr={self.nwr} \n')
        text_file.close()


class EliashbergConfig(ConfigBase):
    '''
        Contains the configuration parameters and flags for the Eliashberg routine.
    '''

    def __init__(self, n_eig=2, k_sym='d-wave'):
        self.gap0_sing = None  # Initial guess for the singlet gap function
        self.gap0_trip = None  # Initial guess for the triplet gap function
        self.n_eig = n_eig  # Number of eigenvalues to be computed
        self.k_sym = k_sym  # k-symmetry of the gap function
        self.sym_sing = True  # Symmetrize the singlet pairing vertex
        self.sym_trip = True  # symmetrize the triplet pairing vertex

    @property
    def k_sym(self):
        return self._k_sym

    @k_sym.setter
    def k_sym(self, value):
        self._k_sym = value
        self.set_gap0()

    def set_gap0(self):
        self.set_gap0_sing()
        self.set_gap0_trip()

    def set_gap0_sing(self):
        if (self.k_sym != 'p-wave'):
            v_sym = 'even'
        else:
            v_sym = 'odd'
        self.gap0_sing = {
            'k': self.k_sym,
            'v': v_sym
        }

    def set_gap0_trip(self):
        if (self.k_sym != 'p-wave'):
            v_sym = 'odd'
        else:
            v_sym = 'even'
        self.gap0_trip = {
            'k': self.k_sym,
            'v': v_sym
        }


class DgaConfig(ConfigBase):
    '''
        Contains the configuration parameters and flags for a DGA run.
    '''

    box_sizes: BoxSizes = None
    lattice_conf: LatticeConfig = None
    output_path: str = None
    do_poly_fitting: bool = False

    def __init__(self, conf_file):
        # Create config file:
        self.build_box_sizes(conf_file)
        self.build_lattice_conf(conf_file)

        # Optional configs, only set if contained in config file:
        # Polyfitting:
        self.n_fit = 4
        self.o_fit = 3
        if('poly_fitting' in conf_file):
            self.do_poly_fitting = True
            self.update_dict(**conf_file['poly_fitting'])

        # Set input parameters:
        self.input_type = 'w2dyn'
        self.input_path = './'
        self.fname_1p = '1p-data.hdf5'
        self.fname_2p = 'g4iw_sym.hdf5'

        self.update_dict(**conf_file['dmft_input'])

        # Set dga routine specifications:
        self.lambda_corr = 'spch'
        self.update_dict(**conf_file['dga'])


    def build_box_sizes(self, conf_file):
        self.box_sizes = BoxSizes()
        if ('box_sizes' in conf_file):
            self.box_sizes.update_dict(**conf_file['box_sizes'])

    def build_lattice_conf(self, conf_file):

        if ('lattice' in conf_file):
            self.lattice_conf = LatticeConfig(conf_file['lattice'])
        else:
            raise ValueError('Lattice must be contained in the config file.')


if __name__ == '__main__':
    box_sizes = BoxSizes()
    box_sizes.set(niv_urange=10, niw_urange=20, _niw_core=10)
    bs_dict = box_sizes.as_dict()

    # box_sizes.niw_core =
