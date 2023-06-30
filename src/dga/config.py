# ------------------------------------------------ COMMENTS ------------------------------------------------------------
# Config object that contains configuration options and handles the task distribution.


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import os
import numpy as np
import dga.matsubara_frequencies as mf
import dga.brillouin_zone as bz
import dga.omega_meshes as omesh
import argparse
from typing import List, Tuple
import dga.hr as hamr
import matplotlib
import dga.util as util
import mpi4py.MPI as mpi
from ruamel.yaml import YAML

# ----------------------------------------------- ARGUMENT PARSER ------------------------------------------------------

DGA_OUPUT_PATH = '/LambdaDga_lc_{}_Nk{}_Nq{}_wcore{}_vcore{}_vshell{}'


def create_dga_argparser(name='dga_config.yaml', path=os.getcwd() + '/'):
    ''' Set up an argument parser for the DGA code. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', default=name, type=str, help=' Config file name. ')
    parser.add_argument('--path', nargs='?', default=path, type=str, help=' Path to the config file. ')
    return parser


def parse_config_file(comm: mpi.Comm):
    '''
        Parse config file and return the dga_config
    '''
    parser = create_dga_argparser()
    if (comm.rank == 0):
        args = parser.parse_args()
        assert hasattr(args, 'config'), 'Config file location must be provided.'
        conf_file = YAML().load(open(args.path + args.config))
    else:
        conf_file = None

    conf_file = comm.bcast(conf_file, root=0)
    dga_config = DgaConfig(conf_file,comm=comm)

    return dga_config, conf_file

def save_config_file(conf_file,output_path):
    with open(output_path + "/dga_config.yaml", "w+") as file:
        yaml = YAML()
        yaml.dump(conf_file, file)
        file.close()


# ----------------------------------------------- CLASSES --------------------------------------------------------------
class ConfigBase():
    '''
        Base config class.
    '''

    def update_dict(self, **kwargs):
        self.__dict__.update(kwargs)

    def set_from_obj(self, obj):
        for key in self.__dict__.keys():
            if hasattr(obj, key):
                setattr(self, key, obj.__dict__[key])
        return None

    def as_dict(self, obj=None):
        if obj is None:
            obj = self
        if not hasattr(obj, "__dict__"):
            return obj
        result = {}
        for key, val in obj.__dict__.items():
            if key.startswith("_"):
                continue
            element = []
            if isinstance(val, list):
                for item in val:
                    element.append(self.as_dict(item))
            else:
                element = self.as_dict(val)
            result[key] = element
        return result

    def set(self, obj):
        if (type(obj) == dict):
            self.update_dict(**obj)
        else:
            self.set_from_obj(obj)
        return None


class OutputConfig(ConfigBase):
    '''
        Builds ontop of the ConfigBase class and adds a method for automatic generation of output folders
    '''

    def __init__(self):
        self.output_path = None

    def set_output_path(self, base_name, comm=None):
        ''' comm is for mpi applications '''
        self.output_path = util.uniquify(base_name)

        if (not os.path.exists(self.output_path)):
            if (comm is not None):
                if (comm.rank == 0): os.mkdir(self.output_path)
            else:
                os.mkdir(self.output_path)

    def save_data(self, mat, name):
        np.save(self.output_path + '/' + name + '.npy', mat, allow_pickle=True)

    def load_data(self, name):
        try:
            data = np.load(self.output_path + '/' + name + '.npy', allow_pickle=True).item()
        except:
            data = np.load(self.output_path + '/' + name + '.npy', allow_pickle=True)
        return data


class BoxSizes(ConfigBase):
    '''
        Contains the grid size parameters for a DGA run.
    '''

    def __init__(self, config_dict=None):
        self.niw_core: int = -1  # Number of bosonic Matsubaras for Gamma
        self.niv_core: int = -1  # Number of fermionic Matsubaras for Gamma
        self.niv_shell: int = 0  # Number of fermionic Matsubaras for Gamma = U

        if (config_dict is not None):
            self.update_dict(**config_dict)  # forwards parameters from config

    @property
    def niv_full(self):
        return self.niv_shell + self.niv_core

    @property
    def vn_full(self):
        return mf.vn(n=self.niv_full)

    @property
    def niv_asympt(self):
        return self.niv_full + self.niw_core * 2

    @property
    def niv_pp(self):
        '''Number of fermionic Matsubaras for the singlet/triplet vertex'''
        return np.min((self.niw_core // 2, self.niv_core // 2))

    @property
    def wn(self):
        return mf.wn(self.niw_core)


class LatticeConfig(ConfigBase):
    '''
        Contains the information about the Lattice and Brillouin zone
    '''

    def __init__(self, config_dict):
        self.nk: Tuple[int, int, int] = (16, 16, 1)  # (nkx,nky,nkz) tuple of linear momenta. Used for fermionic quantities
        self.nq: Tuple[int, int, int] = (16, 16, 1)  # (nkx,nky,nkz) tuple of linear momenta. Used for bosonic quantities
        self.symmetries = []  # Lattice symmetries. Either string or tuple of strings
        self.tb_params = None  # Tight binding parameter. Loading Hr will maybe be implemented later.
        self.type = None

        self.update_dict(**config_dict)  # forwards parameters from config
        if ('nq' not in config_dict):
            print('Notification: nq not set in config. Setting nq = nk')
            self.nq = self.nk

        self.check_symmetries()

        # set k-grids:
        self.nk = tuple(self.nk)
        self.nq = tuple(self.nq)

        self.k_grid = bz.KGrid(self.nk, self.symmetries)
        self.q_grid = bz.KGrid(self.nq, self.symmetries)

        if (self.tb_params is None):
            raise ValueError('tb_params cannot be none. Tight-binding parameters must be supplied.')

        # Build the real-space Hamiltonian
        self.hr = self.set_hr()

    def check_symmetries(self):
        ''' Set symmetries if known: '''
        if (self.symmetries == "two_dimensional_square"):
            self.symmetries = bz.two_dimensional_square_symmetries()

    @property
    def nk_tot(self):
        return np.prod(self.nk)

    @property
    def nq_tot(self):
        return np.prod(self.nq)

    def set_hr(self):
        ''' Return the tight-binding hamiltonian.'''
        if (self.tb_params is not None):
            if (self.type == 't_tp_tpp'):
                return hamr.one_band_2d_t_tp_tpp(*self.tb_params)
        else:
            raise NotImplementedError('Currently only t_tp_tpp tight-binding model implemented.')


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


class MaxEntConfig(OutputConfig):
    '''
        Settings for the analytical continuation.
    '''

    def __init__(self, t, beta, config_dict, output_path_loc='./', output_path_nl_s='./', output_path_nl_g='./'):
        super().__init__()
        self.cut = 0.04  # Relevant for the lorentzian mesh
        self.t = t  # Unit of energy. Everything is scaled by this. We use t for this here.
        self.nwr = 501  # Number of frequencies on the real axis
        self.wmax = 15  # frequency range on the real axis
        self.use_preblur = True  # whether o not to use the pre-blur feature
        self.err = 1e-3  # Error for the analytical continuation
        self.beta = beta  # Inverse Temperature
        self.alpha_det_method = 'chi2kink'  # alpha determination method
        self.optimizer = 'newton'  # alpha determination method
        self.mesh_type = 'tan'  # mesh type
        self.n_fit = int(beta * 3 + 10)  # Number of frequencies used for the analytic continuation
        self.bw_fit_position = 10  # Fit position for estimating the optimal blur width
        self.bw_dga = [0.1, ]  # Blur width for DGA continuation

        # Flags what continuation to perform:
        self.cont_g_loc = True
        self.cont_s_nl = True
        self.cont_g_nl = False
        self.output_path_loc = output_path_loc
        self.output_path_nl_s = output_path_nl_s
        self.output_path_nl_g = output_path_nl_g
        self.bw_range_loc = np.array([0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1])

        self.update_dict(**config_dict)

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


class EliashbergConfig(OutputConfig):
    '''
        Contains the configuration parameters and flags for the Eliashberg routine.
    '''

    def __init__(self, config_dict=None):
        super().__init__()
        self.do_pairing_vertex = False  # Flag whether to compute the pairing vertex
        self.do_eliash = False  # Flag whether to perform the eliashberg routine afterwards
        self.gap0_sing = None  # Initial guess for the singlet gap function
        self.gap0_trip = None  # Initial guess for the triplet gap function
        self.n_eig = 2  # Number of eigenvalues to be computed
        self.k_sym = 'd-wave'  # k-symmetry of the gap function
        self.sym_sing = True  # Symmetrize the singlet pairing vertex
        self.sym_trip = True  # symmetrize the triplet pairing vertex
        if (config_dict is not None): self.update_dict(**config_dict)

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


class DgaConfig(OutputConfig):
    '''
        Contains the configuration parameters and flags for a DGA run.
    '''

    box_sizes: BoxSizes = None
    lattice: LatticeConfig = None
    eliash: EliashbergConfig = None
    output_path: str = None
    do_poly_fitting: bool = False
    poly_fit_dir: str = None

    def __init__(self, conf_file, comm: mpi.Comm = None):
        super().__init__()
        # Create config file:
        self.build_box_sizes(conf_file)
        self.build_lattice_conf(conf_file)
        self.build_eliash_conf(conf_file, comm)

        # Optional configs, only set if contained in config file:
        # Polyfitting:
        self.n_fit = 4
        self.o_fit = 3
        if ('poly_fitting' in conf_file):
            self.do_poly_fitting = True
            self.update_dict(**conf_file['poly_fitting'])

        # Set input parameters:
        self.input_type = 'w2dyn'
        self.input_path = './'
        self.fname_1p = '1p-data.hdf5'
        self.fname_2p = 'g4iw_sym.hdf5'

        if('dmft_input' in conf_file):
            self.update_dict(**conf_file['dmft_input'])

        # Set dga routine specifications:
        self.lambda_corr = 'spch'
        if('dga' in conf_file):
            self.update_dict(**conf_file['dga'])
            if('gui' not in conf_file['dga']):
                matplotlib.use('Agg')
            else:
                pass # Setting a non-Agg backend does not work on my machine. Why is unclear, but this should do the trick.
                # matplotlib.use(conf_file['dga']['gui'])
        else:
            matplotlib.use('Agg')  # non-gui backend. Particularly usefull for use on cluster.



    def build_box_sizes(self, conf_file):
        self.box_sizes = BoxSizes()
        if ('box_sizes' in conf_file):
            self.box_sizes.update_dict(**conf_file['box_sizes'])

    def build_lattice_conf(self, conf_file):

        if ('lattice' in conf_file):
            self.lattice = LatticeConfig(conf_file['lattice'])
        else:
            raise ValueError('Lattice must be contained in the config file.')

    def build_eliash_conf(self, conf_file, comm=None):
        self.eliash = EliashbergConfig()
        if ('pairing' in conf_file):
            self.eliash.update_dict(**conf_file['pairing'])
            self.eliash.set_output_path(self.output_path + 'Eliashberg/', comm)

    def create_dga_ouput_folder(self, comm=None):
        ''' Create the name of the dga output directory and create it '''
        base_name = self.input_path + DGA_OUPUT_PATH.format(self.lambda_corr, self.lattice.nk_tot,
                                                            self.lattice.nq_tot, self.box_sizes.niw_core,
                                                            self.box_sizes.niv_core, self.box_sizes.niv_shell)
        self.set_output_path(base_name, comm=comm)

    def create_poly_fit_folder(self,comm=None):
        ''' Create the output folder for the polynomial fits. '''
        if(self.do_poly_fitting):
            self.poly_fit_dir = self.output_path + '/PolyFits/'
            self.poly_fit_dir = util.uniquify(self.poly_fit_dir)

            # Create the folder
            if (not os.path.exists(self.poly_fit_dir)):
                if (comm is not None):
                    if (comm.rank == 0): os.mkdir(self.poly_fit_dir)
                else:
                    os.mkdir(self.poly_fit_dir)


def get_default_config_dict():

    conf_dict = {
        'box_sizes': {
            'niv_core': -1,
            'niw_core': -1,
            'niv_shell': 0,
        },
        'lattice': {
            'symmetries': 'two_dimensional_square',
            'type': 't_tp_tpp',
            'tb_params': [1.0, -0.2, 0.1],
            'nk': [24,24,1]
        },
        'dga': {
            'gui': 'TkAgg'
        }
    }
    return conf_dict


if __name__ == '__main__':
    box_sizes = BoxSizes()
    box_sizes.set(niv_urange=10, niw_urange=20, _niw_core=10)
    bs_dict = box_sizes.as_dict()

    # box_sizes.niw_core =
