# ------------------------------------------------ COMMENTS ------------------------------------------------------------
'''
    Config object that contains configuration options and handles the task distribution.
'''

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------
import os
import sys
from typing import Tuple

import argparse
import matplotlib
import numpy as np
from mpi4py import MPI as mpi
from ruamel.yaml import YAML

from dga import matsubara_frequencies as mf
from dga import brillouin_zone as bz
from dga import wannier as wannier
from dga import util as util
from dga import loggers

# ----------------------------------------------- ARGUMENT PARSER ------------------------------------------------------
DGA_OUPUT_PATH = '/LDGA_{}_Nk{}_Nq{}_wc{}_vc{}_vs{}'


def get_dga_output_folder_name(lambda_corr, nk_tot, nq_tot, niw_core, niv_core, niv_shell):
    return DGA_OUPUT_PATH.format(lambda_corr, nk_tot, nq_tot, niw_core, niv_core, niv_shell)


def create_dga_argparser(name='dga_config.yaml', path=os.getcwd() + '/'):
    ''' Set up an argument parser for the DGA code. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='?', default=name, type=str, help=' Config file name. ')
    parser.add_argument('-p', '--path', nargs='?', default=path, type=str, help=' Path to the config file. ')
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
    dga_config = DgaConfig(conf_file, comm=comm)

    return dga_config, conf_file


def save_config_file(conf_file, output_path):
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
        # for key,value in kwargs:
        #     if(key in self.__dict__):
        #         self.__dict__[key] = value
        #     else:
        #         warnings.warn(f'Unknown key {key} in config ')
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
        if (comm is not None):
            if (comm.rank == 0):
                self.output_path = util.uniquify(base_name)
            self.output_path = comm.bcast(self.output_path, root=0)
        else:
            self.output_path = util.uniquify(base_name)

        if (not os.path.exists(self.output_path)):
            if (comm is not None):
                if (comm.rank == 0): os.mkdir(self.output_path)
            else:
                os.mkdir(self.output_path)
        return self.output_path

    def save_data(self, mat, name):
        np.save(self.output_path + '/' + name + '.npy', mat, allow_pickle=True)

    def load_data(self, name):
        try:
            data = np.load(self.output_path + '/' + name + '.npy', allow_pickle=True).item()
        except:
            data = np.load(self.output_path + '/' + name + '.npy', allow_pickle=True)
        return data

    def clean_data(self, name):
        fname = self.output_path + '/' + name + '.npy'
        if (os.path.isfile(fname)):
            os.remove(fname)

    def to_yaml(self, name='config'):
        with open(self.output_path + '/' + f'{name}.yaml', 'w') as outfile:
            ddict = self.as_dict()
            YAML().dump(ddict, outfile)


class BoxSizes(ConfigBase):
    '''
        Contains the grid size parameters for a DGA run.
    '''

    def __init__(self, config_dict=None):
        self.niw_core: int = -1  # Number of bosonic Matsubaras for Gamma
        self.niv_core: int = -1  # Number of fermionic Matsubaras for Gamma
        self.niv_shell: int = 0  # Number of fermionic Matsubaras for Gamma = U
        self.niv_asympt: int = -1  # Number of fermionic for the single-particle Green's function

        if (config_dict is not None):
            self.update_dict(**config_dict)  # forwards parameters from config

        if (self.niv_asympt == -1):
            self.niv_asympt = self.niv_full + self.niw_core * 2

    @property
    def niv_full(self):
        return self.niv_shell + self.niv_core

    @property
    def vn_full(self):
        return mf.vn(n=self.niv_full)

    @property
    def niv_pp(self):
        '''Number of fermionic Matsubaras for the singlet/triplet vertex'''
        return np.min((self.niw_core // 2, self.niv_core // 2))

    @property
    def wn(self):
        return mf.wn(self.niw_core)

    def set_from_lfp(self, lfp):
        # set core fermionic frequency range:
        if self.niv_core == -1:
            self.niv_core = lfp.niv
        elif self.niv_core > lfp.niv:
            raise ValueError(f'niv_core ({self.niv_core}) cannot be '
                             f'larger than the available frequencies in four-point ({lfp.niv})')

        # set core fermionic frequency range:
        if self.niw_core == -1:
            self.niw_core = len(lfp.wn) // 2
        elif self.niw_core > len(lfp.wn) // 2:
            raise ValueError(f'niv_core ({self.niw_core}) cannot be '
                             f'larger than the available frequencies in four-point ({len(lfp.wn) // 2})')

        # reset the asymptotic frequency range if not set before:
        if (self.niv_asympt <= 0):
            self.niv_asympt = self.niv_full + self.niw_core * 2


class LatticeConfig(ConfigBase):
    '''
        Contains the information about the Lattice and Brillouin zone
    '''

    def __init__(self, config_dict):
        self.nk: Tuple[int, int, int] = (16, 16, 1)  # (nkx,nky,nkz) tuple of linear momenta. Used for fermionic quantities
        self.nq: Tuple[int, int, int] = (16, 16, 1)  # (nkx,nky,nkz) tuple of linear momenta. Used for bosonic quantities
        self.symmetries = []  # Lattice symmetries. Either string or tuple of strings
        self.hr_input = None  # Tight binding parameter.
        self.type = None

        self.update_dict(**config_dict)  # forwards parameters from config
        if ('nq' not in config_dict):
            print('Notification: nq not set in config. Setting nq = nk')
            self.nq = self.nk

        # check if symmetries are known:
        self.check_symmetries()

        # set k-grids:
        self.nk = tuple(self.nk)
        self.nq = tuple(self.nq)

        self.k_grid = bz.KGrid(self.nk, self.symmetries)
        self.q_grid = bz.KGrid(self.nq, self.symmetries)

        if (self.type is None):
            raise ValueError('Method of hr retrieval must be specified!')

        # Build the real-space Hamiltonian
        self.set_hr()

    def check_symmetries(self):
        ''' Set symmetries if known: '''
        if (self.symmetries == "two_dimensional_square"):
            self.symmetries = bz.two_dimensional_square_symmetries()
        elif (self.symmetries == "quasi_one_dimensional_square"):
            self.symmetries = bz.quasi_one_dimensional_square_symmetries()
        elif (self.symmetries == "simultaneous_x_y_inversion"):
            self.symmetries = bz.simultaneous_x_y_inversion()
        elif (not self.symmetries):
            pass
        elif (self.symmetries == 'none'):
            self.symmetries = ()
        elif (isinstance(self.symmetries, tuple)):
            for sym in self.symmetries:
                if (sym not in bz.KNOWN_SYMMETRIES):
                    NotImplementedError(f'Symmetry {self.symmetries} is not yet implemented.')
        else:
            raise NotImplementedError(f'Symmetry {self.symmetries} is not yet implemented.')

    @property
    def nk_tot(self):
        return np.prod(self.nk)

    @property
    def nq_tot(self):
        return np.prod(self.nq)

    def set_hr(self):
        ''' Set the tight-binding hamiltonian.'''
        self.hr: wannier.WannierHr  # type declaration for nice autocompletes
        if (self.type == 't_tp_tpp'):
            self.hr = wannier.WannierHr(*wannier.wannier_one_band_2d_t_tp_tpp(*self.hr_input))
        elif (self.type == 'from_wannier90'):
            assert isinstance(self.hr_input, str)
            self.hr = wannier.create_wannier_hr_from_file(self.hr_input)
        else:
            raise NotImplementedError(f'Input type {self.type} is not supported. Currently "t_tp_tpp" and "from_wannier90" are '
                                      f'supported.')

    def get_ek(self):
        ''' Return the k-space Hamiltonian '''
        return self.hr.get_ek(self.k_grid)


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


class DmftInput(ConfigBase):
    '''
        System specific parameters
    '''

    def __init__(self, n, beta, u, mu_dmft, giw, siw):
        self.n = n  # Average number of particles per spin-orbital
        self.beta = float(beta)  # Inverse temperature
        self.u = u  # Hubbard interaction 'U'
        self.mu_dmft = mu_dmft  # DMFT chemical potential
        self.giw = giw
        self.siw = siw


def get_dmft_input_config_from_dict(ddict):
    return DmftInput(ddict['n'], ddict['beta'], ddict['u'],
                     ddict['mu_dmft'], ddict['giw'], ddict['siw'])


class MaxEntConfig(OutputConfig):
    '''
        Settings for the analytical continuation.
    '''

    def __init__(self, t, beta, config_dict):
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
        self.bw_dga = [0.1, ]  # Blur width for DGA

        # Flags what continuation to perform:
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
        import dga.analytic_continuation as a_cont
        return a_cont.get_w_mesh(self.mesh_type, -self.wmax, self.wmax, self.nwr, self.cut)

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
        self.k_sym = 'random'  # k-symmetry of the gap function
        self.sym_sing = True  # Symmetrize the singlet pairing vertex
        self.sym_trip = True  # symmetrize the triplet pairing vertex
        self.eps = 1e-6
        self.max_count = 10000
        if (config_dict is not None): self.update_dict(**config_dict)
        self.set_gap0()

    def set_gap0(self):
        self.set_gap0_sing()
        self.set_gap0_trip()

    def set_gap0_sing(self):
        if (self.k_sym == 'd-wave'):
            v_sym = 'even'
        elif (self.k_sym == 'p-wave-x' or self.k_sym == 'p-wave-y'):
            v_sym = 'odd'
        else:
            v_sym = 'random'
        self.gap0_sing = {
            'k': self.k_sym,
            'v': v_sym
        }

    def set_gap0_trip(self):
        if (self.k_sym == 'd-wave'):
            v_sym = 'odd'
        elif (self.k_sym == 'p-wave-x' or self.k_sym == 'p-wave-y'):
            v_sym = 'even'
        else:
            v_sym = 'random'
        self.gap0_trip = {
            'k': self.k_sym,
            'v': v_sym
        }


class OpticsConfig(OutputConfig):
    '''
        Contains the configuration parameters and flags for the optical conductivity routines.
    '''

    def __init__(self, config_dict=None):
        super().__init__()
        self.do_bubble = False  # Flag whether to compute the bubble contribution to the optical conductivity
        self.do_vertex = False  # Flag whether to compute the vertex contribution to the optical conductivity
        self.der_a = 0  # Derivative of the dispersion with respect to kx
        self.der_b = 0  # Derivative of the dispersion with respect to kx
        self.niw_cond = -1  # number of bosonic frequencies for the optical conductivity
        self.niw_bubble = -1  # number of bosonic frequencies included in the bubble
        self.niw_vert = -1  # number of bosonic frequencies included in the vertex
        self.niv_bubble = -1  # number of fermionic frequencies used in the summation of the bubble
        self.niv_vert = -1  # number of fermionic frequencies used in the summation of the vertex
        if (config_dict is not None): self.update_dict(**config_dict)

    def wn_cond(self, pos=False):
        return mf.wn(self.niw_cond, pos=pos)

    def wn_bubble(self, pos=False):
        return mf.wn(self.niw_bubble, pos=pos)

    def wn_vert(self, pos=False):
        return mf.wn(self.niw_vert, pos=pos)

    def set_frequency_ranges(self, d_cfg):
        ''' Infer the frequencies which were not set in the config file '''

        if (self.niw_vert == -1):
            self.niw_vert = d_cfg.box.niw_core
        else:
            assert self.niw_vert <= d_cfg.box.niw_core, 'Number of bosonic frequencies for the vertex must be smaller ' \
                                                        'or equal to the number of frequencies available.'

        if (self.niw_cond == -1):
            self.niw_cond = d_cfg.box.niw_core // 2
        else:
            assert self.niw_cond < d_cfg.box.niv_core, 'Number of bosonic frequencies for the optical conductivity must be ' \
                                                       'smaller than the number of fermionic frequencies available.'
        if (self.niv_bubble == -1):
            self.niv_bubble = d_cfg.box.niv_asympt - self.niw_cond * 2
        # else:
        #     message = f'Number of fermionic frequencies for the bubble {self.niv_bubble}' \
        #               f' must be smaller than the number of fermionic frequencies ' \
        #               f'available in the Greens function.  {d_cfg.box.niv_asympt}'
        #     assert self.niv_bubble < d_cfg.box.niv_asympt, message
        self.niv_vert = d_cfg.box.niv_core - self.niw_cond  # set remaining frequencies for (v-w)

        # set the niw_bubble if not set before:
        if (self.niw_bubble == -1):
            self.niw_bubble = self.niw_cond

        # update the niv_asympt range if it is set too small:
        if (self.niv_bubble + self.niw_bubble >= d_cfg.box.niv_asympt):
            d_cfg.box.niv_asympt = self.niv_bubble + self.niw_bubble


class DebugConfig(ConfigBase):
    '''
        Contains options for debugging and additional verbosity.
    '''

    def __init__(self, config_dict=None):
        super().__init__()
        self.keep_rank_files = False  # Keep the files that were created by the different mpi processes; default: False

        if (config_dict is not None): self.update_dict(**config_dict)


class DgaConfig(OutputConfig):
    '''
        Contains the configuration parameters and flags for a DGA run.
    '''

    sys: DmftInput = None
    box: BoxSizes = None
    lattice: LatticeConfig = None
    eliash: EliashbergConfig = None
    output_path: str = None
    do_poly_fitting: bool = False
    poly_fit_dir: str = None
    logger: loggers.MpiLogger = None

    def __init__(self, conf_file, comm=None):
        super().__init__()

        self.comm = comm
        # Create config file:
        self.build_box_sizes(conf_file)
        self.build_lattice_conf(conf_file)
        self.build_eliash_conf(conf_file)
        self.build_optics_conf(conf_file)
        self.build_debug_conf(conf_file)
        # Optional configs, only set if contained in config file:
        # Polyfitting:
        self.n_fit = 4
        self.o_fit = 3
        if ('poly_fitting' in conf_file):
            self.do_poly_fitting = True
            self.update_dict(**conf_file['poly_fitting'])

        # Set input parameters:
        self.type = 'w2dyn'
        self.input_path = './'
        self.fname_1p = '1p-data.hdf5'
        self.fname_2p = 'g4iw_sym.hdf5'
        self.do_sym_v_vp = True  # Symmetrize g2 with respect to v-vp

        if ('dmft_input' in conf_file):
            self.update_dict(**conf_file['dmft_input'])

        # set the ouput flags:
        self.save_vrg = False
        self.save_fq = False
        self.keep_fq_optics = False
        self.keep_pairing_vertex = True
        self.verbosity = 1

        if ('output' in conf_file):
            self.update_dict(**conf_file['output'])

        # Set dga routine specifications:
        self.lambda_corr = 'spch'
        if ('dga' in conf_file):
            self.update_dict(**conf_file['dga'])
            if ('gui' not in conf_file['dga']):
                matplotlib.use('agg')
            else:
                pass  # Setting a non-Agg backend does not work on my machine. Why is unclear, but this should do the trick.
                # matplotlib.use(conf_file['dga']['gui'])
        else:
            matplotlib.use('agg')  # non-gui backend. Particularly usefull for use on cluster.

        # Check that number of processes is not larger than number of points in the irreducible BZ:
        if (comm is not None and comm.size > self.lattice.k_grid.nk_irr):
            raise ValueError('Number of processes may not be larger than points in the irreducible BZ for distribution.')

    def build_box_sizes(self, conf_file):
        if 'box_sizes' in conf_file:
            self.box = BoxSizes(conf_file['box_sizes'])
        else:
            self.box = BoxSizes()

    def build_lattice_conf(self, conf_file):
        if 'lattice' in conf_file:
            self.lattice = LatticeConfig(conf_file['lattice'])
        else:
            raise ValueError('Lattice must be contained in the config file.')

    def build_eliash_conf(self, conf_file):
        if 'pairing' in conf_file:
            self.eliash = EliashbergConfig(conf_file['pairing'])
        else:
            self.eliash = EliashbergConfig()

    def build_optics_conf(self, conf_file):
        if 'optics' in conf_file:
            self.optics = OpticsConfig(conf_file['optics'])
        else:
            self.optics = OpticsConfig()

    def build_debug_conf(self, conf_file):
        if 'debug' in conf_file:
            self.debug = DebugConfig(conf_file['debug'])
        else:
            self.debug = DebugConfig()

    def create_dga_ouput_folder(self):
        ''' Create the name of the dga output directory and create it '''
        base_name = self.input_path + get_dga_output_folder_name(self.lambda_corr, self.lattice.nk_tot,
                                                                 self.lattice.nq_tot, self.box.niw_core,
                                                                 self.box.niv_core, self.box.niv_shell)
        self.set_output_path(base_name, comm=self.comm)

    def create_plotting_folder(self):
        ''' Create the output folder for the plots. '''

        if (self.comm is not None):
            if (self.comm.rank == 0):
                assert os.path.exists(self.output_path), 'Output path must be set before creating the plotting folder.'
                self.pdir = self.output_path + '/Plots/'
                self.pdir = util.uniquify(self.pdir)
        else:
            assert os.path.exists(self.output_path), 'Output path must be set before creating the plotting folder.'
            self.pdir = self.output_path + '/Plots/'
            self.pdir = util.uniquify(self.pdir)

        # Create the folder
        if (self.comm is not None):
            if (self.comm.rank == 0):
                if (not os.path.exists(self.pdir)):
                    os.mkdir(self.pdir)
        else:
            if (not os.path.exists(self.pdir)):
                os.mkdir(self.pdir)

    def create_poly_fit_folder(self):
        ''' Create the output folder for the polynomial fits. '''
        if (self.do_poly_fitting):
            self.poly_fit_dir = self.output_path + '/PolyFits/'
            self.poly_fit_dir = util.uniquify(self.poly_fit_dir)

            # Create the folder
            if (not os.path.exists(self.poly_fit_dir)):
                if (self.comm is not None):
                    if (self.comm.rank == 0): os.mkdir(self.poly_fit_dir)
                else:
                    os.mkdir(self.poly_fit_dir)

    def create_folders(self):
        ''' Create the output folders '''
        self.comm.barrier()
        self.create_dga_ouput_folder()
        self.create_plotting_folder()
        if self.do_poly_fitting: self.create_poly_fit_folder()
        self.eliash.set_output_path(self.output_path + 'Eliashberg/', self.comm)
        self.optics.set_output_path(self.output_path + 'Optics/', self.comm)
        self.comm.barrier()

    def create_logger(self):
        ''' Create the logger for the DGA run '''
        assert os.path.exists(self.output_path), 'Output path must be set before creating the logger.'
        self.logger = loggers.MpiLogger(logfile=self.output_path + '/dga.log', comm=self.comm, output_path=self.output_path)

    def set_system_parameter(self, dmft_input):
        self.sys = DmftInput(dmft_input['n'], dmft_input['beta'], dmft_input['u'],
                             dmft_input['mu_dmft'], dmft_input['giw'], dmft_input['siw'])

    def log_sys_params(self):
        self.logger.log_message(f'Running calculation for: '
                                f'n={self.sys.n}; beta={self.sys.beta}; u={self.sys.u}; mu_dmft={self.sys.mu_dmft:.4f}')

    def check_gloc_dmft(self, giwk_obj):
        niv_min = np.min((mf.niv_from_mat(self.sys.giw), mf.niv_from_mat(giwk_obj.g_loc)))
        max_diff = np.max(np.abs(mf.cut_v(self.sys.giw, niv_min) - mf.cut_v(giwk_obj.g_loc, niv_min)))
        self.logger.log_message(f'Checking gloc: max_diff={max_diff:.4f}')
        self.logger.log_message(f'Checking mu: mu_input={self.sys.mu_dmft:.4f}; mu_dmft={giwk_obj.mu:.4f}')

    def log_estimated_memory_consumption(self):
        nq_tot = self.lattice.nq_tot
        nk_tot = self.lattice.nk_tot
        vn_size_core = self.box.niv_core * 2
        wn_size_core = self.box.niw_core * 2 + 1
        vn_size_full = self.box.niv_full * 2
        vn_size_asympt = (self.box.niv_full + self.box.niv_asympt)* 2 # this is just an upper bound
        size_complex = np.zeros(1,dtype=complex).itemsize
        vrg_mem = nq_tot * wn_size_core * vn_size_core * size_complex * 1e-9
        chi0q_urange = nq_tot * wn_size_core * vn_size_full * size_complex * 1e-9
        giwk_dga = nk_tot * vn_size_asympt * size_complex * 1e-9
        f_cond = nq_tot * wn_size_core * vn_size_core**3 * size_complex * 1e-9
        message = f'Estimated memory consumption in GB per occourance: giwk: [{giwk_dga}]; vrg: [{vrg_mem}]; ' \
                  f'chi0q: [{chi0q_urange}]; f_cond: [{f_cond}]'
        self.logger.log_message(message)


def get_default_config_dict():
    conf_dict = {
        'box': {
            'niv_core': -1,
            'niw_core': -1,
            'niv_shell': 0,
        },
        'lattice': {
            'symmetries': 'two_dimensional_square',
            'type': 't_tp_tpp',
            'hr_input': [1.0, -0.2, 0.1],
            'nk': [24, 24, 1]
        },
        'dga': {
            'lambda_corr': 'spch',
            'gui': 'Agg'
        },
        'dmft_input': {
            'fname_1p': '1p-data.hdf5',
            'fname_2p': 'g4iw_sym.hdf5',
            'input_path': './',
            'type': 'w2dyn'
        },
        'poly_fitting': {
            'n_fit': 4,
            'o_fit': 3,
        },
        'optics': {
            'do_bubble': False,
            'do_vertex': False,
            'der_a': 0,
            'der_b': 0,
            'niw_cond': -1,
            'niw_vert': -1,
            'niv_bubble': -1,
            'niv_vert': -1,
        },
        'pairing': {
            'do_eliash': False,
            'do_pairing_vertex': False,
            'k_sym': 'random',
        },
        'output': {
            'save_vrg': False,
            'save_fq': False
        },
        'debug': {
            'keep_rank_files': False
        }
    }
    return conf_dict


if __name__ == '__main__':
    pass
