from ruamel.yaml import YAML
import numpy as np
import argparse
fname = './config_test.yaml'
file = open(fname)
data = YAML().load(file)

nk = np.array(data['box_sizes']['nk'])

print(nk)

#%%
parser  = argparse.ArgumentParser()
parser.add_argument('--config', help='Config file and location.')

args = parser.parse_args(args=['--config','config_test.yaml'])
print(args)

config = YAML().load(open(args.config))

#%%

class Test():

    def __init__(self, config=None):
        self.config = 'tmp'

        if(config is not None):
            for key in self.__dict__.keys():
                if hasattr(config, key):
                    setattr(self,key,config.__dict__[key])


test = Test(args)
print(test.config)

#%%
from mpi4py import MPI as mpi
import Config as config
comm = mpi.COMM_WORLD
parser = config.create_dga_argparser()
if(comm.rank == 0):
    args = parser.parse_args()
    # args = parser.parse_args(args=['--config','dga_config.yaml'])
    assert hasattr(args,'config'), 'Config file location must be provided.'
    conf_file = YAML().load(open(args.config))
comm.barrier()
conf_file = comm.bcast(conf_file,root=0)
