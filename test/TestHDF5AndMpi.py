import h5py
import numpy as np
from mpi4py import MPI as mpi
import dga.mpi_aux as mpi_aux
comm = mpi.COMM_WORLD

if(comm.rank == 0):
    file = h5py.File('./Test.hdf5','a')
    if('test' not in file):
        file['test'] = np.zeros((100,100,100), dtype=complex)
comm.Barrier()
if(comm.rank != 0):
    file = h5py.File('./Test.hdf5','a')
mpi_dist = mpi_aux.MpiDistributor(ntasks=100,comm=comm)
my_ind = np.arange(0,100)[mpi_dist.my_slice]

for i in my_ind:
    file['test'][i] = np.random.rand(100,100).astype(complex)


file.close()



