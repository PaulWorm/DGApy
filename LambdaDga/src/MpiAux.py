# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np
import h5py
from mpi4py import MPI as mpi
import os, sys
# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------


def cen2lin(val=None, start=0):
    return val - start


# ------------------------------------------------ OBJECTS -------------------------------------------------------------

class MpiDistributor():
    ''' Distributes size tasks among cores
    '''

    def __init__(self, ntasks=1, comm: mpi.Comm=None, output_path = None, name=''):
        self._comm = comm
        self._ntasks = ntasks
        self.distribute_tasks()
        self.file = None

        if(output_path is not None):
            # Read/write file. Create if it does not exist.
            self.fname = output_path + name + 'Rank{0:05d}'.format(self.my_rank) + '.hdf5'
            self.file = h5py.File(self.fname,'a')

    def __del__(self):
        # self.__del__()
        if(self.file is not None):
            try:
                self.file.close()
            except:
                pass

    @property
    def comm(self):
        return self._comm

    @property
    def ntasks(self):
        return self._ntasks

    @property
    def sizes(self):
        return self._sizes

    @property
    def my_rank(self):
        return self._comm.Get_rank()

    @property
    def mpi_size(self):
        return self._comm.Get_size()

    @property
    def my_size(self):
        return self._my_size

    @property
    def my_slice(self):
        return self._my_slice

    def distribute_tasks(self):
        n_per_rank = self.ntasks // self.mpi_size
        n_excess = self.ntasks - n_per_rank * self.mpi_size
        self._sizes = n_per_rank * np.ones(self.mpi_size, int)

        if n_excess:
            self._sizes[-n_excess:] += 1

        slice_ends = self._sizes.cumsum()
        self._slices = list(map(slice, slice_ends - self._sizes, slice_ends))

        self._my_size = self._sizes[self.my_rank]
        self._my_slice = self._slices[self.my_rank]


    def allgather(self, rank_result = None):
        tot_shape = (self.ntasks,) + rank_result.shape[1:]
        tot_result = np.empty(tot_shape, rank_result.dtype)
        #tot_result[...] = np.nan
        other_dims = np.prod(rank_result.shape[1:])

        # The sizes argument needs the total number of elements rather than
        # just the first axis. The type argument is inferred.
        self.comm.Allgatherv(rank_result,[tot_result, self.sizes * other_dims])
        return tot_result


if __name__ == '__main__':

    comm = mpi.COMM_WORLD
    print(comm.Get_rank())
    niw = 11
    iw = np.arange(-niw, niw+1)
    ntasks = iw.size
    mpi_distributor = MpiDistributor(ntasks=ntasks,comm=comm,output_path='./', name='Test')
    print(f'{mpi_distributor.my_rank} and I am doing slice: {mpi_distributor.my_slice}')
    print(f'My iw: {iw[mpi_distributor.my_slice]}')
    my_iw = iw[mpi_distributor.my_slice]

    gather_iw = mpi_distributor.allgather(rank_result=my_iw)
    mpi_distributor.file['iw/my_iw'] = my_iw
    mpi_distributor.file['iw/gather_iw'] = gather_iw
    mpi_distributor.file['iw/complex'] = gather_iw*1j
    mpi_distributor.file.close()

    print(f'Full iw: {gather_iw}')


















