# ------------------------------------------------ COMMENTS ------------------------------------------------------------

'''
    Module for distributing tasks among cores
'''

# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import os
import numpy as np
import h5py
from mpi4py import MPI as mpi


# ------------------------------------------------ OBJECTS -------------------------------------------------------------


class MpiDistributor():
    ''' Distributes size tasks among cores
    '''

    def __init__(self, ntasks=1, comm: mpi.Comm = None, output_path=None, name=''):
        self._comm = comm
        self._ntasks = ntasks
        self.distribute_tasks()
        self.file = None

        if output_path is not None:
            # Read/write file. Create if it does not exist.
            if output_path[-1] != '/': output_path += '/'
            self.fname = output_path + name + f'Rank{self.my_rank:05d}' + '.hdf5'
            self.file = h5py.File(self.fname, 'a')
            self.file.close()

    def __del__(self):
        # self.__del__()
        if self.file is not None:
            try:
                self.close_file()
            except: # pylint: disable=bare-except
                pass

    def __enter__(self):
        self.open_file()
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.close_file()

    @property
    def comm(self):
        return self._comm

    @property
    def is_root(self):
        return self.my_rank == 0

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
    def my_tasks(self):
        return np.arange(0, self.ntasks)[self.my_slice]

    @property
    def mpi_size(self):
        return self._comm.Get_size()

    @property
    def my_size(self):
        return self._my_size

    @property
    def my_slice(self):
        return self._my_slice

    def open_file(self):
        try:
            self.file = h5py.File(self.fname, 'r+')
        except: # pylint: disable=bare-except
            pass

    def close_file(self):
        try:
            self.file.close()
        except: # pylint: disable=bare-except
            pass

    def delete_file(self):
        try:
            os.remove(self.fname)
        except: # pylint: disable=bare-except
            pass

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

    def allgather(self, rank_result=None):
        tot_shape = (self.ntasks,) + rank_result.shape[1:]
        tot_result = np.empty(tot_shape, rank_result.dtype)
        # tot_result[...] = np.nan
        other_dims = np.prod(rank_result.shape[1:])

        # The sizes argument needs the total number of elements rather than
        # just the first axis. The type argument is inferred.
        self.comm.Allgatherv(rank_result, [tot_result, self.sizes * other_dims])
        return tot_result

    def gather(self, rank_result=None, root=0):
        ''' Gather numpy array from ranks. '''
        tot_shape = (self.ntasks,) + rank_result.shape[1:]
        tot_result = np.empty(tot_shape, rank_result.dtype)
        other_dims = np.prod(rank_result.shape[1:])
        self.comm.Gatherv(rank_result, [tot_result, self.sizes * other_dims], root=root)
        return tot_result

    def scatter(self, full_data=None, root=0):
        ''' Scatter full_data among ranks using the first dimension. '''
        if full_data is not None:
            assert isinstance(full_data,np.ndarray), 'full_data must be a numpy array'
            rest_shape = np.shape(full_data)[1:]
            data_type = full_data.dtype
        else:
            rest_shape = None
            data_type = None

        data_type = self.comm.bcast(data_type, root)
        rest_shape = self.comm.bcast(rest_shape, root)
        rank_shape = (self.my_size,) + rest_shape
        rank_data = np.empty(rank_shape, dtype=data_type)
        other_dims = np.prod(rank_data.shape[1:])
        self.comm.Scatterv([full_data, self.sizes * other_dims], rank_data, root=root)
        return rank_data

    def bcast(self, data, root=0):
        ''' Broadcast data to all ranks. '''
        return self.comm.bcast(data, root=root)

    def allreduce(self, rank_result=None):
        tot_result = np.zeros(np.shape(rank_result), dtype=rank_result.dtype)
        self.comm.Allreduce(rank_result, tot_result)
        return tot_result


def create_distributor(ntasks, comm=None, output_path=None, name=''):
    ''' Create a distributor object. '''
    if comm is None:
        comm = mpi.COMM_WORLD
    return MpiDistributor(ntasks=ntasks, comm=comm, output_path=output_path, name=name)

if __name__ == '__main__':
    pass
