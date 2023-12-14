import numpy as np
from mpi4py import MPI as mpi

from dga import mpi_aux
from test_util import util_for_testing as t_util


def test_mpi_distrbiutor(comm):
    ntasks = 10

    mpi_dist = mpi_aux.MpiDistributor(ntasks, comm=comm)
    t_util.test_statement(mpi_dist.my_rank == comm.Get_rank(), 'rank in MpiDistributor')
    t_util.test_statement(mpi_dist.ntasks == ntasks, 'ntasks in MpiDistributor')

    np.random.seed(1)
    if comm.rank == 0:
        data_1d = np.array(np.random.rand(ntasks))
        data_2d = np.array(np.random.rand(ntasks, ntasks * 2))
        data_3d = np.array(np.random.rand(ntasks, ntasks * 2, ntasks * 3))
    else:
        data_1d = None
        data_2d = None
        data_3d = None

    # test 1d array scatter
    my_data_1d = mpi_dist.scatter(data_1d, root=0)
    data_1d = comm.bcast(data_1d, root=0)
    t_util.test_array(my_data_1d, data_1d[mpi_dist.my_slice], f'scatter_data_1d_rank_{mpi_dist.my_rank}')

    # test 2d array scatter
    my_data_2d = mpi_dist.scatter(data_2d, root=0)
    data_2d = comm.bcast(data_2d, root=0)
    t_util.test_array(my_data_2d, data_2d[mpi_dist.my_slice], f'scatter_data_2d_rank_{mpi_dist.my_rank}')

    # test 3d array scatter
    my_data_3d = mpi_dist.scatter(data_3d, root=0)
    data_3d = comm.bcast(data_3d, root=0)
    t_util.test_array(my_data_3d, data_3d[mpi_dist.my_slice], f'scatter_data_3d_rank_{mpi_dist.my_rank}')

    # test array gather
    data_1d_gather = mpi_dist.gather(my_data_1d, root=0)
    data_2d_gather = mpi_dist.gather(my_data_2d, root=0)
    data_3d_gather = mpi_dist.gather(my_data_3d, root=0)
    if mpi_dist.is_root:
        t_util.test_array(data_1d_gather, data_1d, f'gather_data_1d_rank_root')
        t_util.test_array(data_2d_gather, data_2d, f'gather_data_2d_rank_root')
        t_util.test_array(data_3d_gather, data_3d, f'gather_data_3d_rank_root')

    # test array allgather
    data_1d_allgather = mpi_dist.allgather(my_data_1d)
    data_2d_allgather = mpi_dist.allgather(my_data_2d)
    data_3d_allgather = mpi_dist.allgather(my_data_3d)
    t_util.test_array(data_1d_allgather, data_1d, f'allgather_data_1d_rank_{mpi_dist.my_rank}')
    t_util.test_array(data_2d_allgather, data_2d, f'allgather_data_2d_rank_{mpi_dist.my_rank}')
    t_util.test_array(data_3d_allgather, data_3d, f'allgather_data_3d_rank_{mpi_dist.my_rank}')

    # test array allreduce
    rank_sum = mpi_dist.allreduce(np.array((mpi_dist.my_rank,)))
    t_util.test_array(rank_sum, np.sum(np.arange(mpi_dist.mpi_size)), f'allreduce_rank_{mpi_dist.my_rank}')


if __name__ == '__main__':
    mpi_comm = mpi.COMM_WORLD
    test_mpi_distrbiutor(mpi_comm)
