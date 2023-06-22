from mpi4py import MPI as mpi
import dga.loggers as loggers
import dga.mpi_aux as mpi_aux
import numpy as np
comm = mpi.COMM_WORLD

print(f'I am rank {comm.rank}')

output_dir = './'
ntot = 10
mpi_dist = mpi_aux.MpiDistributor(ntasks=ntot, comm=comm, output_path=output_dir + '/',
                                      name='FBZ')

if(comm.rank == 0):
    tmp = np.arange(ntot**3)
    full_data = np.reshape(tmp,(ntot,ntot,ntot)).astype(complex)
else:
    full_data = None

rank_data = mpi_dist.scatter(full_data)

print(rank_data)
