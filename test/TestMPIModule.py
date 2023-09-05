import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpi4py import MPI as mpi

import dga.mpi_aux as mpi_aux
import dga.util as util

matplotlib.use('agg')

comm = mpi.COMM_WORLD

if(comm.rank == 0): print(f'Commsize: {comm.size}')
print(f'My rank is: {comm.rank}')

nk = (5,100,1)
nw = 501
size = np.prod(nk) * nw

if(comm.rank == 0):
    swk_test = np.random.rand(*nk,nw) + 1j * np.random.rand(*nk,nw)
    print('-----------')
    print(f'Size: {util.mem(swk_test)} GB')
    print('-----------')
else:
    swk_test = None


if(comm.rank == 0): print('Mpi distributor')
mpi_dist = mpi_aux.MpiDistributor(ntasks=nk[0],comm=comm,name='test')

comm.Barrier()
print(f'My size is: {mpi_dist.my_size} and my rank is {comm.rank}')

if(comm.rank == 0): print('Scatter')
comm.Barrier()
swk_scattered = mpi_dist.scatter(swk_test)

if(comm.rank == 0): print('Add')
swk_scattered += comm.rank

if(comm.rank == 0): print('Gather')
swk_test = mpi_dist.gather(swk_scattered)


if(comm.rank == 0): print('Create figure:')
if(comm.rank == 0):
    # print(swk_test.shape)
    plt.figure()
    # print(swk_test.shape)
    plt.imshow(swk_test[:,:,0,251].real,'RdBu')
    # print(swk_test.shape)
    plt.colorbar()
    # print(swk_test.shape)
    plt.savefig('./TestMpit.png')
    # print('Saving figure:')
    plt.close()


if(comm.rank == 0): print('Barrier')
comm.Barrier()
if(comm.rank == 0): print('Closing program.')
mpi.Finalize()
