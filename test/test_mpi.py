from mpi4py import MPI as mpi

comm = mpi.COMM_WORLD

print(f'I am rank {comm.rank}')