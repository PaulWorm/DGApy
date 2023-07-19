import numpy as np
import h5py

file = h5py.File('./1p-data.hdf5','a')
file['.config'].attrs['general.totdens'] = 1.0
file.close()


