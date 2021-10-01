# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------


def cen2lin(val=None, start=0):
    return val - start


# ------------------------------------------------ OBJECTS -------------------------------------------------------------

class qiw():
    ''' Class for handling {q,iw} four-indizes
        Indizes must be of increasing value. q's must start at 0, iw can start in the negative
    '''

    def __init__(self, qgrid=None, iw=None, mpi_size=1, mpi_rank=0):
        self._qx = qgrid[0]
        self._qy = qgrid[1]
        self._qz = qgrid[2]
        self._iw = iw
        self._mpi_size = mpi_size
        self._mpi_rank = mpi_rank
        self.set_qiw()
        self.distribute_workload()

    @property
    def qx(self):
        return self._qx

    @property
    def qy(self):
        return self._qy

    @property
    def qz(self):
        return self._qz

    @property
    def iw(self):
        return self._iw

    @property
    def nqx(self) -> int:
        return self.qx.size

    @property
    def nqy(self) -> int:
        return self.qy.size

    @property
    def nqz(self) -> int:
        return self.qz.size

    @property
    def niw(self) -> int:
        return self.iw.size

    @property
    def mpi_size(self):
        return self._mpi_size

    @property
    def mpi_rank(self):
        return self._mpi_rank

    @property
    def ntot(self):
        return self.nqx * self.nqy * self.nqz * self.niw

    @property
    def qiw(self):
        return self._qiw

    @property
    def my_size(self):
        return self._my_size

    @property
    def my_slice(self):
        return self._my_slice

    @property
    def my_indizes(self):
        return self._my_indizes

    @property
    def my_qiw(self):
        return self._my_qiw


    @property
    def my_qx(self):
        return self.my_qiw[:, 0]

    @property
    def my_qy(self):
        return self.my_qiw[:, 1]

    @property
    def my_qz(self):
        return self.my_qiw[:, 2]

    @property
    def my_iw(self):
        return self.my_qiw[:, 3]

    @property
    def my_wn(self):
        return cen2lin(self.my_iw, self.iw[0])

    def nq(self):
        return self.nqx * self.nqy * self.nqz

    def wn(self, ind):
        return int(cen2lin(val=self.my_qiw[ind,-1],start=self.iw[0]))

    def set_qiw(self):
        self._qiw = np.array(np.meshgrid(self.qx, self.qy, self.qz, self.iw)).reshape(4, -1).T

    def distribute_workload(self):
        n_per_rank = self.ntot // self._mpi_size
        n_excess = self.ntot - n_per_rank * self._mpi_size
        self._sizes = n_per_rank * np.ones(self.mpi_size, int)

        if n_excess:
            self._sizes[-n_excess:] += 1

        slice_ends = self._sizes.cumsum()
        self._slices = list(map(slice, slice_ends - self._sizes, slice_ends))

        self._my_size = self._sizes[self.mpi_rank]
        self._my_slice = self._slices[self.mpi_rank]
        self._my_qiw = self._qiw[self.my_slice]
        self._my_indizes = np.arange(slice_ends[self.mpi_rank] - self.my_size, slice_ends[self.mpi_rank])


if __name__ == '__main__':
    qx = np.arange(0, 9)
    qy = np.arange(0, 7)
    qz = np.arange(0, 1)
    iw = np.arange(-10, 11)
    qgrid = [qx,qy,qz]
    indizes = qiw(qgrid=qgrid, iw=iw, mpi_size=4, mpi_rank=0)
