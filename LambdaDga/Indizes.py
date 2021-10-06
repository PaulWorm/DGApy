# ------------------------------------------------ COMMENTS ------------------------------------------------------------


# -------------------------------------------- IMPORT MODULES ----------------------------------------------------------

import numpy as np


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------


def cen2lin(val=None, start=0):
    return val - start


# ------------------------------------------------ OBJECTS -------------------------------------------------------------

class qiw():
    ''' Class for handling {q,iw_core} four-indizes
        Indizes must be of increasing value. q's must start at 0, iw_core can start in the negative
    '''

    def __init__(self, qgrid=None, iw=None):
        self._qx = qgrid[0]
        self._qy = qgrid[1]
        self._qz = qgrid[2]
        self._iw = iw
        self.set_qiw()
        self._my_qiw = None

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

    @my_qiw.setter
    def my_qiw(self, slice):
        self._my_qiw = self.qiw[slice]

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



if __name__ == '__main__':
    qx = np.arange(0, 9)
    qy = np.arange(0, 7)
    qz = np.arange(0, 1)
    iw = np.arange(-10, 11)
    qgrid = [qx,qy,qz]
    indizes = qiw(qgrid=qgrid, iw=iw)
    my_slice = slice(1,5,None)
    indizes.my_qiw = my_slice

    print(indizes.my_qiw)