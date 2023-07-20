import numpy as np
import dga.matsubara_frequencies as mf

n = 10
vn_1 = mf.vn(10)
vn_1 = mf.vn(int(10))
vn_1 = mf.vn(np.int64(10))

vn_2 = mf.vn(10.,int(10),pos=True)
vn_2 = mf.vn(np.float64(10.),int(10),pos=True)

vn_3 = mf.vn(10., 10)