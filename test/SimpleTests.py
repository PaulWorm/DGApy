import sys
import numpy as np
import dga.loggers as loggers
from dga.loggers import get_largest_vars


tmp1 = np.array((10,10))
tmp2 = np.array((20,10))
tmp3 = np.array((30,10))
tmp4 = np.random.rand(100,100,100)

# print(loggers.get_largest_vars(5))
print(get_largest_vars(5))


