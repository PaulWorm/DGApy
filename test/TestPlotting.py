import matplotlib.pyplot as plt
import numpy as np

name = 'chi'

plt.figure()
plt.ylabel(r'$\Re ' + '\\' + name + '$')
plt.show()

tmp = np.ones((5,10))
tmp2 = np.ones((10,))


print(tmp + tmp2)