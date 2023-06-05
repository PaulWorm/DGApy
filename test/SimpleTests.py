import numpy as np


list = ['1', '2', '3']

value = '1'

assert value in list, "value not in list"


print(f"{list}")

class TestClass():

    def __init__(self,a,b,c):

        self._a = a
        self._b = b
        self._c = c

    @property
    def a(self):
        return self._a

    def method(self):
        return self.a+self._b+self._c

tc = TestClass(1,2,3)

dict = tc.__dict__

# tc2 = TestClass(**dict)

print(dict)


#%%

import matplotlib.pyplot as plt
def find_zeros(mat):
    ''' Finds the zero crossings of a matrix.
        Not sure if this should be transfered to the Plotting module.
    '''
    ind_x = np.arange(mat.shape[0])
    ind_y = np.arange(mat.shape[1])

    cs1 = plt.contour(ind_x, ind_y, mat.real, cmap='RdBu', levels=[0, ])
    paths = cs1.collections[0].get_paths()
    plt.close()
    paths = np.atleast_1d(paths)
    vertices = []
    for path in paths:
        vertices.extend(path.vertices)
    vertices = np.array(vertices,dtype=int)
    return vertices

kx = np.linspace(0,2*np.pi,100)
mat = np.cos(kx[:,None]) * np.cos(kx[None,:])

zeros = find_zeros(mat)
print(zeros.shape)

plt.pcolormesh(kx,kx,mat,cmap='RdBu')
plt.scatter(kx[zeros[:,0]],kx[zeros[:,1]],c='k',s=1)
plt.show()



