import numpy as np

def index2component_band(Nbands, N, ind):
  b=[]; ind_tmp = ind - 1
  for i in range(N):
    b.append(ind_tmp//(Nbands**(N-i-1)))
    ind_tmp = ind_tmp - b[i]*(Nbands**(N-i-1))
  return b

def component2index_band(Nbands, N, b):
  ind = 1
  for i in range(N):
     ind = ind + Nbands**(N-i-1)*b[i]
  return ind

b = index2component_band(2,4,30)
print(b)
ind = component2index_band(2,4,[2,2,2,2])
print(ind)