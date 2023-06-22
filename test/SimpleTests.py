import numpy as np

def index2component_general(Nbands, N, ind):
  b=np.zeros((N),dtype=np.int_)
  s=np.zeros((N),dtype=np.int_)
  bs=np.zeros((N),dtype=np.int_)
  # the proposed back conversion assumes the indices are
  # given form 0 to max-1
  ind_tmp = ind - 1
  tmp=np.zeros((N+1),dtype=np.int_)
  for i in range(0,N+1):
    tmp[i] = (2*Nbands)**(N-i)

  for i in range(N):
    bs[i] = ind_tmp//tmp[i+1]
    s[i] = bs[i]%2
    b[i] = (bs[i]-s[i])//2
    ind_tmp = ind_tmp - tmp[i+1]*bs[i]

  return bs,b,s

def component2index_band(Nbands, N, b):
  ind = 1
  for i in range(N):
     ind = ind + Nbands**(N-i-1)*b[i]
  return ind


print(index2component_general(1,4,7))