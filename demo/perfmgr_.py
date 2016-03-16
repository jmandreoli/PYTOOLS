# numpy array perfomance tests

def MatrixDeterminant(x,manual=None):
  from numpy.linalg import det
  return _timing((_manualdet if manual else det),_somematrix(x))

def MatrixSquared(x,manual=None):
  from numpy import dot
  return _timing((lambda a,dot=(_manualdot if manual else dot): dot(a,a)),_somematrix(x))

def _timing(f,*a,**ka):
  from time import time,clock
  p = time(),clock()
  f(*a,**ka)
  p = time()-p[0],clock()-p[1]
  return dict(time=p[0],clock=p[1]),p[0]

def _somematrix(x):
  from numpy import linspace, newaxis
  a = linspace(0,1000,int(x))[:,newaxis]
  return a+a.T

def _manualdot(a,b):
  from numpy import empty, sum
  m,n = a.shape; n,p = b.shape
  c = empty((m,p))
  for i in range(m):
    for j in range(p):
      c[i,j] = sum(a[i]*b[:,j])
  return c

def _manualdet(a):
  from numpy import empty, argmax, abs
  n,n = a.shape
  r = empty((n,))
  d = 1.
  for i in range(n):
    k = argmax(abs(a[i:,i]))
    r[...] = a[i+k]; p = r[i]
    if p==0: return 0.
    r/=p
    d *= p
    if k: d *= -1; a[i+k] = a[i]
    for j in range(i+1,n): a[j] -= a[j,i]*r
  return d
