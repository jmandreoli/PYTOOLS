# numpy array perfomance tests

def MatrixDeterminant(sz,manual=None):
  from numpy.linalg import det
  from numpy import linspace, newaxis
  a = linspace(0,1000,int(sz))[:,newaxis]
  a = a+a.T
  if manual: det = _manualdet
  yield
  det(a)
  yield

def MatrixSquared(sz,manual=None):
  from numpy import linspace, newaxis, dot
  a = linspace(0,1000,int(sz))[:,newaxis]
  a = a+a.T
  if manual: dot = _manualdot
  yield
  dot(a,a)
  yield

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
