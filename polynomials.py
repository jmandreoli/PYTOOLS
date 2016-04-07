# File:           polynomial.py
# Creation date:  March 2016
# Contributors:   Jean-Marc Andreoli
# Language:       python
# Purpose:        Light support for polynomials
#
# *** Copyright (c) 2016 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

from functools import singledispatch
from fractions import Fraction
from math import gcd

#==================================================================================================
class Polynomial:
  r"""
Instances of this class are polynomials in an arbitrary field (scalars) supporting 0,1 as zero and unit. Polynomial multiplication, addition and subtraction are supported.

:param coefs: a sequence :math:`(c_n)_{n\in0:N}` of scalars
:type coefs: list(scalar)
:param roots: a family :math:`(m_r)_{r\in R}` of integers indexed by a subset :math:`R` of scalars
:type roots: dict(scalar:\ :class:`int`\ )
:param cst: a scalar :math:`z` in the field
:type cst: scalar

The represented polynomial is

.. math::

   \begin{equation*}
   z\prod_{r\in R}(X-r)^{m_r}\left(\sum_{n=0}^Nc_nX^{N-n}\right)
   \end{equation*}

Root 0 (if it is a root) is automatically factorised.

For latex representation, scalars are passed to function :func:`latexc` of this module. It is a single-dispatch function, which can thus be configured for non standard scalar types.
  """
#==================================================================================================
  __slots__ = ('coefs','roots','cst','latex_')

  def __new__(cls,coefs,roots={},cst=1,check=True):
    roots = roots.copy()
    if check:
      s = tuple(i for i,c in enumerate(coefs) if c)
      if not cst or not s: return ZeroPolynomial.singleton
      m = max(s)
      s = slice(min(s),m+1)
      m = len(coefs)-1-m
      if m: roots[0] = roots.get(0,0)+m
      coefs = coefs[s]
    self = super().__new__(cls)
    self.roots = roots
    self.coefs = coefs
    self.cst = cst
    self.latex_ = None
    return self

  def factorise(self,x):
    r"""
Factorises this poynomial with root *x* up to its multiplicity (does nothing if *x* is not a root or is already fully factorised).
    """
    m,coefs = reduce(self.coefs,x)
    if m:
      self.coefs = coefs
      self.roots[x] = self.roots.get(x,0)+m
      self.latex_ = None
    return m

  def __call__(self,x):
    r"""
Evaluates this polynomial at *x*.
    """
    q = peval(self.coefs,x)
    for r,m in self.roots.items(): q *= (x-r)**m
    return q*self.cst

  def __mul__(self,other):
    if not issubclass(self.__class__,other.__class__): return NotImplemented
    roots = self.roots.copy()
    for r,m in other.roots.items(): roots[r] = roots.get(r,0)+m
    cst = self.cst*other.cst
    coefs = list(convolution(self.coefs,other.coefs))
    return self.__class__(coefs,roots,cst,check=False)
  def __rmul__(self,other): return self.__mul__(other)

  def __add__(self,other,trs=(lambda c:c),tro=(lambda c:c)):
    if not issubclass(self.__class__,other.__class__): return NotImplemented
    rs,ro = self.roots.copy(), other.roots.copy()
    roots = {}
    for r in set(rs).intersection(set(ro)):
      ms,mo = rs[r],ro[r]
      roots[r] = m = min(ms,mo)
      rs[r] -= m; ro[r] -= m
    cs = expand(self.coefs,rs,trs(self.cst))
    co = expand(other.coefs,ro,tro(other.cst))
    ns,no = len(cs),len(co)
    if ns<no: cs = (no-ns)*[0]+cs
    elif no<ns: co = (ns-no)*[0]+co
    coefs = tuple(cs_+co_ for cs_,co_ in zip(cs,co))
    return self.__class__(coefs,roots)
  def __radd__(self,other): return self.__add__(other)

  def __sub__(self,other): return self.__add__(other,tro=(lambda c: -c))
  def __rsub__(self,other): return self.__add__(other,trs=(lambda c:-c))

  @property
  def latex(self):
    r"""
Returns a format string with one placeholder. When formatted with a variable name, it returns a latex representation of the polynomial.
    """
    if self.latex_ is None:
      roots = ''.join(latexr(r,m) for r,m in sorted(self.roots.items(),key=(lambda r: abs(r[0]))))
      K = len(self.coefs)-1
      if K>0:
        latex = ''.join(latexm(k,c,K) for k,c in enumerate(self.coefs) if c)
        if roots: latex = '({})'.format(latex)
        latex = roots+latex
        if self.cst == -1: latex = '-'+latex
        elif self.cst!=1: latex = latexesc(latexc(self.cst))+latex
      else:
        c = self.coefs[0]*self.cst
        latex = ('' if roots else '1') if c==1 else ('-' if roots else '-1') if c==-1 else latexesc(latexc(c))
        latex += roots
      self.latex_ = latex
    return self.latex_

  dvar = 'X'
  def _repr_latex_(self): return '${}$'.format(self)
  def __str__(self): return self.latex.format(self.dvar)

#==================================================================================================
class ZeroPolynomial:
  r"""
Single-instance class containing the zero polynomial.
  """
#==================================================================================================
  def __call__(self,x): return 0
  def __add__(self,other): return other
  def __radd__(self,other): return other
  def __mul__(self,other): return self
  def __rmul__(self,other): return self
  def __sub__(self,other): return other.__class__(other.coefs,other.roots,-other.cst)
  def __rsub__(self,other): return other
  def _repr_latex_(self): return '$0$'
  def __str__(self): return '0'
  latex = '0'
ZeroPolynomial.singleton = ZeroPolynomial()

#==================================================================================================
class FracPolynomial (Polynomial):
  r"""
Instances of this class are polynomials with fractional coefficients, where the coefficients are reduced to a common denominator which is factorised into *cst*.
  """
#==================================================================================================
  def __init__(self,coefs,roots={},cst=1):
    super().__init__(list(map(Fraction,coefs)),dict((Fraction(x),m) for x,m in roots.items()),Fraction(cst))
    self.reshape()

  def factorise(self,x):
    if super(FracPolynomial,self).factorise(Fraction(x)): self.reshape()

  def reshape(self):
    d = lcm(*(c.denominator for c in self.coefs if c))
    self.coefs = tuple(d*c.numerator//c.denominator for c in self.coefs)
    self.cst /= d

#==================================================================================================
# Utilities
#==================================================================================================

def peval(coefs,x,q=0):
  for c in coefs: q = c+q*x
  return q

def reduce(coefs,x):
  def reduc(coefl):
    q = 0
    for c in coefl: q = c+q*x; yield q
  m = 0
  while True:
    ncoefs = tuple(reduc(coefs))
    if ncoefs[-1]: return m,coefs
    m += 1; coefs = ncoefs[:-1]

def expand(coefs,roots,cst):
  for r,m in roots.items():
    if not m: continue
    coefs = list(convolution(coefs,[binom(m,k)*(-r)**k for k in range(m+1)]))
  return [cst*c for c in coefs]

def convolution(coefs1,coefs2,max_1=(lambda x: None if x<=-1 else x)):
  n1,n2 = len(coefs1),len(coefs2)
  for k in range(n1+n2-1):
    yield sum(c1*c2 for c1,c2 in zip(coefs1[max(0,k-n2+1):min(k+1,n1)],coefs2[min(k,n2-1):max_1(k-n1):-1]))

def lcm(n0,*L):
  n = n0
  for m in L: n = (n*m)//gcd(n,m)
  return n

def binom(n,k,cache=[(1,)]):
  N = len(cache)
  if n<N: return cache[n][k]
  else:
    row = cache[N-1]
    for m in range(N,n+1):
      row = (1,)+tuple(x+y for x,y in zip(row[:-1],row[1:]))+(1,)
      cache.append(row)
    return row[k]

@singledispatch
def latexc(x): return str(x)

@latexc.register(float)
def _(x): return '{:.3g}'.format(x)

@latexc.register(Fraction)
def _(x): return str(x.numerator) if x.denominator==1 else r'{}\frac{{{}}}{{{}}}'.format(('' if x.numerator>0 else '-'),abs(x.numerator),x.denominator)

@latexc.register(complex)
def _(x): return (('({:.3g}+{:.3g}j)'.format(x.real,x.imag) if x.real>0 else '-({:.3g}-{:.3g}j)'.format(-x.real,x.imag)) if x.imag>0 else ('({:.3g}-{:.3g}j)'.format(x.real,-x.imag) if x.real>0 else '-({:.3g}+{:.3g}j)'.format(-x.real,-x.imag)) if x.imag<0 else '{:.3g}'.format(x.real)) if x.real else '{:.3g}j'.format(x.imag)

def latexm(k,c,K):
  def sgn(v):
    return (v[1:] if v.startswith('+') else v) if k==0 else (v if v.startswith('+') or v.startswith('-') else '+'+v)
  coef = sgn(('1' if k==K else '') if c==1 else ('-1' if k==K else '-') if c==-1 else latexesc(latexc(c)))
  expo = '' if k==K else '{0}' if k==K-1 else '{{0}}^{{{{{}}}}}'.format(K-k)
  return coef+expo

def latexr(r,m):
  expo = '' if m==1 else '^{{{{{}}}}}'.format(m)
  body = '{0}'
  if r:
    r = latexesc(latexc(-r)); sgn = '+'
    if r.startswith('-'): r = r[1:]; sgn = '-'
    body = '({{0}}{}{})'.format(sgn,r)
  return body+expo

def latexesc(x,esc={ord('{'):'{{',ord('}'):'}}'}): return x.translate(esc)
