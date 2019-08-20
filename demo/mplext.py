# File:                 demo/mplext.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the mplext module

if __name__=='__main__':
  import sys
  from PYTOOLS.demo.mplext import demo # properly import this module
  demo()
  sys.exit(0)

#--------------------------------------------------------------------------------------------------

from pathlib import Path
from scipy.special import betainc, beta, gamma, erf
from numpy import sqrt, square, exp, infty, pi, linspace
from ..mplext import Cell
automatic = False

from collections import namedtuple
Distr = namedtuple('Distr',('name','dom','domv','mean','std','pdf','cdf'))

def display(*l,**ka): # l must be a list of Distr instances (probability distributions)
  view = Cell.create(**ka)
  with view.clearm():
    # turn the cell *view* into a grid-cell:
    view.make_grid(5*len(l)+1,2)
    # S(i) specifies rowspan for i-th entry (4/5)
    S = lambda i: slice(5*i,5*i+4)
    for i,D in enumerate(l):
      x = linspace(D.domv[0],D.domv[1],500)
      # automagically create an axes-cell for i-th entry pdf: rowspan S(i), left col
      ax = view[S(i),0].make_axes()
      ax.plot(x,D.pdf(x),'b')
      ax.axvline(D.mean,c='r')
      ax.axvline(D.mean+D.std,c='r',ls=':')
      ax.axvline(D.mean-D.std,c='r',ls=':')
      ax.set_xlim(D.domv)
      ax.set_title('pdf: {}'.format(D.name))
      # automagically create an axes-cell for i-th entry cdf: rowspan S(i), right col
      ax = view[S(i),1].make_axes(sharex=ax)
      ax.plot(x,D.cdf(x))
      ax.set_title('cdf: {}'.format(D.name))
    # automagically create an axes-cell for the message: bottom row, both cols
    ax = view[-1,:].make_axes()
    ax.text(.5,.5,'Distribution details from Wikipedia.',ha='center',va='center')
    ax.set_xticks(()) ; ax.set_yticks(())
    view.figure.tight_layout()
  if automatic: view.figure.savefig(str(Path(__file__).parent.resolve()/'mplext.png'))

#--------------------------------------------------------------------------------------------------

def demo():
  from matplotlib.pyplot import show
  display(Dbeta(),Dweibull(),Dnormal(),figsize=(10,8))
  if not automatic: show()

def Dbeta(a=1.5,b=2.5): # the beta distribution
  return Distr(
    name='beta[a={0},b={1}]'.format(a,b),
    dom=(0.,1.), domv=(1.e-10,1.-1.e-10),
    mean=a/(a+b), std=sqrt(a*b/(a+b+1))/(a+b),
    pdf=lambda x: x**(a-1.)*(1.-x)**(b-1.)/beta(a,b),
    cdf=lambda x: betainc(a,b,x),
  )

def Dweibull(k=1.2): # the weibull distribution
  m = gamma(1.+1./k)
  return Distr(
    name='weibull[shape={0}]'.format(k),
    dom=(0.,infty), domv=(0.,2.5),
    mean=m, std=sqrt(gamma(1.+2./k)-square(m)),
    pdf=lambda x: k*x**(k-1.)*exp(-x**k),
    cdf=lambda x: 1-exp(-x**k),
  )

def Dnormal(mu=0.,sigma=1.): # the normal distribution
  return Distr(
    name='normal',
    dom=(-infty,infty), domv=(mu-4*sigma,mu+4*sigma),
    mean=mu, std=sigma,
    pdf=lambda x,K=sigma*sqrt(2*pi): exp(-square(x-mu)/2)/K,
    cdf=lambda x,K=sigma*sqrt(2): .5*(1+erf((x-mu)/K)),
  )
