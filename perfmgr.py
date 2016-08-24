# File:           perfmgr.py
# Creation date:  2016-02-25
# Contributors:   Jean-Marc Andreoli
# Language:       python
# Purpose:        Manages performance records
#
# *** Copyright (c) 2016 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import os, sys, subprocess, time, logging
from datetime import datetime
from socket import gethostname
from collections import defaultdict
from itertools import cycle
from sqlalchemy import Column, Index, ForeignKey
from sqlalchemy.types import Text, Integer, Float, DateTime, PickleType
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.orm import relationship
from . import SQLinit, zipaxes, ormsroot, html_table, HtmlPlugin

logger = logging.getLogger(__name__)

SPATH = os.path.splitext(os.path.abspath(__file__))[0]+'_.py'

@as_declarative(name=__name__+'.Base')
class Base:
  @declared_attr
  def __tablename__(cls): return cls.__name__
  oid = Column(Integer(),primary_key=True)

Base.metadata.info.update(origin=__name__,version=1)

#==================================================================================================
class Context (Base,HtmlPlugin):
  r"""
Instances of this class are persistent and represent a bunch of performance experiments over a shared testbed.
  """
#==================================================================================================
  testbed = Column(Text(),nullable=False)
  version = Column(DateTime(),nullable=False)
  tests = Column(Text(),nullable=False)

  experiments = relationship('Experiment',back_populates='context',cascade='all, delete, delete-orphan')

#--------------------------------------------------------------------------------------------------
  def run(self,name,seq,smax,host=None,exc=None,**ka):
    r"""
:param name: the name of the test in the testbed
:param seq: a generator of test inputs (floats) which must be in increasing order
:param smax: threshold on test size
:param host: a machine name accessible by ssh (default: current machine)
:param exc: the path to the python executable to run on the host (default: current executable)

Launches one experiment, i.e. a sequence of tests. Each test takes one input (a number) and produces a pair of a result (any object) and a size (a number). The latter must be an increasing function of the input. The test loop stops when the test size exceeds a threshold.
    """
#--------------------------------------------------------------------------------------------------
    def rcall(sub,m,protocol=2):
      import pickle
      pickle.dump(m,sub.stdin,protocol=protocol)
      sub.stdin.flush()
      r = pickle.load(sub.stdout)
      if isinstance(r,Exception): raise Exception('Exception in executor') from r
      return r
    assert name in self.tests.split(' ')
    cmd = [] if host is None else ['ssh','-T','-q','-x',host]
    cmd += [(sys.executable if exc is None else exc),'-u',SPATH]
    with subprocess.Popen(cmd,bufsize=0,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as sub:
      host_,exc_ = rcall(sub,(self.testbed,name,ka))
      L = []
      for x in seq:
        y,sz = rcall(sub,x)
        logger.info('Perf(%.2f)=%s',x,y)
        L.append(Perf(xval=x,yval=y,size=sz))
        if sz > smax: break
    return Experiment(created=datetime.now(),name=name,args=ka,smax=smax,host=host_,exc=exc_,perfs=L,context=self)

#--------------------------------------------------------------------------------------------------
  def display(self,fig,target,meter=(lambda x: x),filtr=(lambda exp: True),fmt={},**kfiltr):
    r"""
:param fig: a matplotlib figure
:param target: the slot to plot
:type target: ``host`` | ``name`` | ``args`` | ``exc``
:param meter: a function which maps a test output into a number, or a key if test output is of type :class:`dict`
:type meter: callable|\ :class:`str`
:param filtr: a function which maps an experiment into a boolean
:param kfiltr: a dict with keys in ``host``, ``name``, ``args``, ``exc``

Displays the results of selected experiments in this context. The selection test is the conjunction of the filter function *filtr* applied to the experiment object, and for each item *s*,\ *v* of the dictionary *kfiltr*, whether slot *k* of the experiment is equal to *v*. If *v* is :const:`None`, it is replaced by a default value: the current host for ``host``, the current python executable for ``exc``, an empty dictionary for ``args`` and an empty string for ``name``.
    """
#--------------------------------------------------------------------------------------------------
    def mkfiltr(u,dflt=dict(host=gethostname(),exc=sys.executable,args={},name='')):
      slot,x = u
      d = dflt.get(slot)
      assert d is not None, 'Unknown slot: {}'.format(slot)
      if x is None: x = d
      return lambda exp: getattr(exp,slot)==x
    fmtd = dict(args=(lambda x:'{{{}}}'.format(','.join('{}={}'.format(k,v) for k,v in sorted(x.items())))))
    fmtd.update(fmt)
    if isinstance(meter,str): meterf = lambda r,m=meter: r[m]
    else: meterf = meter; meter = 'meter' if meter.__name__=='<lambda>' else meter.__name__
    kfiltr = list(map(mkfiltr,kfiltr.items()))
    slots = 'name','args','exc','host'
    fmts = tuple(fmtd.get(slot,str) for slot in slots)
    targeti = slots.index(target)
    rngs = [set() for slot in slots]
    tests = defaultdict(dict)
    for exp in self.experiments:
      if not(filtr(exp) and all(f(exp) for f in kfiltr)): continue
      c = list(f(getattr(exp,slot)) for f,slot in zip(fmts,slots))
      for x,r in zip(c,rngs): r.add(x)
      x = c[targeti]; c[targeti] = None
      tests[tuple(c)][x] = tuple(zip(*((p.xval,meterf(p.yval)) for p in exp.perfs)))
    assert tests, 'No experiment passed the filter'
    targets = tuple(zip(cycle('bgrcmyk'),sorted(rngs[targeti])))
    nontargeti = [i for i,r in enumerate(rngs) if i!=targeti and len(r)>1]
    for (c,D),ax in zipaxes(sorted(tests.items()),fig,sharex=True,sharey=True):
      ax.set_title(','.join('{}={}'.format(slots[i],c[i]) for i in nontargeti),fontsize='small')
      for col,val in targets:
        t = D.get(val)
        if t is None: continue
        ax.plot(t[0],t[1],c=col,label=val)
      ax.set_ylabel(meter,fontsize='small')
      ax.legend(fontsize='x-small',loc='upper left')

#--------------------------------------------------------------------------------------------------
  @staticmethod
  def fromfile(initf):
#--------------------------------------------------------------------------------------------------
    with open(initf) as u: x = u.read(); v = os.stat(u.fileno()).st_mtime
    t = {}
    exec(x,t) # checks syntax
    t = ' '.join(k for k in t if not k.startswith('_'))
    v = datetime.fromtimestamp(v)
    return Context(testbed=x,version=v,tests=t)

#--------------------------------------------------------------------------------------------------
  @property
  def title(self): return self.testbed.split('\n',1)[0][2:]

  def __repr__(self): return '{}.Context<{}:{}>'.format(__name__,self.oid,self.title)

  def as_html(self,incontext):
    H = ('created','host','name','args','exc','smax','nperf')
    return html_table(((exp.oid,[getattr(exp,h) for h in H]) for exp in self.experiments),hdrs=H,fmts=[str for h in H],title='{0.oid}: {0.title} {{{0.tests}}} {0.version}'.format(self))

#==================================================================================================
class Experiment (Base):
#==================================================================================================
  created = Column(DateTime(),nullable=False)
  host = Column(Text(),nullable=False)
  name = Column(Text(),nullable=False)
  args = Column(PickleType(),nullable=False)
  exc = Column(Text(),nullable=False)
  smax = Column(Float(),nullable=False)

  context_oid = Column(Integer(),ForeignKey('Context.oid'))
  context = relationship('Context',back_populates='experiments')
  perfs = relationship('Perf',back_populates='experiment',cascade='all, delete, delete-orphan')

  @property
  def nperf(self): return len(self.perfs)

#==================================================================================================
class Perf (Base):
#==================================================================================================
  xval = Column(Float())
  yval = Column(PickleType())
  size = Column(Float())

  experiment_oid = Column(Integer(),ForeignKey('Experiment.oid'))
  experiment = relationship('Experiment',back_populates='perfs')

#==================================================================================================
class Root (ormsroot):
  r"""
Instances of this class give access to the main :class:`Context` persistent class.
  """
#==================================================================================================

  base = Context

  def addexperiment(self,e):
    with self.session.begin_nested():
      c = e.context
      try:
        e.context = None # unlink
        exp = self.session.merge(e)
        exp.context = self.addcontext(c)
      finally: e.context = c # relink
      self.session.flush()
      logger.info('Created %s',exp)
    return exp

  def addcontext(self,c):
    with self.session.begin_nested():
      context = self.session.query(Context).filter_by(testbed=c.testbed,version=c.version).first()
      if context is None:
        context = self.session.merge(c)
        self.session.flush()
        logger.info('Created %s',context)
    return context

sessionmaker = Root.sessionmaker

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def geometric(xo,a):
  r"""
A generator enumerating the values of a geometric sequence with initial value *xo* and common ratio *a*.
  """
#--------------------------------------------------------------------------------------------------
  x = xo
  while True: yield x; x = a*x
