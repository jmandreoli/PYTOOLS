# File:           perfmgr.py
# Creation date:  2016-02-25
# Contributors:   Jean-Marc Andreoli
# Language:       python
# Purpose:        Manages performance records
#
# *** Copyright (c) 2016 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import os, sys, subprocess, logging
from datetime import datetime
from socket import gethostname
from collections import defaultdict
from itertools import cycle
from sqlalchemy import Column, Index, ForeignKey
from sqlalchemy.types import Text, Integer, Float, DateTime, PickleType
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.orm import relationship
from . import SQLinit, zipaxes, ormsroot, html_table

logger = logging.getLogger(__name__)

SPATH = os.path.splitext(os.path.abspath(__file__))[0]+'_.py'

@as_declarative(name=__name__+'.Base')
class Base:
  @declared_attr
  def __tablename__(cls): return cls.__name__
  oid = Column(Integer(),primary_key=True)

Base.metadata.info.update(origin=__name__,version=1)

#==================================================================================================
class Context (Base):
  r"""
Instances of this class are persistent and represent a bunch of performance experiments over a shared testbed.
  """
#==================================================================================================
  version = Column(DateTime(),nullable=False)
  testbed = Column(Text(),nullable=False)
  tests = Column(Text(),nullable=False)

  experiments = relationship('Experiment',back_populates='context',cascade='all, delete, delete-orphan')

#--------------------------------------------------------------------------------------------------
  def newexperiment(self,name,seq,host=None,exc=sys.executable,tmax=120.,**ka):
    r"""
Launches one experiment.

:param name: the name of the test in the testbed
:param seq: a generator of sizes for the test
:param host: a machine name accessible by ssh
:param exc: the path to the python executable to run on the host
:param tmax: the loop through sizes stops when the test exceeds this duration in sec
    """
#--------------------------------------------------------------------------------------------------
    def rcall(sub,m,protocol=2):
      import pickle
      pickle.dump(m,sub.stdin,protocol=protocol)
      sub.stdin.flush()
      r = pickle.load(sub.stdout)
      if isinstance(r,Exception): raise Exception('Exception in executor') from r
      return r
    exp = Experiment(created=datetime.now(),host=(host or gethostname()),name=name,args=ka,exc=exc,tmax=tmax,perfs=[])
    logger.info('Experiment(host=%s,name=%s,args=%s,exc=%s,tmax=%s)',exp.host,exp.name,exp.args,exp.exc,exp.tmax)
    cmd = [] if host is None else ['ssh','-T','-q','-x',host]
    cmd += [exc,'-u',SPATH]
    with subprocess.Popen(cmd,bufsize=0,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as sub:
      rcall(sub,(self.testbed,name,ka))
      for sz in seq:
        tm = rcall(sub,sz)
        logger.info('Perf(%.2f)=%.2f',sz,tm)
        exp.perfs.append(Perf(size=sz,meter=tm))
        if tm > tmax: break
    self.experiments.append(exp)
    return exp

#--------------------------------------------------------------------------------------------------
  def display(self,fig,inner,outer,**filtr):
    r"""
Displays the results of all the experiments in this context. Strings *inner*, *outer* and the keys in *filtr* must be all distinct and cover exactly the list: ``host``, ``name``, ``args``, ``exc``. The experiments selected for display are those which match the *filter* dictionary.

* For each value of the outer component of the selected experiments, a separate matplotlib axes is displayed for that value. It shows the performance results recorded for all the experiments having that value for the outer component.

* The different experiments are labelled with the value of their inner component, which appears in the legend. Experiments with the same inner component share the same colour across the axes (i.e. different values of the outer component).
    """
#--------------------------------------------------------------------------------------------------
    st0,st1 = set(('host','name','args','exc')),set((inner,outer)+tuple(filtr))
    assert st0 <= st1, 'Missing status: {}'.format(st0-st1)
    assert st1 <= st0, 'Unknown status: {}'.format(st1-st0)
    inner = lambda exp,a=inner: str(getattr(exp,a))
    outer = lambda exp,a=outer: str(getattr(exp,a))
    dflt = dict(host=gethostname(),exc=sys.executable)
    def filtr(exp,L=[(k,(dflt[k] if v is None else v)) for k,v in filtr.items()]):
      return all(getattr(exp,a)==v for a,v in L)
    tests = defaultdict(dict)
    innerL = set()
    for exp in self.experiments:
      if not filtr(exp): continue
      r = inner(exp)
      tests[outer(exp)][r] = tuple(zip(*((p.size,p.meter) for p in exp.perfs)))
      innerL.add(r)
    assert tests, 'No experiment passed the filter'
    innerL = list(zip(cycle('bgrcmyk'),sorted(innerL)))
    for (outerv,D),ax in zipaxes(sorted(tests.items()),fig,sharex=True,sharey=True):
      for col,innerv in innerL:
        r = D.get(innerv)
        if r is None: continue
        ax.plot(r[0],r[1],c=col,label=innerv)
      ax.set_xlabel('size')
      ax.set_ylabel('meter')
      ax.set_title(outerv)
      ax.legend(fontsize='x-small')

#--------------------------------------------------------------------------------------------------
  @property
  def title(self): self.testbed.split('\n',1)[0][2:]

  def __repr__(self): return '{}.Context<{}:{}>'.format(__name__,self.oid,self.title)

  def _repr_html_(self):
    from lxml.etree import tounicode
    return tounicode(self.as_html())
  def as_html(self):
    H = ('created','host','name','args','exc','tmax','nperf')
    return html_table(((exp.oid,[getattr(exp,h) for h in H]) for exp in self.experiments),hdrs=H,fmts=[str for h in H],title='{0.oid}: {0.title} {{{0.tests}}} {0.version}'.format(self))

#==================================================================================================
class Experiment (Base):
#==================================================================================================
  created = Column(DateTime(),nullable=False)
  host = Column(Text(),nullable=False)
  name = Column(Text(),nullable=False)
  args = Column(PickleType(),nullable=False)
  exc = Column(Text(),nullable=False)
  tmax = Column(Float(),nullable=False)

  context_oid = Column(Integer(),ForeignKey('Context.oid'))
  context = relationship('Context',back_populates='experiments')
  perfs = relationship('Perf',back_populates='experiment',cascade='all, delete, delete-orphan')

  @property
  def nperf(self): return len(self.perfs)

#==================================================================================================
class Perf (Base):
#==================================================================================================
  size = Column(Float())
  meter = Column(Float())

  experiment_oid = Column(Integer(),ForeignKey('Experiment.oid'))
  experiment = relationship('Experiment',back_populates='perfs')

#==================================================================================================
class Root (ormsroot):
  r"""
Instances of this class give access to the main :class:`Context` persistent class.
  """
#==================================================================================================

  def getcontext(self,initf):
    r"""
Retrieves or creates a :class:`Context` instance associated with the testbed contained in file *initf*.
    """
    with open(initf) as u: x = u.read()
    t = {}
    exec(x,t) # checks syntax
    t = ' '.join(k for k in t if not k.startswith('_'))
    v = datetime.fromtimestamp(os.stat(initf).st_mtime)
    with self.session.begin_nested():
      r = self.session.query(Context).filter_by(testbed=x,version=v).first()
      if r is None:
        r = Context(version=v,testbed=x,tests=t,experiments=[])
        self.session.add(r)
    return r

Root.set_base(Context)
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
  while True:
    yield x
    x = a*x
