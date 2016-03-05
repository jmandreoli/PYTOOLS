# File:           perfmgr.py
# Creation date:  2016-02-25
# Contributors:   Jean-Marc Andreoli
# Language:       python
# Purpose:        Manages performance records
#
# *** Copyright (c) 2016 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import os, sys, subprocess, logging, threading
from collections.abc import MutableMapping
from functools import partial
from datetime import datetime
from sqlalchemy.engine import Engine
from sqlalchemy.sql import select, insert, delete, func
from . import odict, SQLinit

logger = logging.getLogger(__name__)

SPATH = os.path.splitext(os.path.abspath(__file__))[0]+'_.py'

#==================================================================================================
def schema(meta):
#==================================================================================================
  from sqlalchemy import Table, Column, ForeignKey, Index
  from sqlalchemy.types import Text, Integer, Float, DateTime, PickleType
  Table(
    'Session',meta,
    Column('oid',Integer(),primary_key=True),
    Column('version',DateTime(),nullable=False),
    Column('testbed',Text(),nullable=False),
    Index('session_idx','testbed','version'),
  )
  Table(
    'Experiment',meta,
    Column('oid',Integer(),primary_key=True),
    Column('session',Integer(),ForeignKey('Session.oid',ondelete='CASCADE')),
    Column('created',DateTime(),nullable=False),
    Column('host',Text(),nullable=False),
    Column('name',Text(),nullable=False),
    Column('args',PickleType(),nullable=False),
    Column('exc',Text(),nullable=False),
    Column('tmax',Float(),nullable=False),
  )
  Table(
    'Perf',meta,
    Column('exper',Integer(),ForeignKey('Experiment.oid',ondelete='CASCADE')),
    Column('size',Float()),
    Column('meter',Float()),
  )
schema.status = dict(module=__name__,version=2)

#--------------------------------------------------------------------------------------------------
class TableMixin:
#--------------------------------------------------------------------------------------------------

  def __getitem__(self,oid):
    with self.connect() as conn:
      r = conn.execute(select([self.table]).where(self.table.c.oid==oid)).fetchone()
    if r is None: raise KeyError(oid)
    return self.tableR(**r)

  def __delitem__(self,oid):
    with self.connect() as conn:
      n = conn.execute(self.table.delete().where(self.table.c.oid==oid)).rowcount
    if n==0: raise KeyError(oid)

  def __setitem__(self,oid,v):
    raise Exception('Direct create/update not permitted on {}'.format(self))

  def __iter__(self):
    with self.connect() as conn:
      yield from (oid for oid, in conn.execute(select([self.table.c.oid])))

  def __len__(self):
    with self.connect() as conn:
      return conn.execute(select([func.count(self.table.c.oid)])).fetchone()[0]

  def items(self):
    with self.connect() as conn:
      for r in conn.execute(select([self.table])): yield r.oid,self.tableR(**r)

  def clear(self):
    with self.connect() as conn:
      conn.execute(self.table.delete())

#==================================================================================================
class PerfManager (TableMixin,MutableMapping):
#==================================================================================================

  def __new__(cls,spec,listing={},lock=threading.Lock()):
    if isinstance(spec,PerfManager): return spec
    elif isinstance(spec,Engine) or isinstance(spec,str): engine = spec
    else: raise TypeError('Expected: {}|{}|{}; Found: {}'.format(PerfManager,Engine,str,type(spec)))
    with lock:
      self = listing.get(engine)
      if self is None:
        self = super(PerfManager,cls).__new__(cls)
        self.engine,meta = SQLinit(engine,schema)
        self.table = self.session_t = meta['Session']
        self.tableR = partial(PerfSession,self)
        self.experiment_t = meta['Experiment']
        self.perf_t = meta['Perf']
        self.session = None
        self.history = []
        self.config = None
    return self

  def __del__(self):
    self.engine.dispose()

  def __getnewargs__(self): return self.engine.url,

  def connect(self,**ka):
    conn = self.engine.connect(**ka)
    if self.engine.name == 'sqlite': conn.execute('PRAGMA foreign_keys=1')
    return conn

  def getsession(self,initf):
    with open(initf) as u: x = u.read()
    exec(x,{}) # checks syntax
    t = os.stat(initf).st_mtime
    with self.connect() as conn:
      oid = conn.execute(select([self.session_t.c.oid]).where((self.session_t.c.version==t)&(self.session_t.c.testbed==x))).fetchone()
      if oid is None:
        oid = conn.execute(self.session_t.insert().values(version=datetime.fromtimestamp(t),testbed=x)).inserted_primary_key[0]
    return self[oid]

#--------------------------------------------------------------------------------------------------
# Display
#--------------------------------------------------------------------------------------------------

  def _repr_html_(self):
    from lxml.etree import tounicode
    return tounicode(self.as_html())
  def as_html(self):
    return html_stack(*(v.as_html() for k,v in sorted(self.items())))
  def __str__(self): return 'PerfManager<{}>'.format(self.engine.url)

#==================================================================================================
class PerfSession (TableMixin):
#==================================================================================================

  def __init__(self,mgr,**ka):
    self.__dict__.update(ka)
    self.experiment_t = self.table = mgr.experiment_t
    self.perf_t = mgr.perf_t
    self.connect = mgr.connect
    self.url = mgr.engine.url

  def record(self,name,seq,host=None,exc=sys.executable,tmax=120.,**ka):
    def rcall(sub,m,protocol=2):
      import pickle
      pickle.dump(m,sub.stdin,protocol=protocol)
      sub.stdin.flush()
      r = pickle.load(sub.stdout)
      if isinstance(r,Exception): raise Exception('Exception in executor') from r
      return r
    with self.connect() as conn:
      exp = conn.execute(self.experiment_t.insert().values(session=self.oid,created=datetime.now(),name=name,host=(host or os.environ['HOST']),exc=exc,args=ka,tmax=tmax)).inserted_primary_key[0]
    n = 0
    logger.info('Experiment %s:  %s (executor: %s, tmax: %.2fs)',exp,name,exc,tmax)
    try:
      cmd = [] if host is None else ['ssh','-T','-q','-x',host]
      cmd += [exc,'-u',SPATH]
      with subprocess.Popen(cmd,bufsize=0,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as sub:
        rcall(sub,(self.testbed,name,ka))
        for sz in seq:
          tm = rcall(sub,sz)
          try:
            with self.connect() as conn:
              conn.execute(self.perf_t.insert().values(exper=exp,size=sz,meter=tm))
            n += 1
            if tm > tmax: break
          finally: logger.info('Perf(%s,%.2f)=%.2f',exp,sz,tm)
    finally:
      if n==0:
        with self.connect() as conn:
          conn.execute(self.experiment_t.delete().where(self.experiment_t.c.oid==exp))
        exp = None
    return exp

  def report(self,**ka):
    from numpy import array
    e = odict(**ka)
    with self.connect() as conn:
      e.result = array(list(map(tuple,conn.execute(select([self.perf_t.c.size,self.perf_t.c.meter]).where(self.perf_t.c.exper==self.oid)))))
    return e

  tableR = report

#--------------------------------------------------------------------------------------------------
# Display
#--------------------------------------------------------------------------------------------------

  def _repr_html_(self):
    from lxml.etree import tounicode
    return tounicode(self.as_html())
  def as_html(self):
    H = [h for h in self.experiment_t.c.keys() if h != 'oid' and h!='session']
    L = list((k,[v[h] for h in H]) for k,v in sorted(self.items()))
    return html_table(L,[str for h in H],hdrs=H,title='Session {} ')
  def __str__(self): return 'PerfSession<{}.{}>'.format(self.url,self.oid)

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def geometric(xo,step):
#--------------------------------------------------------------------------------------------------
  x = xo
  while True:
    yield x
    x = step*x

#--------------------------------------------------------------------------------------------------
def html_table(irows,fmts,hdrs=None,title=None):
#--------------------------------------------------------------------------------------------------
  from lxml.builder import E
  def thead():
    if title is not None: yield E.TR(E.TD(title,colspan=str(1+len(fmts))),style='background-color: gray; color: white')
    if hdrs is not None: yield E.TR(E.TD(),*(E.TH(hdr) for hdr in hdrs))
  def tbody():
    for ind,row in irows:
      yield E.TR(E.TH(str(ind)),*(E.TD(fmt(v)) for fmt,v in zip(fmts,row)))
  return E.TABLE(E.THEAD(*thead()),E.TBODY(*tbody()))
#--------------------------------------------------------------------------------------------------
def html_stack(*a,**ka):
#--------------------------------------------------------------------------------------------------
  from lxml.builder import E
  return E.DIV(*(E.DIV(x,**ka) for x in a))
