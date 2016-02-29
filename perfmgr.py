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
from . import odict, SQLinit

logger = logging.getLogger(__name__)

SPATH = os.path.splitext(os.path.abspath(__file__))[0]+'_.py'

class PerfManager:

  def __init__(self,engine):
    def schema(meta):
      from sqlalchemy import Table, Column, ForeignKey
      from sqlalchemy.types import Text, Integer, Float, DateTime, PickleType
      Table(
        'Session',meta,
        Column('oid',Integer(),primary_key=True),
        Column('tstamp',DateTime(),nullable=False),
        Column('host',Text(),nullable=False),
        Column('testbed',Text(),nullable=False),
      )
      Table(
        'Experiment',meta,
        Column('oid',Integer(),primary_key=True),
        Column('session',Integer(),ForeignKey('Session.oid',ondelete='CASCADE')),
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
    schema.status = dict(version=1)
    self.engine,meta = SQLinit(engine,schema)
    self.session_t = meta['Session']
    self.experiment_t = meta['Experiment']
    self.perf_t = meta['Perf']

    self.session = None
    self.history = []
    self.config = None

  def open(self,initf,host=None):
    with open(initf) as u: t = u.read()
    exec(t,{})
    self.config = odict(testbed=t,host=host)
    return self

  def close(self):
    self.config = None

  def __enter__(self):
    assert self.config is not None, 'PerfManager is not opened'
    assert self.session is None, 'Cannot open multiple sessions'
    with self.engine.connect() as conn:
      self.session = conn.execute(self.session_t.insert().values(tstamp=datetime.now(),host=self.config.host,testbed=self.config.testbed)).inserted_primary_key[0]
    logger.info('Opening session %s',self.session)
    return self
  def __exit__(self,*a):
    self.history.append(self.session)
    logger.info('Closing session %s',self.session)
    self.session = None

  def record(self,name,seq,exc=sys.executable,tmax=120.,**ka):
    def rcall(sub,m,protocol=2):
      import pickle
      pickle.dump(m,sub.stdin,protocol=protocol)
      sub.stdin.flush()
      r = pickle.load(sub.stdout)
      if isinstance(r,Exception): raise Exception('Exception in executor') from r
      return r
    assert self.session is not None, 'No ongoing session'
    cfg = self.config
    with self.engine.connect() as conn:
      exp = conn.execute(self.experiment_t.insert().values(session=self.session,name=name,exc=exc,args=ka,tmax=tmax)).inserted_primary_key[0]
    n = 0
    logger.info('Experiment %s:  %s (executor: %s, tmax: %.2fs)',exp,name,exc,tmax)
    try:
      cmd = [] if cfg.host is None else ['ssh','-T','-q','-x',cfg.host]
      cmd += [exc,'-u',SPATH]
      with subprocess.Popen(cmd,bufsize=0,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as sub:
        rcall(sub,(cfg.testbed,name,ka))
        for sz in seq:
          tm = rcall(sub,sz)
          try:
            with self.engine.connect() as conn:
              conn.execute(self.perf_t.insert().values(exper=exp,size=sz,meter=tm))
              n += 1
            if tm > tmax: break
          finally: logger.info('Perf(%s,%.2f)=%.2f',exp,sz,tm)
    finally:
      if n==0:
        logger.info('Experiment %s: deleted (no entries)',exp)
        with self.engine.connect() as conn:
          conn.execute(self.experiment_t.delete().where(self.experiment_t.c.oid==exp))
      else:
        logger.info('Experiment %s: completed (%s entries)',exp,n)

  def report(self,session):
    from sqlalchemy.sql import select
    with self.engine.connect() as conn:
      for x in conn.execute(select([self.experiment_t]).where(self.experiment_t.c.session==session)):
        R = conn.execute(select([self.perf_t.c.size,self.perf_t.c.meter]).where(self.perf_t.c.exper==x.oid).order_by('size')).fetchall()
        yield x,R

def geometric(xo,step):
  x = xo
  while True:
    yield x
    x = step*x
