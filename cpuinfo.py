# File:           cpuinfo.py
# Creation date:  2016-02-25
# Contributors:   Jean-Marc Andreoli
# Language:       python
# Purpose:        Records cpu information from machines on the cloud
#
# *** Copyright (c) 2016 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import subprocess, logging
from datetime import datetime
from socket import gethostname
from sqlalchemy import Column, Index, ForeignKey, create_engine
from sqlalchemy.types import Text, Integer, Float, DateTime, PickleType
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.orm import relationship, sessionmaker as basesessionmaker
from . import SQLinit, zipaxes

logger = logging.getLogger(__name__)

@as_declarative(name=__name__+'.Base')
class Base:
  @declared_attr
  def __tablename__(cls): return cls.__name__
  oid = Column(Integer(),primary_key=True)

Base.metadata.info.update(origin=__name__,version=1)

class Context(Base):
  tstamp = Column(DateTime())

  hosts = relationship('Host',back_populates='context',cascade='all, delete, delete-orphan')

  def __repr__(self): return '{}.Context<{}:{}>'.format(__name__,self.oid,self.tstamp)

class Host(Base):
  name = Column(Text(),nullable=False)
  nproc = Column(Integer(),nullable=False)

  context_oid = Column(Integer(),ForeignKey('Context.oid'),nullable=False)
  context = relationship('Context',back_populates='hosts')
  procinfos = relationship('Procinfo',back_populates='host',cascade='all, delete, delete-orphan')

class Procinfo(Base):
  procn = Column(Integer(),nullable=False)
  key = Column(Text(),nullable=False)
  val = Column(Text(),nullable=False)

  host_oid = Column(Integer(),ForeignKey('Host.oid'),nullable=False)
  host = relationship('Host',back_populates='procinfos')

#==================================================================================================
class Listing (MutableMapping):
#==================================================================================================

  def __init__(self,s):
    self.session = s

  def __getitem__(self,k):
    r = self.session.query(Context).get(k).first()
    if r is None: raise KeyError(k)
    self.session.add(r)
    return r

  def __delitem__(self,k):
    with self.session.begin_nested(): self.session.delete(self[k])

  def __setitem__(self,k,v):
    raise Exception('Direct create/update not permitted on Listing')

  def __iter__(self):
    with self.session.begin_nested():
      yield from (r[0] for r in self.session.query(Context.oid))

  def __len__(self):
    return self.session.query(Context).count()

  def items(self):
    with self.session.begin_nested():
      yield from self.session.query(Context.oid,Context)

  def clear(self):
    with self.session.begin_nested():
      for r in self.session.query(Context): self.session.delete(r)

  def newcontext(self,hosts=None):

  def _repr_html_(self):
    from lxml.etree import tounicode
    return tounicode(self.as_html())
  def as_html(self):
    return html_stack(*(v.as_html() for k,v in sorted(self.items())))

def sessionmaker(url,*a,_cache={},**ka):
  engine = _cache.get(url)
  if engine is None: _cache[url] = engine = SQLinit(url,Base.metadata)
  Session_ = basesessionmaker(engine,*a,**ka)
  def Session(**x):
    s = Session_(**x)
    s.listing = Listing(s)
    return s
  return Session

















def makecontext(hosts=None):
  if hosts is None: hosts = knownhosts()
  return Context(tstamp=datetime.now(),hosts=list(makehosts(hosts)))

def makehosts(hosts):
  H = [(hostname,list(makeprocinfos(hostname))) for hostname in hosts]
  H.sort()
  for hostname,L in H:
    yield Host(name=(hostname or gethostname()),nproc=len(set(p.procn for p in L)),procinfos=L)

def makeprocinfos(hostname):
  logger.info('Examining host: %s',hostname)
  cmd = [] if hostname is None else ['ssh','-q','-n','-x','-T',hostname]
  cmd += 'cat','/proc/cpuinfo'
  p = subprocess.run(cmd,stdout=subprocess.PIPE,universal_newlines=True)
  procn = None
  for y in p.stdout.split('\n'):
    y = y.strip()
    if not y: continue
    key,val = y.split(':',1)
    key = key.strip(); val = val.strip()
    if key == 'processor': procn = int(val); continue
    yield Procinfo(procn=procn,key=key,val=val)





def knownhosts(source=None):
  import os
  if source is None: source = os.path.expanduser('~/.ssh/known_hosts')
  with open(source) as u:
    for x in u:
      x = x.strip()
      if x: yield x.split(None,1)[0].split(',',1)[0]





