# File:           cpuinfo.py
# Creation date:  2016-02-25
# Contributors:   Jean-Marc Andreoli
# Language:       python
# Purpose:        Records cpu information from machines on the cloud
#
# *** Copyright (c) 2016 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import subprocess, re, logging
from collections import defaultdict
from datetime import datetime
from socket import getfqdn
from sqlalchemy import Column, Index, ForeignKey
from sqlalchemy.types import Text, Integer, Float, DateTime, PickleType
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.orm import relationship
from . import SQLinit, zipaxes, ormsroot, html_table

logger = logging.getLogger(__name__)

@as_declarative(name=__name__+'.Base')
class Base:
  @declared_attr
  def __tablename__(cls): return cls.__name__
  oid = Column(Integer(),primary_key=True)

Base.metadata.info.update(origin=__name__,version=1)

#==================================================================================================
class Context(Base):
  r"""
Instances of this class are persistent and represent a collection of host machines on the cloud at a given time.
  """
#==================================================================================================
  tstamp = Column(DateTime())

  hosts = relationship('Host',back_populates='context',cascade='all, delete, delete-orphan')

#--------------------------------------------------------------------------------------------------
  @staticmethod
  def load(hosts=''):
#--------------------------------------------------------------------------------------------------
    if isinstance(hosts,str):
      pat = re.compile(hosts)
      hosts = (h for h in knownhosts() if pat.fullmatch(h) is not None)
    context = Context(tstamp=datetime.now())
    for hostname in hosts:
      logger.info('Examining host: %s',hostname)
      cmd = [] if hostname is None else ['ssh','-q','-n','-x','-T',hostname]
      cmd += 'cat','/proc/cpuinfo'
      p = subprocess.run(cmd,stdout=subprocess.PIPE,universal_newlines=True)
      procn = None
      L = []
      n = 0
      for y in p.stdout.split('\n'):
        y = y.strip()
        if not y: continue
        key,val = map(str.strip,y.split(':',1))
        if key == 'processor': n+=1; procn = int(val); continue
        L.append(Procinfo(procn=procn,key=key,val=val))
      context.hosts.append(Host(name=(getfqdn() if hostname is None else getfqdn(hostname)),nproc=n,procinfos=L))
    return context

#--------------------------------------------------------------------------------------------------
  def __repr__(self): return '{}.Context<{}:{}>'.format(__name__,self.oid,self.tstamp)

  def _repr_html_(self):
    from lxml.etree import tounicode
    return tounicode(self.as_html())
  def as_html(self):
    def hist(L):
      from lxml.builder import E
      D = defaultdict(int)
      for v in L: D[v] += 1
      if len(D) == 1: return E.SPAN(list(D.keys())[0])
      return E.TABLE(E.TBODY(*(E.TR(E.TD(' '.join(v.split()),style='border:0'),E.TD(str(n),style='border:0; color:blue')) for v,n in sorted(D.items()))))
    return html_table(((host.oid,[host.name,host.nproc,host.getkey('model name'),host.getkey('cpu cores'),host.getkey('cpu MHz')]) for host in self.hosts),hdrs=('name','nproc','model','cores','MHz'),fmts=(str,str,hist,hist,hist),title='{0.oid}: {0.tstamp}'.format(self))

#==================================================================================================
class Host(Base):
#==================================================================================================
  name = Column(Text(),nullable=False)
  nproc = Column(Integer(),nullable=False)

  context_oid = Column(Integer(),ForeignKey('Context.oid'))
  context = relationship('Context',back_populates='hosts')
  procinfos = relationship('Procinfo',back_populates='host',cascade='all, delete, delete-orphan')

  def getkey(self,key,procn=None):
    if procn is None: filtr = lambda n: True
    elif isinstance(procn,set): filtr = lambda n,s=procn: n in s
    elif isinstance(n,int): lambda n,t=procn: n==t
    else: raise Exception('Invalid process number specification')
    for p in self.procinfos:
      if filtr(p.procn) and p.key==key: yield p.val

#==================================================================================================
class Procinfo(Base):
#==================================================================================================
  procn = Column(Integer(),nullable=False)
  key = Column(Text(),nullable=False)
  val = Column(Text(),nullable=False)

  host_oid = Column(Integer(),ForeignKey('Host.oid'),nullable=False)
  host = relationship('Host',back_populates='procinfos')

#==================================================================================================
class Root (ormsroot):
#==================================================================================================

  base = Context

  def addcontext(self,ctx):
    r"""
Creates a :class:`Context` instance associated with the list of hosts specified by *hosts*. If *hosts* is a string, it is interpreted as a regular expression filtering the host names taken from the list of know hosts. Otherwise, it must be a list of host names.
    """
    with self.session.begin_nested():
      context = self.session.merge(ctx)
      self.session.flush()
    return context

sessionmaker = Root.sessionmaker

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def knownhosts(source=None):
  r"""
Enumerates hostnames from file *source* with the same format as ``~/.ssh/known_hosts`` (default).
  """
#--------------------------------------------------------------------------------------------------
  import os
  if source is None: source = os.path.expanduser('~/.ssh/known_hosts')
  with open(source) as u:
    for x in u:
      x = x.strip()
      if x: yield x.split(None,1)[0].split(',',1)[0]
