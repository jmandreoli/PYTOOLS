import subprocess, logging
from datetime import datetime
from . import SQLinit
logger = logging.getLogger(__name__)

def knownhosts(source=None):
  import os
  if source is None:
    source = os.path.join(os.path.expanduser('~/.ssh/known_hosts'))
  with open(source) as u:
    for x in u:
      x = x.strip()
      if x: yield x.split(None,1)[0].split(',',1)[0]

def schema(meta):
  from sqlalchemy import Table, Column, ForeignKey
  from sqlalchemy.types import Text, Integer, DateTime
  Table(
    'Session',meta,
    Column('oid',Integer(),primary_key=True),
    Column('tstamp',DateTime()),
  )
  Table(
    'Host',meta,
    Column('oid',Integer(),primary_key=True),
    Column('session',Integer(),ForeignKey('Session.oid',ondelete='CASCADE'),nullable=False),
    Column('name',Text(),nullable=False),
    Column('nproc',Integer(),nullable=False),
  )
  Table(
    'Processor',meta,
    Column('host',Integer(),ForeignKey('Host.oid',ondelete='CASCADE'),nullable=False),
    Column('procn',Integer(),nullable=False),
    Column('key',Text(),nullable=False),
    Column('val',Text(),nullable=False),
  )
schema.status = dict(origin=__name__,version=2)

def makebase(engine,hosts=None):
  def parse(D): x = D[0]; assert x[0].lower()=='processor'; return int(x[1]),D[1:]
  K,H = [],[]
  if hosts is None: hosts = knownhosts()
  for host in hosts:
    logger.info('Examining host: %s',host)
    p = subprocess.run(('ssh','-q','-n','-x','-T',host,'cat','/proc/cpuinfo'),stdout=subprocess.PIPE,universal_newlines=True)
    L,D = [],[]
    for y in p.stdout.split('\n'):
      y = y.strip()
      if not y:
        if D: L.append(parse(D))
        D=[]; continue
      key,val = y.split(':',1)
      D.append((key.strip(),val.strip()))
    if D: L.append(parse(D))
    H.append((host,L))
  H.sort()
  engine,meta = SQLinit(engine,schema)
  session_t = meta['Session']
  host_t = meta['Host']
  processor_t = meta['Processor']
  with engine.connect() as conn:
    s = conn.execute(session_t.insert().values(tstamp=datetime.now())).inserted_primary_key[0]
    for host,L in H:
      logger.info('Dumping host %s',host)
      h = conn.execute(host_t.insert().values(session=s,name=host,nproc=len(L))).inserted_primary_key[0]
      for proc,D in L:
        conn.execute(processor_t.insert().values([dict(host=h,procn=proc,key=k,val=v) for k,v in D]))
