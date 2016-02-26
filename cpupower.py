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

def makebase(engine,hosts=None):
  def schema(meta):
    from sqlalchemy import Table, Column, ForeignKey
    from sqlalchemy.types import Text, Integer, DateTime
    cols = [Column(key,Text()) for key in K]
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
      *cols)
  schema.status = dict(version=1)
  K,Kd,H = [],{},[]
  if hosts is None: hosts = knownhosts()
  for host in hosts:
    logger.info('Examining host: %s',host)
    p = subprocess.run(('ssh','-q','-n','-x','-T',host,'cat','/proc/cpuinfo'),stdout=subprocess.PIPE,universal_newlines=True)
    L,D = [],{}
    for y in p.stdout.split('\n'):
      y = y.strip()
      if not y: L.append(D); D={}; continue
      key,val = y.split(':',1)
      key = key.strip().replace(' ','_')
      if Kd.get(key) is None: Kd[key] = 1; K.append(key)
      D[key] = val.strip()
    assert not(D)
    H.append((host,L))
  H.sort()
  engine,meta = SQLinit(engine,schema)
  logger.info('Columns: %s',K)
  sess_t = meta['Session']
  host_t = meta['Host']
  proc_t = meta['Processor']
  with engine.connect() as conn:
    s = conn.execute(sess_t.insert().values(tstamp=datetime.now())).inserted_primary_key[0]
    for host,L in H:
      logger.info('Dumping host %s',host)
      i = conn.execute(host_t.insert().values(session=s,name=host,nproc=len(L))).inserted_primary_key[0]
      for D in L:
        conn.execute(proc_t.insert().values(host=i,**dict((key,D.get(key)) for key in K)))
