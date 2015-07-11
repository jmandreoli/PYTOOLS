import os, subprocess, sqlite3, logging
logger = logging.getLogger(__name__)

def knownhosts(source=None):
  if source is None:
    source = os.path.join(os.environ['HOME'],'.ssh/known_hosts')
  with open(source) as u:
    for x in u:
      x = x.strip()
      if x: yield x.split(None,1)[0].split(',',1)[0]

def makebase(hosts,dbname):
  K = []
  Kd = {}
  H = []
  for host in hosts:
    logger.info('Examining host: %s',host)
    p = subprocess.Popen(('ssh','-q','-n','-x','-T',host,'cat','/proc/cpuinfo'),stdout=subprocess.PIPE,universal_newlines=True)
    L = []
    D = {}
    for y in p.stdout:
      y = y.strip()
      if not y: L.append(D); D={}; continue
      key,val = y.split(':',1)
      key = key.strip()
      if Kd.get(key) is None:
        Kd[key] = 1
        K.append(key)
      D[key] = val.strip()
    assert not(D)
    H.append((host,L))
  H.sort()
  colnames = ','.join(key.replace(' ','_') for key in K)
  colholes = ','.join(len(K)*('?',))
  logger.info('Columns: %s',colnames)
  try: os.remove(dbname)
  except: pass
  conn = sqlite3.connect(dbname)
  c = conn.cursor()
  c.execute('CREATE TABLE Host (name TEXT, nproc INT)')
  c.execute('CREATE TABLE Processor (host INT REFERENCES Host,%s)'%colnames)
  c.close()
  c = conn.cursor()
  for host,L in H:
    logger.info('Dumping host %s',host)
    c.execute('INSERT INTO Host (name, nproc) VALUES (?,?)',(host,len(L)))
    i = c.lastrowid
    for D in L:
      c.execute('INSERT INTO Processor (host,%s) VALUES (?,%s)'%(colnames,colholes),(i,)+tuple(D.get(key) for key in K))
  c.close()
  conn.commit()
