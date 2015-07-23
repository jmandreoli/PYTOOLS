# File:                 main.py
# Creation date:        2014-05-13
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              finding documents in a haystack
#
# *** Copyright (c) 2014 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import os, sys, logging, subprocess, collections, shutil, errno, traceback, time
import whoosh.index, whoosh.fields
from threading import Thread
from pathlib import Path
from random import randrange
from hashlib import md5
from .util import LogStep
logger = logging.getLogger(__name__)

#==================================================================================================
class DocStack:
  """
An object of this class offers an API to manage a set of documents.

Methods:
  """
#==================================================================================================

  TIMEOUT = 600
  SCHEMA = dict(
    oid=whoosh.fields.ID(stored=True,unique=True),
    src=whoosh.fields.STORED(),
    signature=whoosh.fields.ID(stored=True),
    version=whoosh.fields.NUMERIC(stored=True,sortable=True),
    active=whoosh.fields.BOOLEAN(stored=True),
    final=whoosh.fields.BOOLEAN(stored=True),
    )
  METASCHEMA = {}
  # targets MUST BE RESOLVED Path instances (no move) 
  targets = ()

  def __init__(self,dirn):
    self.path = Path(dirn).resolve()
    assert self.path.exists()
    self.pathm = self.path/'metadir'
    self.pathi = self.path/'inddir'
    self.ix_ = None
    queue = Queue()
    metaqueue = Queue()
    threading.Thread(target=self.process_target,args=(queue,),daemon=True).start()
    threading.Thread(target=self.process_meta,args=(metaqueue,),daemon=True).start()
    self.init_target(queue)
    self.init_meta(metaqueue)

  @property
  def ix(self):
    if self.ix_ is None: self.ix_ = whoosh.index.open_dir(str(self.pathi))
    return self.ix_

  def __getstate__(self):
    self.ix_ = None
    return self.__dict__

  def init_target(self,queue):
    def ckalive(t):
      nonlocal alive
      for f in list(self.pathm.glob('*.work')):
        if time.time()-f.stats().st_mtime>t:
          try: f.unlink()
          except OSError as e:
            if e.errno!=errno.ENOENT: raise
          self.metaclear(f)
          queue.put((0,f))
      alive = threading.Timer(t,ckalive,(t,))
      alive.daemon = True
      alive.start()
    alive = None
    ckalive(300)

  def process_target(self,queue):
    D = collections.defaultdict(lambda: [None,None])
    with self.ix.searcher() as ixs:
      for doc in ixs.documents(): fr = doc['src']; D[fr][1] = -1,(doc['oid'],fr)
    self.watch_target(D,queue)
    for fpath,filtr in self.targets:
      for fR in fpath.glob(filtr): D[str(fR)][0] = +1,fR
    for ops in D.values():
      if ops[1] is None: queue.put(ops[0])
      elif ops[0] is None: queue.put(ops[1])
    while True:
      cbuf = accumulate(queue,15)
      new = []
      with self.ix.writer() as ixw:
        for op,x in cbuf:
          if op<0:
            oid,fr = x
            del D[fr]
            f = self.pathm/oid
            for ext in ('.wait','.work','.ready','.error','.log'):
              try: f.with_suffix(ext).unlink()
              except OSError as e:
                if e.errno!=errno.ENOENT: raise
            self.metaclear(f)
            n = ixw.delete_by_term('oid',oid)
            assert n==1, 'unable to delete {}'.format(oid)
            logger.info('deleted %s',oid)
          elif op>0:
            fR = x; fr = str(x)
            with fR.open('rb') as u: x = u.read()
            sig = md5(x).hexdigest()
            while True:
              oid = 'M{:06d}'.format(randrange(0,1000000))
              f = (self.pathm/oid).with_suffix('.log')
              try: n = os.open(str(f),os.O_EXCL|os.O_CREAT,mode=0o666)
              except OSError as e:
                if e.errno==errno.EEXIST: continue
                raise
              else: os.close(n); break
            new.append(f)
            ixw.add_document(oid=oid,src=fr,signature=sig,version=fR.stat().st_mtime,active=False,final=False)
            D[fr][1] = -1,(oid,fr)
            logger.info('inserted %s',oid)
          else: new.append(x)
      for f in new: f.with_suffix('.wait').touch()
      self.launch(len(new))

  def watch_target(self,D,queue):
    import pyinotify
    def process(e):
      if e.mask&(pyinotify.IN_DELETE|pyinotify.IN_MOVED_FROM): queue.put(D[e.pathname][1])
      elif e.mask&pyinotify.IN_MOVED_TO: queue.put((+1,Path(e.pathname)))
    wm = pyinotify.WatchManager()
    for fpath,filtr in self.targets:
      wm.add_watch(str(fpath),pyinotify.IN_DELETE|pyinotify.IN_MOVED_FROM|pyinotify.IN_MOVED_TO)
    nt = pyinotify.ThreadedNotifier(wm,default_proc_fun=process)
    nt.daemon = True
    nt.start()

  def init_meta(self,queue):
    for f in list(self.pathm.glob('*.ready')): queue.put(f)

  def process_meta(self,queue):
    while True:
      cbuf = accumulate(queue,60)
      with self.ix.writer() as ixw:
        with self.ix.searcher() as ixs:
          for f in cbuf:
            try: f.unlink()
            except OSError as e:
              if e.errno!=errno.ENOENT: raise
              continue
            doc = ixs.document(oid=f.name,active=False)
            if doc is None: continue
            logger.info('indexing %s',doc['oid'])
            doc.update(self.metasetdoc(self.pathm/doc['oid'],full=True),active=True)
            ixw.update_document(**doc)
        logger.info('committing index')
      logger.info('optimising index')
      self.ix.optimize()

  def genmeta(self):
    def mkalive(t):
      nonlocal alive
      if current is not None:
        try: current.touch()
        except: pass
      alive = threading.Timer(t,mkalive,(t,))
      alive.daemon = True
      alive.start()
    with self.ix.searcher() as ixs:
      L = list((doc['oid'],doc['src']) for doc in ixs.documents(active=False))
    current = None
    alive = None
    mkalive(60)
    for oid,fr in L:
      f = (self.pathm/oid).with_suffix('.wait')
      f2 = f.with_suffix('.work')
      try: f.rename(f2)
      except OSError as e:
        if e.errno!=errno.ENOENT: raise
        continue
      current = f2
      f2.touch()
      with f.with_suffix('.log').open('w') as v:
        sys.stdout = sys.stderr = v
        try:
          for op,status in self.metamake(f,fr,self.TIMEOUT):
            print('{}[{}]:{}'.format(op,oid,status))
        except:
          traceback.print_exc()
          res = '.error'
        else: res= '.ready'
        sys.stdout.flush()
        sys.stderr.flush()
      f2.rename(f.with_suffix(res))
      current = None

def accumulate(queue,t):
  cbuf = []
  while True:
    while True:
      while True:
        try: cbuf.append(queue.get(False))
        except Empty: break
      try: cbuf.append(queue.get(False,t))
      except Empty: break
    if cbuf: return cbuf
    cbuf.append(queue.get())

#--------------------------------------------------------------------------------------------------
  def load(self,path=None):
    """
Updates all the entries with metadata from a file saved by dump (if any).
    """
#--------------------------------------------------------------------------------------------------
    import pickle
    def newdocs():
      fpath = self.DIR/'ind.pck' if path is None else Path(path).resolve()
      with fpath.open('rb') as u: state = pickle.load(u)
      logger.info('loading from %s',fpath)
      with self.ix.searcher() as ixs:
        for doc in ixs.documents(active=True):
          if doc['final']: continue
          x = state.get(doc['signature'])
          if x is not None:
            oid = doc['oid']
            doc.update(self.metasetdoc(self.META/oid,full=True))
            doc.update(x,final=True)
            yield doc
    with LogStep(logger,'load',showdate=True): self.updateindex(newdocs())

#--------------------------------------------------------------------------------------------------
  def dump(self,path=None):
    """
Dump all final entries stored metadata to a file.
    """
#--------------------------------------------------------------------------------------------------
    import pickle
    fpath = self.DIR/'ind.pck' if path is None else Path(path)
    with LogStep(logger,'dump',showdate=True):
      fields = tuple(k for k,f in self.METASCHEMA.items() if f.stored)
      logger.info('collecting datafields (%s)',','.join(fields))
      with self.ix.searcher() as ixs:
        state = dict((doc['signature'],tuple((k,doc[k]) for k in fields)) for doc in ixs.documents(final=True))
      logger.info('dumping to %s',fpath)
      with fpath.open('wb') as v: pickle.dump(state,v,-1)

#--------------------------------------------------------------------------------------------------
  def editor(self):
    """
Opens an editor window for metadata correction.
    """
#--------------------------------------------------------------------------------------------------
    self.viewer(self.editorwindow())

#--------------------------------------------------------------------------------------------------
  def browser(self):
    """
Opens a browser window for document search.
    """
#--------------------------------------------------------------------------------------------------
    self.viewer(self.browserwindow())

  def viewer(self,vobj):
    from .viewer import run
    sys.exit(run(vobj,sys.argv))

#==================================================================================================
# Utilities
#==================================================================================================

def metaprocess(server,mgr):
  from ..remote import Iterable
  return Iterable(server._pyroDaemon,mgr.metaprocess())
