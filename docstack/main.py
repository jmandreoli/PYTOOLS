# File:                 main.py
# Creation date:        2014-05-13
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              finding documents in a haystack
#
# *** Copyright (c) 2014 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import os, sys, logging, subprocess, collections, shutil, errno
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
  TARGETS = ()

  def __init__(self,dirn):
    self.DIR = Path(dirn).resolve()
    assert self.DIR.exists()
    self.META = self.DIR/'metadir'
    self.INDEX = self.DIR/'inddir'
    self.ix_ = None

  @property
  def ix(self):
    if self.ix_ is None: self.ix_ = whoosh.index.open_dir(str(self.INDEX))
    return self.ix_

  def __getstate__(self):
    self.ix_ = None
    return self.__dict__

#--------------------------------------------------------------------------------------------------
  def reset(self):
    """
Resets the whole document index and attached meta directory.
    """
#--------------------------------------------------------------------------------------------------
    try: shutil.rmtree(str(self.META))
    except: pass
    try: shutil.rmtree(str(self.INDEX))
    except: pass
    self.META.mkdir()
    self.INDEX.mkdir()
    schema = {}
    schema.update(self.SCHEMA)
    schema.update(self.METASCHEMA)
    schema = whoosh.fields.Schema(**schema)
    whoosh.index.create_in(str(self.INDEX),schema)

#--------------------------------------------------------------------------------------------------
  def updateentries(self):
    """
Inserts the new entries into the index and deletes old ones.
    """
#--------------------------------------------------------------------------------------------------
    D = collections.defaultdict(lambda: [None,None])
    with self.ix.searcher() as ixs:
      for doc in ixs.documents(): D[doc['src']][1] = doc['oid']
    for fpath,filtr in self.TARGETS:
      for fr in fpath.glob(filtr): D[str(fr)][0] = fr
    Ladd, Ldel = [],[]
    for fr,oid in D.values():
      if fr is None: Ldel.append(oid)
      elif oid is None: Ladd.append(fr)
    with self.ix.writer() as ixw:
      for oid in Ldel:
        f = (self.META/oid)
        for ext in ('.wait','.log'):
          try: f.with_suffix(ext).unlink()
          except OSError as e:
            if e.errno!=errno.ENOENT: raise
        self.metaclear(self.META/oid)
        n = ixw.delete_by_term('oid',oid)
        assert n==1, 'unable to delete {}'.format(oid)
        logger.info('deleted %s',oid)
      for fr in Ladd:
        try:
          with fr.open('rb') as u: x = u.read()
        except OSError as e:
          logger.warn('error opening %s: %s',fr,e.stderror)
          continue
        sig = md5(x).hexdigest()
        while True:
          oid = 'M{:06d}'.format(randrange(0,1000000))
          f = (self.META/oid).with_suffix('.log')
          try: n = os.open(str(f),os.O_EXCL|os.O_CREAT,mode=0o666)
          except OSError as e:
            if e.errno==errno.EEXIST: continue
            raise
          else: os.close(n); break
        f.with_suffix('.wait').touch()
        ixw.add_document(src=str(fr),signature=sig,version=fr.stat().st_mtime,oid=oid,active=False,final=False)
        logger.info('inserted %s',oid)

#--------------------------------------------------------------------------------------------------
  def updatemeta(self):
#--------------------------------------------------------------------------------------------------
    def worker(w):
      label = '[{}#{}]'.format(w.uname().nodename,w.getpid())
      logger.info('metaprocess%s.start',label)
      for r in IterableProxy(w.metaprocess(self)): logger.info('metaprocess%s.%s',label,r)
      logger.info('metaprocess%s.stop',label)
    ws = self.engines(len(tuple(self.META.glob('*.wait'))))
    assert ws, 'Unable to launch enough metaprocesses'
    for w in ws:
      w.declare(metaprocess,static=False)
      w.declare(os.getpid)
      w.declare(os.uname)
    L = tuple(Thread(target=worker,args=(w,)) for w in ws)
    for t in L: t.start()
    for t in L: t.join()
    ws.shutdown()

  def metaprocess(self):
    with self.ix.searcher() as ixs:
      L = list((doc['oid'],doc['src']) for doc in ixs.documents(active=False))
    for oid,fr in L:
      f = (self.META/oid).with_suffix('.wait')
      try: f.unlink()
      except OSError as e:
        if e.errno!=errno.ENOENT: raise
        continue
      with f.with_suffix('.log').open('w') as v:
        sys.stderr = v
        for op,status in self.metamake(f,fr,self.TIMEOUT): yield '{}[{}]:{}'.format(op,oid,status)

#--------------------------------------------------------------------------------------------------
  def updateindex(self,newdocs=None):
    """
Updates all the index entries with newly computed metadata.
    """
#--------------------------------------------------------------------------------------------------
    def newdocs_initial():
      with self.ix.searcher() as ixs:
        for doc in ixs.documents(active=False):
          doc.update(self.metasetdoc(self.META/doc['oid'],full=True),active=True)
          yield doc
    if newdocs is None: newdocs = newdocs_initial()
    with self.ix.writer() as ixw:
      for doc in newdocs:
        logger.info('indexing %s',doc['oid'])
        ixw.update_document(**doc)
      logger.info('committing index')
    logger.info('optimising index')
    self.ix.optimize()

#--------------------------------------------------------------------------------------------------
  def update(self):
    """
Update all.
    """
#--------------------------------------------------------------------------------------------------
    with LogStep(logger,'update',showdate=True):
      with LogStep(logger,'update.entries'): self.updateentries()
      with LogStep(logger,'update.meta'): self.updatemeta()
      with LogStep(logger,'update.index'): self.updateindex()

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
