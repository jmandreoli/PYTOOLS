# File:                 cache.py
# Creation date:        2015-03-19
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Persistent cache management
#
# *** Copyright (c) 2015 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import logging
logger = logging.getLogger(__name__)

import os, sqlite3, pickle, inspect, threading
from pathlib import Path
from functools import update_wrapper
from collections import namedtuple
from collections.abc import MutableMapping
from . import SQliteNew

# Data associated with each cell is kept in a separate file
# in the same folder as the database

SCHEMA = '''
CREATE TABLE Block (
  oid INTEGER PRIMARY KEY AUTOINCREMENT,
  signature PICKLE UNIQUE NOT NULL,
  hits INTEGER DEFAULT 0,
  misses INTEGER DEFAULT 0,
  maxsize INTEGER DEFAULT 10
  )

CREATE TABLE Cell (
  oid INTEGER PRIMARY KEY AUTOINCREMENT,
  hitdate TIMESTAMP DEFAULT ( datetime('now') ),
  block REFERENCES Block(oid) NOT NULL,
  ckey PICKLE NOT NULL,
  size INTEGER DEFAULT 0
  )

CREATE VIEW Overflow AS
  SELECT Block.oid AS block, count(*)-maxsize AS osize
  FROM Block, Cell
  WHERE block = Block.oid AND Cell.size>0
  GROUP BY Block.oid, maxsize
  HAVING osize>0

CREATE UNIQUE INDEX CellIndex ON Cell (block, ckey)

CREATE INDEX CellIndex2 ON Cell (hitdate)

CREATE TRIGGER BlockDeleteTrigger
  AFTER DELETE ON Block
  BEGIN
    DELETE FROM Cell WHERE block=OLD.oid;
  END

CREATE TRIGGER CellDeleteTrigger
  AFTER DELETE ON Cell
  BEGIN
    SELECT cellrm(OLD.oid,OLD.size);
  END
'''

sqlite3.register_converter('PICKLE',pickle.loads)
sqlite3.enable_callback_tracebacks(True)

#==================================================================================================
class CacheDB (MutableMapping):
  r"""
Instances of this class manage cache folders. There is at most one instance of this class in a process for each absolute folder path. A cache folder contains a set of values stored in some form. Each value has meta information stored in an sqlite database named ``index.db`` in the folder. The index entry attached to a value is called a cell and describes the call event which obtained it. A ``Cell`` entry has the following fields:

- ``oid``: a unique identifier of the cell, which also allows retrieval of the attached value;
- ``block``: reference to another entry in the database (table ``Block``) describing the functor of the call event which produced the value;
- ``ckey``: the argument of the call event which produced the value;
- ``hitdate``: date of creation, or last reuse, of the cell.

``Block`` entries correspond to clusters of ``Cell`` entries (call events) sharing the same functor. They have the following fields

- ``signature``: the persistent functor (callable) producing all the cells in the block; different blocks always have signatures with different pickle byte-strings;
- ``misses``: number of call events with that functor where the argument had not been seen before;
- ``hits``: number of call events with that functor where the argument had previously been seen; such a call does not generate a new ``Cell``, but reuses the existing one;
- ``maxsize``: maximum number of cells attached to the block; when overflow occurs, the cells with the oldest ``hitdate`` are discarded (this amounts to the Least Recently Used policy, a.k.a. LRU, currently hardwired).

Furthermore, a :class:`CacheDB` instance acts as a mapping object, where the keys are block identifiers (:class:`int`) and values are :class:`CacheBlock` objects for the corresponding blocks. Such :class:`CacheBlock` objects are normally deactivated (i.e. heir signatures do not support calls).

Finally, :class:`CacheDB` instances have an HTML ipython display.

Attributes:

.. attribute:: path

   the path of the cache folder, as a :class:`pathlib.Path` instance

.. attribute:: dbpath

   the path of the index path, as a string

Methods:

.. automethod:: __new__
  """
#==================================================================================================

  def __new__(cls,spec,listing={},lock=threading.Lock()):
    """
Generates a :class:`CacheDB` object.

:param spec: specification of the cache folder
:type spec: :class:`CacheDB`\|\ :class:`pathlib.Path`\|\ :class:`str`
    """
    if isinstance(spec,CacheDB): return spec
    elif isinstance(spec,str): path = Path(spec)
    elif isinstance(spec,Path): path = spec
    else: raise TypeError('Expected: {}|{}|{}; Found: {}'.format(str,Path,CacheDB,type(spec)))
    path = path.resolve()
    with lock:
      self = listing.get(path)
      if self is None:
        self = super(CacheDB,cls).__new__(cls)
        self.path = path
        self.dbpath = str(path/'index.db')
        check = (lambda:'cannot create index in non empty folder') if any(path.iterdir()) else (lambda:None)
        r = SQliteNew(self.dbpath,SCHEMA,check)
        if r is not None: raise WrongCacheFolderException(path,r)
        self.storage = Storage(self)
        listing[path] = self
    return self
  
  def __getnewargs__(self): return str(self.path),

  def connect(self,**ka):
    conn = sqlite3.connect(self.dbpath,**ka)
    conn.create_function('cellrm',2,lambda cell,size,s=self.storage: s(cell).remove(size>0))
    return conn

  def getblock(self,sig):
    sigp = pickle.dumps(sig)
    with self.connect(isolation_level='IMMEDIATE') as conn:
      row = conn.execute('SELECT oid FROM Block WHERE signature=?',(sigp,)).fetchone()
      if row is None:
        return conn.execute('INSERT INTO Block (signature) VALUES (?)',(sigp,)).lastrowid
      else:
        return row[0]

#--------------------------------------------------------------------------------------------------
# CacheDB as Mapping
#--------------------------------------------------------------------------------------------------

  def __getitem__(self,block):
    with self.connect(detect_types=sqlite3.PARSE_DECLTYPES) as conn:
      r = conn.execute('SELECT signature FROM Block WHERE oid=?',(block,)).fetchone()
    if r is None: raise KeyError(block)
    return CacheBlock(db=self,signature=r[0],block=block)

  def __delitem__(self,block):
    with self.connect() as conn:
      conn.execute('DELETE FROM Block WHERE oid=?',(block,))
      if conn.total_changes==0: raise KeyError(block)

  def __setitem__(self,block,v):
    raise Exception('Direct create/update not permitted on Block')

  def __iter__(self):
    with self.connect() as conn:
      for block, in conn.execute('SELECT oid FROM Block'): yield block

  def __len__(self):
    with self.connect() as conn:
      return conn.execute('SELECT count(*) FROM Block').fetchone()[0]

  def items(self):
    with self.connect(detect_types=sqlite3.PARSE_DECLTYPES) as conn:
      for block,sig in conn.execute('SELECT oid,signature FROM Block'):
        yield block, CacheBlock(db=self,signature=sig,block=block)

#--------------------------------------------------------------------------------------------------
# Display
#--------------------------------------------------------------------------------------------------

  def _repr_html_(self):
    from lxml.etree import tounicode
    return tounicode(self.as_html())
  def as_html(self):
    return html_stack(*(v.as_html() for k,v in sorted(self.items())))
  def __str__(self): return 'Cache<{}>'.format(self.path)

#==================================================================================================
class CacheBlock (MutableMapping):
  r"""
Instances of this class implements blocks of cells sharing the same functor (signature).

:param db: specification of the cache where the block resides
:type db: :class:`CacheDB`\|\ :class:`str`\|\ :class:`pathlib.Path`
:param signature: signature of the block, describing its functor
:type signature: :class:`Signature`
:param maxsize: if not :const:`None`, resizes the block at initialisation
:type maxsize: :class:`int`\|\ :class:`NoneType`
:param clear: if :const:`True`, clears the block at initialisation
:type clear: :class:`bool`

A block object is callable. When a new call is submitted to a block, with positional arguments *a* and keyword arguments *ka*, the following sequence happens:

- method :meth:`keyfunc` of the signature is invoked with the two arguments *a* and *ka*. It must return a pickable python object. The pickled result is used as ``ckey`` in the corresponding index database cell.
- Then a transaction is begun on the index database.
- If there already exists a cell with the same ``ckey``, and its value has been computed, the transaction is immediately terminated and its value is unpickled and returned.
- If there already exists a cell with the same ``ckey``, but its value is still being computed in another thread/process, the transaction is terminated, then the thread waits until completion of the other thread/process, then the obtained value is unpickled and returned. If the cell has been removed (e.g. when the other thread/process results in failure), an error is raised.
- If there does not exist a cell with the same ``ckey``, a cell with this ``ckey`` is immediately created, then the transaction is terminated. Then method :meth:`valfunc` of the signature is invoked with *a* as positional arguments and *ka* as keyword arguments. The result is pickled. If an error occurs, a compensating transaction removes the created cell and the error is raised. Otherwise a new transaction informs the database cell that its result has been computed.

Furthermore, a :class:`CacheBlock` object acts as a mapping where the keys are cell identifiers (:class:`int`) and values are triples ``hitdate``, ``ckey``, ``size``. When ``size`` is null, the value of the cell is still being computed, otherwise it represents the size in bytes of its pickle.

Finally, :class:`CacheBlock` instances have an HTML ipython display.

Methods:
  """
#==================================================================================================

  def __init__(self,db=None,signature=None,block=None,maxsize=None,clear=False):
    self.sig = signature
    self.db = db = CacheDB(db)
    self.block = db.getblock(signature) if block is None else block
    if maxsize is not None: self.resize(maxsize)
    if clear: self.clear()

  def clear(self):
    with self.db.connect() as conn:
      conn.execute('DELETE FROM Cell WHERE block=?',(self.block,))

  def resize(self,n):
    assert isinstance(n,int) and n>=1
    with self.db.connect() as conn:
      conn.execute('UPDATE Block SET maxsize=? WHERE oid=?',(n,self.block,))
    self.checkmax()

  def info(self,typ=namedtuple('CellInfo',('signature','hits','misses','maxsize','currsize'))):
    r"""
Returns information about this block. Available attributes:
:attr:`signature`, :attr:`hits`, :attr:`misses`, :attr:`maxsize`, :attr:`currsize`
    """
    with self.db.connect(detect_types=sqlite3.PARSE_DECLTYPES) as conn:
      sz = conn.execute('SELECT count(*) FROM Cell WHERE block=?',(self.block,)).fetchone()[0]
      row = conn.execute('SELECT signature, hits, misses, maxsize FROM Block WHERE oid=?',(self.block,)).fetchone()
    return typ(*row,currsize=sz)

#--------------------------------------------------------------------------------------------------
  def __call__(self,*a,**ka):
#--------------------------------------------------------------------------------------------------
    ckey = pickle.dumps(self.sig.keyfunc(a,ka))    
    with self.db.connect(detect_types=sqlite3.PARSE_DECLTYPES,isolation_level='IMMEDIATE') as conn:
      row = conn.execute('SELECT oid,size FROM Cell WHERE block=? AND ckey=?',(self.block,ckey,)).fetchone()
      if row is None:
        cell = conn.execute('INSERT INTO Cell (block,ckey) VALUES (?,?)',(self.block,ckey)).lastrowid
        conn.execute('UPDATE Block SET misses=misses+1 WHERE oid=?',(self.block,))
        size = None
      else:
        conn.execute('UPDATE Block SET hits=hits+1 WHERE oid=?',(self.block,))
        cell,size = row
      store = self.db.storage(cell,size)
    if row is None:
      logger.info('%s MISS(%s)',self,cell)
      try:
        cval = self.sig.valfunc(*a,**ka)
        store.setval(cval)
      except:
        with self.db.connect() as conn:
          conn.execute('DELETE FROM Cell WHERE oid=?',(cell,))
        raise
      with self.db.connect(isolation_level='IMMEDIATE') as conn:
        if conn.execute('SELECT oid FROM Cell WHERE oid=?',(cell,)).fetchone() is None:
          logger.info('%s LOST(%s)',self,cell)
        else:
          size = store.commit()
          conn.execute('UPDATE Cell SET size=?, hitdate=datetime(\'now\') WHERE oid=?',(size,cell))
      self.checkmax()
    else:
      if size==0: logger.info('%s WAIT(%s)',self,cell)
      cval = store.getval()
      logger.info('%s HIT(%s)',self,cell)
      with self.db.connect() as conn:
        conn.execute('UPDATE Cell SET hitdate=datetime(\'now\') WHERE oid=?',(cell,))
    return cval

#--------------------------------------------------------------------------------------------------
  def checkmax(self):
    r"""
Checks whether there is a cache overflow and applies the LRU policy.
    """
#--------------------------------------------------------------------------------------------------
    with self.db.connect(isolation_level='IMMEDIATE') as conn:
      row = conn.execute('SELECT osize FROM Overflow WHERE block=?',(self.block,)).fetchone()
      if row is not None:
        osize = row[0]
        logger.info('%s OVERFLOW(%s)',self,osize)
        conn.execute('DELETE FROM Cell WHERE oid IN (SELECT oid FROM Cell WHERE block=? AND size>0 ORDER BY hitdate ASC, oid ASC LIMIT ?)',(self.block,osize,))

#--------------------------------------------------------------------------------------------------
# CacheBlock as Mapping
#--------------------------------------------------------------------------------------------------

  def __getitem__(self,cell):
    with self.db.connect(detect_types=sqlite3.PARSE_DECLTYPES) as conn:
      r = conn.execute('SELECT hitdate, ckey, size FROM Cell WHERE oid=? AND block=?',(cell,self.block)).fetchone()
    if r is None: raise KeyError(cell)
    return r

  def __delitem__(self,cell):
    with self.db.connect() as conn:
      conn.execute('DELETE FROM Cell WHERE oid=? AND block=?',(cell,self.block))
      if conn.total_changes==0: raise KeyError(cell)

  def __setitem__(self,cell,v):
    raise Exception('Direct create/update not permitted on Cell')

  def __iter__(self):
    with self.db.connect() as conn:
      for cell, in conn.execute('SELECT oid FROM Cell WHERE block=?',(self.block,)): yield cell

  def __len__(self):
    with self.db.connect() as conn:
      return conn.execute('SELECT count(*) FROM Cell WHERE block=?',(self.block,)).fetchone()[0]

  def items(self):
    with self.db.connect(detect_types=sqlite3.PARSE_DECLTYPES) as conn:
      for row in conn.execute('SELECT oid, hitdate, ckey, size FROM Cell WHERE block=?',(self.block,)):
        yield row[0],row[1:]

#--------------------------------------------------------------------------------------------------
# Display
#--------------------------------------------------------------------------------------------------

  def _repr_html_(self):
    from lxml.etree import tounicode
    return tounicode(self.as_html())
  def as_html(self):
    return html_table(sorted(self.items()),hdrs=('hitdate','ckey','size'),fmts=(str,self.sig.html,str),title='{}: {}'.format(self.block,self.sig))
  def __str__(self): return 'Cache<{}:{}>'.format(self.db.path,self.sig)      

#==================================================================================================
# Signatures
#==================================================================================================

#--------------------------------------------------------------------------------------------------
class Signature:
  r"""
An instance of this class defines a functor attached to a python top-level function. Arguments which do not influence the result can be specified (they will be ignored in the caching mechanism).

:param func: a function, defined at the top-level of its module (hence pickable)
:param ignore: a list of arguments among those of *func*

The signature is entirely defined by the name of the function and of its module, as well as the sequence of argument names, marked if ignored. Hence, two functions (possibly in different processes at different times) sharing these components produce the same signature.

The :meth:`keyfunc` method applied to a pair *a*, *ka*, where *a* is a list of positional arguments and *ka* a dict of keyword arguments, will match these arguments against the definition of the function and return the assignment of the argument names of the function in the order in which they appear in its definition, omitting the ignored ones. If an argument name is prefixed by \*\*, its assignment must be a dict, which is replaced by its sorted list of items.

Methods:
  """
#--------------------------------------------------------------------------------------------------
  def __init__(self,func,ignore):
    func = inspect.unwrap(func)
    self.valfunc = func
    self.name, self.params = '{}.{}'.format(func.__module__, func.__name__), tuple(getparams(func,ignore))
  def keyfunc(self,a,ka):
    d = inspect.getcallargs(self.valfunc,*a,**ka)
    return tuple((p,(sorted(d[p].items()) if typ==2 else d[p])) for p,typ in self.params if typ!=-1)
  def html(self,ckey):
    from lxml.builder import E
    disp = (lambda k,v: E.DIV(E.B(k),'=',E.EM(repr(v)),style='display: inline; padding: 5px;'))
    def h(ckey,dparams):
      for p,v in ckey:
        if dparams[p]==2: yield from map(disp,v)
        else: yield disp(p,v)
    return E.DIV(*h(ckey,dict(self.params)))
  def __str__(self,mark={-1:'-',0:'',1:'*',2:'**'}): return '{}({})'.format(self.name,','.join(mark[typ]+p for p,typ in self.params))
  def __getstate__(self): return self.name, self.params
  def __setstate__(self,state): self.name, self.params = state
  def restore(self):
    r"""
Attempts to restore the :attr:`valfunc` attribute from the other attributes. This is useful with instances obtained by unpickling, since the :attr:`valfunc` attribute is not automatically restored. This may be risky however, as the function may have changed since pickling time.
    """
    if hasattr(self,'valfunc'): return
    from importlib import import_module
    fmod,fname = self.name.rsplit('.',1)
    valfunc = inspect.unwrap(getattr(import_module(fmod),fname))
    ignore = tuple(p for p,typ in self.params if typ==-1)
    params = tuple(getparams(valfunc,ignore))
    if params != self.params: raise SignatureMismatchException(params,self.params)
    self.valfunc = valfunc

#--------------------------------------------------------------------------------------------------
class ProcessSignature:
  r"""
An instance of this class defines a functor attached to a list of python top-level functions, executed in sequence, where the first argument of each call is the result of the previous call.

The signature is entirely defined by the components of :class:`Signature` applied to each individual functions.

:param func: a function, defined at the top-level of its module (hence pickable)
:param base: a cache block with a signature which is an instance of :class:`ProcessSignature`, or a callable
:type base: :class:`CacheBlock`\|\ callable
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,func,base):
    func = inspect.unwrap(func)
    self.func = func
    self.base = base
    bnameS,bparamsS,self.keyfuncS  = (),(),(lambda *a: ())
    if isinstance(self.base,CacheBlock):
      bases = self.base.sig
      bnameS,bparamsS,self.keyfuncS = bases.nameS, bases.paramsS, bases.keyfunc
    self.nameS, self.paramsS = bnameS+('{}.{}'.format(func.__module__, func.__name__),), bparamsS+(tuple(getparams(func))[1:],)
  def valfunc(self,*args):
    obj = self.base(*args[:-1])
    a,ka = args[-1]
    return self.func(obj,*a,**ka)
  def keyfunc(self,args,_=None):
    a,ka = args[-1]
    d = inspect.getcallargs(self.func,None,*a,**ka)
    t = tuple((p,(sorted(d[p].items()) if typ==2 else d[p])) for p,typ in self.paramsS[-1])
    return self.keyfuncS(args[:-1])+(t,)
  def html(self,ckeyS):
    from lxml.builder import E
    disp = (lambda k,v: E.DIV(E.B(k),'=',E.EM(repr(v)),style='display: inline; padding: 5px;'))
    def h(ckey,dparams):
      for p,v in ckey:
        if dparams[p]==2: yield from map(disp,v)
        else: yield disp(p,v)
    def hS(ckeyS):
      for params,ckey in zip(self.paramsS,ckeyS):
        yield E.DIV(*h(ckey,dict(params)),style='border-bottom: thin dashed blue')
    return E.DIV(*hS(ckeyS))
  def __str__(self): return ''.join(name+';' for name in self.nameS)
  def __getstate__(self): return self.nameS, self.paramsS
  def __setstate__(self,state): self.nameS, self.paramsS = state

#==================================================================================================
class Storage:
  r"""
Instances of this class manage the storage of cell values of a cache.

:param db: the cache to manage
:type db: :class:`CacheDB`

Method:

.. automethod:: __call__
  """
#==================================================================================================

  def __init__(self,db):
    self.path = db.path
    self.trackers = {},{},{}
    self.watch()

#--------------------------------------------------------------------------------------------------
  def __call__(self,cell,size=-1,typ=namedtuple('StoreAPI',('getval','setval','commit','remove'))):
    r"""
Returns an incarnation of the API to manipulate the value of *cell* (see below).

:param cell: oid of the cell
:type cell: :class:`int`
:param size: current size recorded in the cache
:type size: :class:`int`\|\ :class:`NoneType`

The API to manipulate the value of a cell consists of:

- :func:`getval` invoked (possibly concurrently) to obtain the value of *cell*
- :func:`setval` invoked with an argument *val* to pre-assign the value of *cell* to *val*. Invoked only once, but *cell* may have disappeared from the cache when invoked, or may disappear while executing, so the assignment may have to later be rollbacked.
- :func:`commit` invoked to confirm a pre-assignment made by :func:`setval`. The size of the assigned value (positive number) must be returned. The cache is frozen during execution, and no other cell value operations can happen.
- :func:`remove` invoked with an argument *r* when *cell* is removed (*r* is a boolean indicating whether the cell value is committed). If the cell is being assigned, locally or remotely, this must be eventually rollbacked. The cache is frozen during execution, and no other cell value operations can happen.

The *size* argument gives an indication of the intended use of the API in the current thread:

- When *size* is :const:`None`, the cache expects the cell value to be computed by the current thread, calling :func:`setval` then :func:`commit`.
- When the *size* is :const:`0`, the cache expects the cell value to be computed by another thread/process, which must be waited for, calling :func:`getval`.
- When *size* is a positive number, the cache expects the cell value to have already been computed in the past (with size *size*) and be directly accessible, calling :func:`getval`.
- When *size* is :const:`-1`, the cache intends to remove the cell value, calling :func:`remove`.
    """
#--------------------------------------------------------------------------------------------------
    vpath = self.path/'V{:06d}'.format(cell)
    rpath = vpath.with_suffix('.pck')
    tpath = vpath.with_suffix('.tmp')
    if size is None: # current process is computing the value: setval, commit
      tfile = tpath.open('wb')
    elif size==0:    # other process is computing the value: getval (must wait)
      evt = self.track(tpath.name,rpath.name)
    else:            # value already computed: getval (nowait), or to be removed: remove
      evt = None
    def getval():
      if evt is not None: evt.wait()
      with rpath.open('rb') as u: return pickle.load(u)
    def setval(val):
      pickle.dump(val,tfile)
      tfile.close()
    def commit():
      tpath.rename(rpath)
      return rpath.stat().st_size
    def remove(r):
      try: (rpath if r else tpath).unlink()
      except: pass
    return typ(getval,setval,commit,remove)

  def track(self,tname,rname):
    evt = threading.Event()
    self.trackers[0][evt] = tname, rname
    self.trackers[1][tname] = evt
    self.trackers[2][rname] = evt
    return evt

  def untrack(self,i,name):
    evt = self.trackers[i].get(name)
    if evt is not None:
      tname,rname = self.trackers[0][evt]
      del self.trackers[0][evt], self.trackers[1][tname], self.trackers[2][rname]
      evt.set()

  def watch_darwin(self):
    import fsevents
    def process(e):
      if e.mask&fsevents.IN_DELETE: i=1
      elif e.mask&fsevents.IN_MOVED_TO: i=2
      else: return
      self.untrack(i,os.path.basename(e.name))
    ob = fsevents.Observer()
    ob.schedule(fsevents.Stream(process,str(self.path),file_events=True))
    ob.daemon = True
    ob.start()

  def watch_linux(self):
    import pyinotify
    def process(e):
      if e.mask&pyinotify.IN_DELETE: i=1
      elif e.mask&pyinotify.IN_MOVED_TO: i=2
      else: return
      self.untrack(i,e.name)
    wm = pyinotify.WatchManager()
    wm.add_watch(str(self.path),pyinotify.IN_DELETE|pyinotify.IN_MOVED_TO)
    nt = pyinotify.ThreadedNotifier(wm,default_proc_fun=process)
    nt.daemon = True
    nt.start()

  def watch_win32(self):
    import win32file, win32con, win32event
    D = dict((p.name,1) for p in self.path.iterdir())
    h = win32file.FindFirstChangeNotification(str(self.path),False,win32con.FILE_NOTIFY_CHANGE_FILE_NAME)
    def process():
      while True:
        r = win32event.WaitForSingleObject(h,win32event.INFINITE)
        assert r == win32con.WAIT_OBJECT_0
        DD = D.copy()
        for p in self.path.iterdir():
          if DD.pop(p.name,None) is None: D[p.name]=1; self.untrack(2,p.name)
        for name in DD: del D[name]; self.untrack(1,name)
        win32file.FindNextChangeNotification(h)
    threading.Thread(target=process,daemon=True).start()

  def watch_win32_alt(self): # does not seem to work
    h = win32file.CreateFile(str(self.path),win32con.GENERIC_READ,win32con.FILE_SHARE_READ|win32con.FILE_SHARE_WRITE|win32con.FILE_SHARE_DELETE,None,win32con.OPEN_EXISTING,win32con.FILE_FLAG_BACKUP_SEMANTICS,None)
    def process():
      while True:
        for action,name in win32file.ReadDirectoryChangesW(h,4096,False,win32con.FILE_NOTIFY_CHANGE_FILE_NAME):
          if action == 2: i=1
          elif action == 5: i=2
          else: return
          self.untrack(i,name)
    threading.Thread(target=process,daemon=True).start()

  from sys import platform
  watch = locals()['watch_'+platform]

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def lru_persistent_cache(ignore=(),factory=CacheBlock,**ka):
  r"""
A decorator which applies to a function and replaces it by a persistently cached version. The function must be defined at the top-level of its module, to be compatible with :class:`Signature`.

:param ignore: passed, together with the function, to the :class:`Signature` constructor
:param ka: keyword arguments passed to *factory*
:param factory: the block factory (normally :class:`CacheBlock`)
  """
#--------------------------------------------------------------------------------------------------
  return lambda f: update_wrapper(factory(signature=Signature(f,ignore),**ka),f)

#--------------------------------------------------------------------------------------------------
class PCacheBlock (CacheBlock):
#--------------------------------------------------------------------------------------------------
  def clear(self,recursive=True):
    super(PCacheBlock,self).clear()
    if recursive and isinstance(self.sig.base,PCacheBlock): self.sig.base.clear()

#--------------------------------------------------------------------------------------------------
def lru_persistent_process_cache(*a,factory=PCacheBlock):
  r"""
A function which takes a list of functions and returns a cache callable implementing their composition. All the functions must be defined at the top-level of their modules, to be compatible with :class:`ProcessSignature`.

:param a: list of function cache specifiers
:type a: list(pair(:class:`function`,\ :class:`dict`))
:param factory: the block factory (normally :class:`CacheBlock`)

A cache specifier is a pair consisting of a function (defined at the top-level of its module, to be compatible with :class:`ProcessSignature`) and a disctionary of arguments passed to *factory*.
  """
#--------------------------------------------------------------------------------------------------
  if a:
    base = lru_persistent_process_cache(*a[:-1],factory=factory)
    f,ka = a[-1]
    return factory(signature=ProcessSignature(f,base),**ka)
  else: return State

#--------------------------------------------------------------------------------------------------
class State: pass
#--------------------------------------------------------------------------------------------------
class ARG (tuple):
  r"""
Instances of this (immutable) class hold arbitrary call arguments (both positional and keyword).

Methods:
  """
#--------------------------------------------------------------------------------------------------
  def __new__(cls,*a,**ka):
    return super(ARG,cls).__new__(cls,(a,ka))

  def variant(self,*a,**ka):
    r"""
Returns a variant of *self* where *a* is appended to the positional arguments and *ka* is updated into the keyword arguments.
    """
    a0,ka0 = self
    a = a0+a
    ka1 = ka0.copy(); ka1.update(ka); ka = ka1
    return ARG(*a,**ka1)

#--------------------------------------------------------------------------------------------------
def getparams(func,ignore=(),code={inspect.Parameter.VAR_POSITIONAL:1,inspect.Parameter.VAR_KEYWORD:2}):
#--------------------------------------------------------------------------------------------------
  for p in inspect.signature(func).parameters.values():
    # coding needed because p.kind is not pickable/unpickable before python 3.5
    yield p.name,(-1 if p.name in ignore else code.get(p.kind,0))

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
def html_stack(*a):
#--------------------------------------------------------------------------------------------------
  from lxml.builder import E
  return E.DIV(*(E.DIV(x) for x in a))

#--------------------------------------------------------------------------------------------------
class WrongCacheFolderException (Exception):
  r"""
Exception raised when the :class:`CacheDB` constructor is called with inappropriate arguments.
  """
#--------------------------------------------------------------------------------------------------
  pass
#--------------------------------------------------------------------------------------------------
class SignatureMismatchException (Exception):
  """
Exception raised when restoring a passive signature fails.
  """
#--------------------------------------------------------------------------------------------------
  pass
