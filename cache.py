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
from functools import partial, update_wrapper
from itertools import islice
from collections import namedtuple
from collections.abc import MutableMapping
from time import process_time, perf_counter
from . import SQliteNew, size_fmt, time_fmt, html_stack, html_table, html_parlist, HtmlPlugin, pickleclass, configurable_decorator

SCHEMA = '''
CREATE TABLE Block (
  oid INTEGER PRIMARY KEY AUTOINCREMENT,
  functor PICKLE UNIQUE NOT NULL,
  hits INTEGER DEFAULT 0,
  misses INTEGER DEFAULT 0
  )

CREATE TABLE Cell (
  oid INTEGER PRIMARY KEY AUTOINCREMENT,
  hitdate TIMESTAMP DEFAULT ( datetime('now') ),
  block REFERENCES Block(oid) NOT NULL,
  ckey BLOB NOT NULL,
  size INTEGER DEFAULT 0,
  tprc REAL,
  ttot REAL
  )

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
class CacheDB (MutableMapping,HtmlPlugin):
  r"""
Instances of this class manage cache repositories. There is at most one instance of this class in a process for each normalised repository specification path. A cache repository contains a set of values stored in some form. Each value has meta information stored in an index stored as a sqlite database. The index entry attached to a value is called a cell and describes the call event which obtained it. A ``Cell`` entry has the following fields:

- ``oid``: a unique identifier of the cell, which also allows retrieval of the attached value;
- ``block``: reference to another entry in the database (table ``Block``) describing the functor of the call event which produced the value;
- ``ckey``: the key of the call event which produced the value;
- ``hitdate``: date of creation, or last reuse, of the cell;
- ``size``, ``tprc``, ``ttot``: size (in bytes) of the value and process and total time (in sec) of its computation.

``Block`` entries correspond to clusters of ``Cell`` entries (call events) sharing the same functor. They have the following fields

- ``functor``: the persistent functor producing all the cells in the block; different blocks always have functors with different pickle byte-strings;
- ``misses``: number of call events with that functor where the argument had not been seen before;
- ``hits``: number of call events with that functor where the argument had previously been seen; such a call does not generate a new ``Cell``, but reuses the existing one;

Furthermore, a :class:`CacheDB` instance acts as a mapping object, where the keys are block identifiers (:class:`int`) and values are :class:`CacheBlock` objects for the corresponding blocks.

Finally, :class:`CacheDB` instances have an HTML ipython display.

Attributes:

.. attribute:: path

   the normalised path of this instance, as a :class:`pathlib.Path`

.. attribute:: dbpath

   the path to the sqlite database holding the index, as a string

.. attribute:: storage

   the object managing the actual storage of the values

Methods:

.. automethod:: __new__
  """
#==================================================================================================

  timeout = 120.

  def __new__(cls,spec,listing={},lock=threading.Lock()):
    r"""
Generates a :class:`CacheDB` object.

:param spec: specification of the cache folder
:type spec: :class:`CacheDB`\|\ :class:`pathlib.Path`\|\ :class:`str`

* If *spec* is a :class:`CacheDB` instance, that instance is returned
* If *spec* is a path to a directory, returns a :class:`CacheDB` instance whose storage is an instance of :class:`DefaultStorage` pointing to that directory
* Otherwise, *spec* must be a path to a file, returns a :class:`CacheDB` instance whose storage is unpickled from the file at *spec*.

Note that this constructor is locally cached on the resolved path *spec*.
    """
    if isinstance(spec,CacheDB): return spec
    if isinstance(spec,str): path = Path(spec)
    elif isinstance(spec,Path): path = spec
    else: raise TypeError('Expected: {}|{}|{}; Found: {}'.format(CacheDB,str,Path,type(spec)))
    path = path.resolve()
    with lock:
      self = listing.get(path)
      if self is None:
        if path.is_dir(): storage = DefaultStorage(path)
        elif path.is_file():
          with path.open('rb') as u: storage = pickle.loads(u)
        else: raise Exception('Cache repository specification path must be directory or file')
        dbpath = str(storage.dbpath)
        SQliteNew(dbpath,SCHEMA)
        self = super().__new__(cls)
        self.path = path
        self.dbpath = dbpath
        self.storage = storage
        listing[path] = self
    return self

  def __getnewargs__(self): return self.path,
  def __getstate__(self): return
  def __hash__(self): return hash(self.path)

  def connect(self,**ka):
    conn = sqlite3.connect(self.dbpath,timeout=self.timeout,**ka)
    conn.create_function('cellrm',2,self.storage.remove)
    return conn

  def getblock(self,functor):
    p = pickle.dumps(functor)
    with self.connect() as conn:
      conn.execute('BEGIN IMMEDIATE TRANSACTION')
      row = conn.execute('SELECT oid FROM Block WHERE functor=?',(p,)).fetchone()
      if row is None:
        return conn.execute('INSERT INTO Block (functor) VALUES (?)',(p,)).lastrowid
      else:
        return row[0]

  def clear_obsolete(self,sel=(lambda x: True),*,strict=True):
    if sel is None: sel = (lambda x: 'yes'.startswith(input('DEL {}? '.format(x)).strip().lower()))
    for k,c in list(self.items()):
      o = c.functor.obsolete()
      if not strict: o = o is not None
      if o and sel(c.functor): del self[k]

#--------------------------------------------------------------------------------------------------
# CacheDB as Mapping
#--------------------------------------------------------------------------------------------------

  def __getitem__(self,block):
    with self.connect(detect_types=sqlite3.PARSE_DECLTYPES) as conn:
      r = conn.execute('SELECT functor FROM Block WHERE oid=?',(block,)).fetchone()
    if r is None: raise KeyError(block)
    return CacheBlock(db=self,functor=r[0],block=block)

  def __delitem__(self,block):
    with self.connect() as conn:
      conn.execute('DELETE FROM Block WHERE oid=?',(block,))
      if not conn.total_changes: raise KeyError(block)

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
      for block,functor in conn.execute('SELECT oid,functor FROM Block'):
        yield block, CacheBlock(db=self,functor=functor,block=block)

  def clear(self):
    with self.connect() as conn:
      conn.execute('DELETE FROM Block')

#--------------------------------------------------------------------------------------------------
# Display
#--------------------------------------------------------------------------------------------------

  def as_html(self,incontext):
    return html_stack(*(v.as_html(incontext) for k,v in sorted(self.items())))
  def __repr__(self): return 'Cache<{}>'.format(self.path)

#==================================================================================================
class CacheBlock (MutableMapping,HtmlPlugin):
  r"""
Instances of this class implements blocks of cells sharing the same functor.

:param db: specification of the cache repository where the block resides
:param functor: functor of the block
:type functor: :class:`Functor`
:param cacheonly: if :const:`True`, cell creation is disallowed
:type cacheonly: :class:`bool`

A block object is callable, and calls take a single argument. Method :meth:`__call__` implements the cacheing mechanism.

* The computation of the cache key and cached value attached to an argument is delegated to a dedicated functor object which must implement the following api (e.g. implemented by class :class:`Functor`):

  - method :meth:`getkey` takes as input an argument and must return a byte string which represents it uniquely.
  - method :meth:`getval` takes as input an argument and must return its associated value (to be cached).
  - method :meth:`html` takes as input a byte string as returned by invocation of method :meth:`getkey` and must return an HTML formatted representation of the argument of that invocation.

* The actual storage of values is delegated to a dedicated storage object which must implement the following api (e.g. implemented by class :class:`FileStorage`):

  - method :meth:`insert` is called inside the transaction which inserts a new cell into the index, hence exactly once overall for a given cell. It takes as input the cell id and must return the function to call to set its value. The returned function is called exactly once, but outside the transaction. It is passed the computed value and must return the size in bytes of its storage. The cell may have disappeared from the cache when called, or may disappear while executing, so the assignment may have to later be rolled back.
  - method :meth:`lookup` is called inside the transaction which looks up a cell from the index. There may be multiple such transactions in possibly concurrent threads/processes for a given cell. It takes as input the cell id and a boolean flag indicating whether the cell value is currently being computed by a concurrent thread/process, and must return the function to call to get its value. The returned function is called exactly once, but outside the transaction.
  - method :meth:`remove` is called inside the transaction which deletes a cell from the index. It takes as arguments the cell id and the size of its value as memorised in the index.

Furthermore, a :class:`CacheBlock` object acts as a mapping where the keys are cell identifiers (:class:`int`) and values are triples ``hitdate``, ``ckey``, ``size``. When ``size`` is null, the value of the cell is still being computed, otherwise it represents its size in bytes, possibly with a negative sign if the stored value is an exception.

Finally, :class:`CacheBlock` instances have an HTML ipython display.

Methods:

.. automethod:: __call__
  """
#==================================================================================================

  def __init__(self,db=None,functor=None,block=None,cacheonly=False):
    self.db = db = CacheDB(db)
    self.functor = functor
    self.block = db.getblock(functor) if block is None else block
    self.cacheonly = cacheonly

  def __hash__(self): return hash((self.db,self.block))
  def __eq__(self,other): return isinstance(other,CacheBlock) and self.db is other.db and self.block == other.block

  def clear_error(self):
    r"""
Clears all the cells from this block which cache an exception.
    """
    with self.db.connect() as conn:
      conn.execute('DELETE FROM Cell WHERE block=? AND size<0',(self.block,))
      deleted = conn.total_changes
    if deleted>0: logger.info('%s DELETED(%s)',self,deleted)
    return deleted

  def clear_overflow(self,n):
    assert isinstance(n,int) and n>=1
    with self.db.connect() as conn:
      conn.execute('DELETE FROM Cell WHERE oid IN (SELECT oid FROM Cell WHERE block=? AND size>0 ORDER BY hitdate DESC, oid DESC LIMIT -1 OFFSET ?)',(self.block,n))
      deleted = conn.total_changes
    if deleted>0: logger.info('%s DELETED(%s)',self,deleted)
    return deleted

  def info(self,typ=namedtuple('BlockInfo',('functor','hits','misses','size'))):
    r"""
Returns information about this block. Available attributes:
:attr:`functor`, :attr:`hits`, :attr:`misses`, :attr:`size`
    """
    with self.db.connect(detect_types=sqlite3.PARSE_DECLTYPES) as conn:
      sz = conn.execute('SELECT count(*) FROM Cell WHERE block=?',(self.block,)).fetchone()[0]
      row = conn.execute('SELECT functor, hits, misses FROM Block WHERE oid=?',(self.block,)).fetchone()
    return typ(row[0].info(),*row[1:],size=sz)

#--------------------------------------------------------------------------------------------------
  def __call__(self,arg):
    """
:param arg: arument of the call

Implements cacheing as follows:

- method :meth:`getkey` of the functor is invoked with argument *arg* to obtain a ``ckey``.
- Then a transaction is begun on the index database.
- If there already exists a cell with the same ``ckey``, the transaction is immediately terminated and its value is extracted, using method :meth:`lookup` of the storage, and returned.
- If there does not exist a cell with the same ``ckey``, a cell with that ``ckey`` is immediately created, then the transaction is terminated. Then method :meth:`getval` of the functor is invoked with argument *arg*. The result is stored, even if it is an exception, using method :meth:`insert` of the storage, and completion is recorded in the database.

If the result was an exception, it is raised, otherwise it is returned. In all cases, hit status is updated in the database.
    """
#--------------------------------------------------------------------------------------------------
    ckey = self.functor.getkey(arg)
    with self.db.connect() as conn:
      conn.execute('BEGIN IMMEDIATE TRANSACTION')
      row = conn.execute('SELECT oid,size FROM Cell WHERE block=? AND ckey=?',(self.block,ckey,)).fetchone()
      if row is None:
        if self.cacheonly: raise Exception('Cache cell creation disallowed')
        cell = conn.execute('INSERT INTO Cell (block,ckey) VALUES (?,?)',(self.block,ckey)).lastrowid
        conn.execute('UPDATE Block SET misses=misses+1 WHERE oid=?',(self.block,))
        setval = self.db.storage.insert(cell)
      else:
        cell,size = row
        conn.execute('UPDATE Block SET hits=hits+1 WHERE oid=?',(self.block,))
        getval = self.db.storage.lookup(cell,size==0)
    if row is None:
      logger.info('%s MISS(%s)',self,cell)
      tm = process_time(),perf_counter()
      try: cval = self.functor.getval(arg)
      except BaseException as e: cval = e; size = -1
      else: size = 1
      tm = process_time()-tm[0],perf_counter()-tm[1]
      try: size *= setval(cval)
      except:
        with self.db.connect() as conn:
          conn.execute('DELETE FROM Cell WHERE oid=?',(cell,))
        raise
      with self.db.connect() as conn:
        conn.execute('UPDATE Cell SET size=?, tprc=?, ttot=?, hitdate=datetime(\'now\') WHERE oid=?',(size,tm[0],tm[1],cell))
        if not conn.total_changes: logger.info('%s LOST(%s)',self,cell)
      if size<0: raise cval
    else:
      if size==0: logger.info('%s WAIT(%s)',self,cell)
      cval = getval()
      logger.info('%s HIT(%s)',self,cell)
      with self.db.connect() as conn:
        conn.execute('UPDATE Cell SET hitdate=datetime(\'now\') WHERE oid=?',(cell,))
      if isinstance(cval,BaseException): raise cval
    return cval

#--------------------------------------------------------------------------------------------------
# CacheBlock as Mapping
#--------------------------------------------------------------------------------------------------

  def __getitem__(self,cell):
    with self.db.connect() as conn:
      r = conn.execute('SELECT ckey, hitdate, size, tprc, ttot FROM Cell WHERE oid=?',(cell,)).fetchone()
    if r is None: raise KeyError(cell)
    return r

  def __delitem__(self,cell):
    with self.db.connect() as conn:
      conn.execute('DELETE FROM Cell WHERE oid=?',(cell,))
      if not conn.total_changes: raise KeyError(cell)

  def __setitem__(self,cell,v):
    raise Exception('Direct create/update not permitted on Cell')

  def __iter__(self):
    with self.db.connect() as conn:
      for cell, in conn.execute('SELECT oid FROM Cell WHERE block=?',(self.block,)): yield cell

  def __len__(self):
    with self.db.connect() as conn:
      return conn.execute('SELECT count(*) FROM Cell WHERE block=?',(self.block,)).fetchone()[0]

  def items(self):
    with self.db.connect() as conn:
      for row in conn.execute('SELECT oid, ckey, hitdate, size, tprc, ttot FROM Cell WHERE block=?',(self.block,)):
        yield row[0],row[1:]

  def clear(self):
    with self.db.connect() as conn:
      conn.execute('DELETE FROM Cell WHERE block=?',(self.block,))

#--------------------------------------------------------------------------------------------------
# Display
#--------------------------------------------------------------------------------------------------

  def as_html(self,incontext,size_fmt_=(lambda sz: '*'+size_fmt(-sz) if sz<0 else size_fmt(sz)),time_fmt_=(lambda t: '' if t is None else time_fmt(t))):
    n = len(self)-self._html_limit
    L = self.items(); closing = None
    if n>0: L = islice(L,self._html_limit); closing = '{} more'.format(n)
    return html_table(sorted(L),hdrs=('ckey','hitdate','size','tprc','ttot'),fmts=((lambda ckey,h=self.functor.html: h(ckey,incontext)),str,size_fmt_,time_fmt_,time_fmt_),opening='{}: {}'.format(self.block,self.functor),closing=closing)
  def __repr__(self): return 'Cache<{}:{}>'.format(self.db.path,self.functor)

#==================================================================================================
class Functor:
  r"""
An instance of this class defines a functor attached to a python top-level function.

:param func: a function, defined at the top-level of its module (hence pickable)

A functor is entirely defined by the name of the function, that of its module, its version and its signature. Hence, two functions (possibly in different processes at different times) sharing these components produce the same functor.

Methods:
  """
#===================================================================================================

  __slots__ = 'config', 'sig', 'func'

  def __new__(cls,spec,fromfunc=True):
    self = super().__new__(cls)
    if fromfunc:
      self.func = spec
      self.sig = sig = inspect.signature(spec)
      spec = Shadow(spec),sig_dump(sig)
    else:
      self.sig = sig_load(spec[-1])
    self.config = spec
    return self

  def __getnewargs__(self): return self.config,False
  def __getstate__(self): return
  def __hash__(self): return hash(self.config)
  def __eq__(self,other): return isinstance(other,Functor) and self.config==other.config
  def __repr__(self): return '{}{}'.format(self.config[0],self.sig)

  class fpickle (pickleclass):
    class Pickler (pickle.Pickler):
      def persistent_id(self,obj):
        if inspect.isfunction(obj) and hasattr(obj,'version'): return Shadow(obj)
    class Unpickler (pickle.Unpickler):
      def persistent_load(self,pid): return pid

  def getkey(self,arg):
#--------------------------------------------------------------------------------------------------
    r"""
Argument *arg* must be a pair of a list of positional arguments and a dict of keyword arguments. They are normalised against the signature of the functor and the pickled value of the result is returned. The pickling of versioned function objects is modified to embed their version.
    """
#--------------------------------------------------------------------------------------------------
    a,ka = self.norm(arg)
    return self.fpickle.dumps((a,sorted(ka.items())))

  def getval(self,arg):
#--------------------------------------------------------------------------------------------------
    r"""
Argument *arg* must be a pair of a list of positional arguments and a dict of keyword arguments. Returns the value of calling attribute :attr:`func` with that positional argument list and keyword argument dict.
    """
#--------------------------------------------------------------------------------------------------
    a,ka = arg
    return self.func(*a,**ka)

#--------------------------------------------------------------------------------------------------
  def html(self,ckey,incontext):
#--------------------------------------------------------------------------------------------------
    a,ka = self.fpickle.loads(ckey)
    return html_parlist(a,ka,incontext)

#--------------------------------------------------------------------------------------------------
  def info(self): return self.config[0],self.sig
#--------------------------------------------------------------------------------------------------

  def norm(self,arg):
    a,ka = arg
    b = self.sig.bind(*a,**ka)
    return b.args, b.kwargs

  def obsolete(self): return self.config[0].obsolete()

def sig_dump(sig): return tuple((p.name,p.kind,p.default) for p in sig.parameters.values())
def sig_load(x): return inspect.Signature(inspect.Parameter(name,kind,default=default) for name,kind,default in x)

#==================================================================================================
class FileStorage:
  r"""
Instances of this class manage the persistent storage of cache values using a filesystem directory.

:param path: the directory to store cached values
:type path: :class:`pathlib.Path`

The storage for a cell consists of a content file which contains the value of the cell in pickled format and a cross process mechanism to synchronise access to the content file between writer and possibly multiple readers. The names of the content file as well as the file underlying the synch lock are built from the cell id, so as to be unique to that cell. They inherit the access rights of the *path* directory.

Attributes:

.. attribute:: path

   Directory where values are stored, initialised from *path*

Methods:
  """
#==================================================================================================

  def __init__(self,path):
    self.path = path
    self.umask = ~path.stat().st_mode|0o111

  def __enter__(self): self.umask = os.umask(self.umask)
  def __exit__(self,*a): self.umask = os.umask(self.umask)
  # not thread safe (should at least have a lock)

#--------------------------------------------------------------------------------------------------
  def insert(self,cell):
    r"""
Opens the content file path in write mode, acquires the synch lock, then returns a :func:`setval` function which pickle-dumps its argument into the content file, closes it and releases the synch lock.
    """
#--------------------------------------------------------------------------------------------------
    def setval(val):
      try: pickle.dump(val,vfile)
      except Exception as e: vfile.seek(0); vfile.truncate(); pickle.dump(e,vfile)
      s = vfile.tell()
      vfile.close()
      synch_close()
      return s
    vpath = self.getpath(cell)
    with self:
      vfile = vpath.open('wb')
      synch_close = self.insert_synch(vpath)
    return setval

#--------------------------------------------------------------------------------------------------
  def lookup(self,cell,wait):
    r"""
Opens the content file path in read mode and returns a :func:`getval` function which waits for the synch lock to be released (if *wait* is True), then pickle-loads the content file and returns the obtained value.
    """
#--------------------------------------------------------------------------------------------------
    def getval():
      wait()
      try: return pickle.load(vfile)
      finally: vfile.close()
    vpath = self.getpath(cell)
    vfile = vpath.open('rb')
    wait = self.lookup_synch(vpath) if wait else lambda: None
    return getval

#--------------------------------------------------------------------------------------------------
  def remove(self,cell,size):
    r"""
Removes the content file path and as well as the synch lock.
    """
#--------------------------------------------------------------------------------------------------
    vpath = self.getpath(cell)
    try: vpath.unlink(); self.remove_synch(vpath)
    except: pass

  def getpath(self,cell):
    return self.path/'V{:06d}.pck'.format(cell)

#--------------------------------------------------------------------------------------------------
# Synchronisation mechanism based on sqlite (for portability: could be simplified)
#--------------------------------------------------------------------------------------------------

  @staticmethod
  def insert_synch(vpath):
    tpath = vpath.with_suffix('.tmp')
    tpath.open('w').close()
    synch = sqlite3.connect(str(tpath))
    synch.execute('BEGIN EXCLUSIVE TRANSACTION')
    return synch.close

  @staticmethod
  def lookup_synch(vpath,timeout=600.):
    tpath = vpath.with_suffix('.tmp')
    synch = sqlite3.connect(str(tpath),timeout=timeout)
    def wait():
      while True:
        if not tpath.exists(): synch.close(); raise Exception('synch file lost!')
        try: synch.execute('BEGIN IMMEDIATE TRANSACTION')
        except sqlite3.OperationalError as e:
          if e.args[0] != 'database is locked': synch.close(); raise
          continue
        break
      synch.close()
    return wait

  @staticmethod
  def remove_synch(vpath):
    tpath = vpath.with_suffix('.tmp')
    tpath.unlink()

#==================================================================================================
class DefaultStorage (FileStorage):
  r"""
The default storage class for a cache repository. Stores the index database in the same directory as the values, with name ``index.db``.

Attributes:

.. attribute:: dbpath

   The :class:`pathlib.Path` to the sqlite database holding the index
  """
#==================================================================================================
  def __init__(self,path):
    super().__init__(path)
    self.dbpath = path/'index.db'
    if not self.dbpath.is_file():
      if any(path.iterdir()): raise Exception('Cannot create new index in non empty directory')
      with self: self.dbpath.open('wb').close()

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
@configurable_decorator
def persistent_cache(f,factory=CacheBlock,**ka):
  r"""
A decorator which makes a function persistently cached. The cached function behaves as the original function except that its invocations are cached and reused when possible. The original function must be defined at the top-level of its module, to be compatible with :class:`Functor`. If it does not have a version already, it is assigned version :const:`None`.
  """
#--------------------------------------------------------------------------------------------------
  assert inspect.isfunction(f)
  if not hasattr(f,'version'): f.version = None
  c = factory(functor=Functor(f),**ka)
  F = lambda *a,**ka: c((a,ka))
  F.cache = c
  return update_wrapper(F,f)

#--------------------------------------------------------------------------------------------------
class Shadow:
  r"""
Instances of this class are defined from versioned functions (ie. functions defined at the toplevel of their module, and with an attribute :attr:`version` defined). Their state is composed of the name, the module name and the version of the original function. They can be arbitrarily pickled and unpickled and produce a string representation close to that of their origin. However, unpickling does not restore the calling capacity of their origin.
  """
#--------------------------------------------------------------------------------------------------

  __slots__ = 'config',

  def __new__(cls,spec,fromfunc=True):
    self = super().__new__(cls)
    if fromfunc: spec = spec.__module__,spec.__name__,spec.version
    self.config = spec
    return self
  def __getnewargs__(self): return self.config,False
  def __getstate__(self): return
  def __hash__(self): return hash(self.config)
  def __eq__(self,other): return isinstance(other,Shadow) and self.config==other.config
  def __repr__(self):
    module,name,version = self.config
    return '{}.{}{}'.format(module,name,('' if version is None else '{{{}}}'.format(version)))
  def obsolete(self):
    from importlib import import_module
    module,name,version = self.config
    try:
      f = getattr(import_module(module),name)
      if inspect.isfunction(f) and f.__module__==module and f.__name__==name:
        return None if f.version==version else (f.version,version)
    except: pass
    return ()
