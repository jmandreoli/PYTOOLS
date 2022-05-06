# File:                 cache.py
# Creation date:        2015-03-19
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Persistent cache management
#

r"""
Available types and functions
-----------------------------
"""

from typing import Any, Union, Callable, Iterable, Mapping, Tuple
import logging; logger = logging.getLogger(__name__)

import os, sqlite3, pickle, inspect, threading, abc
from pathlib import Path
from functools import update_wrapper
from itertools import islice, chain
from collections import namedtuple, OrderedDict
from collections.abc import MutableMapping
from weakref import WeakValueDictionary
from time import time
from . import SQliteNew, size_fmt, time_fmt, html_table, html_parlist, HtmlPlugin, pickleclass

SCHEMA = '''
CREATE TABLE Block (
  oid INTEGER PRIMARY KEY AUTOINCREMENT,
  functor PICKLE NOT NULL
  )

CREATE TABLE Cell (
  oid INTEGER PRIMARY KEY AUTOINCREMENT,
  block REFERENCES Block(oid) NOT NULL,
  ckey BLOB NOT NULL,
  tstamp TIMESTAMP DEFAULT ( datetime('now') ),
  hits INTEGER DEFAULT 0,
  size INTEGER DEFAULT 0,
  duration REAL
  )

CREATE UNIQUE INDEX BlockIndex ON Block (functor)

CREATE UNIQUE INDEX CellIndex ON Cell (block, ckey)

CREATE INDEX CellIndex2 ON Cell (tstamp)

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
Instances of this class manage cache repositories. There is at most one instance of this class in a process for each normalised repository specification path. A cache repository contains cells, each cell corresponding to one cached value produced by a unique call, and possibly reused by later calls. Cells are clustered into blocks, each block grouping cells produced by the same call type, called a functor (of class :class:`AbstractFunctor`). Meta-information about blocks and cells are stored in an index in a sqlite3 database. The values themselves are persistently stored by a dedicated storage object (of class :class:`AbstractStorage`).

The index entry attached to a block describes the common functor of all the calls which produced its cells. A ``Block`` entry has the following field:

- ``oid``: a unique integer identifier of the block;
- ``functor``: the functor producing all the cells in the block; each functor is assumed to have a deterministic pickle byte-string which uniquely identifies it.

The index entry attached to a cell describes the call event which produced it. A ``Cell`` entry has the following fields:

- ``oid``: a unique integer identifier of the cell; also used to retrieve the value attached to the cell;
- ``block``: reference to the ``Block`` entry holding the functor of the call event which produced the cell;
- ``ckey``: the key of the call event which produced the cell;
- ``tstamp``: date of creation, last update or last reuse, of the cell;
- ``hits``: number of hits (reuse) since creation;
- ``size``: either 0 if the value is still being computed, or the size in bytes of the computed value, with a negative sign if that value is an exception;
- ``duration``: duration (in float sec) of its computation.

A :class:`CacheDB` instance acts as a mapping object, where the keys are block identifiers (:class:`int`) and values are :class:`CacheBlock` objects for the corresponding blocks.

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

  def __new__(cls,spec:Union['CacheDB',Path,str],listing={},lock=threading.Lock()):
    r"""
Generates a :class:`CacheDB` object.

:param spec: specification of the cache folder

* If *spec* is a :class:`CacheDB` instance, that instance is returned
* If *spec* is a path to a directory, returns a :class:`CacheDB` instance whose storage is an instance of :class:`DefaultStorage` pointing to that directory
* Otherwise, *spec* must be a path to a file, returns a :class:`CacheDB` instance whose storage is unpickled from the file at *spec*.

Note that this constructor is locally cached on the resolved path *spec*.
    """
    if isinstance(spec,CacheDB): return spec
    if isinstance(spec,str): path = Path(spec)
    elif isinstance(spec,Path): path = spec
    else: raise TypeError(f'Expected: {CacheDB}|{str}|{Path}; Found: {type(spec)}')
    path = path.resolve()
    with lock:
      self = listing.get(path)
      if self is None:
        if path.is_dir(): storage = DefaultStorage(path)
        elif path.is_file():
          with path.open('rb') as u: storage = pickle.load(u)
        else: raise ValueError('Cache repository specification path must be directory or file')
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
  # no need to define '__eq__': default 'is' behaviour works due to '__new__' constructor

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

  def clear_obsolete(self,strict,dry_run=False):
    r"""
Clears all the blocks which are obsolete.
    """
    assert isinstance(strict,bool)
    strictf = bool if strict else (lambda o: o is not None)
    L = [k for k,c in list(self.items()) if strictf(c.functor.obsolete())]
    # this may load modules which add new (non obsolete) entries, hence out of transaction
    if dry_run: return L
    if not L: return 0
    with self.connect() as conn:
      conn.create_function('obsolete',1,(lambda cell: cell in L))
      conn.execute('DELETE FROM Block WHERE obsolete(oid)')
      deleted = conn.total_changes
    if deleted>0: logger.info('%s DELETED(%s)',self,deleted)
    return deleted

#--------------------------------------------------------------------------------------------------
# Methods defining the Mapping behaviour
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

  def __setitem__(self,block:str,v:Any):
    raise Exception('Direct create/update not permitted on Block')

  def __iter__(self):
    with self.connect() as conn:
      for block, in conn.execute('SELECT oid FROM Block'): yield block

  def __len__(self):
    with self.connect() as conn:
      return conn.execute('SELECT count(*) FROM Block').fetchone()[0]

  def items(self):
    r"""Returns a view on this instance's blocks."""
    with self.connect(detect_types=sqlite3.PARSE_DECLTYPES) as conn:
      for block,functor in conn.execute('SELECT oid,functor FROM Block'):
        yield block, CacheBlock(db=self,functor=functor,block=block)

  def clear(self):
    r"""Removes all blocks from this instance."""
    with self.connect() as conn:
      conn.execute('DELETE FROM Block')

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  def as_html(self,_):
    n = len(self)-self._html_limit
    L = self.items(); closing = None
    if n>0: L = islice(L,self._html_limit); closing = f'{n} more'
    return html_table(sorted((k,(v,)) for k,v in L),fmts=((lambda x: x.as_html(_)),),opening=repr(self),closing=closing)
  def __repr__(self): return f'Cache<{self.path}>'

#==================================================================================================
class CacheBlock (MutableMapping,HtmlPlugin):
  r"""
Instances of this class implements blocks of cells sharing the same functor.

:param db: specification of the cache repository where the block resides
:param functor: functor of the block
:param cacheonly: if :const:`True`, cell creation is disallowed

A :class:`CacheBlock` instance is callable, and calls take a single argument. Method :meth:`__call__` implements the cross-process cacheing mechanism which produces and reuses cache cells. It also implements a weak cache for local calls (within its process).

Furthermore, a :class:`CacheBlock` instance acts as a mapping where the keys are cell identifiers (:class:`int`) and values are tuples of meta-information about the cells (i.e. not the values of the cells: these are only accessible through calling).

Finally, :class:`CacheBlock` instances have an HTML ipython display.

Attributes:

.. attribute:: db

   the :class:`CacheDB` instance this block belongs to

.. attribute:: functor

   the functor for this block (field ``functor`` in the ``Block`` table of the index is the functor's pickle)

.. attribute:: block

   the :class:`int` identifier of this block (field ``oid`` in the ``Block`` table of the index)

.. attribute:: cacheonly

   whether cell creation is disabled

.. attribute:: memory

   a :class:`weakref.WeakValueDictionary` implementing a local cache of calls within the current process

Methods:

.. automethod:: __call__
  """
#==================================================================================================

  def __init__(self,db:Union[CacheDB,Path,str]='',functor:'AbstractFunctor'=None,block:int=None,cacheonly:bool=False):
    self.db = db = CacheDB(db)
    self.functor = functor
    self.block = db.getblock(functor) if block is None else block
    self.cacheonly = cacheonly
    self.memory = WeakValueDictionary()

  def __hash__(self): return hash((self.db,self.block))
  def __eq__(self,other): return isinstance(other,CacheBlock) and self.db is other.db and self.block == other.block

  def clear_error(self,dry_run:bool=False):
    r"""
Clears all the cells from this block which cache an exception.
    """
    with self.db.connect() as conn:
      if dry_run: return [cell for cell, in conn.execute('SELECT oid FROM Cell WHERE block=? AND size<0',(self.block,))]
      conn.execute('DELETE FROM Cell WHERE block=? AND size<0',(self.block,))
      deleted = conn.total_changes
    if deleted>0: logger.info('%s DELETED(%s)',self,deleted)
    return deleted

  def clear_overflow(self,n:int,dry_run:bool=False):
    r"""
Clears all the cells from this block except the *n* most recent (lru policy).
    """
    assert isinstance(n,int) and n>=1
    with self.db.connect() as conn:
      if dry_run: return [cell for cell, in conn.execute('SELECT oid FROM Cell WHERE block=? AND size>0 ORDER BY tstamp DESC, oid DESC LIMIT -1 OFFSET ?',(self.block,n))]
      conn.execute('DELETE FROM Cell WHERE oid IN (SELECT oid FROM Cell WHERE block=? AND size>0 ORDER BY tstamp DESC, oid DESC LIMIT -1 OFFSET ?)',(self.block,n))
      deleted = conn.total_changes
    if deleted>0: logger.info('%s DELETED(%s)',self,deleted)
    return deleted

  def info(self,typ=namedtuple('BlockInfo',('hits','ncell','ncell_error','ncell_pending'))):
    r"""
Returns information about this block. Available attributes:
:attr:`hits`, :attr:`ncell`, :attr:`ncell_error`, :attr:`ncell_pending`
    """
    with self.db.connect(detect_types=sqlite3.PARSE_DECLTYPES) as conn:
      ncell = dict(conn.execute('SELECT CASE WHEN size ISNULL THEN \'pending\' WHEN size<0 THEN \'error\' ELSE \'\' END AS status, count(*) FROM Cell WHERE block=? GROUP BY status',(self.block,)))
      hits, = conn.execute('SELECT sum(hits) FROM Cell WHERE block=?',(self.block,)).fetchone()
    return typ((hits or 0),sum(ncell.values()),*(ncell.get(k,0) for k in ('error','pending')))

#--------------------------------------------------------------------------------------------------
  def __call__(self,arg:Any):
    """
:param arg: argument of the call

Implements cacheing as follows:

- Method :meth:`getkey` of the functor is invoked with argument *arg* to obtain a ``ckey``.
- If that ``ckey`` is present in the (local) memory mapping of this block, its associated value is returned.
- Otherwise, a transaction is begun on the index database.

  - If there already exists a cell with the same ``ckey``, method :meth:`lookup` of the storage is invoked to obtain a getter for that cell, then the transaction is terminated and the result is extracted, using the obtained getter. The cell's hit count is incremented.
  - If there does not exist a cell with the same ``ckey``, a cell with that ``ckey`` is created, and method :meth:`insert` of the storage is invoked to obtain a setter for that cell, then the transaction is terminated. Then, method :meth:`getval` of the functor is invoked with argument *arg* and its result is stored, even if it is an exception, using the obtained setter.

- If the result is an exception, it is raised.
- Otherwise, the memory mapping of this block is updated at key ``ckey`` with the result (if possible), and the result is returned.
    """
#--------------------------------------------------------------------------------------------------
    ckey = self.functor.getkey(arg)
    cval = self.memory.get(ckey)
    if cval is not None: return cval
    with self.db.connect() as conn:
      conn.execute('BEGIN IMMEDIATE TRANSACTION')
      row = conn.execute('SELECT oid,size FROM Cell WHERE block=? AND ckey=?',(self.block,ckey)).fetchone()
      if row is None:
        if self.cacheonly: raise Exception('Cache cell creation disallowed')
        cell = conn.execute('INSERT INTO Cell (block,ckey) VALUES (?,?)',(self.block,ckey)).lastrowid
        setval = self.db.storage.insert(cell)
      else:
        cell,size = row
        getval = self.db.storage.lookup(cell,size==0)
    if row is None:
      logger.info('%s MISS(%s)',self,cell)
      tm = time()
      try: cval = self.functor.getval(arg)
      except BaseException as e: cval = e; size = -1
      else: size = 1
      tm = time()-tm
      try: size *= setval(cval)
      except:
        with self.db.connect() as conn:
          conn.execute('DELETE FROM Cell WHERE oid=?',(cell,))
        raise
      with self.db.connect() as conn:
        conn.execute('UPDATE Cell SET size=?, duration=?, tstamp=datetime(\'now\') WHERE oid=?',(size,tm,cell))
        if not conn.total_changes: logger.info('%s LOST(%s)',self,cell)
      if size<0: raise cval
    else:
      if size==0: logger.info('%s WAIT(%s)',self,cell)
      cval = getval()
      logger.info('%s HIT(%s)',self,cell)
      with self.db.connect() as conn:
        conn.execute('UPDATE Cell SET hits=hits+1, tstamp=datetime(\'now\') WHERE oid=?',(cell,))
      if isinstance(cval,BaseException): raise cval
    try: self.memory[ckey] = cval
    except: pass
    return cval

#--------------------------------------------------------------------------------------------------
# Methods defining the Mapping behaviour
#--------------------------------------------------------------------------------------------------

  def __getitem__(self,cell):
    with self.db.connect() as conn:
      r = conn.execute('SELECT ckey, tstamp, hits, size, duration FROM Cell WHERE oid=?',(cell,)).fetchone()
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
    r"""Returns a view on this instance's cells."""
    with self.db.connect() as conn:
      for row in conn.execute('SELECT oid, ckey, tstamp, hits, size, duration FROM Cell WHERE block=?',(self.block,)):
        yield row[0],row[1:]

  def clear(self):
    r"""Removes all cells from this instance."""
    with self.db.connect() as conn:
      conn.execute('DELETE FROM Cell WHERE block=?',(self.block,))

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  def as_html(self,_,size_fmt_=(lambda sz: '*'+size_fmt(-sz) if sz<0 else size_fmt(sz)),time_fmt_=(lambda t: '' if t is None else time_fmt(t))):
    n = len(self)-self._html_limit
    L = self.items(); closing = None
    if n>0: L = islice(L,self._html_limit); closing = f'{n} more'
    return html_table(sorted(L),hdrs=('ckey','tstamp','hits','size','duration'),fmts=((lambda ckey,h=self.functor.html: h(ckey,_)),str,str,size_fmt_,time_fmt_),opening=repr(self),closing=closing)
  def __repr__(self): return f'Cache<{repr(self.functor)}>'

#==================================================================================================
class AbstractFunctor (metaclass=abc.ABCMeta):
  r"""
An instance of this class defines a type of (single argument) call to be cached.
  """
#==================================================================================================

  @abc.abstractmethod
  def getkey(self,arg):
    r"""
:param arg: an arbitrary python object.

Returns a byte string which represents *arg* uniquely.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def getval(self,arg):
    r"""
:param arg: an arbitrary python object.

Returns the result of calling this functor with argument *arg*.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def html(self,ckey:bytes,_):
    r"""
:param ckey: a byte string as returned by invocation of method :meth:`getkey`

Returns an HTML formatted representation of the argument of that invocation.
    """
    raise NotImplementedError()

#==================================================================================================
class AbstractStorage (metaclass=abc.ABCMeta):
  r"""
An instance of this class stores cached values on a persistent support.
  """
#==================================================================================================

  @abc.abstractmethod
  def insert(self,cell:int)->Callable[[Any],int]:
    r"""
:param cell: the identifier of a cell

Returns the function to call to set a cell value. This method is called inside the transaction which inserts a new cell into a cache index, hence exactly once overall for a given cell. The returned function is then called, but outside the transaction. It is passed the computed value and must return the size in bytes of its storage. The cell may have disappeared from the cache when called, or may disappear while executing, so the assignment may have to later be rolled back.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def lookup(self,cell:int,wait:bool)->Callable[[],Any]:
    r"""
:param cell: the identifier of a cell
:param wait: whether the cell value is currently being computed by a concurrent thread/process

Returns the function to call to get a cell value. This method is called inside the transaction which looks up a cell from a cache index, which may happens multiple times in possibly concurrent threads/processes for a given cell. The returned function is then called, but outside the transaction.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def remove(self,cell:int,size:int):
    r"""
:param cell: the identifier of a cell
:param size: size of the cell

Frees the storage resources associated with a cell. This method is called inside the transaction which deletes a cell from a cache index.
    """
    raise NotImplementedError()

#==================================================================================================
class Functor (AbstractFunctor):
  r"""
An instance of this class defines a functor attached to a python top-level versioned function. The functor is entirely defined by the name of the function, that of its module, its version and its signature. These components are saved on pickling and restored on unpickling, even if the function has disappeared or changed. This is not checked on unpickling, and method :meth:`getval` is disabled.

Attributes:

.. attribute:: func

   the versioned function characterizing this functor

Methods:

.. automethod:: __new__
  """
#===================================================================================================

  __slots__ = 'config', 'sig', 'func'

  def __new__(cls,spec,fromfunc:bool=True):
    r"""
Generates a functor.

:param spec: a versioned function, defined at the top-level of its module (hence pickable)
    """
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
  def __repr__(self): return f'{self.config[0]}{self.sig}'

  class fpickle (pickleclass):
    class Pickler (pickle.Pickler):
      def persistent_id(self,obj):
        if inspect.isfunction(obj) and hasattr(obj,'version'): return Shadow(obj)
    class Unpickler (pickle.Unpickler):
      def persistent_load(self,pid): return pid

  def getkey(self,arg:Tuple[Iterable[Any],Mapping[str,Any]]):
#--------------------------------------------------------------------------------------------------
    r"""
Argument *arg* must be a pair of a list of positional arguments and a dict of keyword arguments. They are normalised against the signature of the functor and the pickled value of the result is returned. The pickling of versioned function objects is modified to embed their version.
    """
#--------------------------------------------------------------------------------------------------
    a,ka = self.norm(arg)
    return self.fpickle.dumps((a,sorted(ka.items())))

  def getval(self,arg:Tuple[Iterable[Any],Mapping[str,Any]]):
#--------------------------------------------------------------------------------------------------
    r"""
Argument *arg* must be a pair of a list of positional arguments and a dict of keyword arguments. Returns the value of calling attribute :attr:`func` with that positional argument list and keyword argument dict.
    """
#--------------------------------------------------------------------------------------------------
    a,ka = arg
    return self.func(*a,**ka)

#--------------------------------------------------------------------------------------------------
  def html(self,ckey:bytes,_):
#--------------------------------------------------------------------------------------------------
    a,ka = self.fpickle.loads(ckey)
    return html_parlist(_,a,ka)

  def norm(self,arg:Tuple[Iterable[Any],Mapping[str,Any]]):
    a,ka = arg
    b = self.sig.bind(*a,**ka)
    return b.args, b.kwargs

  def obsolete(self): return self.config[0].obsolete()

def sig_dump(sig): return tuple((p.name,p.kind,p.default) for p in sig.parameters.values())
def sig_load(x): return inspect.Signature(inspect.Parameter(name,kind,default=default) for name,kind,default in x)

#==================================================================================================
class FileStorage (AbstractStorage):
  r"""
Instances of this class manage the persistent storage of cached values using a filesystem directory.

:param path: the directory to store cached values

The storage for a cell consists of a content file which contains the value of the cell in pickled format and a cross process mechanism to synchronise access to the content file between writer and possibly multiple readers. The path of the content file as well as that of the file underlying the synch lock are built from the cell id, so as to be unique to that cell. They inherit the access rights of the :attr:`path` directory.

Attributes:

.. attribute:: path

   Directory where values are stored, initialised from *path*

Methods:
  """
#==================================================================================================

  def __init__(self,path:Path):
    self.path = path
    self.mode = path.stat().st_mode

#--------------------------------------------------------------------------------------------------
  def insert(self,cell:int)->Callable[[Any],int]:
    r"""
Opens the content file path for *cell* in write mode, acquires the corresponding synch lock, then returns a setter function which pickle-dumps its argument into the content file, closes it and releases the synch lock.
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
    vfile = vpath.open('wb')
    vpath.chmod(self.mode&0o666)
    synch_close = self.insert_synch(vpath)
    try: os.sync()
    except: pass
    return setval

#--------------------------------------------------------------------------------------------------
  def lookup(self,cell:int,wait:bool)->Callable[[],Any]:
    r"""
Opens the content file path for *cell* in read mode, then returns a getter function which waits for the corresponding synch lock to be released (if *wait* is True), then pickle-loads the content file and returns the obtained value.
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
  def remove(self,cell:int,size:int):
    r"""
Removes the content file path for *cell* as well as the corresponding synch lock.
    """
#--------------------------------------------------------------------------------------------------
    vpath = self.getpath(cell)
    try: vpath.unlink(); self.remove_synch(vpath)
    except: pass

#--------------------------------------------------------------------------------------------------
  def getpath(self,cell:int,masks=tuple((5*n,31*32**n) for n in range(5))):
    r"""
Returns the content file path (as a :class:`pathlib.Path` instance) associated to *cell* (of type :class:`int`). It is composed of two parts (a directory name and a file name), joined to the main :attr:`path` attribute. The directory is created if it does not already exist. The concatenation of the directory name (without its prefix ``X``) and the file name (without its suffix ``.pck``) is the representation of *cell* in base 32 (digits are 0-9A-V). This mapping of cells to paths ensures that no sub-directory holds more than 1024 cells. It assumes that cells are created sequentially (which is what AUTOINCREMENT in sqlite3 does), so the number of sub-directories grows slowly.
    """
#--------------------------------------------------------------------------------------------------
    s = ''.join('0123456789ABCDEFGHIJKLMNOPQRSTUV'[(cell&m)>>n] for n,m in masks)
    p = self.path/('X'+s[:1:-1])
    if not p.exists(): p.mkdir(exist_ok=True); p.chmod(self.mode)
    return (p/s[1::-1]).with_suffix('.pck')

#--------------------------------------------------------------------------------------------------
# Synchronisation mechanism based on sqlite (for portability: could probably be simplified)
#--------------------------------------------------------------------------------------------------

  @staticmethod
  def insert_synch(vpath:Path)->Callable[[],None]:
    tpath = vpath.with_suffix('.tmp')
    tpath.touch()
    tpath.chmod(vpath.stat().st_mode)
    synch = sqlite3.connect(str(tpath))
    synch.execute('BEGIN EXCLUSIVE TRANSACTION')
    return synch.close

  @staticmethod
  def lookup_synch(vpath:Path,timeout:float=600.)->Callable[[],None]:
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
  def remove_synch(vpath:Path):
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
  def __init__(self,path:Path):
    super().__init__(path)
    self.dbpath = path/'index.db'
    if not self.dbpath.exists():
      if any(path.iterdir()): raise Exception('Cannot create new index in non empty directory')
      self.dbpath.touch(exist_ok=True)
      self.dbpath.chmod(self.mode&0o666)

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def persistent_cache(f:Callable,factory=CacheBlock,**kwd):
  r"""
A decorator which makes a function persistently cached. The cached function behaves as the original function except that its invocations are cached and reused when possible. The original function must be defined at the top-level of its module, to be compatible with :class:`Functor`. If it does not have a version already, it is assigned version :const:`None`.
  """
#--------------------------------------------------------------------------------------------------
  assert inspect.isfunction(f)
  if not hasattr(f,'version'): f.version = None
  c = factory(functor=Functor(f),**kwd)
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
    return '{}.{}{}'.format(module,name,('' if version is None else f'{{{version}}}'))
  def obsolete(self):
    r"""
Returns :const:`None` if this instance is up-to-date. It may be obsolete for two reasons:

* the function it refers to (by module name and function name) cannot be imported or is not a versioned function: in that case the empty tuple is returned;
* it can be imported as a versioned function, but has a different version from that of this instance: in that case, the pair of the current version of the imported function and the version of this instance is returned.

Note that this method may import modules which in turn may create cache entries (for new versions of functions). It should therefore not be called in a cache transaction.
    """
    from importlib import import_module
    module,name,version = self.config
    try:
      f = getattr(import_module(module),name)
      if inspect.isfunction(f) and f.__module__==module and f.__name__==name:
        return None if f.version==version else (f.version,version)
    except: pass
    return ()

#--------------------------------------------------------------------------------------------------
def manage(*paths,ivname='db'):
  r"""
A simple tool to manage a set of :class:`CacheDB` instances, specified by their paths.
  """
#--------------------------------------------------------------------------------------------------
  import ipywidgets
  from IPython.display import clear_output, display
  from IPython.core.getipython import get_ipython
  def setdb(c): nonlocal db; db = c.new; interpreter.push({ivname:db}); showdb()
  def showdb():
    wmsg.clear_output()
    with wout: clear_output(); display(db)
  def runc(c):
    x = c.new
    if x == autorun_: return
    wautorun.value = autorun_
    try:
      with wmsg:
        clear_output()
        print('Operation:','clear'+x,f'(dryrun: {ivname} unchanged)' if wdryrun.value else '')
        r = autorun[x]()
        print('Result:',r)
    except: return
    if not wdryrun.value and r:
      with wout: clear_output(); display(db)
  autorun = dict(
    Error = (lambda: [x for x in ((c.block,c.clear_error(dry_run=wdryrun.value)) for c in list(db.values())) if x[1]]),
    Obsolete = (lambda: db.clear_obsolete(False,dry_run=wdryrun.value)),
    ObsoleteStrict = (lambda: db.clear_obsolete(True,dry_run=wdryrun.value)),
  )
  interpreter = get_ipython()
  autorun_ = 'clear...'
  db = ()
  interpreter.push({ivname:db})
  wdb = ipywidgets.Dropdown(description=ivname,options=OrderedDict(chain((('...',()),),((p,CacheDB(p)) for p in paths))),style={'description_width':'initial'})
  wshow = ipywidgets.Button(tooltip='show db',icon='fa-refresh',layout=dict(width='.4cm',padding='0cm'))
  wautorun = ipywidgets.Dropdown(options=(autorun_,'Error','Obsolete','ObsoleteStrict'),layout=dict(width='3cm'))
  wdryrun = ipywidgets.Checkbox(description='dry-run',value=True,style={'description_width':'initial'})
  wout = ipywidgets.Output()
  wmsg = ipywidgets.Output(layout=dict(border='thin solid black'))
  wdb.observe(setdb,'value')
  wautorun.observe(runc,'value')
  wshow.on_click(lambda b: showdb())
  showdb()
  return ipywidgets.VBox(children=(ipywidgets.HBox(children=(wdb,wshow,wautorun,wdryrun)),wmsg,wout))
