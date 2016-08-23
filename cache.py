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
from collections import namedtuple, defaultdict
from collections.abc import MutableMapping
from time import process_time, perf_counter
from . import SQliteNew, size_fmt, time_fmt, html_stack, html_table, html_parlist, HtmlPlugin

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
  ckey BLOB NOT NULL,
  size INTEGER DEFAULT 0,
  tprc REAL,
  ttot REAL
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
class CacheDB (MutableMapping,HtmlPlugin):
  r"""
Instances of this class manage cache repositories. There is at most one instance of this class in a process for each normalised repository specification path. A cache repository contains a set of values stored in some form. Each value has meta information stored in an index stored as a sqlite database. The index entry attached to a value is called a cell and describes the call event which obtained it. A ``Cell`` entry has the following fields:

- ``oid``: a unique identifier of the cell, which also allows retrieval of the attached value;
- ``block``: reference to another entry in the database (table ``Block``) describing the functor of the call event which produced the value;
- ``ckey``: the key of the call event which produced the value;
- ``hitdate``: date of creation, or last reuse, of the cell;
- ``size``, ``tprc``, ``ttot``: size (in bytes) of the value and process and total time (in sec) of its computation.

``Block`` entries correspond to clusters of ``Cell`` entries (call events) sharing the same functor. They have the following fields

- ``signature``: the persistent functor producing all the cells in the block; different blocks always have signatures with different pickle byte-strings;
- ``misses``: number of call events with that functor where the argument had not been seen before;
- ``hits``: number of call events with that functor where the argument had previously been seen; such a call does not generate a new ``Cell``, but reuses the existing one;
- ``maxsize``: maximum number of cells attached to the block; when overflow occurs, the cells with the oldest ``hitdate`` are discarded (this amounts to the Least Recently Used policy, a.k.a. LRU, currently hardwired).

Furthermore, a :class:`CacheDB` instance acts as a mapping object, where the keys are block identifiers (:class:`int`) and values are :class:`CacheBlock` objects for the corresponding blocks. Such :class:`CacheBlock` objects may be deactivated (i.e. their signatures do not support calls).

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
        self = super(CacheDB,cls).__new__(cls)
        self.path = path
        self.dbpath = dbpath
        self.storage = storage
        listing[path] = self
    return self

  def __getnewargs__(self): return self.path,
  def __hash__(self): return hash(self.path)
  # __eq__ not needed because default behaviour ('is') is OK due to spec cacheing

  def connect(self,**ka):
    conn = sqlite3.connect(self.dbpath,timeout=self.timeout,**ka)
    conn.create_function('cellrm',2,self.storage.remove)
    return conn

  def getblock(self,sig):
    sigp = pickle.dumps(sig)
    with self.connect() as conn:
      conn.execute('BEGIN IMMEDIATE TRANSACTION')
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

  def clear(self):
    with self.connect() as conn:
      conn.execute('DELETE FROM Block')

#--------------------------------------------------------------------------------------------------
# Display
#--------------------------------------------------------------------------------------------------

  def as_html(self,incontext):
    return html_stack(*(v.as_html(incontext) for k,v in sorted(self.items())))
  def __str__(self): return 'Cache<{}>'.format(self.path)

#==================================================================================================
class CacheBlock (MutableMapping,HtmlPlugin):
  r"""
Instances of this class implements blocks of cells sharing the same functor (signature).

:param db: specification of the cache repository where the block resides
:param signature: signature of the block, describing its functor
:type signature: :class:`Signature`
:param maxsize: if not :const:`None`, resizes the block at initialisation
:type maxsize: :class:`int`\|\ :class:`NoneType`
:param cacheonly: if :const:`True`, cell creation is disallowed
:type cacheonly: :class:`bool`

A block object is callable, and calls take a single argument. Method :meth:`__call__` implements the cacheing mechanism.

* Duplicate detection of arguments and computation of the value attached to an argument is delegated to a dedicated signature object which must implement the following api (e.g. implemented by class :class:`Signature`):

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

  def __init__(self,db=None,signature=None,block=None,maxsize=None,cacheonly=False):
    self.sig = signature
    self.db = db = CacheDB(db)
    self.block = db.getblock(signature) if block is None else block
    if maxsize is not None: self.resize(maxsize)
    self.cacheonly = cacheonly

  def __hash__(self): return hash((self.db,self.block))
  def __eq__(self,other): return isinstance(other,CacheBlock) and self.db is other.db and self.block == other.block

  def clear_error(self):
    r"""
Clears all the cells from this block which cache an exception.
    """
    with self.db.connect() as conn:
      conn.execute('DELETE FROM Cell WHERE block=? AND size<0',(self.block,))

  def resize(self,n):
    assert isinstance(n,int) and n>=1
    with self.db.connect() as conn:
      conn.execute('UPDATE Block SET maxsize=? WHERE oid=?',(n,self.block,))
    self.checkmax()

  def info(self,typ=namedtuple('BlockInfo',('signature','hits','misses','maxsize','currsize'))):
    r"""
Returns information about this block. Available attributes:
:attr:`signature`, :attr:`hits`, :attr:`misses`, :attr:`maxsize`, :attr:`currsize`
    """
    with self.db.connect(detect_types=sqlite3.PARSE_DECLTYPES) as conn:
      sz = conn.execute('SELECT count(*) FROM Cell WHERE block=?',(self.block,)).fetchone()[0]
      row = conn.execute('SELECT signature, hits, misses, maxsize FROM Block WHERE oid=?',(self.block,)).fetchone()
    return typ(row[0].info(),*row[1:],currsize=sz)

#--------------------------------------------------------------------------------------------------
  def __call__(self,arg):
    """
:param arg: arument of the call

Implements cacheing as follows:

- method :meth:`getkey` of the signature is invoked with argument *arg* to obtain a ``ckey``.
- Then a transaction is begun on the index database.
- If there already exists a cell with the same ``ckey``, the transaction is immediately terminated and its value is extracted, using method :meth:`lookup` of the storage, and returned.
- If there does not exist a cell with the same ``ckey``, a cell with that ``ckey`` is immediately created, then the transaction is terminated. Then method :meth:`getval` of the signature is invoked with argument *arg*. The result is stored, even if it is an exception, using method :meth:`insert` of the storage, and completion is recorded in the database.

If the result was an exception, it is raised, otherwise it is returned. In all cases, hit status is updated in the database (for the LRU policy).
    """
#--------------------------------------------------------------------------------------------------
    ckey = self.sig.getkey(arg)
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
      try: cval = self.sig.getval(arg)
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
      self.checkmax()
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
  def checkmax(self):
    r"""
Checks whether there is a cache overflow and applies the LRU policy (hardwired).
    """
#--------------------------------------------------------------------------------------------------
    with self.db.connect() as conn:
      conn.execute('BEGIN IMMEDIATE TRANSACTION')
      row = conn.execute('SELECT osize FROM Overflow WHERE block=?',(self.block,)).fetchone()
      if row is not None:
        osize = row[0]
        logger.info('%s OVERFLOW(%s)',self,osize)
        conn.execute('DELETE FROM Cell WHERE oid IN (SELECT oid FROM Cell WHERE block=? AND size>0 ORDER BY hitdate ASC, oid ASC LIMIT ?)',(self.block,osize,))

#--------------------------------------------------------------------------------------------------
# CacheBlock as Mapping
#--------------------------------------------------------------------------------------------------

  def __getitem__(self,cell):
    with self.db.connect() as conn:
      r = conn.execute('SELECT hitdate, ckey, size, tprc, ttot FROM Cell WHERE oid=?',(cell,)).fetchone()
    if r is None: raise KeyError(cell)
    return r

  def __delitem__(self,cell):
    with self.db.connect() as conn:
      conn.execute('DELETE FROM Cell WHERE oid=?',(cell,))
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
    with self.db.connect() as conn:
      for row in conn.execute('SELECT oid, hitdate, ckey, size, tprc, ttot FROM Cell WHERE block=?',(self.block,)):
        yield row[0],row[1:]

  def clear(self):
    with self.db.connect() as conn:
      conn.execute('DELETE FROM Cell WHERE block=?',(self.block,))

#--------------------------------------------------------------------------------------------------
# Display
#--------------------------------------------------------------------------------------------------

  def as_html(self,incontext,size_fmt_=(lambda sz: '*'+size_fmt(-sz) if sz<0 else size_fmt(sz)),time_fmt_=(lambda t: '' if t is None else time_fmt(t))):
    return html_table(sorted(self.items()),hdrs=('hitdate','ckey','size','tprc','ttot'),fmts=(str,(lambda ckey,h=self.sig.html: h(ckey,incontext)),size_fmt_,time_fmt_,time_fmt_),title='{}: {}'.format(self.block,self.sig))
  def __str__(self): return 'Cache<{}:{}>'.format(self.db.path,self.sig)

#==================================================================================================
class Signature:
  r"""
An instance of this class defines a functor attached to a python top-level function. Parameters which do not influence the result can be specified (they will be ignored in the caching mechanism).

:param func: a function, defined at the top-level of its module (hence pickable)
:param ignore: a list of parameter names among those of *func*

The signature is entirely defined by the name of the function and of its module, as well as the sequence of parameter names, marked if ignored. Hence, two functions (possibly in different processes at different times) sharing these components produce the same signature.

Methods:
  """
#==================================================================================================
  def __init__(self,func,ignore):
    func = inspect.unwrap(func)
    self.name, self.params = '{}.{}'.format(func.__module__, func.__name__), tuple(getparams(func,ignore))
    self.func = func
#--------------------------------------------------------------------------------------------------
  def getkey(self,arg):
    r"""
Argument *arg* must be a pair of a list of positional arguments and a dict of keyword arguments. They are matched against the definition of :attr:`func`. Returns the pickled representation of the resulting values of the parameter names of the function in the order in which they appear in its definition, omitting the ignored ones. If a parameter name in :attr:`func` is prefixed by \*\* (hence its assignment is a dict), it is replaced by its sorted list of items.
    """
#--------------------------------------------------------------------------------------------------
    return pickle.dumps(tuple(self.genkey(arg)))
  def genkey(self,arg):
    a,ka = arg
    d = inspect.getcallargs(self.func,*a,**ka)
    for p,typ in self.params:
      if typ!=-1: yield sorted(d[p].items()) if typ==2 else d[p]
#--------------------------------------------------------------------------------------------------
  def getval(self,arg):
    r"""
Argument *arg* must be a pair of a list of positional arguments and a dict of keyword arguments. Returns the value of calling attribute :attr:`func` with that positional argument list and keyword argument dict. Note that attribute :attr:`func` is not restored when the signature is obtained by unpickling, so invocation of this method is disabled in that case.
    """
#--------------------------------------------------------------------------------------------------
    a,ka = arg
    return self.func(*a,**ka)

#--------------------------------------------------------------------------------------------------
  def html(self,ckey,incontext):
#--------------------------------------------------------------------------------------------------
    def h(ckey):
      for v,(p,typ) in zip(ckey,((p,typ) for p,typ in self.params if typ!=-1)):
        if typ==2: yield from v
        else: yield p,v
    return html_parlist((),h(pickle.loads(ckey)),incontext)

  def info(self): return self.name
  def __str__(self,mark={-1:'-',0:'',1:'*',2:'**'}): return '{}({})'.format(self.name,','.join(mark[typ]+p for p,typ in self.params))
  def __getstate__(self): return self.name, self.params
  def __setstate__(self,state): self.name, self.params = state

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
def lru_persistent_cache(*a,**ka):
  r"""
A decorator which applies to a function and returns a persistently cached version of it. The function must be defined at the top-level of its module, to be compatible with :class:`Signature`. When not applied to a function (as sole positional argument), returns a version of itself with default keyword arguments given by *ka*.
  """
#--------------------------------------------------------------------------------------------------
  def transf(f,ignore=(),factory=CacheBlock,**ka):
    c = factory(signature=Signature(f,ignore),**ka)
    F = lambda *a,**ka: c((a,ka))
    F.cache = c
    return update_wrapper(F,f)
  return transf(*a,**ka) if a else partial(lru_persistent_cache,**ka)

#--------------------------------------------------------------------------------------------------
def getparams(func,ignore=(),code={inspect.Parameter.VAR_POSITIONAL:1,inspect.Parameter.VAR_KEYWORD:2}):
#--------------------------------------------------------------------------------------------------
  for p in inspect.signature(func).parameters.values():
    yield p.name,(-1 if p.name in ignore else code.get(p.kind,0))
