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
from collections import namedtuple, defaultdict
from collections.abc import MutableMapping
from time import process_time, perf_counter
from . import ARG, SQliteNew, size_fmt, time_fmt

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
class CacheDB (MutableMapping):
  r"""
Instances of this class manage cache repositories. There is at most one instance of this class in a process for each normalised repository specification path. A cache repository contains a set of values stored in some form. Each value has meta information stored in an index stored as a sqlite database. The index entry attached to a value is called a cell and describes the call event which obtained it. A ``Cell`` entry has the following fields:

- ``oid``: a unique identifier of the cell, which also allows retrieval of the attached value;
- ``block``: reference to another entry in the database (table ``Block``) describing the functor of the call event which produced the value;
- ``ckey``: the key of the call event which produced the value;
- ``hitdate``: date of creation, or last reuse, of the cell;
- ``size``, ``tprc``, ``ttot``: size (in bytes) of the value and process and total time (in sec) of its computation.

``Block`` entries correspond to clusters of ``Cell`` entries (call events) sharing the same functor. They have the following fields

- ``signature``: the persistent functor (callable) producing all the cells in the block; different blocks always have signatures with different pickle byte-strings;
- ``misses``: number of call events with that functor where the argument had not been seen before;
- ``hits``: number of call events with that functor where the argument had previously been seen; such a call does not generate a new ``Cell``, but reuses the existing one;
- ``maxsize``: maximum number of cells attached to the block; when overflow occurs, the cells with the oldest ``hitdate`` are discarded (this amounts to the Least Recently Used policy, a.k.a. LRU, currently hardwired).

Furthermore, a :class:`CacheDB` instance acts as a mapping object, where the keys are block identifiers (:class:`int`) and values are :class:`CacheBlock` objects for the corresponding blocks. Such :class:`CacheBlock` objects are normally deactivated (i.e. their signatures do not support calls).

Finally, :class:`CacheDB` instances have an HTML ipython display.

Attributes:

.. attribute:: path

   the normalised path of this instance, as a :class:`pathlib.Path`

.. attribute:: dbpath

   the path to the sqlite database holding the index, as a string

Methods:

.. automethod:: __new__
  """
#==================================================================================================

  def __new__(cls,spec,listing={},lock=threading.Lock()):
    """
Generates a :class:`CacheDB` object.

:param spec: specification of the cache folder
:type spec: :class:`CacheDB`\|\ :class:`pathlib.Path`\|\ :class:`str`

* If *spec* is a :class:`CacheDB` instance, that instance is returned
* If *spec* is a path to a directory, returns a :class:`CacheDB` instance whose storage is an instance of :class:`Storage` pointing to that directory
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
        if path.is_dir(): storage = Storage(path)
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

  def connect(self,**ka):
    conn = sqlite3.connect(self.dbpath,**ka)
    conn.create_function('cellrm',2,lambda cell,size,s=self.storage: s.remove(cell,size))
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

:param db: specification of the cache repository where the block resides
:param signature: signature of the block, describing its functor
:type signature: :class:`Signature`
:param maxsize: if not :const:`None`, resizes the block at initialisation
:type maxsize: :class:`int`\|\ :class:`NoneType`
:param clear: if :const:`True`, clears the block at initialisation
:type clear: :class:`bool`

A block object is callable. When a new call is submitted to a block with argument *arg*, the following sequence happens:

- method :meth:`getkey` of the signature is invoked with argument *arg*. It must return a python byte string used as ``ckey`` in the corresponding index database cell.
- Then a transaction is begun on the index database.
- If there already exists a cell with the same ``ckey``, and its value has been computed, the transaction is immediately terminated and its value is extracted and returned.
- If there already exists a cell with the same ``ckey``, but its value is still being computed in another thread/process, the transaction is terminated, then the current thread waits until completion of the other thread/process, then the obtained value is extracted and returned, or raised if it is an exception.
- If there does not exist a cell with the same ``ckey``, a cell with this ``ckey`` is immediately created, then the transaction is terminated. Then method :meth:`getval` of the signature is invoked with argument *arg*. The result is stored, even if it is an exception, and a new transaction informs the database cell that its result has been computed. If the result was an exception, it is raised, otherwise it is returned.

Furthermore, a :class:`CacheBlock` object acts as a mapping where the keys are cell identifiers (:class:`int`) and values are triples ``hitdate``, ``ckey``, ``size``. When ``size`` is null, the value of the cell is still being computed, otherwise it represents its size in bytes.

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

#==================================================================================================
  def clear_error(self):
    r"""
Clears all the cells from this block which cache an exception.
    """
#==================================================================================================  
    with self.db.connect() as conn:
      conn.execute('DELETE FROM Cell WHERE block=? AND size<0',(self.block,))

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
  def __call__(self,arg):
#--------------------------------------------------------------------------------------------------
    ckey = self.sig.getkey(arg)
    with self.db.connect() as conn:
      conn.execute('BEGIN IMMEDIATE TRANSACTION')
      row = conn.execute('SELECT oid,size FROM Cell WHERE block=? AND ckey=?',(self.block,ckey,)).fetchone()
      if row is None:
        cell = conn.execute('INSERT INTO Cell (block,ckey) VALUES (?,?)',(self.block,ckey)).lastrowid
        size = None
        conn.execute('UPDATE Block SET misses=misses+1 WHERE oid=?',(self.block,))
      else:
        cell,size = row
        conn.execute('UPDATE Block SET hits=hits+1 WHERE oid=?',(self.block,))
      store = self.db.storage(cell,size,self.sig.timeout)
    if row is None:
      logger.info('%s MISS(%s)',self,cell)
      tm = process_time(),perf_counter()
      try: cval = self.sig.getval(arg)
      except BaseException as e: cval = e; size = -1
      else: size = 1
      tm = process_time()-tm[0],perf_counter()-tm[1]
      try: size *= store.setval(cval)
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
      if size==0:
        logger.info('%s WAIT(%s)',self,cell)
        store.waitval()
      cval = store.getval()
      logger.info('%s HIT(%s)',self,cell)
      with self.db.connect() as conn:
        conn.execute('UPDATE Cell SET hitdate=datetime(\'now\') WHERE oid=?',(cell,))
      if isinstance(cval,BaseException): raise cval
    return cval

#--------------------------------------------------------------------------------------------------
  def checkmax(self):
    r"""
Checks whether there is a cache overflow and applies the LRU policy.
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

  def _repr_html_(self):
    from lxml.etree import tounicode
    return tounicode(self.as_html())
  def as_html(self,size_fmt_=(lambda sz: '*'+size_fmt(-sz) if sz<0 else size_fmt(sz))):
    return html_table(sorted(self.items()),hdrs=('hitdate','ckey','size','tprc','ttot'),fmts=(str,self.sig.html,size_fmt_,time_fmt,time_fmt),title='{}: {}'.format(self.block,self.sig))
  def __str__(self): return 'Cache<{}:{}>'.format(self.db.path,self.sig)

#==================================================================================================
# Signatures
#==================================================================================================

#--------------------------------------------------------------------------------------------------
class Signature:
  r"""
An instance of this class defines a functor attached to a python top-level function. Parameters which do not influence the result can be specified (they will be ignored in the caching mechanism).

:param func: a function, defined at the top-level of its module (hence pickable)
:param ignore: a list of parameter names among those of *func*
:param timeout: an upper bound estimate of the execution time in secs of the function

The signature is entirely defined by the name of the function and of its module, as well as the sequence of parameter names, marked if ignored. Hence, two functions (possibly in different processes at different times) sharing these components produce the same signature.

Methods:
  """
#--------------------------------------------------------------------------------------------------
  def __init__(self,func,ignore,timeout):
    func = inspect.unwrap(func)
    self.name, self.params = '{}.{}'.format(func.__module__, func.__name__), tuple(getparams(func,ignore))
    self.func = func
    self.timeout = timeout
  def getkey(self,arg):
    r"""
Argument *arg* must be a pair of a list of positional arguments and a dict of keyword arguments. They are matched against the definition of :attr:`func`. Returns the pickled representation of the resulting assignment of the parameter names of the function in the order in which they appear in its definition, omitting the ignored ones. If a parameter name in :attr:`func` is prefixed by \*\* (hence its assignment is a dict), it is replaced by its sorted list of items.
    """
    return pickle.dumps(tuple(self.genkey(arg)))
  def genkey(self,arg):
    a,ka = arg
    d = inspect.getcallargs(self.func,*a,**ka)
    for p,typ in self.params:
      if typ!=-1: yield p,(sorted(d[p].items()) if typ==2 else d[p])
  def getval(self,arg):
    r"""
Argument *arg* must be a pair of a list of positional arguments and a dict of keyword arguments. Returns the value of calling attribute :attr:`func` with that positional argument list and keyword argument dict. Note that this attribute is not restored when the signature is obtained by unpickling, so invcation of this method fails.
    """
    a,ka = arg
    return self.func(*a,**ka)
  def html(self,ckey):
    r"""
Argument *ckey* must have been obtained by invocation of method :meth:`getkey`. Returns an HTML formatted representation of the argument of that invocation.
    """
    from lxml.builder import E
    return E.DIV(*self.genhtml(pickle.loads(ckey)))
  def genhtml(self,ckey):
    from lxml.builder import E
    disp = (lambda k,v: E.DIV(E.B(k),'=',E.EM(repr(v)),style='display: inline; padding: 5px;'))
    def h(ckey,dparams=dict(self.params)):
      for p,v in ckey:
        if dparams[p]==2: yield from (disp(k,vk) for k,vk in v)
        else: yield disp(p,v)
    return h(ckey)
  def __str__(self,mark={-1:'-',0:'',1:'*',2:'**'}): return '{}({})'.format(self.name,','.join(mark[typ]+p for p,typ in self.params))
  def __getstate__(self): return self.name, self.params
  def __setstate__(self,state): self.name, self.params = state

#--------------------------------------------------------------------------------------------------
class ProcessSignature (Signature):
  r"""
An instance of this class defines a functor attached to a python top-level function and a base signature.

:param func: a function, defined at the top-level of its module (hence pickable)
:param ignore: a list of parameter names among those of *func*
:param timeout: an upper bound estimate of the execution time in secs
:param base: a signature

.. method:: getkey(arg)
.. method:: getval(arg)

   Method :meth:`getkey` (resp. :meth:`getval`) applied to a pair *a*, *ka* returns the same as their counterpart in class :class:`Signature`, except the first positional argument in *a* is replaced by the result of applying to it the :meth:`getkey` (resp. :meth:`getval`) method of the base signature.
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,func,ignore,timeout,base):
    super(ProcessSignature,self).__init__(func,ignore,timeout)
    p,typ = self.params[0]
    self.params = ((p,-1),)+self.params[1:]
    self.base = base
  def genkey(self,arg):
    yield self.base.getkey(arg[0][0])
    yield from super(ProcessSignature,self).genkey(arg)
  def getval(self,args):
    a,ka = args
    return self.func(self.base.getval(a[0]),*a[1:],**ka)
  def html(self,ckey):
    from lxml.builder import E
    ckey = pickle.loads(ckey)
    return E.DIV(self.base.html(ckey[0]),E.DIV(*self.genhtml(ckey[1:]),style='border-top: thin dashed blue'),style='padding:0')
  def __str__(self): return '{};{}'.format(self.base,super(ProcessSignature,self).__str__())
  def __getstate__(self): return super(ProcessSignature,self).__getstate__(), self.base
  def __setstate__(self,state): state, self.base = state; super(ProcessSignature,self).__setstate__(state)

#==================================================================================================
class FileStorage:
  r"""
Instances of this class manage the persistent storage of cache values using a filesystem directory.

:param path: the directory to store cached values
:type path: :class:`pathlib.Path`

Methods:

.. automethod:: __call__
  """
#==================================================================================================

  def __init__(self,path):
    self.path = path

#--------------------------------------------------------------------------------------------------
  def __call__(self,cell,size,timeout,typ=namedtuple('StoreAPI',('waitval','getval','setval'))):
    r"""
Returns an incarnation of the API to manipulate the value of *cell* (see below).

:param cell: oid of the cell
:type cell: :class:`int`
:param size: size status of the cell
:type size: :class:`int`\|\ :class:`NoneType`
:param timeout: max waiting time when cell is being computed in another thread/process

The API to manipulate the value of a cell consists of:

- :func:`waitval` invoked (possibly concurrently) to wait for the value of *cell* to be computed.
- :func:`getval` invoked (possibly concurrently) to obtain the value of *cell*.
- :func:`setval` invoked with an argument *val* to assign the value of *cell* to *val*. Invoked only once, but *cell* may have disappeared from the cache when invoked, or may disappear while executing, so the assignment may have to later be rollbacked.

The *size* parameter has the following meaning:

- When *size* is :const:`None`, the cell has just been created by the current thread, which will compute its value and invoke :func:`setval` to store it.
- When *size* is :const:`0`, the cell is currently being computed by another thread (possibly in another process) and the current thread will invoke :func:`waitval` to wait until the value is available, then :func:`getval` to get its value.
- Otherwise, the cell has been computed in the past and the current thread will invoke :func:`getval` to get its value.
    """
#--------------------------------------------------------------------------------------------------
    tpath,rpath = self.getpaths(cell)
    def setval(val):
      try: pickle.dump(val,vfile)
      except Exception as e: vfile.seek(0); vfile.truncate(); pickle.dump(e,vfile)
      s = vfile.tell()
      vfile.close(); conn.close()
      return s
    def getval():
      try: return pickle.load(vfile)
      finally: vfile.close()
    def waitval():
      try: conn.execute('BEGIN IMMEDIATE TRANSACTION')
      except: pass
      finally: conn.close()
    if size is None:
      vfile = rpath.open('wb')
      conn = sqlite3.connect(str(tpath))
      conn.execute('BEGIN IMMEDIATE TRANSACTION')
    else:
      vfile = rpath.open('rb')
      if size==0: conn = sqlite3.connect(str(tpath),timeout)
    return typ(waitval,getval,setval)

  def remove(self,cell,size):
    tpath,rpath = self.getpaths(cell)
    try: rpath.unlink(); tpath.unlink()
    except: pass

  def getpaths(self,cell):
    vpath = self.path/'V{:06d}'.format(cell)
    tpath = vpath.with_suffix('.tmp')
    rpath = vpath.with_suffix('.pck')
    return tpath,rpath

#==================================================================================================
class Storage (FileStorage):
  r"""
The default storage class for a cache repository. Stores the index database in the same directory as the values, with name ``index.db``.

Attributes:

.. attribute:: dbpath

   The :class:`pathlib.Path` to the sqlite database holding the index
  """
#==================================================================================================
  def __init__(self,path):
    super(Storage,self).__init__(path)
    self.dbpath = path/'index.db'
    if not self.dbpath.is_file() and any(path.iterdir()):
      raise Exception('Cannot create new index in non empty directory')

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def lru_persistent_cache(ignore=(),timeout=86400.,factory=CacheBlock,**ka):
  r"""
A decorator which applies to a function and replaces it by a persistently cached version. The function must be defined at the top-level of its module, to be compatible with :class:`Signature`.

:param ignore,timeout: passed, together with the function, to the :class:`Signature` constructor
:param ka: keyword argument dict passed to :class:`CacheBlock`
  """
#--------------------------------------------------------------------------------------------------
  def transf(f):
    c = factory(signature=Signature(f,ignore,timeout),**ka)
    F = lambda *a,**ka: c((a,ka))
    F.cache = c
    return update_wrapper(F,f)
  return transf

#--------------------------------------------------------------------------------------------------
class DerivedSignature:
#--------------------------------------------------------------------------------------------------
  def __init__(self,cache): self.getval = cache; self.alt = cache.sig
  def getkey(self,ckey): return self.alt.getkey(ckey)
  def getval(self,arg): return self.alt.cache(arg) # overridden at instance level unless unpickled
  def html(self,ckey): return self.alt.html(ckey)
  def __str__(self): return str(self.alt)
  def __getstate__(self): return self.alt
  def __setstate__(self,state): self.alt = state

#--------------------------------------------------------------------------------------------------
def make_process_step(base,f,ignore=(),timeout=86400.,factory=CacheBlock,**ka):
  r"""
Basic process cache factory (invoked to cache each process step).

:param base: a cache block
:type base: :class:`CacheBlock`\|\ :class:`NoneType`
:param f: a function
:param ignore,timeout: passed, together with the function, to the signature constructor

The signature of the returned cache is an instance of :class:`Signature` if *base* is :const:`None`, otherwise an instance of :class:`ProcessSignature`, in which case, its base is a copy of the signature of *base* where :attr:`getval` is overridden by *base* itself. This is what enables the "stacked cache" effect of processes.
  """
#--------------------------------------------------------------------------------------------------
  if base is None: sig = Signature(f,ignore,timeout)
  elif isinstance(base,CacheBlock): sig = ProcessSignature(f,ignore,timeout,base=DerivedSignature(base))
  else: raise TypeError('Cannot create a cache with base of type {} (not {})'.format(type(base),CacheBlock))
  c = factory(signature=sig,**ka)
  c.base = base
  return c

#--------------------------------------------------------------------------------------------------
def make_process(base,*steps,**spec):
  r"""
Basic process factory.

:param base: the base cache block, which must be a process cache (optional, defaults to :const:`None`\)
:type base: :class:`CacheBlock`\|\ :class:`NoneType`
:param steps: list of steps
:type steps: list(\ :class:`ARG`\)

Each step is defined by a :class:`ARG` object whose first positional argument must be a string representing the step name, the rest being called the step configuration. The *spec* dict is used to customise the configuration of each step. First, each *spec* item whose key is not a step name is used to update the keyword argument of the configuration of all the steps. Then, each *spec* item whose key is a step name must have a dict value, which is used to update the keyword argument of that step's configuration. Thus the following expressions are equivalent::

   F= make_process(ARG('s_A',fA,ignore=('z',)),ARG('s_B',fB),clear=True,db=DIR,s_A=dict(clear=False))
   F= make_process(ARG('s_A',fA,ignore=('z',),clear=False,db=DIR),ARG('s_B',fB,clear=True,db=DIR))

Then function :func:`make_process_step` is called for each step with the result of the previous call used as base (initially the *base* argument), and the updated step configuration as other arguments. In the above example, this leads to the following cache construction::

   make_process_step(make_process_step(None,fA,ignore=('z',),clear=False,db=DIR),fB,clear=True,db=DIR)

Finally, the returned value is a function using the cache thus obtained (accessible through the attribute :attr:`cache` of the function). The function must be invoked with keyword arguments only, assigning to each step name a :class:`ARG` object called the step instantiation. Except for the first step, the positional argument of each step instantiation is prepended with the step instantiation of the previous step. The updated instantiation of the last step is then submitted to the cache. Thus, in the above example, the following expressions are equivalent::

   F(s_A=ARG(3,z=22),s_B=ARG(4,u=5),...)
   F.cache(ARG(ARG(3,z=22),4,u=5))
   fB(fA(3,z=22),4,u=5)

where both the inner expression ``fA()`` and the outer expression ``fB()`` are independently persistently cached.
  """
#--------------------------------------------------------------------------------------------------
  def setdflt(ka,v): ka.update(spec); ka.update(v); return ka
  def caller(steps,cache,dflt=ARG()):
    def F(**args):
      arg = args.get(steps[0],dflt)
      for step in steps[1:]:
        a,ka = args.get(step,dflt)
        arg = ARG(arg,*a,**ka)
      return cache(arg)
    return F
  if isinstance(base,ARG): steps = (base,)+steps; base = None
  cfg = [(a[0],a[1:],ka,spec.pop(a[0],())) for a,ka in steps]
  cfg = [(step,a,setdflt(ka.copy(),v)) for step,a,ka,v in cfg]
  cache = base
  bases,steps = ((),()) if cache is None else (cache.bases,cache.steps)
  for step,a,ka in cfg:
    cache = make_process_step(cache,*a,**ka)
    cache.base,base = base,cache
    bases = cache.bases = bases+(cache,)
    steps = cache.steps = steps+(step,)
    F = cache.caller = caller(steps,cache)
    F.cache = cache
  return F

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
def html_stack(*a,**ka):
#--------------------------------------------------------------------------------------------------
  from lxml.builder import E
  return E.DIV(*(E.DIV(x,**ka) for x in a))

#--------------------------------------------------------------------------------------------------
class SignatureMismatchException (Exception):
  """
Exception raised when restoring a passive signature fails.
  """
#--------------------------------------------------------------------------------------------------
  pass
#--------------------------------------------------------------------------------------------------
class State: pass
#--------------------------------------------------------------------------------------------------

