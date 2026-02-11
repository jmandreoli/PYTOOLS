# File:                 cache.py
# Creation date:        2015-03-19
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Persistent cache management
#

import logging; logger = logging.getLogger(__name__)
from typing import Any, Callable, Iterable, Mapping, Sequence

import pickle, inspect, threading, abc
from pathlib import Path
from functools import update_wrapper
from itertools import islice
from collections import namedtuple
from weakref import WeakValueDictionary
from datetime import datetime, timezone
from sqlalchemy import select, func
from .cache_v1 import get_sessionmaker, Block, Cell
from . import size_fmt, time_fmt, pickleclass

__all__ = 'CacheDB', 'CacheBlock', 'AbstractFunctor', 'Functor', 'AbstractStorage', 'FileStorage', 'DefaultStorage'

NOW = lambda tz=timezone.utc: datetime.now(tz=tz) # shorthand for timestamping
BlockInfo = namedtuple('BlockInfo','hits ncell ncell_error ncell_pending')

#==================================================================================================
class CacheDB:
  r"""
Instances of this class manage cache repositories. There is at most one instance of this class in a process for each normalised repository specification path. A cache repository contains cells, each cell corresponding to one cached value produced by a unique call, and possibly reused by later calls. Cells are clustered into blocks, each block grouping cells produced by the same call type, called a functor (of class :class:`AbstractFunctor`). Meta-information about blocks and cells are stored in an index database accessed through :mod:`sqlalchemy`. The values themselves are persistently stored by a dedicated storage object (of class :class:`AbstractStorage`).

:class:`CacheDB` instances have an HTML ipython display.
  """
#==================================================================================================

  path:Path
  url:str
  r"""the sqlalchemy url of the database"""
  session_maker:Callable[[],Any]
  r"""the sqlalchemy session maker for manipulation of this database"""
  timeout:float = 120.

  def __new__(cls,spec:CacheDB|Path|str,listing={},lock=threading.Lock()):
    r"""
:param spec: cache specification

Generates a :class:`CacheDB` object.

* If *spec* is already a :class:`CacheDB` instance, that instance is returned
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
        storage:AbstractStorage
        if path.is_dir(): storage = DefaultStorage(path)
        elif path.is_file():
          with path.open('rb') as u: storage = pickle.load(u); assert isinstance(storage,AbstractStorage)
        else: raise ValueError('Cache repository specification path must be directory or file')
        self = super().__new__(cls)
        self.path = path
        self.url = url = storage.db_url
        self.session_maker = get_sessionmaker(url)
        self.storage = storage
        listing[path] = self
    return self

  def __getnewargs__(self): return self.path,
  def __getstate__(self): return
  def __hash__(self): return hash(self.path)
  # no need to define '__eq__': default 'is' behaviour works due to '__new__' constructor

  def clear_obsolete(self,strict:bool,dry_run=False):
    r"""
Clears all the blocks which are obsolete.
    """
    assert isinstance(strict,bool)
    def obsolete(block,strictf=(bool if strict else (lambda o: o is not None)))->bool:
      return strictf(pickle.loads(block.functor).obsolete()) is True
    return self.clear((lambda L: [block for block in L if obsolete(block)]),dry_run)
    # this may load modules which add new (non obsolete) entries which must not be included in the list

  def clear(self,f:Callable[[Sequence[Block]],Sequence[Block]]=(lambda L: L),dry_run:bool=False):
    r"""Deletes blocks from this instance."""
    with self.session_maker() as session:
      L,L_ = f([block for block, in session.execute(select(Block))]),[]
      if dry_run is True: return L
      if (n:=len(L))>0:
        for block in L: session.delete(block); L_.extend(cell.oid for cell in block.cells)
        session.commit(); logger.info('%s DELETED %s',self,n)
    self.storage.remove(L_)
    return n

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  def _repr_html_(self): from .html import repr_html; return repr_html(self)
  _html_limit = 50
  def as_html(self,_:Callable[[Any],None]):
    from .html import html_table
    n_max = self._html_limit
    with self.session_maker() as session:
      L = [CacheBlock(self,pickle.loads(block.functor),block.oid) for block, in session.execute(select(Block).order_by(Block.oid).limit(n_max))]
      n = session.execute(select(func.count(Block.oid))).scalar()-len(L)
    closing = f'{n} more' if n>0 else None
    return html_table(
      sorted((c.oid,(c,)) for c in L),
      fmts=((lambda x: x.as_html(_,session=session)),),
      opening=repr(self),
      closing=closing
    )
  def __repr__(self): return f'{self.__class__.__name__}<{self.path})>'

#==================================================================================================
class CacheBlock:
  r"""
Instances of this class implements blocks of cells sharing the same functor.

:param db: specification of the cache repository where the block resides
:param functor: functor of the block
:param cacheonly: if :const:`True`, cell creation is disallowed

A :class:`CacheBlock` instance is callable, and calls take a single argument. Method :meth:`__call__` implements the cross-process cacheing mechanism which produces and reuses cache cells. It also implements a weak cache for local calls (within its process).

Furthermore, a :class:`CacheBlock` instance acts as a mapping where the keys are cell identifiers (:class:`int`) and values are tuples of meta-information about the cells (i.e. not the values of the cells: these are only accessible through calling).

Finally, :class:`CacheBlock` instances have an HTML ipython display.

.. automethod:: __call__
  """
#==================================================================================================

  db:CacheDB
  r"""the :class:`CacheDB` instance this block belongs to"""
  functor:AbstractFunctor
  r"""the functor for this block (field ``functor`` in the ``Block`` table of the index is the functor's pickle)"""
  oid:int
  r"""the identifier of this block (field ``oid`` in the ``Block`` table of the index)"""
  cacheonly:bool
  r"""whether cell creation is disabled"""
  memory:WeakValueDictionary
  r"""a local cache of calls within the current process"""

  def __init__(self,db:CacheDB|Path|str='',functor:AbstractFunctor|None=None,oid:int=None,cacheonly:bool=False):
    self.db = db = CacheDB(db)
    self.functor = functor
    if oid is None:
      p = pickle.dumps(functor)
      with db.session_maker() as session:
        oid = session.execute(select(Block.oid).where(Block.functor==p)).scalar()
        if oid is None:
          session.add(block:=Block(functor=p)); session.flush(); oid = block.oid; session.commit()
          logger.info('%s F-MISS(%s)',self,oid)
        else: logger.info('%s F-HIT(%s)',self,oid)
    self.oid = oid
    self.cacheonly = cacheonly
    self.memory = WeakValueDictionary()
  def getblock(self,session)->Block: return session.get(Block,self.oid)

  def __hash__(self): return hash((self.db,self.oid))
  def __eq__(self,other): return isinstance(other,CacheBlock) and self.db == other.db and self.oid == other.oid

  def clear_error(self,dry_run:bool=False):
    r"""
Clears all the cells from this block which cache an exception.
    """
    return self.clear((lambda L: [cell for cell in L if cell.size<0]),dry_run)

  def clear_overflow(self,n:int,dry_run:bool=False):
    r"""
Clears all the cells from this block except the *n* most recent (lru policy).
    """
    assert isinstance(n,int) and n>=1
    return self.clear((lambda L: sorted((cell for cell in L if cell.size>0),key=(lambda cell: cell.tstamp),reverse=True)[n:]),dry_run)

  def clear(self,f:Callable[[Sequence[Cell]],Sequence[Cell]]=(lambda L:L),dry_run:bool=False):
    r"""Deletes cells from this instance."""
    with self.db.session_maker() as session:
      L = f(self.getblock(session).cells)
      if dry_run: return L
      L_ = [cell.oid for cell in L]
      if (n:=len(L))>0:
        for cell in L: session.delete(cell)
        session.commit();logger.info('%s DELETED %s',self,n)
    self.db.storage.remove(L_)
    return n

  def info(self):
    r"""
Returns information about this block. Available attributes:
:attr:`hits`, :attr:`ncell`, :attr:`ncell_error`, :attr:`ncell_pending`
    """
    hits = n = n_error = n_pending = 0
    with self.db.session_maker() as session:
      for cell in self.getblock(session).cells:
        n += 1; hits += cell.hits
        if cell.size==0: n_pending += 1
        elif cell.size<0: n_error += 1
    return BlockInfo(hits,n,n_error,n_pending)

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
    if (cval:=self.memory.get(ckey,self)) is not self: return cval # self is unlikely to be a stored value
    with self.db.session_maker() as session:
      q = select(Cell).where((Cell.block_oid==self.oid)&(Cell.ckey==ckey))
      cell = session.execute(q).scalar()
      newcell = cell is None
      if newcell is True:
        if self.cacheonly: raise Exception('Cache cell creation disallowed')
        cell = Cell(block_oid=self.oid,ckey=ckey,tstamp=NOW())
        session.add(cell); session.flush(); oid = cell.oid
        session.commit()
      else:
        oid,wait = cell.oid,cell.size==0
    if newcell is True:
      logger.info('%s MISS(%s)', self, oid)
      start = NOW()
      try: cval = self.functor.getval(arg)
      except BaseException as e: cval = e; size = -1
      else: size = 1
      duration = (NOW()-start).total_seconds()
      size *= self.db.storage.setval(oid,cval)
      with self.db.session_maker() as session:
        cell = session.get(Cell,oid)
        if cell is None: raise Exception(f'Lost cell {oid}')
        cell.size,cell.duration,cell.tstamp = size,duration,NOW()
        session.commit()
      if size<0: raise cval
    else:
      if wait is True: logger.info('%s WAIT(%s)',self,oid)
      cval = self.db.storage.getval(oid,wait)
      logger.info('%s HIT(%s)',self,oid)
      with self.db.session_maker() as session:
        cell = session.get(Cell,oid); cell.hits += 1; cell.tstamp = NOW(); session.commit()
      if isinstance(cval,BaseException): raise cval
    try: self.memory[ckey] = cval
    except: pass
    return cval

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  def _repr_html_(self): from .html import repr_html; return repr_html(self)
  _html_limit = 50
  def as_html(self,_,session=None,size_fmt_=(lambda sz: '*'+size_fmt(-sz) if sz<0 else size_fmt(sz)),time_fmt_=(lambda t: '' if t is None else time_fmt(t))):
    from .html import html_table
    if session is None: session = self.db.session_maker()
    with session:
      L = sorted([cell for cell, in islice(session.execute(select(Cell).where(Cell.block_oid==self.oid)),self._html_limit)],key=(lambda cell: cell.tstamp))
      n = session.execute(select(func.count(Cell.oid)).where(Cell.block_oid==self.oid)).scalar()-self._html_limit
      closing = f'{n} more' if n>0 else None
      return html_table(
        [(cell.oid,(cell.ckey,cell.tstamp,cell.hits,cell.size,cell.duration)) for cell in L],
        hdrs=('ckey','tstamp','hits','size','duration'),
        fmts=((lambda ckey,h=self.functor.html: h(ckey,_)),str,str,size_fmt_,time_fmt_),
        opening=repr(self),
        closing=closing
      )

  def __repr__(self): return f'{self.__class__.__name__}<{self.functor!r}>'

#==================================================================================================
class AbstractFunctor (metaclass=abc.ABCMeta):
  r"""
An instance of this class defines a type of (single argument) call to be cached.
  """
#==================================================================================================

  @abc.abstractmethod
  def getkey(self,arg:Any):
    r"""
:param arg: an arbitrary python object.

Returns a byte string which represents *arg* uniquely.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def getval(self,arg:Any):
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

  db_url:str
  r"""sqlalchemy url of the index database"""

  @abc.abstractmethod
  def setval(self,oid:int,val:Any)->int:
    r"""
:param oid: the identifier of a cell

Stores the cell value. This method is called inside the transaction which inserts a new cell into a cache index, hence exactly once overall for a given cell.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def getval(self,oid:int,wait:bool)->Any:
    r"""
:param oid: the identifier of a cell
:param wait: whether the cell value is currently being computed by a concurrent thread/process

Retrieves the cell value, possibly waiting for it to be stored. This method is called inside the transaction which looks up a cell from a cache index, which may happens multiple times in possibly concurrent threads/processes for a given cell.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def remove(self,L:Iterable[int]):
    r"""
:param L: an iterable of cell identifiers

Frees the storage resources associated with the cells.
    """
    raise NotImplementedError()

#==================================================================================================
class Functor (AbstractFunctor):
  r"""
An instance of this class defines a functor attached to a python top-level versioned function. The functor is entirely defined by the name of the function, that of its module, its version and its signature. These components are saved on pickling and restored on unpickling, even if the function has disappeared or changed. This is not checked on unpickling, and method :meth:`getval` is disabled.

.. automethod:: __new__
  """
#===================================================================================================

  __slots__ = 'config', 'sig', 'func'

  func:Callable
  r"""the versioned function characterizing this functor"""
  sig:inspect.Signature
  r"""signature of this functor"""
  config:tuple[Shadow,Any]
  r"""configuration of this functor"""

  def __new__(cls,spec,fromfunc:bool=True):
    r"""
Generates a functor.

:param spec: a versioned function, defined at the top-level of its module (hence pickable)
    """
    self = super().__new__(cls)
    if fromfunc:
      self.func = spec
      self.sig = sig = inspect.signature(spec)
      self.config = Shadow(spec),sig_dump(sig)
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

  def getkey(self,arg:tuple[Iterable[Any],Mapping[str,Any]]):
#--------------------------------------------------------------------------------------------------
    r"""
Argument *arg* must be a pair of a list of positional arguments and a dict of keyword arguments. They are normalised against the signature of the functor and the pickled value of the result is returned. The pickling of versioned function objects is modified to embed their version.
    """
#--------------------------------------------------------------------------------------------------
    a,ka = self.norm(arg)
    return self.fpickle.dumps((a,sorted(ka.items())))

  def getval(self,arg:tuple[Iterable[Any],Mapping[str,Any]]):
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
    from .html import html_parlist
    a,ka = self.fpickle.loads(ckey)
    return html_parlist(_,a,ka)

  def norm(self,arg:tuple[Iterable[Any],Mapping[str,Any]]):
    a,ka = arg
    b = self.sig.bind(*a,**ka)
    b.apply_defaults()
    return b.args, b.kwargs

  def obsolete(self): return self.config[0].obsolete()

def sig_dump(sig): return tuple((p.name,p.kind,p.default) for p in sig.parameters.values())
def sig_load(x): return inspect.Signature([inspect.Parameter(name,kind,default=default) for name,kind,default in x])

#==================================================================================================
class FileStorage (AbstractStorage):
  r"""
Instances of this class manage the persistent storage of cached values using a filesystem directory.

:param path: the directory to store cached values

The storage for a cell consists of a content file which contains the value of the cell in pickled format and a cross process mechanism to synchronise access to the content file between writer and possibly multiple readers. The path of the content file as well as that of the file underlying the synch lock are built from the cell id, so as to be unique to that cell. They inherit the access rights of the :attr:`path` directory.
  """
#==================================================================================================

  path:Path
  r"""directory where values are stored"""
  gate: Mapping
  r"""dictionary of threading events"""
  from watchdog.events import FileSystemEventHandler
  class MyEventHandler(FileSystemEventHandler):
    r"""Watchdog event for new value files."""
    def __init__(self):
      gate = {}
      lock = threading.Lock()
      def notify(p):
        with lock:
          if (ev:=gate.get(p)) is not None: ev.set(); del gate[p]
      self.notify = notify
      def wait(p):
        with lock:
          if p.exists(): return
          if (ev:=gate.get(p)) is None: ev = gate[p] = threading.Event()
        ev.wait()
      self.wait = wait
      def remove(L):
        with lock:
          for p in L:
            if gate.get(p) is not None: del gate[p]
      self.remove = remove
      super().__init__()
    def on_moved(self,event):
      r""""""
      p = Path(event.dest_path)
      if p.suffix == '.pck': self.notify(p)
  del FileSystemEventHandler
  def __init__(self,path:Path):
    from watchdog.observers import Observer
    self.path = path
    self.monitor = monitor = self.MyEventHandler()
    observer = Observer()
    observer.schedule(monitor,path,recursive=True)
    observer.start()

#--------------------------------------------------------------------------------------------------
  def setval(self,oid:int,val:Any)->int:
    r"""
Dumps (pickle) *val* into some temporary file and renames it to the content path for *oid*. Renaming will trigger a :mod:`watchdog` event for other processes.
    """
#--------------------------------------------------------------------------------------------------
    vpath = self.getpath(oid)
    p = vpath.with_suffix('.tmp')
    with p.open('wb') as v:
      try: pickle.dump(val,v)
      except Exception as e: v.seek(0); v.truncate(); pickle.dump(e,v)
    p.rename(vpath)
    return vpath.stat().st_size

#--------------------------------------------------------------------------------------------------
  def getval(self,oid:int,wait:bool)->Any:
    r"""
If *wait*, waits for the content file for *oid* to appear. Then loads (pickle) from that file and return the value.
    """
#--------------------------------------------------------------------------------------------------
    vpath = self.getpath(oid)
    if wait: self.monitor.wait(vpath)
    with vpath.open('rb') as u: return pickle.load(u)

#--------------------------------------------------------------------------------------------------
  def remove(self,L:Iterable[int]):
    r"""
Removes the content file path for oids in L.
    """
#--------------------------------------------------------------------------------------------------
    def rm_vpath(oid):
      vpath = self.getpath(oid)
      try: vpath.unlink()
      except FileNotFoundError: logger.warning('%s unable to remove content file path: %s for oid: %s',self,vpath,oid)
      return vpath
    self.monitor.remove([rm_vpath(oid) for oid in L])

#--------------------------------------------------------------------------------------------------
  def clear(self):
    r"""
Clears all storage.
    """
#--------------------------------------------------------------------------------------------------
    from shutil import rmtree
    for f in self.path.iterdir():
      if f.is_dir(): rmtree(f)
      else: f.unlink()

#--------------------------------------------------------------------------------------------------
  def getpath(self,oid:int,masks=tuple((n,31<<n) for n in range(20,-5,-5))):
    r"""
Returns the content file path (as a :class:`pathlib.Path` instance) associated to *oid* (of type :class:`int`). It is composed of two parts (a directory name and a file name), joined to the main :attr:`path` attribute. The directory is created if it does not already exist. The concatenation of the directory name (without its prefix ``X``) and the file name (without its suffix ``.pck``) is the representation of *oid* in base 32 (digits are 0-9A-V). This mapping of oids to paths ensures that no sub-directory holds more than 1024 oids. It assumes that oids are created sequentially (which is what AUTOINCREMENT in databases does), so the number of sub-directories grows slowly.
    """
#--------------------------------------------------------------------------------------------------
    s = ''.join('0123456789ABCDEFGHIJKLMNOPQRSTUV'[(oid&m)>>n] for n,m in masks)
    p = self.path/('X'+s[:3])
    if not p.exists(): p.mkdir(exist_ok=True)
    return (p/s[3:]).with_suffix('.pck')

  def __repr__(self): return f'{self.__class__.__name__}<{self.path}>'

#==================================================================================================
class DefaultStorage (FileStorage):
  r"""
The default storage class for a cache repository. Stores the index as a sqlite3 database in the same directory as the values, with name ``index.db``.
  """
#==================================================================================================
  def __init__(self,path:Path):
    super().__init__(path)
    dbpath = path/'index.db'
    if not dbpath.exists() and any(path.iterdir()): raise Exception('Cannot create new index in non empty directory')
    self.db_url = f'sqlite+pysqlite:///{dbpath}'

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
    return f'{module}.{name}{'' if version is None else f'{{{version}}}'}'
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
  from ipywidgets import Button, Dropdown, Checkbox, Output, VBox, HBox
  from IPython.display import clear_output, display
  from IPython.core.getipython import get_ipython
  def setdb(c): nonlocal db; db = c.new; interpreter.push({ivname:db}); showdb()
  def showdb():
    w_msg.clear_output()
    with w_out: clear_output(); display(db)
  def clear(op):
    if op is None: return
    w_clear.value = None
    try:
      with w_msg:
        clear_output()
        print('Operation:','clear'+op.label,f'(dryrun: {ivname} unchanged)' if w_dryrun.value else '')
        r = op()
        print('Result:',r)
    except: return
    if not w_dryrun.value and r:
      with w_out: clear_output(); display(db)
  clearops = {
    'Error':(lambda: [x for x in ((c.block,c.clear_error(dry_run=w_dryrun.value)) for c in list(db.values())) if x[1]]),
    'Obsolete':(lambda: db.clear_obsolete(False,dry_run=w_dryrun.value)),
    'ObsoleteStrict':(lambda: db.clear_obsolete(True,dry_run=w_dryrun.value)),
  }
  for k,op in clearops.items(): op.label = k
  db:CacheDB|None = None
  interpreter = get_ipython()
  interpreter.push({ivname:db})
  w_db = Dropdown(description=ivname,options={'...':()}|{p:CacheDB(p) for p in paths},style={'description_width':'initial'})
  w_show = Button(tooltip='show db',icon='fa-refresh',layout={'width':'.4cm','padding':'0cm'})
  w_clear = Dropdown(options={'clear...':None}|clearops,layout={'width':'3cm'})
  w_dryrun = Checkbox(description='dry-run',value=True,style={'description_width':'initial'})
  w_out = Output()
  w_msg = Output(layout={'border':'thin solid black'})
  w_db.observe(setdb,'value')
  w_clear.observe((lambda c: clear(c.new)),'value')
  w_show.on_click(lambda b: showdb())
  showdb()
  return VBox([HBox([w_db,w_show,w_clear,w_dryrun]),w_msg,w_out])
