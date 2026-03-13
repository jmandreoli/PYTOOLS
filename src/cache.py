# File:                 cache.py
# Creation date:        2015-03-19
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Persistent cache management
#

import logging; logger = logging.getLogger(__name__)
from typing import Any, Callable, Iterable, Mapping, Sequence

import pickle, inspect, threading
from pathlib import Path
from functools import partial, update_wrapper
from collections import namedtuple
from weakref import WeakValueDictionary
from datetime import datetime, timezone
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from .cache_v1 import get_sessionmaker, Block, Cell, AbstractFunctor, AbstractStorage
from . import qty_format, time_format, pickleclass

__all__ = 'CacheDB', 'CacheBlock', 'Functor', 'FileStorage', 'DefaultStorage', 'Shadow'

NOW = lambda tz=timezone.utc: datetime.now(tz=tz) # shorthand for timestamping
BlockInfo = namedtuple('BlockInfo','hits ncell ncell_error ncell_pending')

#==================================================================================================
class CacheDB:
  r"""
Instances of this class manage cache repositories. A cache repository contains cells, each cell corresponding to one cached value produced by a unique call, and possibly reused by later calls. Cells are clustered into blocks, each block grouping cells produced by the same call type, called a functor (of class :class:`AbstractFunctor`). Meta-information about blocks and cells are stored in an index database accessed through :mod:`sqlalchemy`. The values themselves are persistently stored by a dedicated storage object (of class :class:`AbstractStorage`). Instances of :class:`CacheDB` have an HTML ipython display.
  """
#==================================================================================================

  path:Path
  url:str
  r"""the sqlalchemy url of the database"""
  session_maker:Callable[[],Any]
  r"""the sqlalchemy session maker for manipulation of this database"""

#--------------------------------------------------------------------------------------------------
  def __new__(cls,spec:CacheDB|Path|str,listing={},lock=threading.Lock())->CacheDB:
    r"""
:param spec: cache specification

Generates a :class:`CacheDB` instance so that there is at most one instance of this class in a process for each normalised repository specification path:

* If *spec* is already a :class:`CacheDB` instance, that instance is returned
* If *spec* is a path to a directory, returns a :class:`CacheDB` instance whose storage is an instance of :class:`DefaultStorage` pointing to that directory
* Otherwise, *spec* must be a path to a file, returns a :class:`CacheDB` instance whose storage is unpickled from the file at *spec*.
    """
#--------------------------------------------------------------------------------------------------
    if isinstance(spec,CacheDB): return spec
    if isinstance(spec,str): path = Path(spec)
    elif isinstance(spec,Path): path = spec
    else: raise TypeError(f'Expected: {CacheDB}|{str}|{Path}; Found: {type(spec)}')
    path = path.resolve()
    with lock:
      if (self:=listing.get(path)) is None:
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
  # no need to define '__eq__': default 'is' behaviour works because '__new__' is cacheing

#--------------------------------------------------------------------------------------------------
  def clear_obsolete(self,tol,dry_run:bool=False):
    r"""
Clears all the blocks which are obsolete with tolerance *tol* of non-obsolescence. If *dry_run*, only return the list of obsolete blocks.
    """
#--------------------------------------------------------------------------------------------------
    assert isinstance(dry_run,bool)
    return self.clear((lambda L: [block for block in L if pickle.loads(block.functor).obsolete(tol) is True]),dry_run)
    # this may load modules which add new (non obsolete) entries which must not be included in the list

#--------------------------------------------------------------------------------------------------
  def clear(self,f:Callable[[Sequence[Block]],Sequence[Block]]=(lambda L: L),dry_run:bool=False):
    r"""Deletes blocks from this instance."""
#--------------------------------------------------------------------------------------------------
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

  _html_limit = 50
  def _repr_html_(self,tail=None):
    from .html import repr_html, html_table
    if tail is None: return repr_html(self)
    n_max = self._html_limit
    with self.session_maker() as session:
      L = [CacheBlock(self,pickle.loads(block.functor),block.oid) for block, in session.execute(select(Block).order_by(Block.oid).limit(n_max))]
      n = session.execute(select(func.count(Block.oid))).scalar()-n_max
    return html_table(
      ((str(c.oid),c._repr_html_(tail=tail,session=session)) for c in L),
      hdrs = ('block',),
      opening=(repr(self),),
      closing=((f'{n} more',) if n>0 else ()),
    )
  def __repr__(self): return f'{self.__class__.__name__}<{self.path})>'

#==================================================================================================
class CacheBlock:
  r"""
Instances of this class implements blocks of cells sharing the same functor. They have an HTML ipython display.
  """
#==================================================================================================

  db:CacheDB
  r"""the :class:`CacheDB` instance this block belongs to"""
  functor:AbstractFunctor
  r"""the functor for this block (field ``functor`` in the ``Block`` table of the index is the functor's pickle)"""
  oid:int
  r"""the identifier of this block (field ``oid`` in the ``Block`` table of the index)"""
  cacheonly:bool
  r"""whether cell creation is disabled by this instance"""
  memory:WeakValueDictionary
  r"""a local cache of calls within the current process"""

#--------------------------------------------------------------------------------------------------
  def __init__(self,db:CacheDB|Path|str='',functor:AbstractFunctor|None=None,oid:int=None,cacheonly:bool=False):
    r"""
:param db: specification of the cache repository where the block resides
:param functor: functor of the block
:param cacheonly: if :const:`True`, cell creation is disallowed

A :class:`CacheBlock` instance is callable, and calls take a single argument. Method :meth:`__call__` implements the cross-process cacheing mechanism which produces and reuses cache cells. It also implements a weak cache for local calls (within its process).
    """
#--------------------------------------------------------------------------------------------------
    self.db = db = CacheDB(db)
    self.functor = functor
    if oid is None:
      p = pickle.dumps(functor)
      created = False
      with db.session_maker(expire_on_commit=False) as session:
        oid = session.execute(select(Block.oid).where(Block.functor==p)).scalar()
        if oid is None:
          session.add(block:=Block(functor=p)); session.commit()
          oid = block.oid; created = True
      logger.info('%s F-%s(%s)',self,('MISS' if created is True else 'HIT'),oid)
    self.oid = oid
    self.cacheonly = cacheonly
    self.memory = WeakValueDictionary()
  def getblock(self,session)->Block: return session.get(Block,self.oid)

  def __hash__(self): return hash((self.db,self.oid))
  def __eq__(self,other): return isinstance(other,CacheBlock) and self.db == other.db and self.oid == other.oid

#--------------------------------------------------------------------------------------------------
  def clear_error(self,dry_run:bool=False):
    r"""
Clears all the cells from this block which cache an exception.
    """
#--------------------------------------------------------------------------------------------------
    return self.clear((lambda L: [cell for cell in L if cell.size<0]),dry_run)

#--------------------------------------------------------------------------------------------------
  def clear_overflow(self,n:int,dry_run:bool=False):
    r"""
Clears all the cells from this block except the *n* most recent (lru policy).
    """
#--------------------------------------------------------------------------------------------------
    assert isinstance(n,int) and n>=1
    return self.clear((lambda L: sorted((cell for cell in L if cell.size>0),key=(lambda cell: cell.tstamp),reverse=True)[n:]),dry_run)

#--------------------------------------------------------------------------------------------------
  def clear(self,f:Callable[[Sequence[Cell]],Sequence[Cell]]=(lambda L:L),dry_run:bool=False):
    r"""Deletes cells from this instance."""
#--------------------------------------------------------------------------------------------------
    with self.db.session_maker() as session:
      L = f(self.getblock(session).cells)
      if dry_run: return L
      L_ = [cell.oid for cell in L]
      if (n:=len(L))>0:
        for cell in L: session.delete(cell)
        session.commit();logger.info('%s DELETED %s',self,n)
    self.db.storage.remove(L_)
    return n

#--------------------------------------------------------------------------------------------------
  def info(self):
    r"""
Returns information about this block. Available attributes:
:attr:`hits`, :attr:`ncell`, :attr:`ncell_error`, :attr:`ncell_pending`
    """
#--------------------------------------------------------------------------------------------------
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

- Method :meth:`getkey` of the functor is invoked with argument *arg* to obtain a *ckey*.
- If that *ckey* is present in the (local) memory mapping of this block, its associated value is returned.
- Otherwise, if there already exists a cell with the same *ckey* in the database, its value is retrieved using method :meth:`getval` of the storage with the cell oid. The cell's hit count is incremented.
- Otherwise, a cell with that *ckey* is created, its value is computed and stored using method :meth:`setval` of the storage with the cell oid.

The result can be an exception (stored on miss and reused on hits), in which case it is raised.
    """
#--------------------------------------------------------------------------------------------------
    ckey = self.functor.getkey(arg)
    if (cval:=self.memory.get(ckey,self)) is not self: return cval # self is unlikely to be a stored value
    lookup = select(Cell).where((Cell.block_oid==self.oid)&(Cell.ckey==ckey))
    created:bool = False
    with self.db.session_maker(expire_on_commit=False) as session:
      cell:Cell = session.execute(lookup).scalar()
      if cell is None:
        if self.cacheonly: raise Exception('Cache cell creation disallowed')
        cell = Cell(block_oid=self.oid,ckey=ckey,tstamp=NOW()); session.add(cell)
        try: session.commit(); created = True
        except IntegrityError: # just in case another thread/process created the cell during the session
          session.rollback()
          cell = session.execute(lookup).scalar_one()
      oid,wait = cell.oid,cell.size==0
    if created is True:
      logger.info('%s MISS(%s)',self,oid)
      start = NOW(); size = 1
      try: cval = self.functor.getval(arg)
      except BaseException as e: cval = e; size = -1
      duration = (NOW()-start).total_seconds()
      size *= self.db.storage.setval(oid,cval)
      with self.db.session_maker() as session:
        cell = session.get(Cell,oid)
        if cell is None: raise Exception(f'Lost cell {oid}')
        cell.size,cell.duration,cell.tstamp = size,duration,NOW()
        session.commit()
    else:
      if wait is True: logger.info('%s WAIT(%s)',self,oid)
      cval = self.db.storage.getval(oid,wait)
      logger.info('%s HIT(%s)',self,oid)
      with self.db.session_maker() as session:
        cell = session.get(Cell,oid)
        if cell is None: raise Exception(f'Lost cell {oid}')
        cell.hits += 1; cell.tstamp = NOW(); session.commit()
    try: self.memory[ckey] = cval
    except: pass # in case cval does not support weak references
    if isinstance(cval,BaseException): raise cval
    return cval

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  _html_limit = 50
  def _repr_html_(self,tail=None,session=None,size_fmt=(lambda sz: '*'+qty_format(-sz) if sz<0 else qty_format(sz) if sz>0 else ''),time_fmt=(lambda t: '' if t is None else time_format(t))):
    from .html import repr_html, html_table
    if tail is None: return repr_html(self)
    if session is None: session = self.db.session_maker()
    with session:
      n_max = self._html_limit
      L = [cell for cell, in session.execute(select(Cell).where(Cell.block_oid==self.oid).order_by(Cell.tstamp.desc()).limit(n_max))]
      n = session.execute(select(func.count(Cell.oid)).where(Cell.block_oid==self.oid)).scalar()-n_max
      return html_table(
        ((str(cell.oid),self.functor.html(cell.ckey,tail=tail),str(cell.tstamp),str(cell.hits),size_fmt(cell.size),time_fmt(cell.duration)) for cell in L),
        hdrs=('ckey','tstamp','hits','size','duration'),
        opening=(repr(self),),
        closing=((f'{n} more',) if n>0 else ()),
      )

  def __repr__(self): return f'{self.__class__.__name__}<{self.functor!r}>'

#==================================================================================================
class Functor (AbstractFunctor):
  r"""
An instance of this class defines a functor attached to a python top-level versioned function. The functor is entirely defined by the name of the function, that of its module, its version and its signature (without annotations). These components are saved on pickling and restored on unpickling, even if the function has disappeared or changed. This is not checked on unpickling, but method :meth:`getval` is disabled.
  """
#===================================================================================================

  __slots__ = 'func', 'sig', 'shadow'

  func:Callable
  r"""the versioned function characterizing this functor"""
  sig:inspect.Signature
  r"""signature of this functor"""

#--------------------------------------------------------------------------------------------------
  def __new__(cls,func:Callable,sig:inspect.Signature|None=None)->Functor:
    r"""
:param func: a versioned function, defined at the top-level of its module (hence pickable)
:param sig: only used on unpickling

Generates a functor associated with the function *func*.
    """
#--------------------------------------------------------------------------------------------------
    if sig is None:
      shadow = Shadow(func)
      sig = inspect.Signature([inspect.Parameter(p.name,p.kind,default=p.default) for p in inspect.signature(func).parameters.values()]) # just removes annotations
    else: assert isinstance(func,Shadow); shadow = func
    self = super().__new__(cls)
    self.func,self.sig,self.shadow = func,sig,shadow
    return self

  def __getnewargs__(self): return self.shadow,self.sig
  def __getstate__(self): return
  def __hash__(self): return hash(self.shadow)
  def __eq__(self,other): return isinstance(other,Functor) and self.shadow==other.shadow
  def __repr__(self): return f'{self.shadow!r}{self.sig}'

  class _fpickle (pickleclass):
    class Pickler (pickle.Pickler):
      def persistent_id(self,obj):
        if inspect.isfunction(obj) and hasattr(obj,'version'): return Shadow(obj)
    class Unpickler (pickle.Unpickler):
      def persistent_load(self,pid): return pid

#--------------------------------------------------------------------------------------------------
  def getkey(self,arg:tuple[Iterable[Any],Mapping[str,Any]]):
    r"""
Argument *arg* must be a pair of a list of positional arguments and a dict of keyword arguments. They are normalised against the signature of the functor and the pickled value of the result is returned. The pickling of versioned function objects is modified to embed their version.
    """
#--------------------------------------------------------------------------------------------------
    a,ka = self.normalise(arg)
    return self._fpickle.dumps((a,sorted(ka.items())))

#--------------------------------------------------------------------------------------------------
  def getval(self,arg:tuple[Iterable[Any],Mapping[str,Any]]):
    r"""
Argument *arg* must be a pair of a list of positional arguments and a dict of keyword arguments. Returns the value of calling attribute :attr:`func` with that positional argument list and keyword argument dict.
    """
#--------------------------------------------------------------------------------------------------
    a,ka = arg
    return self.func(*a,**ka)

#--------------------------------------------------------------------------------------------------
  def normalise(self,arg:tuple[Iterable[Any],Mapping[str,Any]]):
#--------------------------------------------------------------------------------------------------
    a,ka = arg
    b = self.sig.bind(*a,**ka)
    b.apply_defaults()
    return b.args, b.kwargs

#--------------------------------------------------------------------------------------------------
  def obsolete(self,tol)->bool:
    r"""
Returns whether this functor is obsolete. This concerns only unpickled functors, and obsolescence is determined by the unpickled attribute :attr:`func`.
    """
#--------------------------------------------------------------------------------------------------
    return self.shadow.obsolete(tol) if self.func is self.shadow else False

#--------------------------------------------------------------------------------------------------
  def html(self,ckey:bytes,tail):
#--------------------------------------------------------------------------------------------------
    from .html import html_parlist
    a,ka = self._fpickle.loads(ckey)
    return html_parlist((),((k,tail(v)) for k,v in self.sig.bind(*a,**dict(ka)).arguments.items()))

#==================================================================================================
class FileStorage (AbstractStorage):
  r"""
Instances of this class manage the persistent storage of cached values using a filesystem directory.
  """
#==================================================================================================

  path:Path
  r"""directory where values are stored"""
#--------------------------------------------------------------------------------------------------
  def __init__(self,path:Path):
    r"""
:param path: the directory to store cached values

The storage for a cell consists of a content file which contains the value of the cell in pickled format and a cross process mechanism to synchronise access to the content file between a writer and possibly multiple readers.
    """
#--------------------------------------------------------------------------------------------------
    self.path = path
    self.monitor = self.monitor_factory(path)

#--------------------------------------------------------------------------------------------------
  @staticmethod
  def monitor_factory(path:Path):
    r"""
:param path: a directory used by an instance of :class:`FileStorage` in this or other thread or process

Returns a monitor of the activity within directory *path*. The target events are any creation (obtained by mv) of a file with extension ``.pck``. A monitor must have method :meth:`ask`, to wait until a target event occurs for a given file, and method :meth:`tell`, to stop monitoring events for a given list of files. The monitor must be thread safe. This implementation uses module :mod:`watchdog`.
    """
#--------------------------------------------------------------------------------------------------
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    class GatingEventHandler(FileSystemEventHandler):
      def __init__(self):
        gate = {}
        lock = threading.Lock()
        def ask(p):
          with lock:
            if p.exists(): return
            if (ev:=gate.get(p)) is None: ev = gate[p] = threading.Event()
          ev.wait()
        self.ask = ask
        def tell(*L):
          with lock:
            for ev in (ev for p in L if (ev:=gate.pop(p,None)) is not None): ev.set()
        self.tell = tell
        super().__init__()
      def on_moved(self,event):
        p = Path(event.dest_path)
        if p.suffix == '.pck': self.tell(p)
    monitor = GatingEventHandler()
    observer = Observer()
    observer.schedule(monitor,path,recursive=True)
    observer.start()
    return monitor

#--------------------------------------------------------------------------------------------------
  def setval(self,oid:int,val:Any)->int:
    r"""
Dumps (pickle) *val* into some temporary file and renames it to the content path for *oid*. Renaming will trigger a monitored event in :class:`Filestorage` instances in this and other threads/processes (if any).
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
If *wait* is :const:`True`, asks the monitor to wait until the content file for *oid* is available. Then loads (unpickle) from that file and returns the unpickled value.
    """
#--------------------------------------------------------------------------------------------------
    vpath = self.getpath(oid)
    if wait: self.monitor.ask(vpath)
    with vpath.open('rb') as u: return pickle.load(u)

#--------------------------------------------------------------------------------------------------
  def remove(self,L:Iterable[int]):
    r"""
Removes the content file path for oids in *L* and tell the monitor to stop monitoring them.
    """
#--------------------------------------------------------------------------------------------------
    def rm_vpath(oid):
      vpath = self.getpath(oid)
      try: vpath.unlink()
      except FileNotFoundError: logger.warning('%s unable to remove content file for oid: %s at path: %s',self,oid,vpath)
      return vpath
    self.monitor.tell(*(rm_vpath(oid) for oid in L))

#--------------------------------------------------------------------------------------------------
  def getpath(self,oid:int,masks=tuple((n,31<<n) for n in range(20,-5,-5))):
    r"""
Returns the content file path (as an absolute :class:`pathlib.Path` instance) associated to *oid* (of type :class:`int`). It is composed of two parts (a directory name and a file name), joined to the main :attr:`path` attribute. The directory is created if it does not already exist. This implementation ensures that no directory holds more than 1024 oids. If oids are created sequentially (which is what AUTOINCREMENT in databases does), the number of directories grows slowly.
    """
#--------------------------------------------------------------------------------------------------
    s = ''.join('0123456789ABCDEFGHIJKLMNOPQRSTUV'[(oid&m)>>n] for n,m in masks)
    p = self.path/('X'+s[:3])
    if not p.exists(): p.mkdir(exist_ok=True)
    return (p/s[3:]).with_suffix('.pck')

  def __repr__(self): return f'{self.__class__.__name__}<{self.path}>'

#==================================================================================================
class DefaultStorage (FileStorage):
#==================================================================================================
#--------------------------------------------------------------------------------------------------
  def __init__(self,path:Path):
    r"""
:param path: path to an existing directory, which must be either empty or contain a sqlite file `index.db` created by class :class:`CacheDB` on its first connection

The default storage class for a cache repository. Stores the index as a sqlite3 database in the same directory as the values, with name ``index.db``.
    """
#--------------------------------------------------------------------------------------------------
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
Instances of this class are defined from versioned functions (ie. functions defined at the toplevel of their module, and with an attribute :attr:`version` defined). Their state is composed of the name, the module name and the version of the original function. They can be arbitrarily pickled and unpickled and produce a string representation close to that of their origin. However, unpickling may fail to restore the calling capacity of the original function, if it cannot be imported with the same version.
  """
#--------------------------------------------------------------------------------------------------

  __slots__ = 'config', 'func', '_origin'
  class VersionMismatchException (Exception): pass

  def __new__(cls,spec,fromfunc=True):
    self = super().__new__(cls)
    if fromfunc:
      assert inspect.isfunction(spec)
      self.func = self._origin = func = spec
      self.config = func.__module__,func.__name__,func.version
    else:
      self.config = spec
      self._origin = None
    return self
  def __getnewargs__(self): return self.config,False
  def __getstate__(self): return
  def __hash__(self): return hash(self.config)
  def __eq__(self,other): return isinstance(other,Shadow) and self.config==other.config
  def __repr__(self):
    module,name,version = self.config
    return f'{module}.{name}{'' if version is None else f'{{{version}}}'}'
  @property
  def origin(self)->Callable|Exception:
    r"""
  Returns the function which was originally shadowed with the same version, if possible, otherwise an exception (returned, not raised).
    """
    o: Callable|Exception|None = self._origin
    if o is None:
      from importlib import import_module
      module,name,version = self.config
      try:
        o = getattr(import_module(module),name)
        if inspect.isfunction(o) and o.__module__==module and o.__name__==name and (v:=getattr(o,'version',o)) is not o:
          if v!=version: raise Shadow.VersionMismatchException(v,version)
        else: raise Exception(f'Failed to import matching function.')
      except Exception as exc: o = exc
      self._origin = o
    return o

  def obsolete(self,tol:int)->bool:
    r"""
:param tol: indicates the degree of tolerance (this implementation accepts only :const:`0` or :const:`1`)

Returns :const:`True` if attribute :attr:`origin` is a non tolerated exception and :const:`False` otherwise. When *tol* is :const:`0` no exception is tolerated, otherwise version mismatch exceptions are tolerated.
    """
    assert isinstance(tol,int) and tol in (0,1)
    match self.origin:
      case Shadow.VersionMismatchException(): return tol==0
      case Exception(): return True
      case _: return False

  def __call__(self,*a,**ka):
    if isinstance((o:=self.origin),Exception): raise o
    return o(*a,**ka)

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
