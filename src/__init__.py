# File:                 __init__.py
# Creation date:        2014-03-16
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities in Python
#

from __future__ import annotations
import logging; logger = logging.getLogger(__name__)
from typing import Any, Callable, Iterable, Sequence, Mapping, Iterator

import re, collections

#==================================================================================================
class owrap:
  r"""
Objects of this class present an attribute oriented proxy interface to an underlying mapping object. Example::

   r = dict(a=3,b=6); x = owrap(r)
   assert x.a == 3
   del x.a; x.b += 7
   assert r == {'b':13}
   r['c'] = 42
   assert x.c == 42
   assert x == r
  """
#==================================================================================================
  __slot__ = '__ref__',
  __ref__:Mapping[str,Any]
  r"""The proxy object"""
  def __init__(self,__ref__:Mapping[str,Any]|None=None,**ka):
    r"""
:param __ref__: a reference to a (possibly mutable) mapping object, of which this instance is to be a proxy

Keys in the reference are turned into attributes of the proxy. If *__ref__* is :const:`None`, the reference is a new empty dictionary, otherwise *__ref__* must be a mapping object, and *__ref__* (or its own reference if it is itself of class :class:`owrap`) is taken as reference. The reference is first updated with the keyword arguments *ka*.
    """
    if __ref__ is None: r = dict(ka)
    else:
      r:Mapping[str,Any] = __ref__
      if isinstance(r,owrap): r = r.__ref__
      else: assert isinstance(r,collections.abc.Mapping)
      if ka: r.update(ka)
    super().__setattr__('__ref__',r)
  def __eq__(self,other):
    return self.__ref__ == (other.__ref__ if isinstance(other,owrap) else other)
  def __ne__(self,other):
    return self.__ref__ != (other.__ref__ if isinstance(other,owrap) else other)
  def __hash__(self): return hash(self.__ref__)
  def __getattr__(self,a):
    try: return self.__ref__[a]
    except KeyError: raise AttributeError(a) from None
  def __setattr__(self,a,v):
    self.__ref__[a] = v
  def __delattr__(self,a):
    try: del self.__ref__[a]
    except KeyError: raise AttributeError(a) from None
  def __str__(self): return str(self.__ref__)
  def __repr__(self): return repr(self.__ref__)

#==================================================================================================
def import_module_from_file(modname:str,filename:str,init_globals:dict[str,Any]|None=None):
  r"""
:param modname: the name of the module to import
:param filename: the name of the file to import from
:param init_globals: the global variables to initialize the module with (if not :const:`None`)

Similar to the standard library function :func:`runpy.run_path`, but declares the created module explicitly.
  """
  from importlib.util import spec_from_file_location,module_from_spec
  spec = spec_from_file_location(modname,filename)
  mod = module_from_spec(spec)
  import sys; sys.modules[modname] = mod
  if init_globals is not None: mod.__dict__.update(init_globals)
  spec.loader.exec_module(mod)
  return mod

#==================================================================================================
def namedtuple(*a,**ka):
  r"""
Returns a class of tuples with named fields. It is identical to :func:`collections.namedtuple`, with only two additions:

* The returned class has an attribute :attr:`_field_index` which is an instance of that class where each field is set to its index in the field list. Example::

   c = namedtuple('c','u v')
   assert isinstance(c._field_index,c) and c._field_index.u == 0 and c._field_index.v == 1

* Furthermore, if *f* and *x* are instances of the returned class *c*, then calling *f* with argument *x* returns a new instance of class *c* in which each field is set to the result of applying the corresponding field in *f* (which must be a callable) to the corresponding field in *x*. Additional keyword arguments can be passed to the calls. Example::

   c = namedtuple('c','u v')
   f = c(u=(lambda z,r=0.: 2.*z+r),v=(lambda z,r=0.: 3.*z+5.*r))
   x = c(u=3,v=5)
   k = {'r':42}
   assert f(x,**k) == c(u=f.u(x.u,**k),v=f.v(x.v,**k))
  """
#==================================================================================================
  cls = collections.namedtuple(*a,**ka)
  cls.__call__ = lambda self,X,**k: cls(*(f(x,**k) for f,x in zip(self,X)))
  cls._field_index = cls(*range(len(cls._fields)))
  return cls

#==================================================================================================
def config_xdg(rsc:str,dflt:Any=None):
  r"""
:param rsc: the name of an XDG resource
:param dflt: a default value

Returns the content of the XDG resource named *rsc* or *dflt* if the resource is not found.
  """
#==================================================================================================
  try: from xdg.BaseDirectory import load_first_config
  except: return dflt
  p = load_first_config(rsc)
  if p is None: return dflt
  with open(p) as u: return u.read()

#==================================================================================================
def config_env(name:str,dflt:str|None=None,asfile:bool=False)->str|None:
  r"""
:param name: the name of an environment variable
:param dflt: a default value
:param asfile: whether to treat the value of the environment variable as a path to a file

Returns the string value of an environment variable (or the content of the file pointed by it if *asfile* is set) or *dflt* if the environment variable is not assigned.
  """
#==================================================================================================
  from os import environ
  x = environ.get(name)
  if x is None: return dflt
  if asfile:
    with open(x) as u: return u.read()
  else: return x

#==================================================================================================
def autoconfig(module:str,name:str,dflt:Any=None,asfile:bool=False):
  r"""
:param module: the name of a module
:param name: the name of a configuration parameter for that module
:param dflt: a default value for the configuration parameter
:param asfile: whether to treat the value of the environment variable as a path to a file

Returns an object obtained from an environment variable whose name is derived from *module* and *name*. For example, if *module* is ``mypackage.mymodule`` and *name* is ``myparam`` then the environment variable is ``MYPACKAGE_MYMODULE_MYPARAM``. The value of that variable (or the content of the file pointed by it if *asfile* is set) is executed in an empty dictionary and the value attached to key *name* is returned. If the variable is not assigned, *dflt* is returned.
  """
#==================================================================================================
  x = config_env(f'{module.replace('.','_')}_{name}'.upper(),asfile=asfile)
  if x: d:dict[str,Any] = {}; exec(x,d); return d[name]
  else: return dflt

#==================================================================================================
def secret_tool(attr_list:Iterable[dict[str,str]])->Iterator[Iterable[tuple[dict[str,str],bytes]]]:
  r"""
:param attr_list: a list of dictionaries
:returns: an iterable of streams of attribute-secret pairs

Retrieves a (possibly empty) stream of attribute-secret pair(s) for each of a list of dictionaries. Each input dictionary is looked up in the default keyring, and the corresponding stream consists of the attribute-secret pair of matching items. The attribute in each pair is the dictionary associated with the keyring item, and, by construction, contains the input dictionary. Example::

   L = {'a':'x'},{'b':'y','c':'z'}
   for d,stream in zip(L,secret_tool(L)):
     for D,secret in stream: assert all(D[k]==v for k,v in d.items()) and isinstance(secret,bytes)
  """
#==================================================================================================
  from secretstorage import dbus_init, get_default_collection
  conn = dbus_init()
  try:
    coll = get_default_collection(conn)
    for attr in attr_list: yield ((item.get_attributes(),item.get_secret()) for item in coll.search_items(attr))
  finally: conn.close()

#==================================================================================================
class SymbolicOperator:
#==================================================================================================
  __slots__ = 'name','op','covariant','mark'
  @staticmethod
  def lookup(n:str,pat=re.compile(r'Same as a (\S+) b')):
    import operator
    op = getattr(operator,f'__{n}__')
    mark = pat.match(op.__doc__).group(1)
    yield SymbolicOperator(n,op,slice(None),mark)
    if hasattr(operator,f'__i{n}__'): yield SymbolicOperator(f'r{n}',op,slice(None,None,-1),mark)
  def __new__(cls,name,op=None,covariant=None,mark=None,listing={}):
    if (self:=listing.get(name)) is None:
      self = super().__new__(cls)
      self.name,self.op,self.covariant,self.mark = name,op,covariant,mark
      listing[name] = self
    return self
  def __call__(self,x,y): return self.op(*(x.__ref__.value,(y.__ref__.value if isinstance(y,Symbolic) else y))[self.covariant])
  def __hash__(self): return hash(self.name)
  # no need to define '__eq__': default 'is' behaviour works because '__new__' is cacheing
  def __getnewargs__(self): return self.name,
  def __getstate__(self): pass
  def __str__(self): return self.mark

#==================================================================================================
class Symbolic:
  r"""
Instances of this class implement closed symbolic expressions. Upon invocation of specific triggers, the value of the expression is computed and stored, and the instance subsequently behaves as a proxy to that value. The proxy is not saved on pickling, hence lost on unpickling, but can of course be restored using any of the triggers. Thus, when unpickling a foreign :class:`Symbolic` instance, a process can decide whether it wants to access its value or only inspect its expression. If recomputing the value is costly, one can use a persistent cache (see module :mod:`.cache`).
  """
#==================================================================================================

  __slots__ = '__ref__',

  def __init__(self,func,*a,**ka):
    r"""
:param func: the function of the symbolic expression
:param a: list of positional arguments of the symbolic expression
:param ka: dictionary of keyword arguments of the symbolic expression

The triple *func*, *a*, *ka* forms the configuration of the :class:`Symbolic` instance. Its value is defined as the result of calling function *func* with positional arguments *a* and keyword arguments *ka*. See code for a list of triggers. Normally, direct access to the value or configuration should not be needed, since the instance is a proxy for it. If required, it is done through attribute :attr:`__ref__` as follows:

* ``x.__ref__.config`` returns the configuration of symbolic expression *x*, in which the third component is converted to a tuple of key-value pairs to be hashable.
* ``x.__ref__.value`` returns the value of symbolic expression *x*, computing it if it has not been computed yet (so this is a trigger).
* ``x.__ref__.check()`` indicates whether the value of symbolic expression *x* has already been computed by a trigger (this is not a trigger).

Caveat: the configuration should be deterministically picklable and hashable (in particular: no dicts nor lists are allowed in the components). Hence, any :class:`Symbolic` instance is itself deterministically picklable and hashable, and can thus be used as argument in the configuration of another :class:`Symbolic` instance.
    """
    assert callable(func)
    from inspect import signature
    signature(func).bind(*a,**ka) # just ensures the signature matches
    super().__setattr__('__ref__',_Symbolic((func,a,tuple(sorted(ka.items())))))

  # non triggers
  def __getstate__(self): return self.__ref__.config
  def __setstate__(self,config): super().__setattr__('__ref__',_Symbolic(config))
  def __hash__(self): return hash(self.__ref__.config)
  def __eq__(self,other): return isinstance(other,Symbolic) and self.__ref__.config == other.__ref__.config

  # triggers:
  def __getattr__(self,k): return getattr(self.__ref__.value,k)
  def __setattr__(self,k,v): return setattr(self.__ref__.value,k,v)
  def __delattr__(self,k): return delattr(self.__ref__.value,k)
  def __getitem__(self,k): return self.__ref__.value[k]
  def __setitem__(self,k,v): self.__ref__.value[k] = v
  def __delitem__(self,k): del self.__ref__.value[k]
  def __contains__(self,k): return k in self.__ref__.value
  def __len__(self): return len(self.__ref__.value)
  def __iter__(self): return iter(self.__ref__.value)
  def __call__(self,*a,**ka): return self.__ref__.value(*a,**ka)
  def __bool__(self): return bool(self.__ref__.value)
  def __int__(self): return int(self.__ref__.value)
  def __float__(self): return float(self.__ref__.value)
  def __complex__(self): return complex(self.__ref__.value)

  # supported operators (non triggers)
  exec('\n'.join(
    f'def __{sop.name}__(self,other,sop=SymbolicOperator({sop.name!r})): return Symbolic(sop,self,other)'
    for n in 'add sub mul matmul truediv floordiv mod pow lshift rshift and xor or ne lt le gt ge'.split() # NOT eq
    for sop in SymbolicOperator.lookup(n)
  ))

  # representation methods (non triggers)
  def __repr__(self): return repr(self.__ref__)
  def _repr_html_(self,tail=None): return self.__ref__._repr_html_(tail)

#==================================================================================================
def symbolic(f):
  r"""
A decorator which adds to its target *f* an attribute :attr:`symbolic` pointing to a symbolic variant of the target.
Example::

   def heavy(*a): from time import sleep; sleep(1); return a # simulates some long computation
   @symbolic
   def f(E,x): return E|{'a':heavy(x)}
   @symbolic
   def g(E,x): return E|{'b':heavy(E['a'],x)}

Note that standard operators, like ``|`` above, are directly interpreted as symbolic calls when at least one of their arguments is a :class:`Symbolic` instance. One exception is operator ``==`` which returns :const:`True` if both arguments are :class:`Symbolic` instances and are equal, and :const:`False` otherwise. Then we have::

   E = g(f({},42),7) # long
   E_symbolic = g.symbolic(f.symbolic({},42),7) # immediate
   assert E.get('u') is None # immediate
   assert E_symbolic.get('u') is None # long
   assert not E != E_symbolic # immediate; use of == is prohibited
  """
#==================================================================================================
  f.symbolic = lambda *a,**ka:Symbolic(f,*a,**ka); return f

#==================================================================================================
class _Symbolic:
# Kept separate from class Symbolic only to avoid attribute name conflicts between the instance itself and its value:
# a Symbolic instance has no non-special attributes, so, when incarnated, behaves exactly as its value (except for special attributes)
#==================================================================================================
  __slots__ = 'config', '_value'

  config: tuple[Callable,tuple[Any,...],tuple[tuple[str,Any],...]]
  _value: Any
  empty = object() # used as dummy, always different from _value

  def __init__(self,config): self.config,self._value = config,_Symbolic.empty
  def check(self): return self._value is not _Symbolic.empty
  @property
  def value(self):
    if (v:=self._value) is _Symbolic.empty:
      func,a,ka = self.config
      self._value = v = func(*a,**dict(ka))
    return v

  # representation methods
  def __repr__(self):
    func,a,ka = self.config
    if isinstance(func,SymbolicOperator): x,y = a[func.covariant]; return f'({x!r} {func} {y!r})'
    sep = ',' if (a and ka) else ''
    a,ka = ','.join(repr(v) for v in a) , ','.join(f'{k}={v!r}' for k,v in ka)
    return f'{func}({a}{sep}{ka})'
  def _repr_html_(self,tail=None):
    from .html import repr_html, html_parlist, E
    if tail is None: return repr_html(self)
    func,a,ka = self.config
    mark = E.span(str(func),style='padding:5px;')
    if isinstance(func,SymbolicOperator): x,y = a[func.covariant]; return E.span(E.b('('),tail(x),mark,tail(y),E.b(')'))
    a,ka = (tail(v) for v in a) , ((k,tail(v)) for k,v in ka)
    return html_parlist(a,ka,opening=(mark,E.b('('),),closing=(E.b(')'),))

#==================================================================================================
def gitcheck(path:str,update:bool=False):
  r"""
:param path: a path to a git repository
:param update: whether to update if stale

Checks directory at *path*, assumed to be a git repository. If *update* is true, attempts to synch with its origin. Returns a tuple of the status  (a string) followed by a list of details. The details are fields ``ref``, ``flags`` and ``note`` from the :class:`git.remote.FetchInfo` object returned by the pull operation. Status is

* ``dirty``: repository is dirty (no details)
* ``uptodate``: repository is up to date (details only if remote exists)
* ``uptodate-now``: repository was stale but has been successfully updated (when *update* is :const:`True`)
* ``stale``: repository is stale but has not been touched (when *update* is :const:`False`)

Use the ``GIT_PYTHON_GIT_EXECUTABLE`` environment variable to set the Git executable if it is not the default ``/usr/bin/git``.
  """
#==================================================================================================
  from git import Repo
  r = Repo(path)
  if r.is_dirty(): return 'dirty',
  try: rm = r.remote()
  except ValueError: return 'uptodate', # clean and no remote
  i, = rm.pull(dry_run=not update)
  d = str(i.ref),i.flags,i.note # detail
  if i.flags & i.HEAD_UPTODATE: return 'uptodate',*d
  if update:
    if i.commit!=r.commit(): raise Exception('Git synch failed',d)
    logger.info('git pull run on %s',r)
    return 'uptodate-now',*d
  return 'stale',*d

#==================================================================================================
def gitcheck_package(pkgname:str,update:bool=False):
  r"""
:param pkgname: full name of a package
:param update: whether to update (git pull) if stale

Assumes that *pkgname* is the name of a python regular (non namespace) package and invokes :meth:`gitcheck` on its path. Reloads the package if ``uptodate-now`` is returned.
  """
#==================================================================================================
  from importlib.util import find_spec
  from importlib import reload
  from sys import modules
  if (pkg:=find_spec(pkgname)) is None or (locs:=pkg.submodule_search_locations) is None or len(locs)!=1: raise ValueError('Not a regular package',pkgname)
  path = locs[0]
  c = gitcheck(path,update)
  if c[0] == 'uptodate-now':
    m = modules.get(pkgname)
    if m is not None: logger.warning('Reloading %s',pkgname); reload(m)
  return c

#==================================================================================================
class ImapFolder:
  r"""
An instance of this class is an iterable enumerating the unread messages in a folder of an IMAP server.
  """
# ==================================================================================================
#--------------------------------------------------------------------------------------------------
  def __init__(self,server:imaplib.IMAP4,folder:str):
    r"""
:param server: the IMAP server
:param folder: the folder to scan

On calling method :meth:`commit`, the messages which have been enumerated since the last commit or rollback (triggered by method :meth:`rollback`) are marked as read. The total number of messages in the folder is held in attribute :attr:`total`, and the number of unread message is given by function :func:`len`.
    """
#--------------------------------------------------------------------------------------------------
    from email import message_from_bytes
    from email.message import EmailMessage
    from email.policy import default as DefaultEmailPolicy
    def message(n):
      t,v = server.fetch(n,'(RFC822)')
      assert t=='OK'
      x = message_from_bytes(v[0][1],_class=EmailMessage,policy=DefaultEmailPolicy)
      seen.append(n)
      return x
    t,v = server.select(folder)
    assert t=='OK'
    self.total = v[0].decode()
    t,v = server.search(None,'UNSEEN')
    assert t=='OK'
    selected = v[0].decode().split()
    self.content_len = len(selected)
    self.content = (message(n) for n in selected)
    seen = []
    def commit():
      server.store(','.join(seen),'+FLAGS',r'\Seen')
      self.content_len -= len(seen)
      seen.clear()
    self.commit = commit
    self.rollback = seen.clear
  def __iter__(self): yield from self.content
  def __len__(self): return self.content_len

#==================================================================================================
class ResettableSimpyEnvironment:
  r"""
An instance of this class wraps a :class:`simpy.Environment` instance, allowing calls to method :meth:`run` to violate the chronological order. When a violation occurs, the wrapped environment is reset and reconfigured before being advanced to the new time. For reconfiguration to work, updates to the wrapped object should not be done directly, but only by invoking method:`add_config` on the wrapper. For stochastic environments, to ensure predictable outcome, make sure to add a configuration which resets the seed to a fixed value.
  """
#==================================================================================================
  configs:list[Callable[[simpy.Environment],None]]
#--------------------------------------------------------------------------------------------------
  def add_config(self,*C:Callable[[simpy.Environment],None]):
    r"""
:param C: the list of configurations to add

Adds configurations to the environment. It is applied to the proxy each time it is reset.
    """
#--------------------------------------------------------------------------------------------------
    self.configs.extend(C)
#--------------------------------------------------------------------------------------------------
  def __init__(self,init_t:float=0.):
    r"""
:param init_t: the initial time of the environment
    """
#--------------------------------------------------------------------------------------------------
    from simpy import Environment
    self.configs = []
    def reset(v:float):
      self._wrapped = env = Environment(init_t)
      for c in self.configs: c(env)
      if v>init_t: self._wrapped.run(v)
    def run(v:float):
      #assert isinstance(v,float) and v >= init_t
      now = self._wrapped.now
      if v>now: self._wrapped.run(v); return
      if v<now: reset(v)
    def run_init(v:float): self.run = run; reset(v)
    self.run = run_init
  def __getattr__(self,name:str): return getattr(self._wrapped,name)

#==================================================================================================
class BoardDisplayer:
  r"""
An instance of this class is a callable which takes as input a board, prepares it for display, and returns a callable which, given a frame, paints it on the board. The board (here an instance of class :class:`Figure`) is divided into panes (here instances of class :class:`Axes`). Each pane can be assigned a list of named pane displayers, corresponding to different views of the displayed phenomenon. The name of each view can be used to configure the view using parameter *view_cfg* in method :meth:`__call__` (invocation of this displayer).
  """
#==================================================================================================
  setup:Sequence[Callable[[Any],None]]
  r"""Invoked before invoking the pane displayers on a frame"""
  displayers:dict[tuple[int,int],list[tuple[str,Callable[[Any],Callable[[Any],None]]]]]
  env:ResettableSimpyEnvironment|None = None
#--------------------------------------------------------------------------------------------------
  def __init__(self):
    r"""
:param pos: the position of the pane on the board
:param ka: the keys are view names and the values are pane displayers
    """
#--------------------------------------------------------------------------------------------------
    self.displayers = collections.defaultdict(list)
    self.setup = []
  
#--------------------------------------------------------------------------------------------------
  def add_displayer(self,pos:tuple[int,int]|None=None,**ka:Callable[[Any],Callable[[Any],None]]):
    r"""
Adds a pane displayer to this board displayer. A pane displayer is a callable which takes as input a pane, prepares it for display, and returns a callable which, given a frame, paints it on the pane according to the view name.
    """
#--------------------------------------------------------------------------------------------------
    if pos is None: pos = (0,0)
    self.displayers[pos].extend((view,D) for view,D in ka.items())
    return self # so other method invocations can be chained
#--------------------------------------------------------------------------------------------------
  def add_setup(self,setup:Callable[[Any],None]):
    r"""
Adds a setup callable to this board displayer.
    """
#--------------------------------------------------------------------------------------------------
    self.setup.append(setup)
    return self # so other method invocations can be chained
#--------------------------------------------------------------------------------------------------
  def add_simpy_setup(self,*C:Callable[[ResettableSimpyEnvironment],None]):
    r"""
:param C: config added to the environment held by attribute :attr:`simpy_env`

Activates (if needed) and configures the environment held by attribute :attr:`simpy_env`.
    """
#--------------------------------------------------------------------------------------------------
    if (env:=self.env) is None:
      env = self.env = ResettableSimpyEnvironment(0.)
      self.add_setup(lambda v: env.run(v))
    env.add_config(*C)
    return self # so other method invocations can be chained
#--------------------------------------------------------------------------------------------------
  call_defaults = {'aspect':'equal'}
  def __call__(self,fig,nrows:int=1,ncols:int=1,sharex:str|bool=False,sharey:str|bool=False,gridspec_kw:Mapping|None=None,view_cfg:Mapping[str,dict]|None=None,gridlines:bool=True,**ka)->Callable[[Any],None]:
    r"""
Prepares the board for display, and returns a callable which takes as input a frame and actually displays it on the board.

:param fig: the board
:param nrows: the number of rows
:param ncols: the number of columns
:param sharex: sharing specification for the x-axis
:param sharey: sharing specification for the y-axis
:param gridspec_kw: a grid specification for the board
:param view_cfg: a dictionary of view configurations (each configuration is a :class:`dict`)
:param gridlines: whether to display gridlines on the axes
:param ka: passed on as subplot keywords to construct the axes
    """
#--------------------------------------------------------------------------------------------------
    from numpy import array
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.axes import Axes
    assert isinstance(fig,(Figure,SubFigure))
    axes = array(nrows*[ncols*[None]],dtype=object)
    share:dict[str|bool,Callable[[int,int],Axes|None]] = {
      'all': (lambda row,col: None if row==col==0 else axes[0,0]),
      True:  (lambda row,col: None if row==col==0 else axes[0,0]),   # alias
      'row': (lambda row,col: None if row==0 else axes[row,0]),
      'col': (lambda row,col: None if col==0 else axes[0,col]),
      'none':(lambda row,col: None),
      False: (lambda row,col: None), # alias
    }
    share_ = tuple((dim,share[s]) for dim,s in (('sharex',sharex),('sharey',sharey)))
    gridspec = fig.add_gridspec(nrows=nrows,ncols=ncols,**(gridspec_kw or {}))
    ka = self.call_defaults|ka
    def _get(row,col):
      ax = axes[row,col]
      if ax is None:
        ax = axes[row,col] = fig.add_subplot(gridspec[row,col],**dict((dim,s(row,col)) for dim,s in share_),**ka)
        ax.grid(gridlines)
      return ax
    get_view_cfg:Callable[[str],dict] = (lambda view: {}) if view_cfg is None else view_cfg.get
    display_list = [*self.setup,*(D(_get(*pos),**kw) for pos,L in self.displayers.items() for view,D in L if (kw:=get_view_cfg(view)) is not None)]
    def display(frm):
      for f in display_list: f(frm)
    return display

#--------------------------------------------------------------------------------------------------
  play_defaults = {'interval':40}
  r"""The default arguments passed to method :meth:`play`"""
  def play(self,displayer_kw=None,**ka):
    r"""
Returns an animation based on this displayer.

:param displayer_kw: a dictionary of keyword arguments passed when invoking this displayer
:param ka: passed to the animation constructor
    """
#--------------------------------------------------------------------------------------------------
    from functools import partial
    from matplotlib import get_backend
    from matplotlib.animation import Animation
    ka = self.play_defaults|ka
    factory:Callable[[str],Callable[[BoardDisplayer],Animation]] = ka.pop('factory')
    return factory(get_backend())((self if displayer_kw is None else partial(self,**displayer_kw)),**ka)

#==================================================================================================
class basic_stats:
  r"""
Instances of this class maintain basic statistics about a group of values.
  """
#==================================================================================================
  def __init__(self,weight:int|float|complex=1.,avg:int|float|complex=0.,var:int|float|complex=0.):
    r"""
:param weight: total weight of the group
:param avg: weighted average of the group
:param var: weighted variance of the group
    """
    self.weight = weight; self.avg = avg; self.var = var
  def __add__(self,other):
    w,a,v = (other.weight,other.avg,other.var) if isinstance(other,basic_stats) else (1.,other,0.)
    W = self.weight+w; r_self = self.weight/W; r_other = w/W; d = a-self.avg
    return basic_stats(weight=W,avg=r_self*self.avg+r_other*a,var=r_self*self.var+r_other*v+(r_self*d)*(r_other*d))
  def __iadd__(self,other):
    w,a,v = (other.weight,other.avg,other.var) if isinstance(other,basic_stats) else (1.,other,0.)
    self.weight += w; r = w/self.weight; d = a-self.avg
    self.avg += r*d; self.var += r*(v-self.var+(1-r)*d*d)
    return self
  def __repr__(self): return f'basic_stats<weight:{self.weight!r},avg:{self.avg!r},var:{self.var!r}>'
  @property
  def std(self):
    r"""standard deviation of the group"""
    from math import sqrt
    return sqrt(self.var)

#==================================================================================================
def iso2date(iso:tuple[int,int,int])->datetime.date:
  r"""
:param iso: triple as returned by :meth:`datetime.date.isocalendar`

Returns the :class:`datetime.date` instance for which the :meth:`datetime.date.isocalendar` method returns *iso*. Example::

   from datetime import datetime
   d = datetime.now().date(); assert iso2date(d.isocalendar()) == d
  """
#==================================================================================================
  from datetime import date,timedelta
  isoyear,isoweek,isoday = iso
  jan4 = date(isoyear,1,4)
  iso1 = jan4-timedelta(days=jan4.weekday())
  return iso1+timedelta(days=isoday-1,weeks=isoweek-1)

#==================================================================================================
def qty_format(val:int|float,precision:int=3,unit:str='iB')->str:
  r"""
:param val: a positive number representing an amount of *unit*
:param precision: number of digits displayed
:param unit: the unit of the measure

Returns the representation of *val* in **IEC binary** format when *unit* is :const:`'iB'` (the default, for bytes), or in **IEC decimal** format when *unit* is :const:`'B'`, or in **SI** format in all other cases. The table below gives for each prefix its name, letter and exponent. Each prefix represents a multiplier equal to the base (constant 1000 or 1024, see below) to the power of the exponent.

+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+--------+
| yocto | zepto | atto  | femto | pico  | nano  | micro | milli | kilo  | Mega  | Giga  | Tera  | Peta  | Exa   | Zetta | Yotta  |
+=======+=======+=======+=======+=======+=======+=======+=======+=======+=======+=======+=======+=======+=======+=======+========+
| ``y`` | ``z`` | ``a`` | ``f`` | ``p`` | ``n`` | ``μ`` | ``m`` | ``k`` | ``M`` | ``G`` | ``T`` | ``P`` | ``E`` | ``Z`` | ``Y``  |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+--------+
| -8    | -7    | -6    | -5    | -4    | -3    | -2    | -1    | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8      |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+--------+
| * `SI`_ (all units except :const:`'B'` and :const:`'iB'`): all prefixes are available, base is 1000                            |
| * `IEC`_ decimal (:const:`'B'`): as SI, but the negative exponents are unavailable & prefix ``k`` (kilo) is uppercased to ``K``|
| * `IEC`_ binary (:const:`'iB'`): as IEC decimal, but base is 1024 instead of 1000                                              |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+--------+

.. _IEC: https://en.wikipedia.org/wiki/Binary_prefix
.. _SI: https://en.wikipedia.org/wiki/Metric_prefix

Example::

   test = [ (2**30,{},'1GiB'), (5300,{},'5.18KiB'), (5300,{'unit':'B'},'5.3KB'), (42897.3,{'unit':'m'},'42.9km') ]
   assert all(qty_format(x,**ka)==v for x,ka,v in test)
  """
#==================================================================================================
  from math import log
  match unit:
    case 'iB': assert val>=1; base = 1024; unit = 'B'; P = qty_format.P_IEC
    case 'B': assert val>=1; base = 1000; P = qty_format.P_IEC10
    case _: assert val>0; base = 1000; P = qty_format.P_SI
  fmt = f'{{:.{precision}g}}{{}}{unit}'.format
  n,v = divmod(log(val,base),1.); n = round(n)+8
  if 0<=n<=16: return fmt(base**v,P[n]) # most likely...
  d = 0 if n<0 else 16; return fmt(base**(v+n-d),P[d])
def _config__(f):
  f.P_IEC = (*9*('',),*(x+'i' for x in 'KMGTPEZY'))
  f.P_IEC10 = (*9*('',),*'KMGTPEZY')
  f.P_SI = (*reversed('mμnpfazy'),'',*'kMGTPEZY')
_config__(qty_format); del _config__

#==================================================================================================
def time_format(val:int|float,precision:int=3)->str:
  r"""
:param val: a number representing a time in seconds
:param precision: number of digits displayed

Returns the representation of *time* in one of day,hr,min,sec,msec,μsec depending on magnitude. Example::

   test = [ (100000,{'precision':4},'1.157day'), (4238.45,{},'1.18hr'), (5.35,{'precision':1},'5sec'), (.932476,{},'932msec') ]
   assert all(time_format(x,**ka)==v for x,ka,v in test)
  """
#==================================================================================================
  fmt = f'{{:.{precision}g}}'.format
  if val < 1.e-6: return '<1μsec'
  if val < 1.e-3: return f'{fmt(1.e6*val)}μsec'
  if val < 1.: return f'{fmt(1e3*val)}msec'
  if val < 60.: return f'{fmt(val)}sec'
  val_:float = val/60.
  if val_ < 60.: return f'{fmt(val_)}min'
  val_ /= 60.
  if val_ < 24.: return f'{fmt(val_)}hr'
  val_ /= 24.
  return f'{fmt(val_)}day'

#==================================================================================================
def versioned(v)->Callable[[Callable],Callable]:
  r"""
A decorator which assigns attribute :attr:`version` of the target function to *v*. The function must be defined at the toplevel of its module. The version must be a simple value.
  """
#==================================================================================================
  def transform(f:Callable):
    from inspect import isfunction
    assert isfunction(f)
    f.version = v; return f
  return transform

#==================================================================================================
class _PDFConverter (dict):
  r"""
The single instance of this class is a dictionary mapping mimetypes to PDF converters. To declare a new converter, which must be of type :class:`bytes`↦:class:`bytes`, decorate it with an invocation of this instance with the list of mimetypes accepted by the converter as positional arguments.
  """
#==================================================================================================
  def __call__(self,*mimetypes:str):
    def transform(f:Callable[[Any],bytes]): assert f.__name__.startswith('pdf_from'); self.update({mtype:f for mtype in mimetypes}); return f
    return transform
PDFConverter = _PDFConverter()

@PDFConverter('application/pdf')
def pdf_frompdf(content:bytes): return content

@PDFConverter(
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/msword',
  'application/vnd.oasis.opendocument.text')
def pdf_fromodf(content:bytes):
  import subprocess
  from tempfile import NamedTemporaryFile
  from pathlib import Path
  with NamedTemporaryFile(prefix='DocConvert_',suffix='.topdf') as v:
    path = Path(v.name); v.write(content); v.flush(); path_ = path.with_suffix('.pdf')
    try:
      subprocess.run(['soffice','--headless','--convert-to','pdf','--outdir',str(path.parent),str(path)],check=True)
      return path_.read_bytes()
    finally: path_.unlink(missing_ok=True)

@PDFConverter('image/svg+xml')
def pdf_fromsvg(content:str):
  import subprocess
  return subprocess.run(['rsvg-convert','-f','pdf'],check=True,capture_output=True,input=content.encode()).stdout

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def unid(post:str='',pre:str=__name__)->str:
  r"""
Returns a "unique" id for miscellanous uses.
  """
#--------------------------------------------------------------------------------------------------
  from time import time
  return f'{pre}{time()}{post}'.replace('.','_')

#--------------------------------------------------------------------------------------------------
def ast_tuple(x:str|ast.AST,top:str='top'):
  r"""
:param x: an AST (python Abstract Syntax Tree) or string (parsed into an AST)
:param top: the name of the top-level node (optional)

returns a representation of the AST object using only tuples and str.
  """
#--------------------------------------------------------------------------------------------------
  import ast
  def gen(x,pre:str):
    if isinstance(x,ast.AST):
      yield f'{pre}:{x.__class__.__name__}',*(z for k,y in ast.iter_fields(x) for z in gen(y,k))
    elif isinstance(x,list):
      for i,y in enumerate(x): yield from gen(y,f'{pre}[{i}]')
    else: yield pre,x
  if isinstance(x,str): x = ast.parse(x)
  else: assert isinstance(x, ast.AST)
  return tuple(gen(x,top))

#--------------------------------------------------------------------------------------------------
class pickleclass:
  r"""
This namespace class defines class methods :meth:`load`, :meth:`loads`, :meth:`dump`, :meth:`dumps`, similar to those of the standard :mod:`pickle` module, but with class specific Pickler/Unpickler which must be defined in subclasses.
  """
#--------------------------------------------------------------------------------------------------

  Pickler: Callable[[Any],pickle.Pickler]
  Unpickler: Callable[[Any],pickle.Unpickler]

  @classmethod
  def dump(cls,obj,v): cls.Pickler(v).dump(obj)

  @classmethod
  def load(cls,u): return cls.Unpickler(u).load()

  @classmethod
  def dumps(cls,obj):
    from io import BytesIO
    with BytesIO() as v: cls.Pickler(v).dump(obj); return v.getvalue()

  @classmethod
  def loads(cls,s):
    from io import BytesIO
    with BytesIO(s) as u: return cls.Unpickler(u).load()

#--------------------------------------------------------------------------------------------------
def einsum_nd(sgn:str,*a,pat=re.compile(r'(\w+)\.?')):
  r"""
Equivalent of numpy.einsum with named dimensions (limited to 26 like the original!). Dimension names must be words in the sense of the :mod:`re` module (unicode allowed) and must be separated by '.' in operand shapes.
  """
#--------------------------------------------------------------------------------------------------
  from numpy import einsum
  D = collections.defaultdict(lambda A=iter('abcdefghijklmnopqrstuvwxyz'): next(A))
  return einsum(pat.sub((lambda m,D=D:D[m[1]]),sgn),*a)
