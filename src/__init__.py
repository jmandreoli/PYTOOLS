# File:                 __init__.py
# Creation date:        2014-03-16
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities in Python
#

from __future__ import annotations
import logging; logger = logging.getLogger(__name__)
import os, re, collections
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, Tuple
if False: import ast,imaplib,pickle,datetime # tricks mypy to import these modules

#==================================================================================================
class owrap:
  r"""
Objects of this class present an attribute oriented proxy interface to an underlying mapping object.

:param __ref__: a reference to a (possibly mutable) mapping object, of which this instance is to be a proxy

Keys in the reference are turned into attributes of the proxy. If *__ref__* is :const:`None`, the reference is a new empty dictionary, otherwise *__ref__* must be a mapping object, and *__ref__* (or its own reference if it is itself of class :class:`owrap`) is taken as reference. The reference is first updated with the keyword arguments *ka*. Example::

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
  def __init__(self,__ref__:Mapping[str,Any]=None,**ka):
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
class _forward:
  r"""
This object allows to name callable members of module/packages without loading them. They are loaded only on actual call. Example::

   from sys import modules
   array = forward.numpy.array
   assert 'numpy' not in modules # numpy is not loaded
   x = array((1,2,3))
   assert 'numpy' in modules and x.shape == (3,) # now numpy is loaded and x is computed
  """
#==================================================================================================
  __slot__ = '__spec__','__value__'
  def __init__(self,s=()): self.__spec__ = s; self.__value__ = None
  def __getattr__(self,a): return _forward(self.__spec__+(a,))
  def __call__(self,*a,**ka):
    from importlib import import_module
    f = self.__value__
    if f is None:
      self.__value__ = f = getattr(import_module('.'.join(self.__spec__[:-1])),self.__spec__[-1])
    return f(*a,**ka)
  def __getstate__(self): return self.__spec__
  def __setstate__(self,s): self.__spec__ = s; self.__value__ = None
  def __repr__(self): return 'forward<{}{}>'.format('.'.join(self.__spec__),('' if self.__value__ is None else ':incarnated'))
forward = _forward()

#==================================================================================================
def namedtuple(*a,**ka):
  r"""
Returns a class of tuples with named fields. It is identical to :func:`collections.namedtuple`, with only two additions:

* The returned class has an attribute :attr:`_field_index` which is an instance of that class where each field is set to its index in the field list. e.g.::

   c = namedtuple('c','u v')
   assert c._field_index.u == 0 and c._field_index.v == 1

* Furthermore, if *f* and *x* are instances of the returned class *c*, then calling *f* with argument *x* returns a new instance of class *c* in which each field is set to the result of applying the corresponding field in *f* (which must be a callable) to the corresponding field in *x*. Additional keyword arguments can be passed to the calls. E.g.::

   c = namedtuple('c','u v')
   f = c(u=(lambda z,r=0.: 2.*z+r),v=(lambda z,r=0.: 3.*z+5.*r))
   x = c(u=3,v=5)
   k = dict(r=42)
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
def config_env(name:str,dflt:str=None,asfile:bool=False)->str|None:
  r"""
:param name: the name of an environment variable
:param dflt: a default value
:param asfile: whether to treat the value of the environment variable as a path to a file

Returns the string value of an environment variable (or the content of the file pointed by it if *asfile* is set) or *dflt* if the environment variable is not assigned.
  """
#==================================================================================================
  x = os.environ.get(name)
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
class Expr:
  r"""
Instances of this class implement closed symbolic expressions.

:param func: the function of the symbolic expression
:param a: list of positional arguments of the symbolic expression
:param ka: dictionary of keyword arguments of the symbolic expression

The triple *func*, *a*, *ka* forms the configuration of the :class:`Expr` instance. Its value is defined as the result of calling function *func* with positional and keyword arguments *a* and *ka*. The value is actually computed only once (and cached), and only when method :meth:`incarnate` is invoked. Subclasses should define automatic triggers of incarnation (see e.g. classes :class:`MapExpr` and :class:`CallExpr`). The incarnation cache can be reset by invoking method :meth:`reset`.

Initially, a :class:`Expr` instance is mutable, and its configuration can be changed. It becomes immutable (frozen) after any of the following operations: incarnation (even after it is reset), hashing, and pickling. Incarnation is not saved on pickling, hence lost on unpickling, but can of course be restored using method :meth:`incarnate`. Thus, when receiving a foreign :class:`Expr` instance, a process can decide whether it wants to access its value or only inspect its configuration. If recomputing the value is costly, use a persistent cache for *func*.

Caveat: function *func* should be defined at the top-level of its module, and the values in *a* and *ka* should be deterministically picklable and hashable (in particular: no dicts nor lists). Hence, any :class:`Expr` instance is itself deterministically picklable and hashable, and can thus be used as argument in the configuration of another :class:`Expr` instance.
  """
#==================================================================================================

  incarnated: bool
  value: Any

  def __init__(self,func:Callable,*a,**ka):
    assert callable(func)
    self.reset()
    self.config = [func,a,ka]
    self.key = None

  def rearg(self,*a,**ka):
    r"""
Updates the config arguments of this instance: *a* is appended to the list of positional arguments and *ka* is updated into the dictionary of keyword arguments. Raises an error if the instance is frozen.
    """
    assert self.key is None, 'attempt to update a frozen Expr instance'
    self.config[1] += a
    self.config[2].update(ka)

  def refunc(self,f:Callable):
    r"""
Set *f* as the config function of this instance. Raises an error if the instance is frozen.
    """
    assert self.key is None, 'attempt to update a frozen Expr instance'
    self.config[0] = f

  def incarnate(self):
    if self.incarnated: return
    self.incarnated = True
    self.freeze()
    func,a,ka = self.config
    self.value = func(*a,**ka)

  def reset(self):
    self.incarnated = False
    self.value = None

  def freeze(self):
    k = self.key
    if k is None:
      func,a,ka = self.config
      self.key = k = func,a,tuple(sorted(ka.items()))
    return k

  def __getstate__(self): return self.freeze()
  def __setstate__(self,key):
    self.reset()
    func,a,ka = self.key = key
    self.config = [func,a,dict(ka)]
  def __hash__(self): return hash(self.freeze())
  def __eq__(self,other): return isinstance(other,Expr) and self.config == other.config
  def __repr__(self):
    if self.incarnated: return repr(self.value)
    func,a,ka = self.config
    sep = ',' if (a and ka) else ''
    a = ','.join(repr(v) for v in a)
    ka = ','.join(f'{k}={repr(v)}' for k,v in sorted(ka.items()))
    return f'{func}({a}{sep}{ka})'
  def _repr_html_(self):
    from .html import repr_html
    return repr_html(self)
  def as_html(self,_):
    from .html import html_parlist, E
    func,a,ka = self.config
    opn,clo = ((E.span if self.incarnated else E.em)(str(func),style='padding:5px;'),E.b('['),), (E.b(']'),)
    return html_parlist(_,a,sorted(ka.items()),opening=opn,closing=clo)

#--------------------------------------------------------------------------------------------------
class MapExpr (Expr,collections.abc.Mapping):
  r"""
Symbolic expressions of this class are also (read-only) mappings and trigger incarnation on all the mapping operations, then delegate such operations to their value (expected to be a mapping).
  """
#--------------------------------------------------------------------------------------------------
  def __getitem__(self,k): self.incarnate(); return self.value[k]
  def __len__(self): self.incarnate(); return len(self.value)
  def __iter__(self): self.incarnate(); return iter(self.value)

#--------------------------------------------------------------------------------------------------
class CallExpr (Expr):
  r"""
Symbolic expressions of this class are also callables, and trigger incarnation on invocation, then delegate the invocation to their value (expected to be a callable).
  """
#--------------------------------------------------------------------------------------------------
  def __call__(self,*a,**ka): self.incarnate(); return self.value(*a,**ka)

#==================================================================================================
class SQliteStack:
  r"""
Objects of this class are aggregation functions which simply collect results in a list, for use with a SQlite database. Example::

   with sqlite3.connect('/path/to/db') as conn:
     rstack = SQliteStack.setup(conn,'stack',2)
     for school,x in conn.execute('SELECT school,stack(age,height) FROM PupilsTable GROUP BY school'):
       print(school,rstack(x))

prints pairs *school*, *L* where *L* is a list of pairs *age*, *height*.
  """
#==================================================================================================
  contents:dict[int,Any] = {}
  def __init__(self): self.content = []
  def step(self,*a): self.content.append(a)
  def finalize(self): n = id(self.content); self.contents[n] = self.content; return n
  @staticmethod
  def setup(conn,name,n):
    conn.create_aggregate(name,n,SQliteStack)
    return SQliteStack.contents.pop

#==================================================================================================
def SQliteNew(path:str,schema:str):
  r"""
:param path: a path to an sqlite database
:param schema: the schema specification

Makes sure the file at *path* is a SQlite3 database with schema exactly equal to *schema*.
  """
#==================================================================================================
  import sqlite3
  from string import whitespace
  sep = whitespace+';'
  with sqlite3.connect(path,isolation_level='EXCLUSIVE') as conn:
    S = [sql for sql, in conn.execute('SELECT sql FROM sqlite_master WHERE name NOT LIKE \'sqlite%\'')]
    if S:
      schema = schema.strip()
      for sql in S:
        if not schema.startswith(sql): raise Exception('database has a version conflict')
        schema = schema[len(sql):].lstrip(sep)
      else:
        if schema.lstrip(sep): raise Exception('database has a version conflict')
    else: conn.executescript(schema)

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
def gitcheck_package(pkgname:str,update=False):
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
class spark:
  r"""
If a resource ``spark/pyspark.py`` exists in an XDG configuration file, that resource is executed (locally) and should define a function :func:`init` used as class method :meth:`init` and meant to initialise Spark. The initialisation must assign a :class:`dict` (even an empty one) to attribute :attr:`conf`.
  """
#==================================================================================================
  conf = None

  @classmethod
  def display_monitor_link(cls,sc):
    r"""
Displays a link to the monitor of :class:`pyspark.SparkContext` *sc*. Recall that the monitor is active only between the creation of *sc* and its termination (when method :meth:`stop` is invoked).
    """
    from IPython.display import display_html
    display_html(f'<a target="_blank" href="{sc.uiWebUrl}">SparkMonitor[{sc.appName}@{sc.master}]</a>', raw=True)

  @classmethod
  def SparkContext(cls,display=True,debug=False,conf={},**ka):
    r"""
Returns an instance of :class:`pyspark.SparkContext` created with the predefined configuration held in attribute :attr:`conf` of this class, updated by *conf* (its value must then be a :class:`dict` of :class:`str`). If *debug* is :const:`True`, prints the exact configuration used. If *display* is :const:`True`, displays a link to the monitor of the created context.
    """
    cls._init()
    from pyspark import SparkContext, SparkConf
    cfg = SparkConf().setAll(cls.conf.items()).setAll(conf.items())
    if debug: print(cfg.toDebugString())
    sc = SparkContext(conf=cfg,**ka)
    if display: cls.display_monitor_link(sc)
    return sc

  @classmethod
  def _init(cls):
    if cls.conf is None:
      cls.init()
      assert isinstance(cls.conf,dict) and all((isinstance(k,str) and isinstance(v,str)) for k,v in cls.conf.items()), f'Class method {cls.__module__}.{cls.__name__}.init must assign a str-str dictionary to class attribute \'conf\''

  init = config_xdg('spark/pyspark.py')
  if init is None: init = classmethod(lambda cls,**ka: None)
  else:
    D = {}
    exec(init,D)
    init = classmethod(D['init'])
    del D

#==================================================================================================
class ImapFolder:
  r"""
An instance of this class is an iterable enumerating the unread messages in a folder of an IMAP server. On calling method :meth:`commit`, the messages which have been enumerated since the last commit or rollback (triggered by method :meth:`rollback`) are marked as read. The total number of messages in the folder is held in attribute :attr:`total`, and the number of unread message is given by function :func:`len`.
  """
# ==================================================================================================
  def __init__(self,server:imaplib.IMAP4,folder:str):
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
class basic_stats:
  r"""
Instances of this class maintain basic statistics about a group of values.

:param weight: total weight of the group
:param avg: weighted average of the group
:param var: weighted variance of the group
  """
#==================================================================================================
  def __init__(self,weight:int|float|complex=1.,avg:int|float|complex=0.,var:int|float|complex=0.):
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
  def __repr__(self): return f'basic_stats<weight:{repr(self.weight)},avg:{repr(self.avg)},var:{repr(self.var)}>'
  @property
  def std(self):
    r"""standard deviation of the group"""
    from math import sqrt
    return sqrt(self.var)

#==================================================================================================
def iso2date(iso:Tuple[int,int,int])->datetime.date:
  r"""
:param iso: triple as returned by :meth:`datetime.date.isocalendar`

Returns the :class:`datetime.date` instance for which the :meth:`datetime.date.isocalendar` method returns *iso*::

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
def size_fmt(size:int|float,binary:bool=True,precision:int=4,suffix:str='B')->str:
  r"""
:param size: a positive number representing a size
:param binary: whether to use IEC binary or decimal convention
:param precision: number of digits displayed (at least 4)

Returns the representation of *size* with IEC prefix. Each prefix is *K* times the previous one for some constant *K* which depends on the convention: *K* =1024 with the binary convention (marked with an ``i`` after the prefix); *K* =1000 with the decimal convention. Example::

   print(size_fmt(2**30), size_fmt(5300), size_fmt(5300,binary=False), size_fmt(42897.3,binary=False,suffix='m'))
   #> 1GiB 5.176KiB 5.3KB 42.9Km
  """
#==================================================================================================
  thr,mark = (1024.,'i') if binary else (1000.,'')
  if size<thr: return f'{size}{suffix}'
  fmt = f'{{:.{precision}g}}{{}}{mark}{suffix}'.format
  size_:float = size/thr
  for prefix in 'KMGTPEZ':
    if size_ < thr: return fmt(size_,prefix)
    size_ /= thr
  return fmt(size_,'Y') # :-)

#==================================================================================================
def time_fmt(time:int|float,precision:int=2)->str:
  r"""
:param time: a number representing a time in seconds
:param precision: number of digits displayed

Returns the representation of *time* in one of days,hours,minutes,seconds,milli-seconds (depending on magnitude). Example::

   print(time_fmt(100000,4),time_fmt(4238.45),time_fmt(5.35,0),time_fmt(.932476))
   #> 1.1574day 1.18hr 5sec 932msec
  """
#==================================================================================================
  fmt = f'{{:.{precision}f}}'.format
  if time < 1.: return f'{1000 * time:3.3g}msec'
  if time < 60.: return f'{fmt(time)}sec'
  time_:float = time/60.
  if time_ < 60.: return f'{fmt(time_)}min'
  time_ /= 60.
  if time_ < 24.: return f'{fmt(time_)}hr'
  time_ /= 24.
  return f'{fmt(time_)}day'

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
A decorator which interprets the target function as a PDF converter.

:param mimetypes: list of mimetypes
  """
#==================================================================================================
  def assign(self,*mimetypes:Sequence[str]):
    def t(f:Callable[[Any],bytes]): self.update({mtype:f for mtype in mimetypes}); return f
    return t
PDFConverter = _PDFConverter()

@PDFConverter.assign('application/pdf')
def pdf_frompdf(content:bytes): return content

@PDFConverter.assign(
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

@PDFConverter.assign('image/svg+xml')
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
  return (pre+str(time())+post).replace('.','_')

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
