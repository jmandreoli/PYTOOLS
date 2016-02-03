from collections.abc import MutableMapping
import logging
logger = logging.getLogger(__name__)

#==================================================================================================
class ondemand (object):
  r"""
Use as a decorator to declare, in a class, a computable attribute which is computed only once (its value is then cached). Example::

   class c:
     def __init__(self,u): self.u = u
     @ondemand
     def att(self): print('Computing att...'); return self.u+1
   x = c(3); print(x.att)
   #>>> Computing att...
   #>>> 4
   print(x.att)
   #>>> 4
   x.u = 6; print(x.att)
   #>>> 4
   del x.att; print(x.att)
   #>>> Computing att...
   #>>> 7
  """
#==================================================================================================

  __slots__ = ('get',)

  def __init__(self,get):
    if get.__code__.co_argcount != 1:
      raise Exception('ondemand attribute definition must be a function with a single argument')
    self.get = get

  def __get__(self,instance,typ):
    if instance is None:
      return self
    else:
      val = self.get(instance)
      setattr(instance,self.get.__name__,val)
      return val

#==================================================================================================
class odict (MutableMapping):
  r"""
Objects of this class act as dict objects, except their keys are also attributes. Keys must be strings and must not override dict standard methods, but no check is performed. Example::

   x = odict(a=3,b=6); print(x.a)
   #>>> 3
   print(x['a'])
   #>>> 3
   del x.a; print(x)
   #>>> {'b':6}
   x.b += 7; print(x['b'])
   #>>> 13
  """
#==================================================================================================
  def __init__(self,**ka): self.__dict__ = ka
  def __getitem__(self,k): return self.__dict__[k]
  def __setitem__(self,k,v): self.__dict__[k] = v
  def __delitem__(self,k): del self.__dict__[k]
  def __iter__(self): return iter(self.__dict__)
  def __len__(self): return len(self.__dict__)
  def __str__(self): return str(self.__dict__)
  def __repr__(self): return repr(self.__dict__)

#==================================================================================================
class ARG (tuple):
  r"""
Instances of this (immutable) class are pairs of a tuple of positional arguments and a dict of keyword arguments. Useful to manipulate function invocation arguments without making the invocation.

Methods:
  """
#==================================================================================================
  def __new__(cls,*a,**ka):
    return super(ARG,cls).__new__(cls,(a,ka))

  def variant(self,*a,**ka):
    r"""
Returns a variant of *self* where *a* is appended to the positional arguments and *ka* is updated into the keyword arguments.
    """
    a0,ka0 = self
    a = a0+a
    ka1 = ka0.copy(); ka1.update(ka); ka = ka1
    return ARG(*a,**ka)

  def __repr__(self):
    a,ka = self
    a = ','.join(repr(v) for v in a) if a else ''
    ka = ','.join('{}={}'.format(k,repr(v)) for k,v in ka.items()) if ka else ''
    sep = ',' if a and ka else ''
    return 'ARG({}{}{})'.format(a,sep,ka)

  def __str__(self):
    return self.__repr__()

#==================================================================================================
def ARGgrid(*a,**ka):
  r"""
Generates a grid of :class:`ARG` instances. Each positional argument in *a* and keyword argument in *ka* must be assigned a list of alternative values.
  """
#==================================================================================================
  def g(c,a):
    if not a: yield c; return
    for cc in g(c,a[:-1]):
      for v in a[-1]:
        yield cc+(v,)
  if ka:
    keys,vals = zip(*ka.items())
    kal = [dict(zip(keys,valc)) for valc in g((),vals)]
  else: kal = [{}]
  for ac in g((),a):
    for kac in kal:
      yield ARG(*ac,**kac)

#==================================================================================================
def zipaxes(L,fig,sharex=False,sharey=False,**ka):
  r"""
:param fig: a figure
:type fig: :class:`matplotlib.figure.Figure`
:param L: an arbitrary sequence
:param sharex: whether all the axes share the same x-axis scale
:type sharex: :class:`bool`
:param sharey: whether all the axes share the same y-axis scale
:type sharey: :class:`bool`
:param ka: passed to :meth:`add_subplot` generating new axes

Yields the pair of *o* and an axes on *fig* for each item *o* in sequence *L*. The axes are spread more or less uniformly.
  """
#==================================================================================================
  from math import sqrt,ceil
  L = list(L)
  N = len(L)
  nc = int(ceil(sqrt(N)))
  nr = int(ceil(N/nc))
  axrefx,axrefy = None,None
  for i,o in enumerate(L,1):
    ax = fig.add_subplot(nr,nc,i,sharex=axrefx,sharey=axrefy,**ka)
    if sharex and axrefx is None: axrefx = ax
    if sharey and axrefy is None: axrefy = ax
    yield o,ax

#==================================================================================================
class Tree:
  r"""
Objects of this class are trees with IPython navigation facilities.

Attributes (must be defined in sub-classes):

.. attribute:: children

   A sequence of :class:`Tree` objects, the offspring of this tree.

.. attribute:: summary

   A string or tuple whose first component is a string and the other components (if any) are name-value pairs. In the latter case, the tuple is formatted into a string. The string must be an HTML representation of the root of this tree.

Methods:
  """
#==================================================================================================

  style_folded =   dict(width='3mm',height='3mm',padding='0cm',font_size='xx-small',label='+',background_color='gray')
  style_unfolded = dict(width='3mm',height='3mm',padding='0cm',font_size='xx-small',label='-',background_color='darkblue')
  style_summary = 'border-left: thin solid blue; padding-top: 0mm; padding-bottom: 0mm; padding-left: 1mm; padding-right: 1mm;'

  def ipynav(self):
    r"""
Displays a widget for nagigating this tree.
    """
    from IPython.display import display
    from ipywidgets.widgets import HBox, VBox, Button, HTML
    def lines(node,pre):
      s = exp.get(pre)
      if s is None: s = exp[pre] = [False]; s.append(HBox(children=tuple(getline(node,pre,s))))
      yield s[1]
      if s[0]:
        for k,nodec in node.children: yield from lines(nodec,pre+(k,))
    def getline(node,pre,s):
      yield HTML('<div style="padding-left: {}cm">&nbsp;</div>'.format(len(pre)+.2))
      buttons = Button(**node.style_folded), Button(visible=False,**node.style_unfolded)
      for i,b in enumerate(buttons): b.dual = buttons[1-i]; b.cell = s; b.on_click(nav); yield b
      if pre:
        yield HTML('<div style="padding-left:1mm;"><span style="background-color: lavender;" title="{}">{}</style></div>'.format('.'.join(pre),pre[-1]))
      x = node.summary
      try:
        if isinstance(x,tuple):
          t = htmlsafe(x[0]); x = x[1:]
          if x: x = ' '.join('<span style="{}"><em>{}</em>: {}</span>'.format(node.style_summary,k,htmlsafe(v)) for k,v in x)
          else: x = ''
          x = '<b>{}</b> {}'.format(t,x)
        elif not isinstance(x,str): raise Exception('Unknown summary format')
      except Exception as e: x = '<span style="color: red;">{}</span>'.format(e)
      yield HTML('<div style="padding-left:3mm;">{}</div>'.format(x))
    def nav(b):
      b.cell[0] = not b.cell[0]
      b.visible = False
      b.dual.visible = True
      display(next(wdgs))
    def wdggen():
      while True:
        wdg = VBox(children=tuple(lines(self,())))
        yield wdg
        wdg.close()
    exp = {}
    wdgs = wdggen()
    display(next(wdgs))

#==================================================================================================
class ObjTree (Tree):
  r"""
Objects of this class are :class:`Tree` instances which represent arbitrary objects as trees.

:param obj: an arbitrary object

Methods:
  """
#==================================================================================================

  Register = [],[]
  @staticmethod
  def register(spec,Register=Register):
    r"""
:param spec: the type of objects to which the hook applies
:type spec: :class:`type`\|\ :class:`str`

Registers a function as a hook. Use as decorator. Example::

  @ObjTree.register('numpy.ndarray')
  def get_children(obj): ...

will replace the default :attr:`get_children` function attribute at creation time, whenever the argument *obj* is of class :class:`numpy.ndarray` (only if module :mod:`numpy` is already loaded). Note that it is important to keep :attr:`get_children` as a function and not set the :attr:`children` attribute directly, so offspring computation is done only on demand (otherwise, it may loop).
    """
    if isinstance(spec,str): i,spec = 1,tuple(spec.rsplit('.',1))
    elif isinstance(spec,type): i=0
    else: raise Exception('Argument must be a string or a class')
    return lambda f,i=i,spec=spec: Register[i].append((f.__name__,spec,f))

  def __init__(self,obj):
    from sys import modules
    L0,L1 = self.Register
    for i,(a,(mod,klass),f) in reversed(list(enumerate(L1))): # update independent of self,obj
      m = modules.get(mod)
      if m is not None: L0.insert(0,(a,getattr(m,klass),f)); del L1[i]
    self.obj = obj
    for a,klass,f in L0:
      if isinstance(obj,klass): setattr(self,a,f)

  @staticmethod
  def get_children(obj):
    r"""
Returns the offspring of an arbitrary object *obj* as an iterator of pairs, each with a string label and a :class:`Tree` instance. By default, returns the sorted content of the attribute dictionary of *obj* (if any), where each value is encapsulated in an :class:`ObjTree` instance. Can be overriden by subclassing, or by registering a hook using :meth:`register`.
    """
    if hasattr(obj,'__dict__'):
      for k,x in sorted(obj.__dict__.items()): yield k,ObjTree(x)

  @staticmethod
  def get_summary(obj):
    r"""
Returns the description of an arbitrary object *obj* as an iterator of pairs, each with a string label and an arbitrary value. By default, returns the :func:`len` of *obj* if it is a sequence, or its signature if it is a routine, nothing otherwise. Can be overriden by subclassing, or by registering a hook using :meth:`register`.
    """
    import inspect
    if inspect.isroutine(obj):
      yield 'signature',inspect.signature(obj)
    else:
      try: yield 'len',len(obj)
      except: pass

  @ondemand
  def children(self): return tuple(self.get_children(self.obj))
  @ondemand
  def summary(self):
    t = '{0.__module__}.{0.__qualname__}'.format(type(self.obj))
    return (t,)+tuple(self.get_summary(self.obj))

@ObjTree.register('numbers.Number')
def get_children(obj): return ()
@ObjTree.register('numbers.Number')
def get_summary(obj): yield 'value',obj

@ObjTree.register('pandas.core.base.PandasObject')
def get_children(obj):
  if hasattr(obj,'columns'):
    for c in obj.columns: yield c,ObjTree(obj[c])
@ObjTree.register('pandas.core.base.PandasObject')
def get_summary(obj):
  if hasattr(obj,'shape'): yield 'shape', obj.shape
  if hasattr(obj,'dtype'): yield 'dtype', obj.dtype

@ObjTree.register('numpy.ndarray')
def get_children(obj): return ()
@ObjTree.register('numpy.ndarray')
def get_summary(obj): yield 'shape',obj.shape; yield 'dtype',obj.dtype

@ObjTree.register('scipy.sparse.base.spmatrix')
def get_children(obj): return ()
@ObjTree.register('scipy.sparse.base.spmatrix')
def get_summary(obj):
  n = 1
  for d in obj.shape: n *= d
  yield 'shape',obj.shape
  yield 'density',obj.nnz/n

#==================================================================================================
def ipybrowse(D,start=1,pgsize=10):
  r"""
:param D: a sequence object (must have :func:`len`\ ).
:param start: the index of the initial page
:param pgsize: the size of the pages

A simple utility to browse sliceable objects page per page in IPython.
  """
#==================================================================================================
  from IPython.display import display
  from ipywidgets.widgets.interaction import interact
  P = (len(D)-1)//pgsize + 1
  if start<1: start = 1
  if start>P: start = P
  if P==1: display(D)
  else: interact((lambda page=start:display(D[(page-1)*pgsize:page*pgsize])),page=(1,P))

#==================================================================================================
def ipylist(name,columns,fmt=None):
  r"""
:param name: name of the type
:type name: :class:`str`
:param columns: a tuple of column names
:type columns: :class:`tuple`\ (\ :class:`str`)
:param fmt: a formatting function

Returns a subclass of :class:`list` with an IPython pretty printer for columns. Function *fmt* takes an object and a column name (from *columns*) as input and returns the string representation of that column for that object. It defaults to :func:`getattr`.
  """
#==================================================================================================
  from lxml.builder import E
  from lxml.etree import tounicode
  if fmt is None: fmt = getattr
  t = type(name,(list,),{})
  t._repr_html_ = lambda self,columns=columns,fmt=fmt: tounicode(E.TABLE(E.THEAD(E.TR(*(E.TH(c) for c in columns))),E.TBODY(*(E.TR(*(E.TD(str(fmt(x,c))) for c in columns)) for x in self))))
  t.__getitem__ = lambda self,s,t=t: t(super(t,self).__getitem__(s)) if isinstance(s,slice) else super(t,self).__getitem__(s)
  return t

#==================================================================================================
def htmlsafe(x):
  r"""
Returns an HTML safe representation of *x*.
  """
#==================================================================================================
  return x._repr_html_() if hasattr(x,'_repr_html_') else str(x).replace('<','&lt;').replace('>','&gt;')

#==================================================================================================
class SQliteHandler (logging.Handler):
  r"""
:param conn: a connection to the database

A logging handler class which writes the log messages into a SQLite database.
  """
#==================================================================================================
  def __init__(self,conn,*a,**ka):
    self.conn = conn
    super(SQliteHandler,self).__init__(*a,**ka)
  def emit(self,rec):
    self.format(rec)
    self.conn.execute('INSERT INTO Log (level,created,module,funcName,message) VALUES (?,?,?,?,?)',(rec.levelno,rec.created,rec.module,rec.funcName,rec.message))
    self.conn.commit()

#==================================================================================================
class SQliteStack:
  r"""
Objects of this class are aggregation functions which simply collect results in a list, for use with a SQlite database. Example::

   with sqlite3.connect('/path/to/db') as conn:
     rstack = SQliteStack(conn,'stack',2)
     for school,x in conn.execute('SELECT school,stack(age,height) FROM DataTable GROUP BY school'):
       print(school,rstack(x))

prints pairs *school*, *L* where *L* is a list of pairs *age*, *height*.
  """
#==================================================================================================
  contents = {}
  def __init__(self): self.content = []
  def step(self,*a): self.content.append(a)
  def finalize(self): n = id(self.content); self.contents[n] = self.content; return n
  @staticmethod
  def setup(conn,name,n):
    conn.create_aggregate(name,n,SQliteStack)
    return SQliteStack.contents.pop

#==================================================================================================
def SQliteNew(path,schema):
  r"""
Makes sure the file at *path* is a SQlite3 database with schema exactly equal to *schema*.
  """
#==================================================================================================
  import sqlite3
  if isinstance(schema,str): schema = list(sql.strip() for sql in schema.split('\n\n'))
  with sqlite3.connect(path,isolation_level='EXCLUSIVE') as conn:
    S = list(sql for sql, in conn.execute('SELECT sql FROM sqlite_master WHERE name NOT LIKE \'sqlite%\''))
    if S:
      if S!=schema: raise Exception('database has a version conflict')
    else:
      for sql in schema: conn.execute(sql)

#==================================================================================================
def type_annotation_autocheck(f):
  r"""
Decorator which uses type annotations in the signature of *f* to include a type-check in it. Each argument in the signature can be annotated by

* a type
* a set, taken to be an enumeration of all the allowed values
* a callable, passed the value of the argument and expected to return a boolean indicating whether it is valid
* a tuple thereof, interpreted disjunctively

The doc string of function *f* is also modified so that each sphinx ``:param:`` declaration is augmented with the corresponding ``:type:`` declaration (whenever possible).
  """
#==================================================================================================
  from functools import update_wrapper
  import re
  def addtype(m):
    t = argcheck.get(m.group(2))
    if t is None: return m.group(0)
    return '{}:type {}: {}{}'.format(m.group(1),m.group(2),t.rst,m.group(0))
  check,argcheck = type_annotation_checker(f)
  def F(*a,**ka):
    check(*a,**ka)
    return f(*a,**ka)
  F = update_wrapper(F,f)
  if F.__doc__ is not None: F.__doc__ = re.sub(r'(\s*):param\s+(\w+)\s*:',addtype,F.__doc__)
  return F

#==================================================================================================
def type_annotation_checker(f):
#==================================================================================================
  import inspect
  def nstr(c): return c.__qualname__ if c.__module__ in ('builtins',f.__module__) else c.__module__+'.'+c.__qualname__
  def nrst(c): return ':class:`{}`'.format(nstr(c))
  def ravel(c):
    for cc in c:
      if isinstance(cc,tuple): yield from ravel(cc)
      else: yield cc
  def testera(c):
    if isinstance(c,set):
      t = lambda v,c=c: v in c
      t.str = t.rst = str(c)
    elif callable(c):
      t = lambda v,c=c: c(v)
      t.str,t.rst = nstr(c),nrst(c)
    else: raise TypeError('Expected type|set|callable|tuple thereof')
    return t
  def tester(c):
    if isinstance(c,tuple):
      c = [(cc,(None if isinstance(cc,type) else testera(cc))) for cc in ravel(c)]
      T = tuple(tt for cc,tt in c if tt is not None)
      C = tuple(cc for cc,tt in c if tt is None)
      if C:
        if T: t = lambda v,C=C,T=T: isinstance(v,C) or any(tt(v) for tt in T)
        else: t = lambda v,C=C: isinstance(v,C)
      else: t = lambda v,T=T: any(tt(v) for tt in T)
      t.str = '|'.join((nstr(cc) if tt is None else tt.str) for cc,tt in c)
      t.rst = r'\|\ '.join((nrst(cc) if tt is None else tt.rst) for cc,tt in c)
    elif isinstance(c,type):
      t = lambda v,c=c: isinstance(v,c)
      t.str,t.rst = nstr(c),nrst(c)
    else: t = testera(c)
    return t
  sig = inspect.signature(f)
  argcheck = dict((k,tester(p.annotation)) for k,p in sig.parameters.items() if p.annotation is not p.empty)
  def check(*a,**ka):
    b = sig.bind(*a,**ka)
    for k,v in b.arguments.items():
      t = argcheck.get(k)
      if t is not None and not t(v): raise TypeError('Argument {} failed to match {}'.format(k,t.str))
  return check, argcheck

#==================================================================================================
def set_qtbinding(b=None):
  r"""
:param b: name of the qt interface
:type b: 'pyside'\|\ 'pyqt4'

Declares a module :mod:`pyqt` in the package of this file equivalent to :mod:`PySide` or :mod:`PyQt4` (depending on the value of *b*).
  """
#==================================================================================================
  import sys
  def pyside():
    import PySide
    from PySide import QtCore, QtGui
    QtCore.pyqtSignal = QtCore.Signal
    QtCore.pyqtSlot = QtCore.Slot
    return PySide
  def pyqt4():
    import PyQt4
    from PyQt4 import QtCore, QtGui
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    QtGui.QFileDialog.getOpenFileName = QtGui.QFileDialog.getOpenFileNameAndFilter
    QtGui.QFileDialog.getOpenFileNames = QtGui.QFileDialog.getOpenFileNamesAndFilter
    QtGui.QFileDialog.getSaveFileName = QtGui.QFileDialog.getSaveFileNameAndFilter
    return PyQt4
  if b is None:
    try: mod = pyside()
    except ImportError:
      try: mod = pyqt4()
      except ImportError:
        raise ImportError('No QT binding found (PyQt4 or PySide)',name='pyqt')
  else:
    b = b.lower()
    if b=='pyside': mod = pyside()
    elif b=='pyqt4': mod = pyqt4()
    else: raise ValueError('QT binding must be \'pyside\' or \'pyqt4\'')
  sys.modules[__name__+'.pyqt'] = mod
  return mod

#==================================================================================================
def set_gitexecutable(*exes):
  r"""
:param exes: possible git executable paths
:type exes: tuple(\ :class:`str`)

Sets the git executable path in module :mod:`git` to the first existing path from *exes*.
  """
#==================================================================================================
  from git.cmd import Git
  import os
  for ex in exes:
    if os.path.exists(ex):
      Git.GIT_PYTHON_GIT_EXECUTABLE = ex
      return ex
#==================================================================================================
def gitcheck(pkgname):
  r"""
:param pkgname: full name of a package
:type pkgname: :class:`str`

Assumes that *pkgname* is the name of a python package contained in a git repository which is a local, passive copy of a remote repository. Checks that the package is up-to-date, and updates it if needed using the git pull operation. Call before loading the package...
  """
#==================================================================================================
  from git import Repo, InvalidGitRepositoryError
  from importlib.machinery import PathFinder
  def check(path):
    try: r = Repo(path)
    except InvalidGitRepositoryError: return
    if r.is_dirty(): return 'target', r
    rr = Repo(r.remote().url)
    if rr.is_dirty(): return 'source', rr
    if r.commit() != rr.commit():
      logger.info('Synching (git pull) %s ...',r)
      r.remote().pull()
    if r.commit() != rr.commit(): raise Exception('Unsuccessful git synching')
  p = PathFinder.find_spec(pkgname)
  for path in p.submodule_search_locations:
    c = check(path)
    if c is not None: return c


#==================================================================================================
def iso2date(iso):
  r'''
:param iso: triple as returned by :meth:`datetime.isocalendar`

Returns the :class:`datetime` instance for which the :meth:`datetime.isocalendar` method returns *iso*.
  '''
#==================================================================================================
  from datetime import date,timedelta
  isoyear,isoweek,isoday = iso
  jan4 = date(isoyear,1,4)
  iso1 = jan4-timedelta(days=jan4.weekday())
  return iso1+timedelta(days=isoday-1,weeks=isoweek-1)

#==================================================================================================
def size_fmt(size,binary=True,precision=4,suffix='B'):
  r"""
:param size: a positive number representing a size
:type size: :class:`int`
:param binary: whether to use IEC binary or decimal convention
:type binary: :class:`bool`
:param precision: number of digits displayed (at least 4)
:type precision: :class:`int`

Returns the representation of *size* with IEC prefix. Each prefix is ``K`` times the previous one for some constant ``K`` which depends on the convention: ``K``\ =1024 with the binary convention (marked with an ``i`` before the prefix); ``K``\ =1000 with the decimal convention.
  """
#==================================================================================================
  thr,mark = (1024.,'i') if binary else (1000.,'')
  fmt = '{{:.{}g}}{{}}{}{}'.format(precision,mark,suffix).format
  if size<thr: return str(size)+suffix
  size /= thr
  for prefix in 'KMGTPEZ':
    if size < thr: return fmt(size,prefix)
    size /= thr
  return fmt(size,'Y') # :-)

#==================================================================================================
def time_fmt(time):
  r"""
:param time: a number representing a time in seconds
:type time: :class:`int`\|\ :class:`float`

Returns the representation of *time* in one of days,hours,minutes,seconds (depending on magnitude).
  """
#==================================================================================================
  if time < 60.: return '{:.0f}sec'.format(time)
  time /= 60.
  if time < 60.: return '{:.1f}min'.format(time)
  time /= 60.
  if time < 24.: return '{:.1f}hr'.format(time)
  time /= 24.
  return '{:.1f}day'.format(time)

