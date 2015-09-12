from collections.abc import MutableMapping
import logging
logger = logging.getLogger(__name__)

#==================================================================================================
class ondemand (object):
  """
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
  """
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
def zipaxes(L,fig,sharex=False,sharey=False,**ka):
  r"""
Yields the pair of *o* and an axes on *fig* for each item *o* in sequence *L*. The axes are spread more or less uniformly.

:param fig: a figure
:type fig: :class:`matplotlib.figure.Figure`
:param L: an arbitrary sequence
:param sharex: whether all the axes share the same x-axis scale
:type sharex: :class:`bool`
:param sharey: whether all the axes share the same y-axis scale
:type sharey: :class:`bool`
:param ka: passed to :meth:`add_subplot` generating new axes
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
def unfold(tree,exp={}):
  r"""
A simple utility to fold/unfold a tree in IPython.
  """
#==================================================================================================
  from IPython.display import display, clear_output
  from IPython.html.widgets import HBox, VBox, Button, Text, HTML
  def nav(b):
    p,r = b.args
    if r: del exp[p]
    else: exp[p] = True
    clear_output(wait=True)
    w.close()
    display(unfold(tree,exp))
  def button(p,r):
    b = Button(width='10',font_size='xx-small',description='-' if r else '+',background_color='green' if r else 'gray')
    b.args = (p,r)
    b.on_click(nav)
    return b
  def label(p,x):
    if hasattr(x,'_repr_html_'): x = x._repr_html_
    else: x = str(x).replace('<','&lt;').replace('>','&gt;')
    return HTML('<div><b>{}</b> <span style="padding-left:1cm;">{}</span></div>'.format('<span style="color: red; font-size: xx-large;">.</span>'.join(p),x))
  w = VBox(children=tuple(HBox(children=(button(pre,r),label(pre,x))) for pre,x,r in tree(exp=exp)))
  display(w)

#==================================================================================================
def browse(D,start=1,pgsize=10):
  """
A simple utility to browse sliceable objects page per page in IPython.
  """
#==================================================================================================
  from IPython.display import display
  from IPython.html.widgets.interaction import interact
  P = (len(D)-1)//pgsize + 1
  assert start>=1 and start<=P
  if P==1: display(D)
  else: interact((lambda page=start:display(D[(page-1)*pgsize:page*pgsize])),page=(1,P))

#==================================================================================================
class SQliteHandler (logging.Handler):
  r"""
A logging handler which writes the log messages in a database.

:param conn: a connection to the database
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
A aggregation function which simply collects results in a list, for use with a SQlite database. Example::

   with sqlite3.connect('/path/to/db') as conn:
     rstack = SQliteStack(conn,'stack',2)
     for school,x in conn.execute('SELECT school,stack(age,height) FROM DataTable GROUP BY school'):
       x = rstack(x)
       print(school,x)

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
def SQliteNew(path,schema,check=lambda:None):
  r"""
Makes sure the file at *path* is a SQlite3 database with schema exactly equal to *schema*. Returns :const:`None` if successful, otherwise a :class:`str` instance describing the problem. If *check* is present, it must be a callable with no argument, invoked just before the creation of the database (when it does not already exist). If it returns anything but :const:`None`, the creation is aborted.
  """
#==================================================================================================
  import sqlite3
  if isinstance(schema,str): schema = list(sql.strip() for sql in schema.split('\n\n'))
  with sqlite3.connect(path,isolation_level='EXCLUSIVE') as conn:
    S = list(sql for sql, in conn.execute('SELECT sql FROM sqlite_master WHERE name NOT LIKE \'sqlite%\''))
    if S:
      return None if S==schema else 'index has a version conflict'
    else:
      c = check()
      if c is not None: return c
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
  F.__doc__ = re.sub(r'(\s*):param\s+(\w+)\s*:',addtype,F.__doc__)
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
def set_gitexecutable(git):
  """
:param git: git executable path
:type git: :class:`str`

Sets the git command to *git* in module :mod:`git`.
  """
#==================================================================================================
  from git.cmd import Git
  Git.GIT_PYTHON_GIT_EXECUTABLE = git
#==================================================================================================
def gitcheck(pkgname):
  """
:param pkgname: full name of a package
:type pkgname: :class:`str`

Assumes that *pkgname* is a package contained in a git repository which is a local, passive copy of a remote repository. Checks that the package is up-to-date, and updates it if needed using the git pull operation.
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
  for path in p.submodule_search_locations._path:
    c = check(path)
    if c is not None: return c

#==================================================================================================
def infosql(table=None,full=False,driver=None):
  """
:param table: A sql table name
:type table: :class:`str`\|\ :class:`NoneType`

Returns the SQL query needed to extract information on *table*, if specified, or on all the tables otherwise.
  """
#==================================================================================================
  i = Infosql.listing.get(driver.__name__ if hasattr(driver,'__name__') else driver,Infosql)
  if table is None: return i.tables(full)
  else: return i.columns(table,full)

#--------------------------------------------------------------------------------------------------
class Infosql:
  """Generic info class, works with most flavours of SQL. Not meant to be instantiated."""
#--------------------------------------------------------------------------------------------------

  listing = {}

  table_mainfields = 'TABLE_NAME,TABLE_TYPE'
  table_view = 'INFORMATION_SCHEMA.TABLES ORDER BY TABLE_NAME'

  column_mainfields = 'COLUMN_NAME,DATA_TYPE,IS_NULLABLE'

  @classmethod
  def column_view(cls,table):
    return 'INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME=\'{}\' ORDER BY ORDINAL_POSITION'.format(table)

  @classmethod
  def tables(cls,full):
    sel = '*' if full else cls.table_mainfields
    return 'SELECT {} FROM {}'.format(sel,cls.table_view)

  @classmethod
  def columns(cls,table,full):
    sel = '*' if full else cls.column_mainfields
    return 'SELECT {} FROM {}'.format(sel,cls.column_view(table))

  @classmethod
  def declare(cls,c):
    assert issubclass(c,cls)
    cls.listing[c.__name__] = c
    return c

  @classmethod
  def declare_multi(cls,*L):
    assert all(isinstance(x,str) for x in L)
    def tr(c):
      assert issubclass(c,cls)
      for x in L: cls.listing[x] = c
      return c
    return tr

#--------------------------------------------------------------------------------------------------
@Infosql.declare
class sqlite3 (Infosql):
#--------------------------------------------------------------------------------------------------

  @classmethod
  def column(cls,table,full):
    return 'PRAGMA table_info({})'.format(table)

