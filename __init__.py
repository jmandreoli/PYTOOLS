from collections.abc import MutableMapping
import logging

#==================================================================================================
class ondemand (object):
  """
Use as a decorator to declare a computable attribute in a class
which is computed only once (cached value).

Example:

class c:
  def __init__(self,u):
    self.u = u
  @ondemand
  def att(self):
    print 'Some complex computation here'
    return self.u+1
x = c(3)
x.att
>>> Some complex computation here
>>> 4
x.att
>>> 4
x.u=6
x.att
>>> 4
c.att
>>> <....ondemand object at 0x...>
  """
#==================================================================================================

  __slots__ = ('get',)

  def __init__(self,get):
    if get.__code__.co_argcount != 1:
      raise Exception('OnDemand attribute must have a single argument')
    self.get = get

  def __get__(self,instance,owner):
    if instance is None:
      return self
    else:
      val = self.get(instance)
      setattr(instance,self.get.__name__,val)
      return val

#==================================================================================================
class odict (MutableMapping):
  """
Objects of this class act as dict objects, except their keys are also attributes.

Example:

x = odict(a=3,b=6)
x.a
>>> 3
x['a']
>>> 3
del x.a
x
>>> {'b':6}
x.b += 7
x['b']
>>> 13
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
def set_qtbinding(b=None):
  """
:param b: name of the qt interface
:type b: ":const:`pyside`" | ":const:`pyqt4`"

Declares a module "qtbinding" in the package of this file
equivalent to PySide or PyQt4 (depending on the value of *b*\ ).
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
def gitcheck(pkgname):
  """
:param pkgname: full name of a package
:type pkgname: :const:`str`

Assumes that *pkgname* is a package contained in a git repository which is a local, passive copy of a remote repository. Checks that the package is up-to-date.
  """
#==================================================================================================
  from git import Repo, InvalidGitRepositoryError
  from importlib.machinery import PathFinder
  def check(path):
    try: r = Repo(path)
    except InvalidGitRepositoryError: return
    assert not r.is_dirty(), '{} should not be changed; please sync manually.'.format(r)
    rr = Repo(r.remote().url)
    assert not rr.is_dirty(), 'There are uncommitted changes in {}; please commit first.'.format(rr)
    if r.commit() != rr.commit():
      print('Updating (git pull) {} ...'.format(r))
      r.remote().pull()
      print('done')
    assert r.commit() == rr.commit(), 'Updating {} does not seem to have worked :-('.format(r)
  p = PathFinder.find_spec(pkgname)
  for path in p.submodule_search_locations._path: check(path)
#==================================================================================================
def gitsetcmd(git):
  """
:param git: git executable path
:type git: :const:`str`

Sets the git command to *git* in module :const:`git`\ .
  """
#==================================================================================================
  from git.cmd import Git
  Git.GIT_PYTHON_GIT_EXECUTABLE = git

#==================================================================================================
def infosql(table=None,full=False,driver=None):
  """
:param table: A sql table name
:type table: :class:`str` | :class:`NoneType`

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

#==================================================================================================
def addaxes(fig,L,sharex=False,**ka):
  """
:param fig: a figure
:type fig: :class:`matplotlib.figure.Figure`
:param L: an arbitrary sequence (must support :func:`len`\)
:param sharex: if :const:`True`\, all the axes share the same x-axis scale
:type sharex: :class:`bool`
:param **ka: passed to :meth:`add_subplot` generating new axes

Yields an axes on *fig* for each item in sequence *L*\. The axes are spread more or less uniformly.
  """
#==================================================================================================
  from numpy import sqrt,ceil
  N = len(L)
  nc = int(ceil(sqrt(N)))
  nr = int(ceil(N/nc))
  axref = None
  for i,x in enumerate(L,1):
    ax = fig.add_subplot(nr,nc,i,sharex=axref,**ka)
    if sharex and axref is None: axref = ax
    yield x,ax

#==================================================================================================
def browse(D,start=1,pgsize=10):
  """
A simple utility to browse sliceable objects page per page.
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
  with sqlite3.connect(path,isolation_level='EXCLUSIVE') as conn:
    if conn.execute('SELECT * FROM sqlite_master').fetchone() is None:
      c = check()
      if c is not None: return c
      for sql in schema.split('\n\n'): conn.execute(sql.strip())
      conn.execute('CREATE TABLE Version ( schema TEXT )')
      conn.execute('INSERT INTO Version (schema) VALUES (?)',(schema,))
    else:
      try: s, = conn.execute('SELECT schema FROM Version').fetchone()
      except: return 'non conformant index'
      if s!=schema: return 'index has a version conflict'

#==================================================================================================
def type_annotation_autocheck(f):
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
  def nstr(c): return c.__name__ if c.__module__ in ('builtins',f.__module__) else c.__module__+'.'+c.__name__
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

