from collections.abc import MutableMapping

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
    except ImportError: mod = pyqt4()
  else:
    b = b.lower()
    if b=='pyside': mod = pyside()
    elif b=='pyqt4': mod = pyqt4()
    else: raise ValueError('QT binding must be \'pyside\' or \'pyqt4\'')
  sys.modules[__name__+'.qtbinding'] = mod
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

