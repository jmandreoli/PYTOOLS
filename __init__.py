from collections.abc import Mapping, MutableMapping
import logging
logger = logging.getLogger(__name__)

#==================================================================================================
class ondemand:
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
    from inspect import signature
    if any(p for p in signature(get).parameters.values() if p.kind!=p.POSITIONAL_OR_KEYWORD):
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
class odict:
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
  def __init__(self,*a,**ka):
    if a:
      n = len(a)
      if n>1: raise TypeError('expected at most 1 positional argument, got {}'.format(n))
      if ka: raise TypeError('expected no keyword arguments, got {}'.format(len(ka)))
      r = a[0]
    else: r = dict(**ka)
    self.__dict__['_ref'] = r
  def __getitem__(self,k): return self._ref[k]
  def __setitem__(self,k,v): self._ref[k] = v
  def __delitem__(self,k): del self._ref[k]
  def __contains__(self,k): return k in self._ref
  def __eq__(self,other): return self._ref == other
  def __ne__(self,other): return self._ref != other
  def __getattr__(self,a): return self._ref[a]
  def __setattr__(self,a,v): self._ref[a] = v
  def __delattr__(self,a): del self._ref[a]
  def __str__(self): return str(self._ref)
  def __repr__(self): return repr(self._ref)
  def __len__(self): return len(self._ref)
  def __iter__(self): return iter(self._ref)
  def __getstate__(self): return self._ref.copy()
  def __setstate__(self,r): self.__dict__['_ref'] = r

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

  def __getnewargs_ex__(self): return tuple(self)

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
class Process:
  r"""
Instances of this class implement closed symbolic expressions. The value of a process is held in attribute :attr:`value`. It can be computed by invoking method :meth:`incarnate`. Incarnation can be cancelled by invoking method :meth:`reset`. Incarnation is lost on pickling, and is not restored on unpickling. Subclasses should define automatic triggers of the incarnation (see e.g. class :class:`MapProcess`). Initially, a process is mutable, and its configuration can be changed. It becomes immutable (frozen) after the following operations: incarnation (even after it is reset), hashing, and pickling.
  """
#==================================================================================================

  def __init__(self,func,*a,**ka):
    self.reset()
    self.config = [func,a,ka.copy()]
    self.key = None

  def rearg(self,*a,**ka):
    r"""
Updates the config argument of this process. Raises an error if the process is frozen.
    """
    assert self.key is None
    self.config[1] += a
    self.config[2].update(ka)

  def refunc(self,f):
    r"""
Replaces the config function of this process. Raises an error if the process is frozen.
    """
    assert self.key is None
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
    if self.key is None:
      from inspect import getcallargs, unwrap, signature
      func,a,ka = self.config
      f = unwrap(func)
      d = getcallargs(f,*a,**ka)
      self.key = func,tuple((tuple(sorted(d[p.name].items())) if p.kind==p.VAR_KEYWORD else d[p.name]) for p in signature(f).parameters.values())

  def as_html(self,incontext):
    func,a,ka = self.config
    f = VERBATIM('{}.{}{}'.format(func.__module__,func.__name__,('' if self.incarnated else '?' if self.key is None else '!')))
    return html_parlist((f,)+a,sorted(ka.items()),incontext,deco=('[|','|',']'))

  def _repr_html_(self):
    from lxml.etree import tounicode
    return tounicode(html_incontext(self))

  def __getstate__(self):
    self.freeze()
    func,a,ka = self.config
    return func,a,tuple(sorted(ka.items())),self.key
  def __setstate__(self,state):
    self.reset()
    func,a,ka,self.key = state
    self.config = func,a,dict(ka)
  def __hash__(self):
    self.freeze()
    return hash(self.key)
  def __eq__(self,other):
    if not isinstance(other,Process): return False
    return self.config == other.config
  def __repr__(self): return repr(self.value) if self.incarnated else super().__repr__()

class MapProcess (Process,Mapping):
  r"""
Processes of this class are also read-only mappings and trigger incarnation on all the map operations, then delegate such operations to their value.
  """
  def __getitem__(self,k): self.incarnate(); return self.value[k]
  def __len__(self): self.incarnate(); return len(self.value)
  def __iter__(self): self.incarnate(); return iter(self.value)

class CallProcess (Process):
  r"""
Processes of this class are also callables, and trigger incarnation on invocation, then delegate the invocation to their value.
  """
  def __call__(self,*a,**ka): self.incarnate(); return self.value(*a,**ka)

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
def ipylist(name,columns,parse=None):
  r"""
:param name: name of the type
:type name: :class:`str`
:param columns: a tuple of column names
:type columns: :class:`tuple`\ (\ :class:`str`)
:param fmt: a parsing function

Returns a subclass of :class:`list` with an IPython pretty printer for columns. Function *parse* takes an object and a column name (from *columns*) as input and returns the representation of that column for that object. It defaults to :func:`getattr`.
  """
#==================================================================================================
  from lxml.etree import tounicode
  if fmt is None: fmt = getattr
  t = type(name,(list,),{})
  fmts = len(columns)*((lambda s:s),)
  parsec=(lambda x: (parse(x,c) for c in columns))
  t._repr_html_ = lambda self: tounicode(html_table(enumerate(map(parsec(self))),fmts,hdrs=columns))
  t.__getitem__ = lambda self,s,t=t: t(super(t,self).__getitem__(s)) if isinstance(s,slice) else super(t,self).__getitem__(s)
  return t

#==================================================================================================
def ipysetup(D,helpers={},types={},_hstyle='display: inline; padding-left: 3mm; vertical-align: middle',_sort=(lambda x:x),**buttons):
  r"""
Sets values in a dictionary through an interface.

:param D: target dictionary
:type D: :class:`dict`
:param helpers: dictionary of helpers (same keys as target dictionary)
:param types: dictionary of vtypes (same keys as target dictionary)
:param buttons: a dictionary of dictionaries with same keys as the target (see below)
:type buttons: :class:`dict`

A helper is any string. A vtype is either a python basic type (\ :class:`bool`, :class:`str`, :class:`int`, :class:`float`), or a tuple of strings, or an instance of :class:`slice` (possibly with float arguments). When a button is specified by a key and a dict value, a button labeled with that key is displayed, which, when clicked, updates the target dictionary with its value.
  """
#==================================================================================================
  from ipywidgets.widgets import HTML, Text, IntText, FloatText, IntSlider, FloatSlider, Dropdown, Checkbox, Button, HBox, VBox
  from functools import partial
  def upd(ev):
    w = ev['owner']; D[w.description] = w.value
  def upda(b):
    for w in W:
      x = b.data.get(w.description)
      if x is not None: w.value = x
  def ClickButton(label,action,data):
    b = Button(description=label); b.on_click(action); b.data = data; return b
  def row():
    for k,v in _sort(D.items()):
      typ = types.get(k,type(v))
      if typ is bool: F = Checkbox
      elif typ is str: F = Text
      elif typ is int: F = IntText
      elif typ is float: F = FloatText
      elif isinstance(typ,slice):
        slider = IntSlider if isinstance(v,int) else FloatSlider
        F = partial(slider,min=(typ.start or 0),max=typ.stop,step=(typ.step or 0))
      elif isinstance(typ,tuple) or isinstance(typ,list):
        F = partial(Dropdown,options=typ)
      else: raise TypeError('Key: {}'.format(k))
      w = F(description=k,value=v)
      W.append(w)
      w.observe(upd, 'value')
      yield HBox(children=(w,HTML(description='help',value='<span style="{}" title="{}">?</span>'.format(_hstyle,helpers.get(k,'')))))
    if buttons:
      yield HBox(children=[ClickButton(label,upda,data) for label,data in sorted(buttons.items())])
  W = []
  return VBox(children=list(row()))

#==================================================================================================
def html_incontext(x):
#==================================================================================================
  from lxml.builder import E
  def format(L):
    L = sorted(L)
    t = html_table(((k,(x,)) for k,x in L[1:]),((lambda x:x),))
    t.insert(0,E.COL(style='background-color: cyan'))
    return E.DIV(L[0][1],t,style='border:thin solid cyan')
  L = []
  ctx = {}
  def incontext(v):
    if isinstance(v,VERBATIM): return v.verbatim
    try: k = ctx.get(v)
    except: return E.SPAN(repr(v))
    if k is None:
      if hasattr(v,'as_html'):
        ctx[v] = k = len(ctx)
        L.append((k,v.as_html(incontext)))
      else: return E.SPAN(repr(v))
    return E.SPAN('?{}'.format(k),style='color: blue')
  e = incontext(x)
  n = len(L)
  return e if n==0 else L[0][1] if n==1 else format(L)

class VERBATIM:
  def __init__(self,x): self.verbatim = x

#==================================================================================================
def html_parlist(La,Lka,incontext,deco=('','',''),style='display: inline; padding: 5px;'):
#==================================================================================================
  from lxml.builder import E
  opn,sep,cls = deco
  def h():
    yield opn
    for v in La: yield E.SPAN(incontext(v),style=style); yield sep
    for k,v in Lka: yield E.DIV(E.B(k),'=',incontext(v),style=style); yield sep
    yield cls
  return E.DIV(*h(),style='padding:0')

#==================================================================================================
def html_safe(x):
  r"""
Returns an HTML safe representation of *x*.
  """
#==================================================================================================
  return x._repr_html_() if hasattr(x,'_repr_html_') else str(x).replace('<','&lt;').replace('>','&gt;')

#==================================================================================================
def html_table(irows,fmts,hdrs=None,title=None):
  r"""
Returns an HTML table object.

:param irows: a generator of pairs of an object (key) and a tuple of objects (value)
:param fmts: a tuple of format functions matching the length of the value tuples
:param hdrs: a tuple of strings matching the length of the value tuples
:param title: a string
  """
#==================================================================================================
  from lxml.builder import E
  def thead():
    if title is not None: yield E.TR(E.TD(title,colspan=str(1+len(fmts))),style='background-color: gray; color: white')
    if hdrs is not None: yield E.TR(E.TD(),*(E.TH(hdr) for hdr in hdrs))
  def tbody():
    for ind,row in irows:
      yield E.TR(E.TH(str(ind)),*(E.TD(fmt(v)) for fmt,v in zip(fmts,row)))
  return E.TABLE(E.THEAD(*thead()),E.TBODY(*tbody()))

#==================================================================================================
def html_stack(*a,**ka):
  r"""
Returns a stack of HTML objects.

:param a: a list of HTML objects
:param ka: a dictionary of HTML attributes for the DIV encapsulating each object
  """
#==================================================================================================
  from lxml.builder import E
  return E.DIV(*(E.DIV(x,**ka) for x in a))

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
def gitcheck(pkgname):
  r"""
:param pkgname: full name of a package
:type pkgname: :class:`str`

Assumes that *pkgname* is the name of a python package contained in a git repository which is a local, passive copy of a remote repository. Checks that the package is up-to-date, and updates it if needed using the git pull operation. Call before loading the package... Use the ``GIT_PYTHON_GIT_EXECUTABLE`` environment variable to set the Git executable if it is not the default ``/usr/bin/git``.
  """
#==================================================================================================
  from git import Repo
  from importlib.machinery import PathFinder
  from importlib import reload
  from sys import modules
  for path in PathFinder.find_spec(pkgname).submodule_search_locations:
    try: r = Repo(path)
    except OSError: continue
    break
  else:
    raise GitException('target-not-found')
  if r.is_dirty(): raise GitException('target-dirty',r)
  rr = Repo(r.remote().url)
  if rr.is_dirty(): raise GitException('source-dirty',r,rr)
  if r.commit() != rr.commit():
    logger.info('Synching (git pull) %s ...',r)
    r.remote().pull()
    if r.commit() != rr.commit(): raise GitException('synch-failed',r,rr)
    m = modules.get(pkgname)
    if m is not None:
      logger.warn('Reloading %s ...',pkgname)
      reload(m)
class GitException (Exception): pass

#==================================================================================================
def SQLinit(engine,meta):
  r"""
:param engine: a sqlalchemy engine (or its url)
:param meta: a sqlalchemy metadata structure

* When the database is empty, is is populated using *meta*

* When the database is not empty, it must contain a ``Metainfo`` table with a single row matching exactly the ``info`` attribute of *meta*, otherwise an exception is raised.
  """
#==================================================================================================
  from datetime import datetime
  from sqlalchemy import MetaData, Table, Column, create_engine, event
  from sqlalchemy.types import DateTime, Text, Integer
  from sqlalchemy.sql import select, insert, update, delete, and_
  if isinstance(engine,str):
    engine = create_engine(engine)
    if engine.driver == 'pysqlite':
      # fixes a bug in the pysqlite driver; to be removed if fixed
      # see http://docs.sqlalchemy.org/en/rel_1_0/dialects/sqlite.html
      def do_connect(conn,rec): conn.isolation_level = None
      def do_begin(conn): conn.execute('BEGIN')
      event.listen(engine,'connect',do_connect)
      event.listen(engine,'begin',do_begin)
  meta_ = MetaData(bind=engine)
  meta_.reflect()
  if meta_.tables:
    try:
      metainfo_table = meta_.tables['Metainfo']
      metainfo = dict(engine.execute(select((metainfo_table.c))).fetchone())
      del metainfo['created']
    except: raise SQLinitMetainfoException('Not found')
    for k,v in meta.info.items():
      if metainfo.get(k) != str(v):
        raise SQLinitMetainfoException('{}[expected:{},found:{}]'.format(k,v,metainfo.get(k)))
  else:
    metainfo_table = Table(
      'Metainfo',meta_,
      Column('created',DateTime()),
      *(Column(key,Text()) for key in meta.info)
    )
    meta_.create_all()
    engine.execute(insert(metainfo_table).values(created=datetime.now(),**meta.info))
    meta.create_all(bind=engine)
  return engine

class SQLinitMetainfoException (Exception): pass

#==================================================================================================
class SQLHandler (logging.Handler):
  r"""
:param engine: a sqlalchemy engine (or its url)
:param label: a text label 

A logging handler class which writes the log messages into a database.
  """
#==================================================================================================
  def __init__(self,engine,label,*a,**ka):
    from datetime import datetime
    from sqlalchemy.sql import select, insert, update, delete, and_
    meta = SQLHandlerMeta()
    engine = SQLinit(engine,meta)
    session_table = meta.tables['Session']
    log_table = meta.tables['Log']
    with engine.begin() as conn:
      if engine.dialect.name == 'sqlite': conn.execute('PRAGMA foreign_keys = 1')
      conn.execute(delete(session_table).where(session_table.c.label==label))
      c = conn.execute(insert(session_table).values(label=label,started=datetime.now()))
      session = c.inserted_primary_key[0]
      c.close()
    def dbrecord(rec,log_table=log_table,session=session):
      with engine.begin() as conn:
        conn.execute(insert(log_table).values(session=session,level=rec.levelno,created=datetime.fromtimestamp(rec.created),module=rec.module,funcName=rec.funcName,message=rec.message))
    self.dbrecord = dbrecord
    super(SQLHandler,self).__init__(*a,**ka)

  def emit(self,rec):
    self.format(rec)
    self.dbrecord(rec)

def SQLHandlerMetadata(info=dict(origin=__name__+'.SQLHandler',version=1)):
  from sqlalchemy import Table, Column, ForeignKey, MetaData
  from sqlalchemy.types import DateTime, Text, Integer
  meta = MetaData(info=info)
  Table(
    'Session',meta,
    Column('oid',Integer(),primary_key=True,autoincrement=True),
    Column('started',DateTime()),
    Column('label',Text(),index=True,unique=True,nullable=False),
    )
  Table(
    'Log',meta,
    Column('oid',Integer(),primary_key=True,autoincrement=True),
    Column('session',ForeignKey('Session.oid',ondelete='CASCADE'),index=True,nullable=False),
    Column('level',Integer(),nullable=False),
    Column('created',DateTime(),nullable=False),
    Column('module',Text(),index=True,nullable=False),
    Column('funcName',Text(),nullable=False),
    Column('message',Text(),nullable=False),
    )
  return meta

#==================================================================================================
class ormsroot (MutableMapping):
  r"""
Instances of this class implement very simple persistent object managers based on the sqlalchemy ORM. This class should not be instantiated directly.

Each subclass *C* of this class should invoke the following class method: :meth:`set_base` with a single argument *R*, an sqlalchemy declarative persistent class. Once this is done, a specialised session maker can be obtained by invoking method :meth:`sessionmaker` of class *C*, with the url of an sqlalchemy supported database as first argument, the other arguments being the same as those for :meth:`sqlalchemy.orm.sessionmaker`. This sessionmaker will produce sessions with the following characteristics:

* they are attached to an sqlalchemy engine at this url (note that engines are reused across multiple sessionmakers with the same url)

* they have a :attr:`root` attribute, pointing to an instance of *C*, which acts as a dictionary where the keys are the primary keys of *R* and the values the corresponding ORM entries. The root object has a convenient ipython HTML representation. Direct update to the dictionary is not supported, except deletion, which is made persitent on session commit.

Class *C* should provide a method to insert new objects in the persistent class *R*, and they will then be reflected in the session root (and made persitent on session commit). Example::

   from sqlalchemy.ext.declarative import declarative_base; Base = declarative_base()
   from sqlalchemy import Column, Text, Integer

   class Employee (Base):
     __tablename__ = 'Employee'
     oid = Column(Integer(),primary_key=True); name = Column(Text()); position = Column(Text())

   class Root(ormsroot):
     def new(self,name,position): r = Employee(name=name,position=position); self.session.add(r); return r

   Root.set_base(Employee)

   Session = Root.sessionmaker('sqlite://') # memory db for the example
   s = Session() # first session
   jack = s.root.new('jack','manager'); joe = s.root.new('joe','writer')
   s.commit(); s.close() # for persistency; assigns oids (jack:1, joe:2)
   s = Session() # second session (possibly in another process if real db is used)
   jack = s.root.pop(1); joe = s.root[2]
   assert set(s.root) == set([joe.oid])
   s.commit(); s.close()
  """
#==================================================================================================

  def __init__(self,session,dclass):
    self.dclass = dclass
    self.pk = pk = self.dclass.__table__.primary_key.columns.values()
    self.pkindex = (lambda r,k=pk[0].name: getattr(r,k)) if len(pk)==1 else (lambda r,pk=pk: tuple(getattr(r,c.name) for c in pk))
    self.session = session

  def __getitem__(self,k):
    r = self.session.query(self.dclass).get(k)
    if r is None: raise KeyError(k)
    return r

  def __delitem__(self,k):
    self.session.delete(self[k])

  def __setitem__(self,k,v):
    raise Exception('Direct create/update not permitted')

  def __iter__(self):
    for r in self.session.query(*self.pk): yield self.pkindex(r)

  def __len__(self):
    return self.session.query(self.base).count()

  def _repr_html_(self):
    from lxml.etree import tounicode
    return tounicode(html_incontext(self))
  def as_html(self,incontext):
    return html_stack(*(v.as_html(incontext) for k,v in sorted(self.items())))

  cache = {}
  @classmethod
  def sessionmaker(cls,url,execution_options={},*a,**ka):
    from sqlalchemy import event
    from sqlalchemy.orm import sessionmaker
    engine = cls.cache.get(url)
    if engine is None: cls.cache[url] = engine = SQLinit(url,cls.base.metadata)
    if execution_options:
      trace = execution_options.pop('trace',{})
      engine = engine.execution_options(**execution_options)
      for evt,log in trace.items():
        if isinstance(log,int): log = lambda _lvl=log,_fmt='SQA:{}%s'.format(evt),**ka:logger.log(_lvl,_fmt,ka)
        event.listen(engine,evt,log,named=True)
    Session_ = sessionmaker(engine,*a,**ka)
    def Session(**x):
      s = Session_(**x)
      s.root = cls(s,cls.base)
      return s
    return Session

#==================================================================================================
class spark:
  r"""
When class method :meth:`default_init` is invoked, it creates a :class:`pyspark.context.SparkContext` instance with the same keyword arguments as the invocation, and stores it as attribute :attr:`sc`.

By default, method :meth:`init` of this class is identical to method :meth:`default_init`. If a resource ``spark/pyspark.py`` exists in an XDG configuration file, that resource is executed (locally) and should define a function :func:`init` used as method :meth:`init` instead. The defined function can of course use method :meth:`default_init`.
  """
#==================================================================================================
  sc = None

  @classmethod
  def default_init(cls,**ka):
    import atexit
    from pyspark.context import SparkContext
    if cls.sc is not None: cls.sc.stop()
    cls.sc = sc = SparkContext(**ka)
    atexit.register(sc.stop)

  def config():
    try: from xdg.BaseDirectory import load_first_config
    except: return
    p = load_first_config('spark/pyspark.py')
    if p is None: return
    d = {}
    with open(p) as u: exec(u.read(),d)
    return classmethod(d['init'])
  init = config() or default_init
  del config

#==================================================================================================
def config_env(name,deflt=None,asfile=False):
  r"""
:param name: the name of an environment variable
:type name: :class:`str`
:param deflt: an arbitrary python object (optional)

Returns a value obtained by executing the string value of an environment variable in a dictionary, then returning the content of key `value` in that dictionary.
  """
#==================================================================================================
  import os
  x = os.environ.get(name)
  if x is None: return deflt
  if asfile:
    with open(x) as u: x = u.read(x)
  d = {}
  exec(x,d)
  return d['value']

#==================================================================================================
def iso2date(iso):
  r"""
:param iso: triple as returned by :meth:`datetime.isocalendar`

Returns the :class:`datetime` instance for which the :meth:`datetime.isocalendar` method returns *iso*.
  """
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

def loggingBasicConfig(level=None,**ka):
  if isinstance(level,str): level = getattr(logging,level)
  logger = logging.getLogger()
  for h in logger.handlers: logger.removeHandler(h)
  logging.basicConfig(level=level,**ka)
  return logger
