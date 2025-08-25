# File:                 sql.py
# Creation date:        2025-08-21
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some SQL utilities based on sqlalchemy
#

from __future__ import annotations
import logging; logger = logging.getLogger(__name__)
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import collections, sqlalchemy
from sqlalchemy import Engine, MetaData, create_engine
from datetime import datetime

#==================================================================================================
def SQLinit(engine:str|Engine,meta:MetaData)->Engine:
  r"""
:param engine: a sqlalchemy engine (or its url)
:param meta: a sqlalchemy metadata structure

* When the database is empty, a ``Metainfo`` table with a single row matching exactly the :attr:`info` attribute of *meta* is created, then the database is populated using *meta*.

* When the database is not empty, it must contain a ``Metainfo`` table with a single row matching exactly the :attr:`info` attribute of *meta*, otherwise an exception is raised.
  """
#==================================================================================================
  from sqlalchemy import Table, Column, event
  from sqlalchemy.types import DateTime, Text, Integer
  from sqlalchemy.sql import select, insert, update, delete, and_
  if isinstance(engine,str): engine = create_engine(engine)
  if engine.driver == 'pysqlite':
    # fixes a bug in the pysqlite driver; to be removed if fixed
    # see https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#transactions-with-sqlite-and-the-sqlite3-driver
    def do_connect(conn,rec): conn.autocommit = False
    event.listen(engine,'connect',do_connect)
  meta_ = MetaData()
  meta_.reflect(bind=engine)
  if meta_.tables: # engine has some tables
    try:
      with engine.connect() as conn: metainfo, = conn.execute(select(meta_.tables['Metainfo'])).mappings()
      metainfo = dict(metainfo); del metainfo['created']
    except: raise SQLinit.MetainfoException('Metainfo table not found or wrong format')
    for k,v in meta.info.items():
      if (v_:=metainfo.get(k)) != str(v):
        raise SQLinit.MetainfoException(f'Metainfo field mismatch {k}[expected:{v},found:{v_}]')
    # assumes that this is sufficient to entail that meta is entirely contained in meta_
  else: # engine is empty
    metainfo_table = Table(
      'Metainfo',meta_,
      Column('created',DateTime()),
      *(Column(key,Text()) for key in meta.info)
    )
    meta_.create_all(bind=engine) # creates just the Metainfo table
    with engine.begin() as conn: conn.execute(insert(metainfo_table).values(created=datetime.now(),**meta.info))
    meta.create_all(bind=engine) # creates all the other tables defined in meta
  return engine
SQLinit.MetainfoException = type('MetainfoException',(Exception,),{})

#==================================================================================================
class SQLupgrade:
  r"""
:param modules: schema modules for the new and old database
:param urls: urls for the new and old database
:param name_to_old_name: a renaming of the new db into the old db (see below)

An instance of this class is a context which helps transfer some database from an old schema to a new (created) one. A schema module must have at least the following members:

  * ``__version__``: the version of the schema as an :class:`int`
  * ``Base``: a subclass of class :class:`sqlalchemy.orm.DeclarativeBase`
  * ``get_sessionmaker``: a callable which takes one input (typically a url) and returns a session maker (typically obtained by calling function :func:`sqlalchemy.orm.sessionmaker` on an engine bound to that url)
  * all the mapped tables, as subclasses of ``Base``

Each key in *name_to_old_name* should be a table name of the new db, and its value should be either a table name in the old db, or a dictionary mapping new column names in the new table to old ones in the old table (or a pair thereof). By default, tables in the new db map to old tables with the same name (if any), and columns of the new table map to columns in the old table if they have the same name and python type (or, in the case of :class:`Enum` types, strictly equivalent ones).
  """
#==================================================================================================
  session:sqlalchemy.orm.Session
  r"""The session opened for access to the new database (only when the context is entered)"""
  old_session:sqlalchemy.orm.Session
  r"""The session opened for access to the old database (only when the context is entered)"""
  transfer: Callable[[],None]
  r"""Used to initiate the transfer from the old to the new database"""
  register: Callable[[str,str,Callable[[Any],None]],None]
  r"""Registers an entry in the converter (the arguments are a table and column name of the new db)"""

  def __init__(self,modules,urls,*,versions=None,name_to_old_name:Mapping[str,str|dict[str,str]|tuple[str,dict[str,str]]]|None=None):
    assert len(modules) == 2
    assert len(urls)==2 and all(isinstance(url,str) for url in urls)
    if versions is not None: assert len(versions)==2 and all((v is None or mod.__version__==v) for mod,v in zip(modules,versions))
    if name_to_old_name is None: name_to_old_name = {}
    def old_name_of(name:str,cname:str|None=None):
      x = name_to_old_name.get(name)
      old_name,D = (name,{}) if x is None else (x,{}) if isinstance(x,str) else (name,x) if isinstance(x,dict) else x
      return old_name if cname is None else (old_name,D.get(cname,cname))
    copy = (lambda x:x); copy.__doc__ = 'COPY'
    def converter(old_t,t,enum_d=(lambda e:{x.name:x.value for x in e})):
      from sqlalchemy import Enum
      if t.python_type == old_t.python_type: return copy
      if isinstance(t,Enum) and isinstance(old_t,Enum):
        old_e,e = (t_.enum_class for t_ in (old_t,t))
        if enum_d(old_e)==enum_d(e):
          old_n,n = (f'{e_.__module__}.{e_.__qualname__}' for e_ in (old_e,e))
          return translate({None:None}|{old_x:e(old_x.value) for old_x in old_e},doc=f'TRANSLATE_ENUM[{old_n}->{n}]')
      return None
    def retrieve(old_cname,conv,doc):
      if conv is copy: F = (lambda old: getattr(old,old_cname))
      else: F = (lambda old: conv(getattr(old,old_cname)))
      F.__doc__ = doc ; return F
    Mapper,OldMapper = ({m.class_.__tablename__:m for m in mod.Base.registry.mappers} for mod in modules)
    TR = {
      name:(mapper,old_mapper,{
        cname:retrieve(old_cname,conv,f'{old_cname}|{conv.__doc__}') for cname,c in mapper.columns.items()
        if (old_c:=old_mapper.columns.get(old_cname:=old_name_of(name,cname)[1])) is not None and (conv:=converter(old_c.type,c.type)) is not None
      })
      for name,mapper in Mapper.items()
      if (old_mapper:=OldMapper.get(old_name:=old_name_of(name))) is not None
    }
    self._Sessions = tuple(mod.get_sessionmaker(u) for mod,u in zip(modules,urls))
    def transfer(dryrun=False):
      if dryrun:
        for name in Mapper:
          x = TR.get(name)
          if x is None: print(f'*** {name}: NO_TRANSFER')
          else:
            mapper,old_mapper,tr = x
            print(f'*** {name}: TRANSFERRED[{old_mapper.class_.__tablename__}]')
            for c in mapper.columns:
              F = tr.get(c.name)
              assert F is not None, f'Missing transfer function for {c.name}'
              print(f'  {c.name}: {F.__doc__}')
        return
      else:
        for name,(mapper,old_mapper,tr) in TR.items():
          for c in mapper.columns: assert c.name in tr, f'Missing transfer function for {name}:{c.name}'
          for old in self.old_session.query(old_mapper.class_): self.session.add(mapper.class_(**{cname:F(old) for cname,F in tr.items()}))
    self.transfer = transfer
    def register(name:str,cname:str,f:Callable[[Any],Any]|None=None,conv:Callable[[Any],Any]|None=None):
      assert cname in Mapper[name].columns
      old_name,old_cname = old_name_of(name,cname)
      if f is None:
        assert conv is not None
        F = retrieve(old_cname,conv,f'{old_cname}|{conv.__doc__ or repr(conv)}')
      else:
        assert conv is None
        if f.__doc__: F = f
        else: F = (lambda old: f(old)); F.__doc__ = repr(f)
      TR[name][2][cname] = F
    self.register = register
  def __enter__(self):
    self.session,self.old_session = (Session() for Session in self._Sessions)
    return self
  def __exit__(self,exc,*a):
    if exc is None: self.session.commit()
    for s in self.session,self.old_session: s.close()
    del self.session,self.old_session

def translate(m:Mapping[Any,Any],doc:str|None=None)->Callable[[Any],Any]:
  f = (lambda v:m[v]); f.__doc__ = f'TRANSLATE[{','.join(f'{k!r}->{v!r}' for k,v in m.items())}]' if doc is None else doc; return f

#==================================================================================================
class SQLHandler (logging.Handler):
  r"""
:param engine: a sqlalchemy engine (or its url)
:param label: a text label

A logging handler class which writes the log messages into a database.
  """
#==================================================================================================
  def __init__(self,engine:str|sqlalchemy.Engine,label:str,*a,**ka):
    from datetime import datetime
    from sqlalchemy.sql import select, insert, update, delete, and_
    meta = self.metadata()
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
    super().__init__(*a,**ka)

  def emit(self,rec):
    self.format(rec)
    self.dbrecord(rec)

  #--------------------------------------------------------------------------------------------------
  @staticmethod
  def metadata(info=tuple({'origin':__name__+'.SQLHandler','version':1}.items()))->sqlalchemy.MetaData:
  #--------------------------------------------------------------------------------------------------
    from sqlalchemy import Table, Column, ForeignKey, MetaData
    from sqlalchemy.types import DateTime, Text, Integer
    meta = MetaData(info=dict(info))
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
class ormsroot (collections.abc.MutableMapping):
  r"""
Instances of this class implement very simple persistent object managers based on the sqlalchemy ORM. This class should not be instantiated directly.

Each subclass *C* of this class should define a class attribute :attr:`base` assigned an sqlalchemy declarative persistent class *B*. Once this is done, a specialised session maker can be obtained by invoking class method :meth:`sessionmaker` of class *C*, with the url of an sqlalchemy supported database as first argument, the other arguments being the same as those for :meth:`sqlalchemy.orm.sessionmaker`. This sessionmaker will produce sessions with the following characteristics:

* they are attached to an sqlalchemy engine at this url (note that engines are reused across multiple sessionmakers with the same url)

* they have a :attr:`root` attribute, pointing to an instance of *C*, which acts as a mapping where the keys are the primary keys of *B* and the values the corresponding ORM entries. The root object has a convenient ipython HTML representation.

Class *C* should provide a method to insert new objects in the persistent class *B*, and they will then be reflected in the session root (and made persitent on session commit). Example::

   from sqlalchemy.ext.declarative import declarative_base; Base = declarative_base()
   from sqlalchemy import Column, Text, Integer

   class Entry (Base): # example of sqlalchemy declarative persistent class definition
     __tablename__ = 'Entry'
     oid = Column(Integer(),primary_key=True)
     name = Column(Text())
     age = Column(Integer())
     def __repr__(self): return 'Entry<{0.oid},{0.name},{0.age}>'.format(self)

   class Root(ormsroot): # manager for class Entry
     base = Entry

   sessionmaker = Root.sessionmaker

Example of use (assuming :func:`sessionmaker` as above has been imported)::

   Session = sessionmaker('sqlite://') # memory db for the example

   from contextlib import contextmanager
   @contextmanager
   def mysession(): # basic sessions as simple block of instructions
     s = Session(autocommit=True)
     try: yield s
     finally: s.close()

   with mysession() as s: assert len(s.root)==0

   with mysession() as s:
     # first session, define two entries. Use key None in assignment because Entry is autokey
     self.root[None] = Entry(name='jack',age=42); self.root[None] = Entry(name='jill',age=29)
     assert len(s.root) == 2 and s.root[1].name=='jack' and s.root[2].name=='jill'

   with mysession() as s: # second session, possibly on another process (if not memory db)
     jack = s.root.pop(1) # also removes from db
     assert len(s.root) == 1 and jack.name=='jack'

   with mysession() as s: # But of course direct sqlalchemy operations are available
     assert len(s.root) == 1
     x, = list(s.query(Entry.name).filter(Entry.age>25))
   assert x.name == 'jill'
    """
#==================================================================================================

  base = None # must be defined in subclasses, NOT at the instance level

  def __init__(self,session):
    self.pk = pk = self.base.__table__.primary_key.columns.values()
    if len(pk) == 1:
      getpk = lambda r,kn=pk[0].name: getattr(r,kn)
      def setpk(k,r,kn=pk[0].name):
        if k is not None: r[kn] = k # does nothing if k is None (autokeys)
        return r
    else:
      getpk = lambda r,kns=tuple(c.name for c in pk): tuple(getattr(r,kn) for kn in kns)
      def setpk(ks,r,kns=tuple(c.name for c in pk)):
        assert len(ks) in (0,)
        for kn,k in zip(kns,ks): r[kn] = k # does nothing if ks==() (autokeys)
        return r
    self.getpk,self.setpk = getpk,setpk
    self.session = session

  def __getitem__(self,k):
    r = self.session.query(self.base,k)
    if r is None: raise KeyError(k)
    return r

  def __delitem__(self,k):
    self.session.delete(self[k])

  def __setitem__(self,k,r):
    assert isinstance(r,self.base)
    self.session.add(self.setpk(k,r))

  def __iter__(self):
    for r in self.session.query(*self.pk): yield self.getpk(r)
  def items(self): return ((self.getpk(r),r) for r in self.session.query(self.base))

  def __len__(self):
    return self.session.query(self.base).count()

  def __hash__(self): return hash((self.session,self.base))

  def __repr__(self): return f'{self.__class__.__name__}<{self.base.__name__}>'

  def _repr_html_(self): from .html import repr_html; return repr_html(self)
  _html_limit = 50
  def as_html(self,_):
    from .html import html_table
    from itertools import islice
    n = len(self)-self._html_limit
    L = self.items(); closing = None
    if n>0: L = islice(L,self._html_limit); closing = f'{n} more'
    return html_table(sorted((k,(v,)) for k,v in L),fmts=(repr,),opening=repr(self),closing=closing)

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
        if isinstance(log,int): log = lambda _lvl=log,_fmt=f'SQA:{evt}%s',**ka:logger.log(_lvl,_fmt,ka)
        event.listen(engine,evt,log,named=True)
    Session_ = sessionmaker(engine,*a,**ka)
    def Session(**x):
      s = Session_(**x)
      s.root = cls(s)
      return s
    return Session
