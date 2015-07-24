# File:                 chrono.py
# Creation date:        2014-06-19
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Tools to build event recorders
#
# *** Copyright (c) 2014 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import logging
logger = logging.getLogger(__name__)

import re, sqlite3, traceback, time, threading, pickle
from pathlib import Path
from collections import namedtuple
from collections.abc import MutableMapping
from contextlib import contextmanager
from itertools import islice
from functools import partial
from . import type_annotation_autocheck, SQliteHandler, SQliteStack, SQliteNew

# IMPORTANT:
# In order for the foreign key clauses to operate, one must set
# PRAGMA foreign_keys = ON

SCHEMA = '''
CREATE TABLE Block (
  oid INTEGER PRIMARY KEY AUTOINCREMENT,
  flow PICKLE UNIQUE NOT NULL
  )

CREATE TABLE Session (
  oid INTEGER PRIMARY KEY AUTOINCREMENT,
  block REFERENCES Block (oid) NOT NULL,
  started DATETIME DEFAULT ( datetime('now') ),
  istart INTEGER,
  istop INTEGER,
  istep INTEGER,
  status BOOLEAN
  )

CREATE VIEW Activity AS
  SELECT Block.oid AS block, Session.oid AS current
  FROM Block LEFT JOIN Session
  ON Session.block = Block.oid AND Session.status IS NULL

CREATE INDEX SessionIndex ON Session (block)

CREATE TRIGGER BlockDeleteTrigger
  AFTER DELETE ON Block
  BEGIN
    DELETE FROM Session WHERE block=OLD.oid;
  END
'''

DATA_SCHEMA = '''
CREATE TABLE Field (
  oid INTEGER PRIMARY KEY AUTOINCREMENT,
  sticky BOOLEAN,
  name TEXT
  )

CREATE TABLE Record (
  oid INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp REAL
  )

CREATE TABLE Component (
  record REFERENCES Record(oid) ON DELETE CASCADE,
  field REFERENCES Field(oid) ON DELETE CASCADE,
  value BLOB
  )

CREATE TABLE Log (oid INTEGER PRIMARY KEY AUTOINCREMENT, level INTEGER, created DATETIME, module TEXT, funcName TEXT, message TEXT )

CREATE TABLE Interrupt (oid INTEGER PRIMARY KEY AUTOINCREMENT, cause TEXT)

CREATE INDEX ComponentIndex ON Component (record)

CREATE INDEX ComponentIndex2 ON Component (field)
'''

sqlite3.register_converter('PICKLE',pickle.loads)
sqlite3.enable_callback_tracebacks(True)

#==================================================================================================
class ChronoDB (MutableMapping):
  r"""
Objects of this class manage chrono folders. There is at most one instance of this class in a process for each absolute folder path. A chrono folder contains a set of sessions. A session is an excerpt of an infinite, persistent flow of events. Each session has meta information stored in an sqlite database named ``index.db`` in the folder. The index entry attached to a session describes the flow from which it was obtained. A ``Session`` entry has the following fields:

- ``oid``: a unique identifier of the session, which also allows retrieval of its records;
- ``block``: reference to another entry in the database (table ``Block``) describing the event flow which produced the session;
- ``started``: date when the recording process started;
- ``status``: NULL if the process is ongoing, otherwise :const:`0` if the process terminated normally and :const:`1` otherwise.

``Block`` entries correspond to clusters of ``Session`` entries sharing the same event flow. They have the following fields:

- ``flow``: the persistent flow (iterable) producing all the sessions in the block.

The content of a session is itself recorded in a sqlite3 database of triples (record-field-value) with the following tables.fields:

- ``Field``.\ ``name``: the name of the field (unique in the session);
- ``Field``.\ ``sticky``: a boolean indicating whether the field is sticky; sticky fields are recorded only when they change;
- ``Record``.\ ``timestamp``: the time (in float seconds) elapsed since the beginning of the session;
- ``Component``.\ ``record``: a pointer to a record entry;
- ``Component``.\ ``field``: a pointer to a field entry;
- ``Component``.\ ``value``: the value (any type is accepted, stored as-is).

Furthermore, a :class:`ChronoDB` instance acts as a mapping object, where the keys are flow objects and values are :class:`ChronoBlock` objects for the corresponding blocks.

Finally, :class:`ChronoDB` instances have an HTML ipython display.

Attributes:

.. attribute:: path

   the path of the chrono folder, as a :class:`pathlib.Path` instance

Methods:

.. automethod:: __new__
  """
#==================================================================================================

  def __new__(cls,spec,listing={},lock=threading.Lock()):
    r"""
Generates a :class:`ChronoDB` object.

:param spec: specification of the ChronoDB folder
:type spec: :class:`ChronoDB`\|\ :class:`pathlib.Path`\|\ :class:`str`
    """
    if isinstance(spec,ChronoDB): return spec
    elif isinstance(spec,str): path = Path(spec)
    elif isinstance(spec,Path): path = spec
    else: raise TypeError('Expected: {}|{}|{}; Found: {}'.format(str,Path,ChronoDB,type(spec)))
    path = path.resolve()
    with lock:
      self = listing.get(path)
      if self is None:
        self = super(ChronoDB,cls).__new__(cls)
        self.path = path
        dbpath = str(path/'index.db')
        check = (lambda:'cannot create index in non empty folder') if any(path.iterdir()) else (lambda:None)
        r = SQliteNew(dbpath,SCHEMA,check)
        if r is not None: raise WrongCacheFolderException(path,r)
        listing[path] = self
    return self
  
  def __getnewargs__(self): return str(self.path),

  def dbpath(self,session):
    db = 'index.db' if session is None else 'session-{}.db'.format(session)
    return self.path/db

  def connect(self,session,**ka):
    return sqlite3.connect(str(self.dbpath(session)),**ka)

  def getblock(self,flow):
    flowp = pickle.dumps(flow)
    with self.connect(None,isolation_level='IMMEDIATE') as conn:
      row = conn.execute('SELECT oid FROM Block WHERE flow=?',(flowp,)).fetchone()
      if row is None:
        return conn.execute('INSERT INTO Block (flow) VALUES (?)',(flowp,)).lastrowid
      else:
        return row[0]

#--------------------------------------------------------------------------------------------------
  def start(self,session,flow,slc,**ka):
    r"""
Starts the recording of the slice *slc* of the flow *flow* into the database attached to *session*.

:param session: the id of the session receiving the records
:type session: :class:`int`
:param flow: the flow to record
:type flow: an iterable
:param slc: the slice of the flow to record
:type slc: :class:`slice`

This method is normally called on a remote process (see method :meth:`ChronoBlock.launch`).
    """
#--------------------------------------------------------------------------------------------------
    def getfield(conn,f):
      fnam,sticky = f
      fid = fields.get(fnam)
      if fid is None:
        r = conn.execute('INSERT INTO Field (sticky,name) VALUES (?,?)',(sticky,fnam)).lastrowid
        fid = fields[fnam] = r
      return fid
    fields = {}
    status = 0
    reft = time.perf_counter()
    with self.connect(session) as conn:
      logging.basicConfig(handlers=(SQliteHandler(conn),),**ka)
      try:
        for x in islice(flow,slc.start,slc.stop,slc.step):
          tstamp = time.perf_counter()-reft
          record = conn.execute('INSERT INTO Record (timestamp) VALUES (?)',(tstamp,)).lastrowid
          comps = tuple((record,getfield(conn,f),val) for f,val in x)
          conn.executemany('INSERT INTO Component (record,field,value) VALUES (?,?,?)',comps)
          conn.commit()
          r = conn.execute('SELECT cause FROM Interrupt').fetchone()
          if r is not None: raise InterruptException(r[0])
      except BaseException:
        status = 1
        logger.error(traceback.format_exc())
    with self.connect(None) as conn:
      conn.execute('UPDATE Session SET status=? WHERE oid=?',(status,session))

#--------------------------------------------------------------------------------------------------
# ChronoDB as Mapping
#--------------------------------------------------------------------------------------------------

  def __getitem__(self,block):
    with self.connect(None,detect_types=sqlite3.PARSE_DECLTYPES) as conn:
      r = conn.execute('SELECT flow FROM Block WHERE oid=?',(block,)).fetchone()
    if r is None: raise KeyError(block)
    return ChronoBlock(db=self,flow=r[0],block=block)

  def __delitem__(self,block):
    with self.connect(None) as conn:
      conn.execute('DELETE FROM Block WHERE oid=?',(block,))
      if conn.total_changes==0: raise KeyError(block)

  def __setitem__(self,block,v):
    raise Exception('Direct create/update not permitted on Block')

  def __iter__(self):
    with self.connect(None) as conn:
      for block, in conn.execute('SELECT oid FROM Block'): yield block

  def __len__(self):
    with self.connect(None) as conn:
      return conn.execute('SELECT count(*) FROM Block').fetchone()[0]

  def items(self):
    with self.connect(None,detect_types=sqlite3.PARSE_DECLTYPES) as conn:
      for block,flow in conn.execute('SELECT oid,flow FROM Block'):
        yield block, ChronoBlock(db=self,flow=flow,block=block)

#--------------------------------------------------------------------------------------------------
# Display
#--------------------------------------------------------------------------------------------------

  def _repr_html_(self):
    from lxml.etree import tounicode
    return tounicode(self.as_html())
  def as_html(self):
    return html_stack(*(v.as_html() for k,v in self.items()))
  def __str__(self): return 'ChronoDB<{}>'.format(self.path)

#==================================================================================================
class ChronoBlock (MutableMapping):
  r"""
Instances of this class implements blocks of sessions sharing the same flow.

:param db: specification of the cache where the block resides
:type db: :class:`ChronoDB`\|\ :class:`str`\|\ :class:`pathlib.Path`
:param flow: the block's persistent flow of events
:type flow: iterable

Furthermore, a :class:`ChronrecBlock` object acts as a mapping where the keys are session identifiers (:class:`int`) and values are named tuples with the following attributes (in order):

* :attr:`status`, :attr:`started`, holding the fields ``status`` and ``started``
* :attr:`slice`, obtained from the fields ``istart``, ``istop``, ``istep``
* :attr:`nrec`, :attr:`nfield`, :attr:`log`, :attr:`nint`, the number of records, fields, logs, interruptions, respectively, in the session
* :attr:`streamer`, a function to explore the content of the session

Finally, :class:`ChronoBlock` instances have an HTML ipython display.

Methods:
  """
#==================================================================================================

  def __init__(self,flow=None,db=None,block=None):
    self.db = db = ChronoDB(db)
    self.flow = flow
    self.block = db.getblock(flow) if block is None else block

  @property
  def current(self):
    with self.db.connect(None) as conn:
      return conn.execute('SELECT current FROM Activity WHERE block=?',(self.block,)).fetchone()[0]

#--------------------------------------------------------------------------------------------------
  def activate(self,slc=slice(0,None),**ka):
#--------------------------------------------------------------------------------------------------
    with self.db.connect(None,isolation_level='IMMEDIATE') as conn:
      session, = conn.execute('SELECT current FROM Activity WHERE block=?',(self.block,)).fetchone()
      if session is not None: raise Exception('Block already activated')
      session = conn.execute('INSERT INTO Session (block,istart,istop,istep) VALUES (?,?,?,?)',(self.block,slc.start,slc.stop,slc.step)).lastrowid
      logger.info('session created: %s',session)
    with self.db.connect(session) as conn:
      for sql in DATA_SCHEMA.split('\n\n'): conn.execute(sql.strip())
    self.launch(session,self.flow,slc,**ka)

#--------------------------------------------------------------------------------------------------
  def pause(self,cause):
#--------------------------------------------------------------------------------------------------
    session = self.current
    if session is None: raise Exception('Block not activated')
    with self.db.connect(session) as conn:
      conn.execute('INSERT INTO Interrupt (cause) VALUES (?)',(cause,))

#--------------------------------------------------------------------------------------------------
  def stream(self,session,reverse=False,limit=(-1,0)):
    r"""
Iterates over the records of one session.

:param session: id of the session to list.
:type session: :const:`int`
:param reverse: whether the list order should be latest first instead of the default earliest first
:type reverse: :const:`bool`
:param limit: pair of (max number of rows , offset) limiting the listing
:type limit: (\ :const:`int`,\ :const:`int`\ )
    """
#--------------------------------------------------------------------------------------------------
    def content(conn):
      rstack = SQliteStack.setup(conn,'Stack',2)
      for rec,ts,comps in conn.execute('SELECT Record.oid,Record.timestamp,Stack(Component.field,Component.value) FROM Record,Component WHERE Component.record=Record.oid GROUP BY Record.oid,Record.timestamp ORDER BY Record.timestamp {} LIMIT {} OFFSET {}'.format(('DESC' if reverse else 'ASC'),*limit)):
        comps = dict(rstack(comps))
        yield (rec,ts)+tuple(comps.get(fid) for fid,fnam,sticky in fields)
    with self.db.connect(session) as conn:
      fields = conn.execute('SELECT oid,name,sticky FROM Field ORDER BY sticky DESC, name').fetchall()
      return WrapIterator(content(conn),attributes=('record','timestamp',)+tuple(fnam for fid,fnam,sticky in fields),sticky=dict((fnam,sticky) for fid,fnam,sticky in fields))

#--------------------------------------------------------------------------------------------------
# ChronoBlock as Mapping
#--------------------------------------------------------------------------------------------------

  def __getitem__(self,session):
    with self.db.connect(None) as conn:
      r = conn.execute('SELECT status,started,istart,istop,istep FROM Session WHERE oid=? AND block=?',(session,self.block)).fetchone()
    if r is None: raise KeyError(session)
    return self.full(session,r)

  def __delitem__(self,session):
    with self.db.connect(None) as conn:
      status, = conn.execute('SELECT status FROM Session WHERE oid=? AND block=?',(session,self.block)).fetchone()
      if status is None: raise Exception('Unable to delete an ongoing session')
      conn.execute('DELETE FROM Session WHERE oid=?',(session,))
      self.db.dbpath(session).unlink()

  def __setitem__(self,session,v):
    raise Exception('Direct create/update not permitted on Session')

  def __iter__(self):
    with self.db.connect(None) as conn:
      for session, in conn.execute('SELECT oid FROM Session WHERE block=? ORDER BY started DESC',(self.block,)): yield session

  def __len__(self):
    with self.db.connect(None) as conn:
      return conn.execute('SELECT count(*) FROM Session WHERE block=?',(self.block,)).fetchone()[0]

  def items(self):
    with self.db.connect(None) as conn:
      for row in conn.execute('SELECT oid,status,started,istart,istop,istep FROM Session WHERE block=? ORDER BY started DESC',(self.block,)):
        yield row[0], self.full(row[0],row[1:])

  def full(self,session,row,rtype=namedtuple('Session',('status','started','slice','nrec','nfield','nlog','nint','streamer'))):
    with self.db.connect(session) as conn:
      count = dict(conn.execute('SELECT name,seq FROM sqlite_sequence'))
    return rtype(*(row[:-3]+(slice(*row[-3:]),)+tuple(count.get(k) for k in ('Record','Field','Log','Interrupt'))+(partial(self.stream,session),)))

#--------------------------------------------------------------------------------------------------
# Display
#--------------------------------------------------------------------------------------------------

  def _repr_html_(self):
    from lxml.etree import tounicode
    return tounicode(self.as_html())
  def as_html(self):
    return html_table(((k,v[:-1]) for k,v in self.items()),hdrs=('status','started','slice','nrec','nfield','nlog','nint'),fmts=(str,str,(lambda slc:'{}:{}:{}'.format(slc.start,('' if slc.stop is None else slc.stop),('' if slc.step is None else slc.step))),str,str,str,str),title=str(self.flow))
  def __str__(self): return 'ChronoDB<{}:{}>'.format(self.db.path,self.flow)

#--------------------------------------------------------------------------------------------------
# Base launcher (uses multiprocessing)
#--------------------------------------------------------------------------------------------------
  def launch(self,*a,**ka):
    from multiprocessing import get_context
    get_context('fork').Process(target=self.db.start,args=a,kwargs=ka).start()

#==================================================================================================
class Formatter (object):
  r"""
An object of this class defines a function which extracts information from records.

:param fields: a list of fields (see below)

* A field is a triple (*fmtk*,\ *sticky*,\ *fmtv*) where *sticky* is a Boolean and *fmtk*,\ *fmtv* are two 1-input 1-output functions.
* A record is assumed to be a list (or iterator) of (*key*,\ *value*) pairs.

The information extracted from a record is obtained by filtering and processing the list of (*key*,\ *value*) pairs as follows. For each pair, *fmtk* is evaluated on *key*. If the result is None, the field is not extracted. Otherwise the result is taken to be the field name, and a field value is obtained by applying *fmtv* to *value*. If *sticky* is true and the field value is the same as the last value extracted for that field name, the field is not extracted. Otherwise, the pair of the field spec (pair of the field name and the *sticky* flag) and field value is extracted. The extracted fields are returned as an iterator.

Methods:
  """
#==================================================================================================

  def __init__(self,*fields):
    def lookup(nam,cache={},null=()):
      l = cache.get(nam,null)
      if l is null:
        for fmtk,sticky,fmtv in fields:
          fnam = fmtk(nam)
          if fnam is not None: cache[nam] = l = fnam,sticky,fmtv; break
        else: cache[nam] = l = None
      return l
    self.lookup = lookup
    self.sfields = {}

  def __call__(self,x):
    for nam,val in x:
      l = self.lookup(nam)
      if l is not None:
        fnam,sticky,fmtv = l
        if sticky:
          if val == self.sfields.get(fnam): continue
          self.sfields[fnam] = val
        yield (fnam,sticky),fmtv(val)

  @staticmethod
  @type_annotation_autocheck
  def Field(trig:str,sticky:(bool,set((0,1)))=None,fmtk:(callable,str,type(None))=None,fmtv:(callable,type(None))=None,ID=(lambda x: x)):
    r"""
A convenience function to specify a :class:`Formatter` field.

:param trig: a regular expression
:param sticky: whether the field should be sticky
:param fmtk: 1-input, 1-output function
:param fmtv: 1-input, 1-output function
:rtype: a triple *fmtkr*, *sticky*, *fmtv*

The field name formatting function *fmtkr* of the result field is defined as follows: its argument is matched against the regular expression *trig*; :const:`None` is returned if the match fails, otherwise, the result of applying *fmtk* to the groups extracted by the regular expression. If *fmtk* is unspecified or :const:`None`, identity is assumed, and if *fmtk* is a string, it must be a format string (curly brackets notation) and its formatting function is assumed. If *fmtv* is unspecified or :const:`None`, identity is assumed. For example::

   f = Formatter(
    Formatter.Field(r'\.voltage\.(line-\d+)',fmtv=float),
    Formatter.Field(r'\.name\.line-(\d+)',sticky=True,fmtk='L{}')
    )
   list(f([('.name.line-6','Tom'),('.voltage.line-6','42.0'),('.voltage.line-23','35')]))
   #>>> [(('L6',True),'Tom'),(('line-6',False),42.0),(('line-23',False),35.0)]
   list(f([('.name.line-6','Tom'),('.voltage.line-6','45.0'),('.current.line-6','14')]))
   #>>> [(('line-6',False),45.0)]
   list(f([('.name.line-6','Jerry'),('.voltage.line-6','44.0')]))
   #>>> [(('L6',True),'Jerry'),(('line-6',False),44.0)]
    """
    trig = re.compile(trig)
    if fmtk is None: fmtk = ID
    elif isinstance(fmtk,str): fmtk = fmtk.format
    elif not callable(fmtk): raise TypeError('Expected callable, found {}'.format(type(fmtk)))
    def fmtk(nam,patmatch=trig.fullmatch,fmt=fmtk):
      m = patmatch(nam)
      return None if m is None else fmt(*m.groups()) 
    if sticky is None: sticky = False
    elif not (isinstance(sticky,bool) or sticky in (0,1)): raise TypeError('Expected {}|0|1, found {}'.format(bool,type(sticky)))
    if fmtv is None: fmtv = ID
    elif not callable(fmtv): raise TypeError('Expected callable, found {}'.format(type(fmtv)))
    return fmtk,sticky,fmtv

  @staticmethod
  @type_annotation_autocheck
  def UField(trig:str,fmts:str,unit:str=None,delunit=lambda x:float(x.split()[0]),**ka):
    r"""
A convenience function to specify a :class:`Formatter` field. For example::

   f = Formatter(Formatter.UField(r'\.voltage\.line-(\d+)','L{:0>2}',unit='V'))
   list(f([('.voltage.line-6','42.0 V'),('.voltage.line-23','35 V')]))
   #>>> [(('L06 (V)',False),42.0), (('L23 (V)',False),35.0)]
    """
    if unit is not None: fmts = '{} ({})'.format(fmts,unit)
    return Formatter.Field(trig,fmtk=fmts.format,fmtv=(float if unit is None else delunit),**ka)

#==================================================================================================
# Utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def flatten(x,pre=''):
  r"""
Turns a simple structure *x* built from :const:`dict`, :const:`list` and :const:`tuple` into an iterator of key-value pairs. For example:: 

   list(flatten({'a':{'b':({'ux':3,'uy':4.2},{'ux':5,'uy':6.7}),'c':'abcd'},'d':38.3}))
   #>>> [('.a.b.0.ux',3),('.a.b.0.uy',4.2),('.a.b.1.ux',5),('.a.b.1.uy',6.7),('.a.c','abcd'),('.d',38.3)]
  """
#--------------------------------------------------------------------------------------------------
  if isinstance(x,dict):
    for k,xx in x.items(): yield from flatten(xx,'{}.{}'.format(pre,k))
  elif isinstance(x,(list,tuple)):
    for k,xx in enumerate(x): yield from flatten(xx,'{}.{}'.format(pre,k))
  else: yield pre,x

#--------------------------------------------------------------------------------------------------
def atinterval(it,period):
  r"""
Returns an iterator which enemerates the elements of *it* at regular real time interval.

:param it: iterator
:param period: period (in secs) between two consecutive extractions from *it*
:type period: :const:`float`
  """
#--------------------------------------------------------------------------------------------------
  reft = time.perf_counter()
  nextt = 0
  for x in it:
    yield x
    nextt += period
    d = nextt-(time.perf_counter()-reft)
    if d>0: time.sleep(d)

#--------------------------------------------------------------------------------------------------
def html_table(irows,fmts,hdrs=None,title=None):
#--------------------------------------------------------------------------------------------------
  from lxml.builder import E
  def thead():
    if title is not None: yield E.TR(E.TD(title,colspan=str(1+len(fmts))),style='background-color: gray; color: white')
    if hdrs is not None: yield E.TR(E.TD(),*(E.TH(hdr) for hdr in hdrs))
  def tbody():
    for ind,row in irows:
      yield E.TR(E.TH(str(ind)),*(E.TD(fmt(v)) for fmt,v in zip(fmts,row)))
  return E.TABLE(E.THEAD(*thead()),E.TBODY(*tbody()))
#--------------------------------------------------------------------------------------------------
def html_stack(*a):
#--------------------------------------------------------------------------------------------------
  from lxml.builder import E
  return E.DIV(*(E.DIV(x) for x in a))

#--------------------------------------------------------------------------------------------------
class WrapIterator:
#--------------------------------------------------------------------------------------------------
  def __init__(self,it,**ka): self._it = it; self.__dict__.update(**ka)
  def __iter__(self): return self._it

#--------------------------------------------------------------------------------------------------
class WrongChronoFolderException (Exception):
  """
Exception raised when the :class:`ChronoDB` constructor is called with inappropriate arguments.
  """
#--------------------------------------------------------------------------------------------------
  pass
#--------------------------------------------------------------------------------------------------
class InterruptException (Exception):
#--------------------------------------------------------------------------------------------------
  """
Exception raised when an interruption of a session has been requested.
  """
  pass

