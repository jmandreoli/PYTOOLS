# File:                 polling.py
# Creation date:        2018-05-18
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              a thread which polls code at regular intervals
#

from __future__ import annotations
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence
import logging; logger = logging.getLogger(__name__)

import os,socket,sqlite3,time,threading,traceback
from datetime import datetime
from pathlib import Path

class PollingThread (threading.Thread):
  """
Objects of this class are python contexts which can be used to encapsulate any piece of code into a monitor executing on a separate thread. The monitor polls the code at regular intervals and stores a report. Only one report is stored at all time (any new report replaces the previous one).

:param path: location of the monitor reporting file
:param _interval: polling rate in seconds
:param _maxerror: maximum number of consecutive errors before giving up
:param fields: dictionary of field descriptors

Each field descriptor is either a value or a dictionary of a value (key: `value`), its SQL type (key: `type`), and optionally a callable to use in case of error (key: `error`). The field is dynamic if the value is callable (with no argument) or static otherwise. The SQL type, if not specified, is inferred either from the value for a static field or from a single call to the value otherwise.
  """
  def __init__(self,path:str|Path,_interval:float=1.,_maxerror:int=3,**fields):
    def run_():
      try: self.open_()
      except Exception as exc: logger.warning('Unable to open status file %s (giving up): %s',path,exc); return
      error = 0
      lasterr = None
      ongoing = True
      while ongoing:
        ongoing = not self.stop_requested.wait(_interval)
        try:
          try: self.updates_(); error=0; continue
          except sqlite3.Error: raise
          except Exception as exc:
            error += 1
            if error < _maxerror:
              self.updates_error()
              continue
            else: lasterr = str(exc)
        except sqlite3.Error as exc:
          error += 1
          if error < _maxerror:
            self.close_()
            try: self.open_(); continue
            except Exception as exc2: lasterr = str(exc2)
          else: lasterr = str(exc)
        logger.warning('Unable to record status in %s (giving up after %d errors): %s',path,error,lasterr)
        return
      self.close_()
    path = Path(path)
    self.config(path,fields)
    super().__init__(target=run_,daemon=True)
  def __enter__(self):
    self.stop_requested = threading.Event()
    self.start()
    return self
  def __exit__(self,*a):
    self.stop_requested.set()
    self.join()

  def config(self,path:Path,fields:dict[str,Any]):
    started = time.time(); elapsed = lambda started=started:time.time()-started
    fields_ = [
      Field('started', datetime.fromtimestamp(started)),
      Field('pid',f'{socket.getfqdn()}:{os.getpid()}'),
      Field('elapsed',elapsed,sqltype='FLOAT',error=elapsed),
      Field('error',(lambda:None),error=traceback.format_exc),
      *(Field(name,**(x if isinstance(x,dict) else {'value':x})) for name,x in fields.items())
    ]
    fields_s = [f for f in fields_ if f.static is True]
    fields_c = [f for f in fields_ if f.static is False]
    sql_create = 'CREATE TABLE Status ({})'.format(', '.join(f'{f.name} {f.sqltype}' for f in fields_))
    sql_init = 'INSERT INTO Status ({}) VALUES ({})'.format(','.join(f.name for f in fields_s),','.join(len(fields_s)*'?'))
    sql_update = 'UPDATE Status SET {}'.format(', '.join(f'{f.name}=?' for f in fields_c))
    def open_(sql_create=sql_create,sql_init=sql_init,initv=tuple(f.value for f in fields_s)):
      nonlocal conn
      if path.exists(): path.unlink()
      conn = sqlite3.connect(path)
      try:
        conn.execute(sql_create)
        conn.execute(sql_init,initv)
        conn.commit()
      except: close_(); raise
    def updates_(updf=tuple(f.value for f in fields_c),sql_update=sql_update):
      conn.execute(sql_update,tuple(u() for u in updf))
      conn.commit()
    def updates_error(updf=tuple(f.error for f in fields_c)): updates_(updf)
    def close_():
      try: conn.close()
      except: pass
    self.open_,self.updates_,self.updates_error,self.close_ = open_,updates_,updates_error,close_
    conn:sqlite3.Connection|None = None

class Field:
  __slots__ = 'name','value','sqltype','error','static'
  DefaultTypes = {int:'INTEGER',float: 'FLOAT',str:'TEXT',datetime:'DATETIME',bytes:'BLOB'}
  def __init__(self,name:str,value:Any,sqltype:str|None=None,error:Callable[[],Any]|None=None):
    static = not callable(value)
    if sqltype is None: sqltype = self.DefaultTypes.get(type(value if static else value()),'BLOB')
    if static: assert error is None
    elif error is None: error = (lambda:None)
    else: assert callable(error)
    self.name,self.value,self.sqltype,self.error,self.static = name,value,sqltype,error,static

def status(path:str|Path):
  r"""
Returns the contents of a monitor reporting file.

:param path: location of the monitor reporting file
  """
  path = Path(path)
  assert path.is_file()
  with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as conn:
    conn.row_factory = sqlite3.Row
    r = conn.execute('SELECT * FROM Status').fetchone()
    assert r is not None
    return dict(r)
