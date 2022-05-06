# File:                 polling.py
# Creation date:        2018-05-18
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              a thread which polls code at regular intervals
#
r"""
Available types and functions
-----------------------------
"""

from __future__ import annotations
from typing import Any, Union, Callable, Iterable, Mapping, Tuple, Optional
import logging; logger = logging.getLogger(__name__)

import os,socket,sqlite3,time,threading,traceback
from datetime import datetime
from pathlib import Path

class PollingThread (threading.Thread):
  """
Objects of this class are python contexts which can be used to encapsulate any piece of code into a monitor executing on a separate thread. The monitor polls the code at regular intervals and stores a report. Only one report is stored at all time (any new report replaces the previous one).

:param path: location of the monitor reporting file
:param interval: polling rate in seconds
:param maxerror: maximum number of consecutive errors before giving up
:param fields: list of field descriptors (see below) specifying what to record at each polling
:param staticfields: dictionary of static values specifying what to record initially

Each field descriptor is a pair of an sql column specification and a function with no input which returns a value compatible with the column type. An optional third component can specify an other function to be used in case of error.
  """
  def __init__(self,path:Union[str,Path],*fields,interval:float=1.,maxerror:int=3,**staticfields):
    def open_():
      nonlocal conn
      if path.exists(): path.unlink()
      conn = sqlite3.connect(str(path))
      try: init_()
      except: close_(); raise
    def close_():
      try: conn.close()
      except: pass
    def run_():
      try: open_()
      except Exception as exc: logger.warning('Unable to open status file %s (giving up): %s',path,exc); return
      error = 0
      lasterr = None
      ongoing = True
      while ongoing:
        ongoing = not self.stop_requested.wait(interval)
        try:
          try: updates_(); error=0; continue
          except sqlite3.Error: raise
          except Exception as exc:
            error += 1
            if error < maxerror:
              updates_error()
              continue
            else: lasterr = str(exc)
        except sqlite3.Error as exc:
          error += 1
          if error < maxerror:
            close_()
            try: open_(); continue
            except Exception as exc2: lasterr = str(exc2)
          else: lasterr = str(exc)
        logger.warning('Unable to record status in %s (giving up after %d errors): %s',path,error,lasterr)
        return
      try: conn.execute('PRAGMA user_version = 1'); conn.commit()
      except: pass
      close_()
    super().__init__(target=run_,daemon=True)
    NoneFunc = lambda: None
    DefaultTypes={int:'INTEGER',float:'FLOAT',str:'TEXT',datetime:'DATETIME',bytes:'BLOB'}
    def field(cdef:str,upd:Callable[[],Any],upd_error:Callable[[],Any]=NoneFunc)->tuple[str,str,Callable[[],Any],Callable[[],Any]]:
      cdef_ = cdef.strip().split(' ',1)
      return cdef_[0],(DefaultTypes.get(type(upd()),'BLOB') if len(cdef_)==1 else cdef_[1]),upd,upd_error
    def staticfield(name:str,value)->tuple[str,str,Any]:
      return name,DefaultTypes.get(type(value),'BLOB'),value
    started = time.time(); elapsed = lambda started=started:time.time()-started
    staticfields_ = [staticfield(*x) for x in (('started',datetime.fromtimestamp(started)), ('pid', f'{socket.getfqdn()}:{os.getpid()}'),*staticfields.items())]
    fields_ = [field(*x) for x in (('elapsed',elapsed,elapsed),('error TEXT',NoneFunc,traceback.format_exc),*fields)]
    sql_create = 'CREATE TABLE Status ({})'.format(', '.join(f'{f[0]} {f[1]}' for l in (staticfields_, fields_) for f in l))
    sql_init = 'INSERT INTO Status ({}) VALUES ({})'.format(','.join(f[0] for f in staticfields_),','.join(len(staticfields_)*'?'))
    sql_update = 'UPDATE Status SET {}'.format(', '.join(f'{f[0]}=?' for f in fields_))
    conn:Optional[sqlite3.Connection] = None
    def init_(sql_create=sql_create,sql_init=sql_init,initv=tuple(f[2] for f in staticfields_)):
      conn.execute(sql_create)
      conn.execute(sql_init,initv)
      conn.commit()
    def updates_(updf=tuple(f[2] for f in fields_),sql_update=sql_update):
      conn.execute(sql_update,tuple(u() for u in updf))
      conn.commit()
    def updates_error(updf=tuple(f[3] for f in fields_)): updates_(updf)
    self.stop_requested = threading.Event()
    if isinstance(path,str): path = Path(path)
  def __enter__(self):
    self.start()
    return self
  def __exit__(self,*a):
    self.stop_requested.set()
    self.join()
