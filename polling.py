# File:                 polling.py
# Creation date:        2018-05-18
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              a thread which polls code at regular intervals
#

import os,logging,sqlite3,time,threading,traceback
from datetime import datetime
from pathlib import Path
logger = logging.getLogger(__name__)

class PollingThread (threading.Thread):
  """
Objects of this class are python contexts which can be used to encapsulate any piece of code into a monitor executing on a separate thread. The monitor polls the code at regular intervals and stores a report. Only one report is stored at all time.

:param path: location of the monitor reporting file
:type param: :class:`Union[str,pathlib.Path]`
:param interval: polling rate in seconds
:type interval: :class:`float`
:param maxerror: maximum number of consecutive errors before giving up
:type maxerror: :class:`int`
:param fields: list of field descriptor (see below) specifying what to record at each polling
:param staticfields: dictionary of static values specifying what to record initially

Each field descriptor is a pair of an sql column specification and a function with no input which returns a value compatible with the column type.
  """
  def __init__(self,path,*fields,interval=1.,maxerror=3,**staticfields):
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
      except Exception as exc: logger.warn('Unable to open status file %s (giving up): %s',path,exc); return
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
              try: updates_error()
              except: pass
              continue
            else: lasterr = str(exc)
        except sqlite3.Error as exc:
          error += 1
          if error < maxerror:
            close_()
            try: open_(); continue
            except Exception as exc2: lasterr = str(exc2)
          else: lasterr = str(exc)
        logger.warn('Unable to record status in %s (giving up after %d errors): %s',path,error,lasterr)
        return
      try: conn.execute('PRAGMA user_version = 1'); conn.commit()
      except: pass
      close_()
    super().__init__(target=run_,daemon=True)
    conn = None
    NoneFunc = lambda: None
    DefaultTypes={int:'INTEGER',float:'FLOAT',str:'TEXT',datetime:'DATETIME',bytes:'BLOB'}
    def field(cdef,upd,upd_error=NoneFunc):
      cdef = cdef.strip().split(' ',1); name = cdef[0]
      return name,(DefaultTypes.get(type(upd()),'BLOB') if len(cdef)==1 else cdef[1]),upd,upd_error
    def staticfield(name,value):
      return name,DefaultTypes.get(type(value),'BLOB'),value
    started = time.time(); elapsed = lambda started=started:time.time()-started
    staticfields = list(staticfields.items())
    staticfields[0:0] = ('started',datetime.fromtimestamp(started)), ('pid',os.getpid())
    staticfields = [staticfield(*x) for x in staticfields]
    fields = list(fields)
    fields[0:0] = ('elapsed',elapsed,elapsed),('error TEXT',NoneFunc,traceback.format_exc)
    fields = [field(*x) for x in fields]
    sql_create = 'CREATE TABLE Status ({})'.format(', '.join('{} {}'.format(f[0],f[1]) for l in (staticfields,fields) for f in l))
    sql_init = 'INSERT INTO Status ({}) VALUES ({})'.format(','.join(f[0] for f in staticfields),','.join(len(staticfields)*'?'))
    def init_(sql_create=sql_create,sql_init=sql_init,initv=[f[2] for f in staticfields]):
      conn.execute(sql_create)
      conn.execute(sql_init,initv)
      conn.commit()
    sql_update = 'UPDATE Status SET {}'.format(', '.join('{}=?'.format(f[0]) for f in fields))
    def updates_(u=[f[2] for f in fields],sql_update=sql_update):
      conn.execute(sql_update,tuple(p() for p in u))
      conn.commit()
    def updates_error(u=[f[3] for f in fields]): return updates_(u)
    self.stop_requested = threading.Event()
    if isinstance(path,str): path = Path(path)
  def __enter__(self):
    self.start()
    return self
  def __exit__(self,*a):
    self.stop_requested.set()
    self.join()
