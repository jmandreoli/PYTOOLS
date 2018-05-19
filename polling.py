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
Objects of this class are python environments which can be used to encapsulate any piece of code into a monitor executing on a separate thread. The monitor polls the code at regular intervals and stores a report. Only one report is stored at all time.

:param path: location of the monitor reporting file
:type param: :class:`Union[str,pathlib.Path]`
:param interval: polling rate in seconds
:type interval: :class:`float`
:param maxerror: maximum number of consecutive errors before giving up
:type maxerror: :class:`int`
:param fields: list of field descriptor to record at each polling
:type fields: :class:`List[Dict[str,object]]`

Each field descriptor is a dictionary with the following keys:

* ``name``: name of the field (:class:`str`)
* ``type``: sql type of the field as recognised by sqlite (:class:`str`)
* ``value``: either a single value compatible with field type (used as initial value and never updated) or a function with no input which returns a value compatible with the field type (evaluated at each poll and used as current value)
  """
  def __init__(self,path,*fields,interval=1.,maxerror=3):
    def open_():
      nonlocal conn
      if path.exists(): path.unlink()
      conn = sqlite3.connect(str(path))
      try: conn.execute(sql_create);conn.execute(sql_init,inits_);conn.commit()
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
          conn.execute(sql_update,updates_())
          conn.commit()
          error = 0
          continue
        except Error as exc:
          error += 1
          if error < maxerror:
            try:
              conn.execute(sql_update,exc.args[0])
              conn.commit()
            except sqlite3.Error: pass
            continue
          else: lasterr = str(exc.__context__)
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
    DefaultTypes = {int:'INTEGER',float:'FLOAT',str:'TEXT',datetime:'DATETIME',bytes:'BLOB'}
    nonefunc = lambda: None
    started = time.time(); elapsed = lambda started=started:time.time()-started
    for f in fields:
      assert all(k in ('name','type','value','error_value') for k in f)
      f.setdefault('type',DefaultTypes.get(type(f['value'])))
    fields = (
      dict(name='started',type='DATETIME',value=datetime.fromtimestamp(started)),
      dict(name='pid',type='INTEGER',value=os.getpid()),
      dict(name='elapsed',type='FLOAT',value=elapsed,error_value=elapsed),
      dict(name='error',type='TEXT',value=nonefunc,error_value=traceback.format_exc),
    ) + fields
    sql_create = 'CREATE TABLE Status ({})'.format(', '.join('{name} {type}'.format(**f) for f in fields))
    sql_init,inits_ = zip(*((f['name'],f['value']) for f in fields if not callable(f['value'])))
    sql_init = 'INSERT INTO Status ({}) VALUES ({})'.format(','.join(sql_init),','.join(len(sql_init)*'?'))
    sql_update,setf,error_setf = zip(*(('{name}=?'.format(**f),f['value'],f.get('error_value',nonefunc)) for f in fields if callable(f['value'])))
    def updates_(setf=setf,error_setf=error_setf):
      try: return tuple(p() for p in setf)
      except: raise Error(tuple(p() for p in error_setf))
    sql_update = 'UPDATE Status SET {}'.format(', '.join(sql_update))
    self.stop_requested = threading.Event()
    conn = None
    if isinstance(path,str): path = Path(path)
  def __enter__(self):
    self.start()
    return self
  def __exit__(self,*a):
    self.stop_requested.set()
    self.join()

class Error (Exception): pass
