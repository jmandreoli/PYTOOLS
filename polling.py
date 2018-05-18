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
* ``init``: initial value of the field
* ``update`` (optional): updater function (no input); must return a value compatible with the field type; if omitted, initial value is never updated
  """
  def __init__(self,path,*fields,interval=1.,maxerror=3):
    def open_():
      nonlocal conn
      if path.exists(): path.unlink()
      conn = sqlite3.connect(str(path))
      try: conn.execute(sql_create);conn.execute(sql_init,sql_init_p);conn.commit()
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
          try:
            conn.execute(sql_update, updates_())
            conn.commit()
            error = 0
            continue
          except sqlite3.Error: raise
          except Exception as exc:
            error += 1
            if error < maxerror:
              conn.execute(sql_error,(traceback.format_exc(),))
              conn.commit()
              continue
            else: lasterr = str(exc); close_()
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
    sql_create = 'CREATE TABLE Status (started DATETIME, pid INTEGER, elapsed FLOAT, error TEXT{})'.format(''.join(', {name} {type}'.format(**f) for f in fields))
    sql_init = 'INSERT INTO Status (started,pid,elapsed{}) VALUES (?,?,?{})'.format(''.join(',{name}'.format(**f) for f in fields),len(fields)*',?')
    sql_update = 'UPDATE Status SET error=NULL, elapsed=?{}'.format(''.join(', {name}=?'.format(**f) for f in fields if f.get('update') is not None))
    updates_ = lambda L=tuple(f['update'] for f in fields if f.get('update') is not None): (time.time()-started,)+tuple(p() for p in L)
    sql_error = 'UPDATE Status SET error=?'
    started = time.time()
    sql_init_p = (datetime.fromtimestamp(started),os.getpid(),0.)+tuple(f['init'] for f in fields)
    self.stop_requested = threading.Event()
    conn = None
    if isinstance(path,str): path = Path(path)
  def __enter__(self):
    self.start()
    return self
  def __exit__(self,*a):
    self.stop_requested.set()
    self.join()
