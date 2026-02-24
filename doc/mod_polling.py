# File:                 demo/demo_polling.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the polling module

import time, subprocess, sys
if __name__ == '__main__':
  # Master process: spawn a test process and report
  from PYTOOLS.polling import status
  dbpath,source = str(RUN.path('.db')),str(RUN.source); config = {'_dbpath':dbpath}; modname = 'example'
  instr = f'from runpy import run_path; run_path({source!r},{config!r},{modname!r})'
  w = subprocess.Popen([sys.executable,'-c',instr])
  ts = time.time(); time.sleep(.1); ongoing = True
  print('.. list-table::\n   :header-rows: 1\n')
  print('   * -',*(f'     - {x}' for x in status(dbpath)),sep='\n')
  while ongoing: # REPORTING: db lookup every 450msec until end of spawned process
    time.sleep(.45); ongoing = w.poll() is None; flag = '' if ongoing else ' !'
    try: L = status(dbpath).values()
    except: pass
    else: print(f'   * - {time.time()-ts:.2f}{flag}',*(f'     - {x}' for x in L),sep='\n')
else:
  # Spawned process: executes some code monitored by PollingThread
  from PYTOOLS.polling import PollingThread
  value = 0
  with PollingThread(_dbpath,
    value=(lambda: value),
    fortytwo=42,
    _interval=.2, # MONITORING: db update every 200msec (on a *daemon* thread)
  ):
    for i in range(300): # PROGRESSING: state update every 10msec for ca. 3sec
      value += 1
      time.sleep(.01)
