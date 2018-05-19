# File:                 demo/polling.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the polling module

if __name__=='__main__':
  import sys
  from PYTOOLS.demo.polling import demo # properly import this module
  demo(*sys.argv[1:])
  sys.exit(0)

#--------------------------------------------------------------------------------------------------

import logging,time,sqlite3
from pathlib import Path; DIR = Path(__file__).resolve().parent/'polling.dir'
from ..polling import PollingThread

def demo():
  logging.basicConfig(level=logging.INFO)
  current = 0
  with PollingThread(DIR/'test.db',
    dict(name='current',type='INTEGER',value=(lambda: current)),
    dict(name='fortytwo',value=42),
    #dict(name='generate_an_error',type='INTEGER',value=(lambda: 1/0))
  ):
    for i in range(1,61):
      current += 1
      time.sleep(.05)
  with sqlite3.connect(str(DIR/'test.db')) as conn:
    c = conn.execute('SELECT * FROM Status')
    status, = c
    for d,v in zip(c.description,status): print('{:12s}: {}'.format(d[0],v))
