# File:                 demo/demo_polling.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the polling module

import time
from PYTOOLS.polling import PollingThread, status
DB = RUN.path('.db')

def report(final=False,start=time.time()):
  # displays the current monitored status
  try: s = '\n  '.join(f'{d:12s}: {v}' for d,v in status(DB).items())
  except: s = '.'
  print(f'--- {time.time()-start:.2f}{' (final)' if final else ''} ---\n  {s}')

# calls to report (each on a *daemon* thread), one every 400msec up to 9 times
for t in range(1,10): RUN.schedule(t*.4,report)

value = 0
with PollingThread(DB,
  value=(lambda: value),
  #generate_an_error={'type':'FLOAT','value':(lambda: 1/0)}),
  fortytwo=42,
  _interval=.2, # monitors every 200msec (on a *daemon* thread)
):
  for i in range(1,61):
    value += 1
    time.sleep(.05) # state update happens ca. every 50msec up to ca. 3sec
report(final=True)
