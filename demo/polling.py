# File:                 demo/polling.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the polling module
from make import RUN; RUN(__name__,__file__) # trick for documentation
#--------------------------------------------------------------------------------------------------

import logging,time,sqlite3
from PYTOOLS.polling import PollingThread

logging.basicConfig(level=logging.INFO)
current = 0
DB = RUN.file('.db')
with PollingThread(DB,
  ('current',(lambda: current)),
  #('generate_an_error FLOAT',(lambda: 1/0)),
  fortytwo=42,
):
  for i in range(1,61):
    current += 1
    time.sleep(.05)
with sqlite3.connect(str(DB)) as conn:
  conn.row_factory = sqlite3.Row
  status = conn.execute('SELECT * FROM Status').fetchone()
  report = '\n'.join(f'{d:12s}: {v}' for d,v in zip(status.keys(),status))
print(report)
