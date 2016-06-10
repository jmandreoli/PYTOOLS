# File:                 demo/chrono.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the chrono module

if __name__=='__main__':
  import sys
  from myutil.demo.chrono import demo
  demo()
  sys.exit(0)

#--------------------------------------------------------------------------------------------------

from pathlib import Path; DIR = Path(__file__).resolve().parent/'chrono.dir'
import logging, time, functools
from ..chrono import ChronoBlock, Formatter, flatten, atinterval
automatic = False

class PollingFlow:
  def __init__(self,period,**ka):
    self.period = period; self.__dict__.update(ka)
    d = ','.join('{}={}'.format(k,v) for k,v in self.__dict__.items())
    self.str_ = '{}({})'.format(self.__class__.__name__,d)
  def __iter__(self):
    yield from map(self.fformatter,map(flatten,atinterval(self.source(),self.period)))
  def __str__(self): return self.str_

class ProcFlow (PollingFlow):
  fformatter = Formatter(((lambda nam: nam[1:]),False,float))
  def source(self):
    import psutil
    while True:
      cpus = psutil.cpu_times_percent(percpu=True)
      yield dict(
        cpu_times_percent=[dict(user=cpu.user,system=cpu.system) for cpu in cpus],
        virtual_memory_percent=psutil.virtual_memory().percent,
        swap_memory_percent=psutil.swap_memory().percent,
        disk_io = psutil.disk_io_counters()._asdict(),
        net_io = psutil.net_io_counters()._asdict(),
      )

proc_c = ChronoBlock(flow=ProcFlow(period=3),db=DIR)

#--------------------------------------------------------------------------------------------------

def demo():
  for chrono in proc_c,:
    print(80*'-'); print('Clear',chrono); chrono.clear()
    if automatic:
      import threading, _thread; threading.Timer(10,_thread.interrupt_main).start()
    chrono.activate(slice(0,None),level=logging.INFO)
    waitwhile(chrono,lambda s: s is None)
    trace(chrono)
    chrono.pause('Stop')
    waitwhile(chrono,lambda s: s is not None)
    if not automatic:
      try: input('RET: continue; ^-C: stop')
      except: print(); break

def trace(chrono,period=1.,nbuf=2): # trace records from current session of *chrono*
  session = chrono.current
  print('Trace(session: {} frequency: {:.02f}Hz buffer: {}); ^-C to interrupt anytime'.format(session,1/period,nbuf))
  try:
    t = 0
    while True: # can only be interrupted by another thread or signal
      time.sleep(period)
      s = chrono.stream(session,limit=(nbuf,0),reverse=True)
      for r in reversed(list(s)): # last *nbuf* records in chronological order
        if r[1]>t: print(r); t = r[1] # print only new records, based on record timestamp *t*
  except BaseException as e: print('Interrupted',e)
  print(*tuple(a+('*' if s.sticky.get(a) else '') for a in s.attributes),sep=', ')

def waitwhile(chrono,cond,period=1.): # wait while value of *chrono*.current satifies *cond*
  print(('Activate' if cond(None) else 'Pause'),chrono,end=' ',flush=True)
  while cond(chrono.current): print('.',end='',flush=True); time.sleep(period)
  print()

