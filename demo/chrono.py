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
  def __iter__(self):
    yield from map(self.fformatter,map(flatten,atinterval(self.flow(),self.period)))
  def __str__(self):
    d = ','.join('{}={}'.format(k,v) for k,v in self.__dict__.items())
    return '{}({})'.format(self.__class__.__name__,d)

class OpenWeatherFlow (PollingFlow):
  fformatter = Formatter(
    Formatter.Field(r'\.coord\.(lon)',sticky=True),
    Formatter.Field(r'\.coord\.(lat)',sticky=True),
    Formatter.Field(r'\.(dt)'),
    Formatter.Field(r'\.main\.(temp)'),
    Formatter.Field(r'\.main\.(pressure)'),
    Formatter.Field(r'\.main\.(humidity)'),
  )
  def flow(self):
    import requests
    url = 'http://api.openweathermap.org/data/2.5/weather?units=metric&q='+self.location
    s = requests.Session()
    while True: yield s.get(url).json()

class ProcFlow (PollingFlow):
  fformatter = Formatter(((lambda nam: nam[1:]),False,float))
  def flow(self):
    import psutil
    while True:
      cpus = psutil.cpu_times_percent(percpu=True)
      yield dict(
        cpu_times_percent=[dict(user=cpu.user,system=cpu.system) for cpu in cpus],
        virtual_memory_percent=psutil.virtual_memory().percent,
        swap_memory_percent=psutil.swap_memory().percent,
        disk_io = psutil.disk_io_counters().__dict__,
        net_io = psutil.net_io_counters().__dict__,
      )

weatherchrono = ChronoBlock(flow=OpenWeatherFlow(period=2,location='london'),db=DIR)
procchrono = ChronoBlock(flow=ProcFlow(period=3),db=DIR)

def demo1(chrono,period,nbuf):
  print('Activate',chrono)
  chrono.activate(slice(0,None),level=logging.INFO)
  session = chrono.current
  print('Polling(session: {} frequency: {:.02f}Hz buffer: {}); Ctrl-C to interrupt anytime'.format(session,1/period,nbuf))
  try:
    t = 0
    while True:
      time.sleep(period)
      s = chrono.stream(session,limit=(nbuf,0),reverse=True)
      for r in reversed(list(s)):
        if r[1]>t: print(r); t = r[1]
  except BaseException as e: print('Interrupted',e)
  print(*tuple(a+('*' if s.sticky.get(a) else '') for a in s.attributes),sep=', ')
  chrono.pause('Stop')
  print('Pause',chrono,end=' ',flush=True)
  while chrono.current is not None:
    print('.',end='',flush=True)
    time.sleep(period)
  print()

def demo():
  for w in (weatherchrono,procchrono):
    print(80*'-'); print('Clear',w); w.clear()
    if automatic:
      from threading import Timer; from _thread import interrupt_main
      Timer(10.,interrupt_main).start()
    demo1(w,period=1,nbuf=2)
    if not automatic:
      try: input('RET: continue; Ctrl-C: stop')
      except: print(); break

