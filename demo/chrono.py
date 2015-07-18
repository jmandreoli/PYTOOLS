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
    self.period = period
    self.__dict__.update(ka)
  def __iter__(self):
    src = atinterval(self.flow(),self.period)
    src = map(self.fformatter,map(flatten,src))
    yield from src
  def __str__(self):
    d = ','.join('{}={}'.format(k,v) for k,v in self.__dict__.items())
    return '{}({})'.format(self.__class__.__name__,d)

class OpenWeatherFlow (PollingFlow):
  fmt = ((lambda x: x),(lambda x: x))
  fformatter = Formatter(
    (r'^\.coord\.(lon)$',1,fmt),
    (r'^\.coord\.(lat)$',1,fmt),
    (r'^\.(dt)$',0,fmt),
    (r'^.main\.(temp)$',0,fmt),
    (r'^\.main\.(pressure)$',0,fmt),
    (r'^\.main\.(humidity)$',0,fmt),
  )
  def flow(self):
    import requests
    url = 'http://api.openweathermap.org/data/2.5/weather?units=metric&q='+self.location
    s = requests.Session()
    while True: yield s.get(url).json()

class ProcessesFlow (PollingFlow):
  @staticmethod
  def fformatter(x): return map((lambda q: ((q[0][1:],0),float(q[1]))),x)
  def flow(self):
    import psutil
    while True:
      cpus = psutil.cpu_times_percent(percpu=True)
      dio = psutil.disk_io_counters()
      nio = psutil.net_io_counters()
      yield dict(
        cpu_times_percent=[dict(user=cpu.user,system=cpu.system) for cpu in cpus],
        virtual_memory_percent=psutil.virtual_memory().percent,
        swap_memory_percent=psutil.swap_memory().percent,
        disk_io_read_count=dio.read_count,
        disk_io_write_count=dio.write_count,
        disk_io_read_bytes=dio.read_bytes,
        disk_io_write_bytes=dio.write_bytes,
        disk_io_read_time=dio.read_time,
        disk_io_write_time=dio.write_time,
        net_io_bytes_sent=nio.bytes_sent,
        net_io_bytes_recv=nio.bytes_recv,
        net_io_packets_sent=nio.packets_sent,
        net_io_packets_recv=nio.packets_recv,
        net_io_errin=nio.errin,
        net_io_errout=nio.errout,
        net_io_dropin=nio.dropin,
        net_io_dropout=nio.dropout,
      )

weatherchrono = ChronoBlock(flow=OpenWeatherFlow(period=2,location='london'),db=DIR)
processchrono = ChronoBlock(flow=ProcessesFlow(period=3),db=DIR)

def demo1(chrono,period,nbuf):
  print('Activate',chrono)
  chrono.activate(slice(0,None),level=logging.INFO)
  session = chrono.current
  print('Polling(session: {} frequency: {:.02f}Hz buffer: {}); Ctrl-C to interrupt anytime'.format(session,1/period,nbuf))
  try:
    t = 0
    while True:
      time.sleep(period)
      for r in reversed(list(chrono.stream(session,limit=(nbuf,0),reverse=True))):
        if r[1]>t: print(r); t = r[1]
  except BaseException as e: print('Interrupted',e)
  chrono.pause('Stop')
  print('Pause',chrono,end=' ',flush=True)
  while chrono.current is not None:
    print('.',end='',flush=True)
    time.sleep(period)
  print()

def demo():
  for w in (weatherchrono,processchrono):
    print(80*'-'); print('Clear',w)
    w.clear()
    if automatic:
      from threading import Timer; from _thread import interrupt_main
      Timer(10.,interrupt_main).start()
    demo1(w,period=1,nbuf=2)

