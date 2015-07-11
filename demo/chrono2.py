# File:                 demo/chrono2.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the chrono module

if __name__=='__main__':
    import sys
    from chrono import demo
    from chrono2 import processchrono as w
    w.clear()
    demo(w,period=1,nbuf=2)
    sys.exit(0)

#--------------------------------------------------------------------------------------------------

import logging, time
from myutil.remote.main import shclients, sshclients
from myutil.chrono import ChronoBlock, flatten, atinterval

DIR = 'chrono.dir'

class Processes:

    Dformatter=dict(
      basic=(lambda x: map((lambda q: ((q[0][1:],0),float(q[1]))),x)),
      final=(lambda x: map((lambda q: ((q[0][1:],0),float(q[1]))),x)),
      )

    def __init__(self,formatter,target,period):
        self.formatter = formatter
        self.target = target
        self.period = period

    def __iter__(self):
        w = sshclients((1,((self.target,1),)))[0]
        try:
          w.declare(psinfo)
          def pss():
              while True: yield w.psinfo()
          src = atinterval(pss(),self.period)
          src = map(self.Dformatter[self.formatter],map(flatten,src))
          yield from src
        finally:
          w.shutdown()

    def __getstate__(self): return self.formatter,self.target,self.period
    def __setstate__(self,state): self.formatter,self.target,self.period = state
    def __str__(self): return 'Processes({},{})'.format(self.target,self.period)

processchrono = ChronoBlock(flow=Processes(formatter='basic',target='sirac',period=3),db=DIR)

def psinfo():
    import psutil
    cpus = psutil.cpu_times_percent(percpu=True)
    dio = psutil.disk_io_counters()
    nio = psutil.net_io_counters()
    return dict(
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

