# File:                 demo/monitor.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the monitor module

if __name__=='__main__':
  import sys
  from PYTOOLS.demo.monitor import demo # properly import this module
  demo()
  sys.exit(0)

#--------------------------------------------------------------------------------------------------

import logging, time, sys
logger = logging.getLogger(__name__)
automatic = False

from pathlib import Path
from ..monitor import iterc_monitor, stats_monitor, buffer_monitor

def demo1():
  from itertools import count; from math import log
  loop = map(log,count(1)) # the iterable to monitor
  fmts = 'iterc:{0} log:{1.value:.4f}' # the logging format
  m = iterc_monitor(maxcpu=.1,maxiter=100000,show=.1,fmt=fmts.format)
  m.run(loop,logger=logger)

def demo2():
  fmts = 'iterc:{0} x:{1.value[0]:.4f} (mean: {1.stat.avg:.4f}) y:{1.value[1]:.4f}'
  m = stats_monitor(label='stat',targetf=(lambda env: env.value[0]))
  m *= buffer_monitor(label='buf',targetf=(lambda env: (env.value,)))
  m *= iterc_monitor(maxiter=200,fmt=fmts.format,show=.8)
  def initf(fig,bounds=((-1.5,1.5),(-1.5,1.5))):
    ax = fig.add_subplot(1,1,1,xlim=bounds[0],ylim=bounds[1],aspect='equal')
    return ax.plot(())
  def updatef(env,artists):
    try: buf = env.buf
    except AttributeError: return
    line, = artists
    line.set_data(*zip(*buf))
  env = m.run(cycloid(a=.3,omega=5.1,step=2.),detach=(None if automatic else dict(delay=1.)),logger=logger)
  display(env,initf,updatef,figsize=(10,8))

def display(env,initf:callable,updatef:callable,**figargs):
  from matplotlib.pyplot import figure, show, close
  from matplotlib.animation import FuncAnimation
  fig = figure(**figargs)
  artists = initf(fig)
  txt = fig.text(.5,.5,'',zorder=1,color='r',ha='center',va='center')
  if automatic:
    updatef(env,artists)
    fig.savefig(str(Path(__file__).resolve().parent/'monitor.png'))
  else:
    a = FuncAnimation(fig,lambda frm:(txt.set_text('OVER') if env.stop else updatef(env,artists)),repeat=False)
    show()
    env.stop = True

def cycloid(a,omega,step):
  from math import sin, cos, pi
  step *= pi/180
  u = 0.
  while True:
    if not automatic: time.sleep(.04) # simulates heavy computing with a delay at each iteration
    yield cos(u)+a*cos(omega*u), sin(u)+a*sin(omega*u)
    u += step

#--------------------------------------------------------------------------------------------------

def demo():
  logging.basicConfig(level=logging.INFO)
  for demo_ in demo1, demo2:
    print(80*'-'); print(demo_.__name__)
    demo_()
    if not automatic:
      try: input('RET: continue; ^-C: stop')
      except: print();break
