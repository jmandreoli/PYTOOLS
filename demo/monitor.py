# File:                 demo/monitor.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the monitor module

if __name__=='__main__':
  import sys, logging
  logging.basicConfig(level=logging.INFO)
  from myutil.demo.monitor import demo1, demo2
  for d in demo1, demo2:
    print('----------------------------\n{}'.format(d.__name__))
    d()
    try: input('RET: continue; Ctrl-C: stop')
    except: print();break
  sys.exit(0)

#--------------------------------------------------------------------------------------------------

import logging, time, sys
logger = logging.getLogger(__name__)

from ..monitor import monitor, iterc_monitor, averaging_monitor, buffer_monitor

def demo1():
  from itertools import count; from math import log
  loop = map(log,count(1)) # the iterable to monitor
  fmts = 'iterc:{0} log:{1.value:.4f}' # the logging format
  m = iterc_monitor(maxcpu=.1,maxiter=100000,show=.1,logger=logger,fmt=fmts.format)
  return m.run(loop)

@monitor
def delay_monitor(env,delay:float=1.):
  # inserts a delay after each iteration
  while True: yield time.sleep(delay)

@monitor
def display_monitor(env,targetf:callable,bounds):
  # assumes targetf returns a list of coordinate pairs, and displays the corresponding plot
  from matplotlib.pyplot import figure, show
  fig = figure(figsize=(10,8))
  ax = fig.add_subplot(1,1,1)
  ax.set_xlim(bounds[0])
  ax.set_ylim(bounds[1])
  ax.set_aspect('equal')
  show(False)
  fig.canvas.blit(fig.bbox)
  background = fig.canvas.copy_from_bbox(ax.bbox)
  line, = ax.plot(())
  while True:
    fig.canvas.restore_region(background)
    line.set_data(*zip(*targetf(env)))
    ax.draw_artist(line)
    fig.canvas.blit(ax.bbox)
    yield

def demo2():
  fmts = 'iterc:{0} x:{1.value[0]:.4f} (mean: {1.stat.mean:.4f}) y:{1.value[1]:.4f}'
  m = averaging_monitor(label='stat',targetf=(lambda env: env.value[0]))
  m *= iterc_monitor(maxiter=200,logger=logger,fmt=fmts.format,show=.8)
  m *= buffer_monitor(label='buf',targetf=(lambda env: env.value))
  m *= display_monitor(targetf=(lambda env: env.buf),bounds=((-1.5,1.5),(-1.5,1.5)))
  m *= delay_monitor(.04) # forces at most 25 frames per seconds (crude)
  return m.run(cycloid(a=.3,omega=5.1,step=2.))

def cycloid(a,omega,step):
  from math import sin, cos, pi
  step *= pi/180
  u = 0.
  while True:
    yield cos(u)+a*cos(omega*u), sin(u)+a*sin(omega*u)
    u += step

