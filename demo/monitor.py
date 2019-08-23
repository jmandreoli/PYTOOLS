# File:                 demo/monitor.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the monitor module
from make import RUN; RUN(__name__,__file__,2)
#--------------------------------------------------------------------------------------------------

import logging, time, sys, subprocess
logger = logging.getLogger(__name__)
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
  env = m.run(cycloid(a=.3,omega=5.1,step=2.),detach=dict(delay=1.),logger=logger)
  display(env,figsize=(10,8))

def display(env,**figargs):
  from matplotlib.pyplot import figure, show, close
  from matplotlib.animation import FuncAnimation
  fig = figure(**figargs)
  ax = fig.add_subplot(1,1,1,xlim=(-1.5,1.5),ylim=(-1.5,1.5),aspect='equal')
  line, = ax.plot(())
  txt = fig.text(.5,.5,'',zorder=1,color='r',ha='center',va='center')
  def updatef(env):
    try: buf = env.buf
    except AttributeError: return
    line.set_data(*zip(*buf))
  anim = FuncAnimation(fig,lambda frm:(txt.set_text('OVER') if env.stop else updatef(env)),interval=40.,frames=150,repeat=False)
  def action():
    anim.save(RUN.dir/'monitor.mp4',fps=25)
    close()
    subprocess.run(['ffmpeg','-loglevel','panic','-y','-i',str(RUN.dir/'monitor.mp4'),str(RUN.dir/'monitor.gif')])
  RUN.play(action)
  show()
  env.stop = 'interrupted'

def cycloid(a,omega,step):
  from math import sin, cos, pi
  step *= pi/180
  u = 0.
  while True:
    time.sleep(.02) # simulates heavy computing with a delay at each iteration
    yield cos(u)+a*cos(omega*u), sin(u)+a*sin(omega*u)
    u += step

#--------------------------------------------------------------------------------------------------

def demo():
  logging.basicConfig(level=logging.INFO)
  for demo_ in demo1, demo2:
    print(80*'-'); print(demo_.__name__)
    demo_()
    RUN.pause()
