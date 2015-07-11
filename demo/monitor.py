# File:                 demo/monitor.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the monitor module

import logging, sys
logger = logging.getLogger(__name__)

import time
from itertools import count
from math import log, sin, cos, pi
from myutil.monitor import monitor, iterc_monitor, averaging_monitor, buffer_monitor

def demo1():
    loop = map(log,count(1))
    m = iterc_monitor(maxcpu=.1,maxiter=100000,show=.1,logger=logger,fmt='iterc:{0} log:{1.value:.4f}'.format)
    env = m.run(loop)

@monitor
def delay_monitor(env,delay=1.):
    # inserts a delay after each iteration
    while True: yield time.sleep(delay)

def demo2():
    rate = 25 # frames per second
    sbuf = [] # sample buffer
    # create the monitors
    m = averaging_monitor(label='stat',targetf=(lambda env: env.value[0]))
    m *= iterc_monitor(maxiter=200,logger=logger,fmt='iterc:{0} x:{1.value[0]:.4f} (mean: {1.stat.mean:.4f}) y:{1.value[1]:.4f}'.format,show=.8)
    m *= buffer_monitor(sbuf,targetf=(lambda env: env.value))
    m *= delay_monitor(1/rate)
    # create an animation to display the sample buffer
    from matplotlib.pyplot import figure, show
    from matplotlib.animation import FuncAnimation
    fig = figure(tight_layout=True)
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    line, = ax.plot(())
    def display(t):
        sys.stderr.flush()
        if sbuf: line.set_data(*zip(*sbuf)); ax.relim(); ax.autoscale_view()
        return (line,)
    anim = FuncAnimation(fig,func=display,interval=1000/rate,repeat=False)
    # launch the sample loop and the animation
    env = m.run(cycloid(),detach=True)
    show()
    env.stop = 'terminated'
    time.sleep(.1)

def cycloid(a=.3,omega=5.1,step=pi/90): # 2 deg
    u = 0.
    while True:
        yield cos(u)+a*cos(omega*u), sin(u)+a*sin(omega*u)
        u += step

#--------------------------------------------------------------------------------------------------

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    for d in demo1, demo2:
        print('----------------------------\n{}'.format(d.__name__),flush=True)
        d()
        try: input('RET: continue; Ctrl-C: stop')
        except: print(flush=True);break
        else: print(flush=True)

