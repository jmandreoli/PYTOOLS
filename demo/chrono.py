# File:                 demo/chrono.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the chrono module

if __name__=='__main__':
    import sys
    from chrono import weatherchrono as w, demo
    print('-------------------------\nClearing',w)
    w.clear()
    demo(w,period=1,nbuf=2)
    sys.exit(0)

#--------------------------------------------------------------------------------------------------

import requests, logging, time, functools
from myutil.chrono import ChronoBlock, Formatter, flatten, atinterval, Startc

DIR = 'chrono.dir'

class OpenWeatherFlow:
    def __init__(self,location,period):
        self.location = location
        self.period = period
    def __iter__(self):
        ID = lambda x: x
        fmt = ID,ID
        fformatter = Formatter(
          (r'^\.coord\.(lon)$',1,fmt),
          (r'^\.coord\.(lat)$',1,fmt),
          (r'^\.(dt)$',0,fmt),
          (r'^.main\.(temp)$',0,fmt),
          (r'^\.main\.(pressure)$',0,fmt),
          (r'^\.main\.(humidity)$',0,fmt),
        )
        src = atinterval(weather(self.location),self.period)
        src = map(fformatter,map(flatten,src))
        yield from src
    def __str__(self): return 'OpenWeather({},{})'.format(self.location,self.period)
    def __getstate__(self): return self.location, self.period
    def __setstate__(self,state): self.location, self.period = state

def weather(location):
    url = 'http://api.openweathermap.org/data/2.5/weather?units=metric&q='+location
    s = requests.Session()
    while True: yield s.get(url).json()

weatherchrono = ChronoBlock(flow=OpenWeatherFlow('london',2),db=DIR)

def demo(chrono,period,nbuf):
    print('Activate',chrono)
    chrono.activate(functools.partial(Startc,None,chrono.db),slice(0,None),level=logging.INFO)
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

