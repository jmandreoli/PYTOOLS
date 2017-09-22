# File:                 demo/cache.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the cache module

if __name__=='__main__':
  import sys
  from myutil.demo.cache import demo # properly import this module
  demo(*sys.argv[1:])
  sys.exit(0)

#--------------------------------------------------------------------------------------------------

import os,time,functools
from pathlib import Path; DIR = Path(__file__).resolve().parent/'cache.dir'
from collections import ChainMap
from .. import MapExpr, versioned
from ..cache import persistent_cache; persistent_cache = functools.partial(persistent_cache,db=DIR)
automatic = False

@persistent_cache
def simplefunc(x,y=3): return x,y

@persistent_cache
def longfunc(x,delay=10):
  time.sleep(delay)
  if x is None: raise Exception('longfunc error')
  return x

V = os.getpid()
@persistent_cache
@versioned(V)
def vfunc(x): return x+V

@persistent_cache
def stepA(d,**ini): return dict((k,v+d) for k,v in ini.items())

@persistent_cache
def stepB(E,fr=None,to=None,r=0):
  p,q = fr
  return ChainMap({to:E[p]+E[q]+r},E)

def proc(rab=1,rbc=2,rabc=3,d=7):
  P_ini = MapExpr(stepA,d,a=1,b=2,c=3)
  P_ab = MapExpr(stepB,P_ini,fr=('a','b'),to='ab',r=rab)
  P_bc = MapExpr(stepB,P_ini,fr=('b','c'),to='bc',r=rbc)
  P_abc = MapExpr(stepB,MapExpr(ChainMap,P_ab,P_bc),fr=('ab','bc'),to='abc',r=rabc)
  return P_abc

#--------------------------------------------------------------------------------------------------

DEMOS = (
  ( 'simplefunc(1,2)', 'simplefunc(1,y=2)', ),
  ( 'longfunc(42,6)', 'longfunc(None,4)', ),
  ( 'vfunc(3)', ),
  ( "proc()['abc']", "proc(rabc=4)['abc']", "proc(rbc=5)['abc']", ),
)

def demo(ind=None):
  if ind is None:
    # we are in the master process: for each entry in DEMOS, launch 2 slave processes at 2 sec interval and wait for them
    import subprocess,os,time,sys
    for f in simplefunc,longfunc,vfunc,stepA,stepB: f.cache.clear()
    for ind in range(len(DEMOS)):
      print(80*'-')
      cmd = [sys.executable,__file__,str(ind)]
      w1 = subprocess.Popen(cmd); time.sleep(2); w2 = subprocess.Popen(cmd)
      w1.wait(); w2.wait()
      if not automatic:
        try: input('RET: continue; ^-C: stop')
        except KeyboardInterrupt: print(); break
  else:
    # we are in a slave process: run the demo for entry *ind* in DEMOS
    import logging, gc
    logging.basicConfig(level=logging.INFO,format='[proc %(process)d @ %(asctime)s] %(message)s',datefmt='%H:%M:%S')
    logger = logging.getLogger()
    for expr in DEMOS[int(ind)]:
      gc.collect()
      logger.info('Computing: %s',expr)
      try: val = eval(expr)
      except Exception as exc: val = 'raised[{}]'.format(exc)
      logger.info('Result: %s = %s',expr,val)
