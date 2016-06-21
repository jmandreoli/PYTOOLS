# File:                 demo/cache.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the cache module

if __name__=='__main__':
  import sys
  from myutil.demo.cache import demo, demo_
  if len(sys.argv) == 1: demo()
  else: demo_(*sys.argv[1:])
  sys.exit(0)

#--------------------------------------------------------------------------------------------------

from pathlib import Path; DIR = Path(__file__).resolve().parent/'cache.dir'
from collections import ChainMap
from .. import MapExpr
from ..cache import lru_persistent_cache
automatic = False

@lru_persistent_cache(db=DIR,ignore=('z',))
def simplefunc(x,y=3,z=8): return x,y

@lru_persistent_cache(db=DIR,ignore=('delay',))
def longfunc(x,delay=10):
  from time import sleep
  sleep(delay)
  if x is None: raise Exception('longfunc error')
  return x

@lru_persistent_cache(db=DIR)
def stepA(**ini): return ini

@lru_persistent_cache(db=DIR)
def stepB(E,fr=None,to=None,r=0):
  p,q = fr
  return ChainMap({to:E[p]+E[q]+r},E)

def proc(rab=1,rbc=2,rabc=3):
  P_ini = MapExpr(stepA,a=1,b=2,c=3)
  P_ab = MapExpr(stepB,P_ini,fr=('a','b'),to='ab',r=rab)
  P_bc = MapExpr(stepB,P_ini,fr=('b','c'),to='bc',r=rbc)
  P_abc = MapExpr(stepB,MapExpr(ChainMap,P_ab,P_bc),fr=('ab','bc'),to='abc',r=rabc)
  return P_abc

#--------------------------------------------------------------------------------------------------

def demo_(t,*L):
  import time, logging
  logging.basicConfig(level=logging.INFO,format='[proc %(process)d @ %(asctime)s] %(message)s',datefmt='%H:%M:%S')
  logger = logging.getLogger()
  time.sleep(float(t))
  for x in L:
    logger.info('Computing: %s',x)
    try: v = eval(x)
    except Exception as e: v = 'raised[{}]'.format(e)
    logger.info('Result: %s = %s',x,v)

def demo():
  import subprocess
  from sys import executable as python
  DEMOS = (
      ((simplefunc.cache,),'simplefunc(1,2) ; simplefunc(1,y=2,z=36)'),
      ((longfunc.cache,),'longfunc(42,6) ; longfunc(None,4)'),
      ((stepA.cache,stepB.cache),'proc()["abc"] ; proc(rabc=4)["abc"] ; proc(rbc=5)["abc"]'),
  )
  for caches,tests in DEMOS:
    print(80*'-')
    for c in caches: print('Clearing',c); c.clear()
    L = tests.split(' ; ')
    for w in [subprocess.Popen([python,__file__,str(t)]+L) for t in (0.,2.)]: w.wait()
    if not automatic:
      try: input('RET: continue; ^-C: stop')
      except: print(); break
