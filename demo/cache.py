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
from time import sleep
from os import getpid
from .. import MapExpr, versioned
from ..cache import persistent_cache
automatic = False

@persistent_cache(db=DIR)
def simplefunc(x,y=3): return x,y

@persistent_cache(db=DIR)
def longfunc(x,delay=10):
  sleep(delay)
  if x is None: raise Exception('longfunc error')
  return x

V = getpid()
@persistent_cache(db=DIR)
@versioned(V)
def vfunc(x): return x+V

@persistent_cache(db=DIR)
def stepA(d,**ini): return dict((k,v+d) for k,v in ini.items())

@persistent_cache(db=DIR)
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

def demo_(*L):
  import logging, gc
  logging.basicConfig(level=logging.INFO,format='[proc %(process)d @ %(asctime)s] %(message)s',datefmt='%H:%M:%S')
  logger = logging.getLogger()
  for x in L:
    gc.collect()
    logger.info('Computing: %s',x)
    try: v = eval(x)
    except Exception as e: v = 'raised[{}]'.format(e)
    logger.info('Result: %s = %s',x,v)

def demo():
  import subprocess,os,time
  from sys import executable as python
  DEMOS = (
    'simplefunc(1,2) ; simplefunc(1,y=2)',
    'longfunc(42,6) ; longfunc(None,4)',
    'vfunc(3)',
    "proc()['abc'] ; proc(rabc=4)['abc'] ; proc(rbc=5)['abc']",
  )
  for f in simplefunc,longfunc,vfunc,stepA,stepB: f.cache.clear()
  for tests in DEMOS:
    print(80*'-')
    L = [python,__file__]+tests.split(' ; ')
    def sched():
      for t in (0.,2.): time.sleep(t); yield subprocess.Popen(L)
    for w in list(sched()): w.wait()
    if not automatic:
      try: input('RET: continue; ^-C: stop')
      except: print(); break
