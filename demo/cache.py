# File:                 demo/cache.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the cache module
from make import RUN; RUN(__name__,__file__,2)
#--------------------------------------------------------------------------------------------------

import os,time,functools
from collections import ChainMap
from PYTOOLS import MapExpr, versioned
from PYTOOLS.cache import persistent_cache
persistent_cache = functools.partial(persistent_cache,db=RUN.dir/'cache.dir')

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
def stepI(d,**ini): return dict((k,'I({}{})'.format(v,d)) for k,v in ini.items())

@persistent_cache
def stepK(E,fr=None,to=None,r=None):
  p,q = fr
  return ChainMap({to:'K({},{},{})'.format(E[p],E[q],r)},E)

def proc(rab='ab',rbc='bc',rabc='abc',d='*'):
  P_ini = MapExpr(stepI,d,a='a',b='b',c='c')
  P_ab = MapExpr(stepK,P_ini,fr=('a','b'),to='ab',r=rab)
  P_bc = MapExpr(stepK,P_ini,fr=('b','c'),to='bc',r=rbc)
  P_abc = MapExpr(stepK,MapExpr(ChainMap,P_ab,P_bc),fr=('ab','bc'),to='abc',r=rabc)
  return P_abc

#--------------------------------------------------------------------------------------------------

DEMOS = (
  ( 'simplefunc(1,2)', 'simplefunc(1,y=2)', ),
  ( 'longfunc(42,6)', 'longfunc(None,4)', ),
  ( 'vfunc(3)', ),
  ( "proc()['abc']", "proc(rabc='abc2')['abc']", "proc(rbc='bc2')['abc']", ),
)

def demo(ind=None,ref=None):
  if ind is None:
    # we are in the master process: for each entry in DEMOS, launch 2 slave processes at 2 sec interval and wait for them
    import subprocess,os,time,sys
    for f in simplefunc,longfunc,vfunc,stepI,stepK: f.cache.clear()
    cmd = [sys.executable,__file__,None,None]
    for ind in range(len(DEMOS)):
      print(80*'-')
      cmd[-2] = str(ind)
      cmd[-1] = 'A'; w1 = subprocess.Popen(cmd); time.sleep(2); cmd[-1] = 'B'; w2 = subprocess.Popen(cmd)
      w1.wait(); w2.wait()
      RUN.pause()
  else:
    # we are in a slave process: run the demo for entry *ind* in DEMOS
    import logging, gc
    logging.basicConfig(level=logging.INFO,format='[proc{} @ t=%(asctime)s] %(message)s'.format(ref),datefmt='%S')
    logger = logging.getLogger()
    for expr in DEMOS[int(ind)]:
      gc.collect()
      logger.info('Computing: %s',expr)
      try: val = eval(expr)
      except Exception as exc: val = 'raised[{}]'.format(exc)
      logger.info('Result: %s = %s',expr,val)
