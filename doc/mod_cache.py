# File:                 demo/demo_cache.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the cache module

import os,time,sys

DEMOS = {
  ' simple': ('simplefunc(1,2)', 'simplefunc(1,y=2)',),
  '   long': ('longfunc(42,6)', 'longfunc(None,4)',),
  'version': ('vfunc(3)',),
  'process': ("proc()['abc']", "proc(rabc='abc2')['abc']", "proc(rbc='bc2')['abc']",),
}

if __name__=='__main__':
  # Master process. For each entry in DEMOS,
  #   * execute it in 2 separate processes launched at 2 sec interval and
  #   * wait for them both to complete
  from myutil.make import RUN  # doc gen trick
  from PYTOOLS.cache import CacheDB
  import subprocess
  dbpath = RUN.path('.dir')
  CacheDB(dbpath).clear()
  os.environ['cache_db'] = str(dbpath)
  def spawn(key,runid):
    return subprocess.Popen([sys.executable,'-c',f'from mod_cache import demo; demo(\'{key}\',\'{runid}\')'])
  for key in DEMOS.keys():
    print(80*'-',flush=True)
    wA = spawn(key,'A'); time.sleep(2); wB = spawn(key,'B')
    wA.wait(); wB.wait()
else:
  # Spawned process. Although each execution is in a separate process, the cache is shared.
  from collections import ChainMap
  from PYTOOLS import MapExpr, versioned
  from PYTOOLS.cache import persistent_cache
  import functools; persistent_cache = functools.partial(persistent_cache,db=os.environ['cache_db'])

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
  def stepI(d,**ini): return {k:f'I({v}{d})' for k,v in ini.items()}

  @persistent_cache
  def stepK(E,fr=None,to=None,r=None):
    p,q = fr
    return ChainMap({to:f'K({E[p]},{E[q]},{r})'},E)

  def proc(rab='ab',rbc='bc',rabc='abc',d='*'):
    P_ini = MapExpr(stepI,d,a='a',b='b',c='c')
    P_ab = MapExpr(stepK,P_ini,fr=('a','b'),to='ab',r=rab)
    P_bc = MapExpr(stepK,P_ini,fr=('b','c'),to='bc',r=rbc)
    P_abc = MapExpr(stepK,MapExpr(ChainMap,P_ab,P_bc),fr=('ab','bc'),to='abc',r=rabc)
    return P_abc

  def demo(key,runid):
    import logging
    logging.basicConfig(level=logging.INFO,format=f'[{key} {runid} @ t=%(asctime)s] %(message)s',datefmt='%S')
    logger = logging.getLogger()
    for expr in DEMOS[key]:
      logger.info('Computing: %s',expr)
      try: val = eval(expr)
      except Exception as exc: val = 'raised[{}]'.format(exc)
      logger.info('Result: %s = %s',expr,val)
      time.sleep(.1)
