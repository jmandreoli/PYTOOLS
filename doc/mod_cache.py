# File:                 demo/demo_cache.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the cache module

import time,sys

DEMOS = {
  'simplefunc': ('(1,2)', '(1,y=2)',),
  'longfunc': ('(42,6)', '(None,4)',),
  'vfunc': ('(3)',),
  'proc': (f"{a}['abc']" for a in ("()", "(rabc='abc2')", "(rbc='bc2')",)),
}

if __name__ == '__main__':
  # Master process. For each entry in DEMOS,
  #   * execute it in 2 separate processes launched at 2 sec interval and
  #   * wait for them both to complete
  from PYTOOLS.cache import CacheDB
  import subprocess
  source,dbpath = RUN.source,RUN.path('.dir')
  CacheDB(dbpath).clear()
  modname = 'example' # __name__ of module loaded in spawned processes (must be the same for all for cacheing to work)
  source,dbpath = map(str,(source,dbpath))
  def spawn(key,runid):
    # instruction to dynamically import this file as a module and launch demo; must be a standard module (not exec-dict or runpy)
    instr = f'from PYTOOLS import import_module_from_file; import_module_from_file({modname!r},{source!r},dict(_dbpath={dbpath!r},_version={runid!r})).demo({key!r},{runid!r})'
    return subprocess.Popen([sys.executable,'-c',instr])
  for key in DEMOS.keys():
    print(80*'-',flush=True)
    wA = spawn(key,'A'); time.sleep(2); wB = spawn(key,'B')
    if any(rc:=(wA.wait(),wB.wait())): raise Exception(f'rc:{rc}')
else:
  # Spawned process. The cache is shared (persistent) across all spawned processes
  from collections import ChainMap
  from PYTOOLS import MapExpr, versioned
  from PYTOOLS.cache import persistent_cache
  import functools; persistent_cache = functools.partial(persistent_cache,db=_dbpath)

  @persistent_cache
  def simplefunc(x,y=3): return x,y

  @persistent_cache
  def longfunc(x,delay=10):
    time.sleep(delay)
    if x is None: raise Exception('longfunc error')
    return x

  @persistent_cache
  @versioned(_version)
  def vfunc(x): return x*x

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
    logging.basicConfig(level=logging.INFO,format=f'[{runid}@%(asctime)s] %(message)s',datefmt='%S')
    logger = logging.getLogger()
    for expr in DEMOS[key]:
      logger.info('??? %s%s',key,expr)
      try: val = eval(key+expr)
      except Exception as exc: val = f'raised[{exc}]'
      logger.info('>>> %s%s = %s',key,expr,val)
      time.sleep(.1)
