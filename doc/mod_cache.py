# File:                 demo/mod_cache.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the cache module

import time,sys

DEMOS = {
  'simplefunc': ('(1,)','(1,3)','(1,y=3)',),
  'longfunc': ('(42,6)', '(None,4)',),
  'vfunc': ('(3)',),
  'proc': (f"{a}['abc']" for a in ("()", "(rabc='abc2')", "(rbc='bc2')",)),
}

if __name__ == '__main__':
  # Master process. For each entry in DEMOS,
  #   * execute it in 2 separate processes launched at 1 sec interval and
  #   * wait for them both to complete
  from PYTOOLS.cache import CacheDB
  import subprocess
  def spawn(key,runid): # dynamically imports this file as a module and launches demo in separate process
    config = {'_dbpath':dbpath,'_version':runid}
    loglevel,demo = ('INFO','') if key is None else ('WARNING',f'.demo({key!r})')
    instr = f'''import logging,sys
logging.basicConfig(level=logging.{loglevel},stream=sys.stdout,format="[{runid}@%(asctime)s] %(message)s",datefmt="%S")
from PYTOOLS import import_module_from_file
import_module_from_file({modname!r},{source!r},{config!r}){demo}'''
    return subprocess.Popen([sys.executable,'-c',instr])
  source,dbpath = str(RUN.source),str(RUN.path('.dir',rm=True))
  modname = 'example' # __name__ of module loaded in spawned processes (must be the same for all for cacheing to work)
  for key in (None,*DEMOS.keys()):
    print(80*'-',flush=True)
    wA = spawn(key,'A'); time.sleep(1); wB = spawn(key,'B')
    assert not any(rc:=(wA.wait(),wB.wait())), f'rc:{rc}'
else:
  # Spawned process. The cache is shared (persistent) across all spawned processes
  from PYTOOLS import symbolic, versioned
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

  @symbolic
  @persistent_cache
  def stepI(d,**ini): return {k:f'I({v}{d})' for k,v in ini.items()}

  @symbolic
  @persistent_cache
  def stepK(E,fr=None,to=None,r=None):
    p,q = fr
    return E|{to:f'K({E[p]},{E[q]},{r})'}

  def proc(rab='ab',rbc='bc',rabc='abc',d='*'):
    P_ini = stepI.symbolic(d,a='a',b='b',c='c')
    P_ab = stepK.symbolic(P_ini,fr=('a','b'),to='ab',r=rab)
    P_bc = stepK.symbolic(P_ini,fr=('b','c'),to='bc',r=rbc)
    P_abc = stepK.symbolic(P_ab|P_bc,fr=('ab','bc'),to='abc',r=rabc)
    return P_abc

  def demo(key):
    import logging; (logger:=logging.getLogger()).setLevel(logging.INFO)
    for expr in DEMOS[key]:
      expr = key+expr
      logger.info('??? %s',expr)
      try: val = eval(expr)
      except Exception as exc: val = f'raised[{exc}]'
      logger.info('>>> %s = %s',expr,val)
      time.sleep(.1)
