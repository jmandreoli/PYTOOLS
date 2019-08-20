import os,sys, importlib, time
from pathlib import Path
from threading import Thread
from itertools import cycle
from shutil import rmtree

def clock(module,v):
  print(' ',module,end='\r',file=v,flush=True)
  for c in cycle(r'\|/-'):
    time.sleep(.3)
    print(c,end='\r',file=v,flush=True)
def fordoc(module):
  spec = importlib.util.find_spec(module)
  modpath = Path(spec.origin)
  pdir = modpath.with_suffix('.dir')
  if pdir.exists(): rmtree(str(pdir)); pdir.mkdir()
  pout = modpath.with_suffix('.out')
  prefix = modpath
  for _ in range(len(module.split('.'))): prefix = prefix.parent
  prefix = str(prefix)
  mod = spec.loader.load_module()
  mod.automatic = True
  Thread(target=clock,args=(module,os.fdopen(os.dup(1),'w'),),daemon=True).start()
  with pout.open('w') as v:
    n = v.fileno()
    for m in 1,2: os.close(m);os.dup2(n,m)
    print('python',module,flush=True)
    mod.demo()
  with pout.open('r') as u: r = u.read()
  r = r.replace(prefix,'...')
  with pout.open('w') as v: v.write(r)

if __name__=='__main__': fordoc(*sys.argv[1:])
