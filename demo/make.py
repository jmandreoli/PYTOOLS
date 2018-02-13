import os,sys, importlib, time
from pathlib import Path
from threading import Thread
from itertools import cycle
from shutil import rmtree
DIR = Path(__file__).parent.resolve()

def clock(module,v):
  print(' ',module,end='\r',file=v,flush=True)
  for c in cycle(r'\|/-'):
    time.sleep(.3)
    print(c,end='\r',file=v,flush=True)
def fordoc(module):
  prefix = str(DIR.parent)
  pout = (DIR/module).with_suffix('.out')
  pdir = pout.with_suffix('.dir')
  if pdir.exists():
    for f in list(pdir.iterdir()):
      if f.is_file(): f.unlink()
      else: rmtree(str(f))
  mod = importlib.import_module('PYTOOLS.demo.'+module)
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

if __name__=='__main__':
  module = sys.argv[1]
  fordoc(module)
