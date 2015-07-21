import os,sys, importlib, time
from pathlib import Path
from threading import Thread
from itertools import cycle
DIR = Path(__file__).parent.resolve()

def clock(module,v):
  print(' ',module,end='\r',file=v,flush=True)
  for c in cycle(r'\|/-'):
    time.sleep(.3)
    print(c,end='\r',file=v,flush=True)
def fordoc(module):
  prefix = str(DIR.parent)
  pout = (DIR/module).with_suffix('.out')
  if pout.with_suffix('.dir').exists():
    for f in list(pout.with_suffix('.dir').iterdir()):
      if f.is_file(): f.unlink()
  mod = importlib.import_module('myutil.demo.'+module)
  mod.automatic = True
  Thread(target=clock,args=(module,os.fdopen(os.dup(1),'w'),),daemon=True).start()
  with pout.open('w') as v:
    os.close(1); os.close(2)
    os.dup2(v.fileno(),1); os.dup2(v.fileno(),2)
    print('python',module,flush=True)
    mod.demo()
  with pout.open('r') as u: r = u.read()
  r = r.replace(prefix,'...')
  with pout.open('w') as v: v.write(r)

if __name__=='__main__':
  module = sys.argv[1]
  fordoc(module)

