import os,sys, importlib, time
from pathlib import Path; DIR = Path(__file__).resolve().parent
from threading import Thread, Timer
from itertools import cycle
from shutil import rmtree

def clock(name,v):
  print(' ',name,end='\r',file=v,flush=True)
  for c in cycle(r'\|/-'):
    time.sleep(.3)
    print(c,end='\r',file=v,flush=True)
def import_module(p,N):
  mod = [p.stem]
  for _ in range(N): p = p.parent; mod.insert(0,p.name)
  sys.path.insert(0,str(p.parent))
  return importlib.import_module('.'.join(mod))
def RUNdoc(name,N):
  p = DIR/name
  pdir = p.with_suffix('.dir')
  if pdir.exists(): rmtree(str(pdir)); pdir.mkdir()
  pout = p.with_suffix('.out')
  mod = import_module(p,int(N))
  prefix = str(sys.path[0])
  mod.RUN.pause = (lambda: None)
  def play(*s):
    for x in s:
      if isinstance(x,tuple): Timer(*x).start()
      else: x()
  mod.RUN.play = play
  Thread(target=clock,args=(mod.__name__,os.fdopen(os.dup(1),'w'),),daemon=True).start()
  with pout.open('w') as v:
    n = v.fileno()
    for m in 1,2: os.close(m);os.dup2(n,m)
    print('python',mod.__name__,flush=True)
    mod.demo()
  with pout.open('r') as u: r = u.read()
  r = r.replace(prefix,'...')
  with pout.open('w') as v: v.write(r)
def RUN(name,p,N):
  if name=='__main__':
    import sys
    mod = import_module(Path(p).resolve().with_suffix(''),N)
    def pause():
      try: input('RET: continue; ^-C: stop')
      except: print(); sys.exit(0)
    mod.RUN.pause = pause
    mod.RUN.play = (lambda *a,now=None: None)
    mod.demo(*sys.argv[1:])
    sys.exit(0)
RUN.dir = DIR

if __name__=='__main__': RUNdoc(*sys.argv[1:])
