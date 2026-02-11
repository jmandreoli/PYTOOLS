# File:                 make.py
# Creation date:        2026-01-27
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Helper functions to facilitate doc generation
#

import os,sys,time
from threading import Thread, Timer

class RUN:

  source = None
  target = None

  @classmethod
  def schedule(cls,*x,daemon=True):
    t = Timer(*x); t.daemon = daemon; t.start()

  @classmethod
  def path(cls,suf,ext=None,rm=False):
    p = cls.target
    if ext is not None: p = p.with_stem(f'{p.stem}-{ext}')
    p = p.with_suffix(suf)
    if rm:
      if p.is_file(): p.unlink()
      elif p.is_dir(): rmtree(p)
    if suf=='.dir': p.mkdir(exist_ok=True)
    return p

if __name__=='__main__':
  from pathlib import Path
  from itertools import cycle
  from shutil import rmtree
  from socket import getfqdn
  p_in,p_out = (Path(p).resolve() for p in sys.argv[1:])
  p_local = Path('.').resolve()
  RUN.source,RUN.target = p_in,p_out.relative_to(p_local)
  def clock(name,v):
    print(' ',name,end='\r',file=v,flush=True)
    for c in cycle(r'\|/-'):
      time.sleep(.3)
      print(c,end='\r',file=v,flush=True)
  Thread(target=clock,args=(p_in.stem,os.fdopen(os.dup(1),'w'),),daemon=True).start()
  with p_out.open('w') as v:
    os.dup2(v.fileno(),1)
    exec(p_in.read_text())
  p_out.write_text(p_out.read_text().replace(getfqdn(),'localhost').replace(str(p_local),'/localdir...'))
