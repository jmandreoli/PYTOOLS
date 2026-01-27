# from make import RUN
# at the beginning of a python file to access various doc generation utilities

import os,sys,time,subprocess
from threading import Thread, Timer

class RUN:

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

  @classmethod
  def record(cls,player,ext=None):
    from matplotlib.pyplot import close
    for a in 'get_size_inches','set_size_inches','savefig': setattr(player.board,a,getattr(player.main,a))  # very ugly trick because board is a subfigure
    p_mp4,p_gif = str(cls.path('.mp4',ext)), str(cls.path('.gif',ext))
    player.save(p_mp4)
    close()

if __name__=='__main__':
  from importlib import import_module
  from pathlib import Path
  from itertools import cycle
  from shutil import rmtree
  from socket import getfqdn
  p_in,p_out = (Path(p).resolve() for p in sys.argv[1:])
  RUN = import_module('.make',__package__).RUN
  RUN.target = p_out
  def clock(name,v):
    print(' ',name,end='\r',file=v,flush=True)
    for c in cycle(r'\|/-'):
      time.sleep(.3)
      print(c,end='\r',file=v,flush=True)
  Thread(target=clock,args=(p_in.stem,os.fdopen(os.dup(1),'w'),),daemon=True).start()
  with p_out.open('w') as v:
    n = v.fileno()
    for m in 1,2: os.dup2(n,m)
    exec(p_in.read_text())
  p_out.write_text(p_out.read_text().replace(getfqdn(),'localhost').replace(str(p_out.parent),'/localdir...'))
