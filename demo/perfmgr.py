# File:                 demo/perfmgr.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the perfmgr module

if __name__=='__main__':
  import sys
  from myutil.demo.perfmgr import demo, demo_
  if len(sys.argv) == 1: demo()
  else: demo_(*sys.argv[1:])
  sys.exit(0)

#--------------------------------------------------------------------------------------------------

import logging
from ..perfmgr import sessionmaker, geometric
from pathlib import Path; DIR = Path(__file__).resolve().parent
automatic = False

#--------------------------------------------------------------------------------------------------

def demo_(phase):
  logging.basicConfig(level=logging.INFO,format='[proc %(process)d] %(message)s',datefmt='%H:%M:%S')
  logger = logging.getLogger()
  Session = sessionmaker('sqlite:///{}'.format(DIR/'perfmgr.dir'/'tests.db'))
  orms = Session()
  ctx = orms.root.getcontext(str(DIR/'perfmgr_.py'))
  logger.info('%s performances',phase)
  if phase=='compute':
    for test in 'MatrixSquared','MatrixDeterminant':
      for m in True,False:
        ctx.newexperiment(test,geometric(100,1.2),tmax=5.,manual=m)
  else:
    from matplotlib.pyplot import figure, show
    fig = figure(figsize=(16,8))
    ctx.display(fig,inner='args',outer='name',exc=None,host=None)
    fig.tight_layout()
    fig.savefig(str(DIR/'perfmgr.png'))
    show(not automatic)
  orms.commit()

def demo():
  import subprocess
  from sys import executable as python
  for phase in 'compute','display': subprocess.run([python,__file__,phase])
