# File:                 demo/perfmgr.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the perfmgr module
from make import RUN; RUN(__name__,__file__,2)
#--------------------------------------------------------------------------------------------------

import logging
from ..perfmgr import sessionmaker, geometric, Context, Experiment
from pathlib import Path; DIR = Path(__file__).resolve().parent

def demo_(compute=False):
  logging.basicConfig(level=logging.INFO,format='[proc %(process)d] %(message)s',datefmt='%H:%M:%S')
  logger = logging.getLogger()
  Session = sessionmaker('sqlite:///{}'.format(DIR/'perfmgr.dir'/'tests.db'),execution_options=dict(isolation_level='SERIALIZABLE'))
  orms = Session()
  ctx = Context.fromfile(str(DIR/'perfmgr_.py'))
  if compute:
    for test in 'MatrixSquared','MatrixDeterminant':
      for m in True,False:
        exp = ctx.run(test,geometric(100,1.2),smax=5.,manual=m)
        orms.root.addexperiment(exp)
        orms.commit()
  else: # display
    from matplotlib.pyplot import figure, show, close
    fig = figure(figsize=(16,8))
    orms.root.addcontext(ctx).display(fig,'args',meter='time')
    fig.tight_layout()
    def action(): fig.savefig(str(RUN.dir/'perfmgr.png')); close()
    RUN.play(action)
    show()
  orms.close()

def demo(phase=None):
  import subprocess
  from sys import executable as python
  if phase is None:
    subprocess.run([python,__file__,'compute'])
    demo_(False)
  else:
    demo_(dict(compute=True,display=False)[phase])
