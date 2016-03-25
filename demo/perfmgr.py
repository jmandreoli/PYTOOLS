# File:                 demo/perfmgr.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the perfmgr module

if __name__=='__main__':
  import sys
  from myutil.demo.perfmgr import demo
  demo(*sys.argv[1:])
  sys.exit(0)

#--------------------------------------------------------------------------------------------------

import logging
from ..perfmgr import sessionmaker, geometric, Context, Experiment
from pathlib import Path; DIR = Path(__file__).resolve().parent
automatic = False

#--------------------------------------------------------------------------------------------------

def demo_(compute=False):
  logging.basicConfig(level=logging.INFO,format='[proc %(process)d] %(message)s',datefmt='%H:%M:%S')
  logger = logging.getLogger()
  Session = sessionmaker('sqlite:///{}'.format(DIR/'perfmgr.dir'/'tests.db'),execution_options=dict(isolation_level='SERIALIZABLE'))
  orms = Session()
  ctx = Context.fromfile(str(DIR/'perfmgr_.py'))
  if compute:
    for test in 'MatrixSquared','MatrixDeterminant':
      for m in True,False:
        exp = Experiment.run(ctx,test,geometric(100,1.2),smax=5.,manual=m)
        orms.root.addexperiment(exp)
        orms.commit()
  else: # display
    from matplotlib.pyplot import figure, show
    fig = figure(figsize=(16,8))
    orms.root.addcontext(ctx).display(fig,'args',meter='time')
    fig.tight_layout()
    if automatic: fig.savefig(str(DIR/'perfmgr.png'))
    else: show()
  orms.close()

def demo(phase=None):
  import subprocess
  from sys import executable as python
  if phase is None:
    subprocess.run([python,__file__,'compute'])
    demo_(False)
  else:
    demo_(dict(compute=True,display=False)[phase])
