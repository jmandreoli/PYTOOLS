# File:                 sge.py
# Creation date:        2015-01-15
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              An extension supporting SGE
#
# *** Copyright (c) 2014 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

from .main import shellscript, clients
from functools import partial
import sys, os

SGESUBHOST = None
SGEENV = None

def sgesetup(subhost=None,env=None):
  r"""
Configures the SGE controller host and the script to execute (source) to setup SGE on that host.
  """
  global SGESUBHOST, SGEENV
  SGESUBHOST, SGEENV = subhost, env
  
def sgeclients(_N,**ka):
  r"""
Instances of this class are lists of clients obtained by the SGE launcher.

:param _N: number of clients to launch
:param ka: additional keyword arguments passed to each invocation of :class:`Client`
  """
  script='source {}; {}'.format(SGEENV,shellscript())
  w = Client(('ssh','-T','-q',SGESUBHOST,'sh -c \'{}\''.format(script)))
  w.declare(sgeclients_)
  ws = w.sgeclients_(_N,**ka)
  w.shutdown()
  return ws

class sgeclients_ (clients):

  def __init__(self,_N,factory,**ka):
    super(sgeclients_,self).__init__(_N*(sgelauncher(),),factory=partial(sgefactory,(SGESUBHOST,SGEENV),factory),**ka)

def sgelauncher():
  return 'qrsh','-cwd','-v','PYTHONPATH={}'.format(os.environ.get('PYTHONPATH','')),'-v','LD_LIBRARY_PATH={}'.format(os.environ.get('LD_LIBRARY_PATH','')),sys.executable,'-m',clients.__module__

def sgefactory(sgecfg,factory,**ka):
  sgesetup(*sgecfg)
  return factory(**ka)

