# File:                 main.py
# Creation date:        2014-06-15
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              A thin layer around Pyro to launch remote daemons
#
# *** Copyright (c) 2014 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import sys, os, pickle, Pyro4

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
#Pyro4.config.SOCK_REUSE = True
Pyro4.config.METADATA = False

#--------------------------------------------------------------------------------------------------
if __name__=='__main__':
#--------------------------------------------------------------------------------------------------
  ok = True
  try:
    os.setsid()
    ini = pickle.load(sys.stdin.buffer)
    sys.stdin.close()
    daemon = Pyro4.Daemon(host=os.uname().nodename)
    proxy = ini(daemon)
    del ini
  except Exception as e:
    ok = False
    proxy = e
  pickle.dump(proxy,sys.stdout.buffer)
  sys.stdout.close()
  if ok:
    daemon.requestLoop()
    daemon.close()
    sys.exit(0)
  else:
    sys.exit(1)

import subprocess, random, logging, functools

logger = logging.getLogger(__name__)

#--------------------------------------------------------------------------------------------------
class InitialServer:
#--------------------------------------------------------------------------------------------------
  @classmethod
  def getproxy(cls,launchcmd,*a,**ka):
    """
Launches a server, 
    subp = subprocess.Popen(launchcmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    try: pickle.dump(cls(HMAC_KEY,*a,**ka),subp.stdin)
    finally: subp.stdin.close()
    try: proxy = pickle.load(subp.stdout)
    except Exception as e: raise ClientException(launchcmd,subp.poll(),e)
    if isinstance(proxy,Exception): raise ClientException(launchcmd,subp.poll(),proxy)
    return proxy
#--------------------------------------------------------------------------------------------------
  def __init__(self,hmackey,factory,*a,**ka):
    self.hmackey = hmackey
    self.factory = resolve(factory)
    self.args = a,ka
#--------------------------------------------------------------------------------------------------
  def __call__(self,daemon):
    global HMAC_KEY
    daemon._pyroHmacKey = HMAC_KEY = self.hmackey
    a,ka = self.args
    srv = self.factory(daemon,*a,**ka)
    assert isinstance(srv,BaseServer)
    return srv.proxy

HMAC_KEY = bytes(random.randrange(256) for i in range(64))

def shellscript():
  r"""
Returns the shell script which launches a python process with the same PYTHONPATH and current directory, then runs this module using the -m option (hence with name __main__).
  """
  return 'cd {}; export PYTHONPATH={}; exec {} -m {}'.format(os.path.abspath(os.getcwd()),os.environ.get('PYTHONPATH',''),sys.executable,__name__)

#--------------------------------------------------------------------------------------------------
class clients (list):
  r"""
An instance of this class is simply a list of :class:`Client` instances.

:param L: a list of launchers (argument *launchcmd* of :class:`Client`)
:param ka: additional keyword arguments passed to each invocation of :class:`Client`
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,L,check=False,**ka):
    Lexc = []
    def tryall(L,**ka):
      for launcher in L:
        try: yield Client(launcher,**ka)
        except Exception as e:
          Lexc.append(e.with_traceback(sys.exc_info()[2]))
          if check: break
    super(clients,self).__init__(tryall(L,**ka))
    if Lexc:
      if check:
        self.shutdown()
        raise Lexc[0]
      else:
        logger.warn('Unable to launch %s.Client\n%s',Client.__module__,''.join('\t{}\n'.format(e) for e in Lexc))

  def shutdown(self):
    r"""
Invoke :meth:`shutdown` on all the clients in the list.
    """
    for w in self: w.shutdown()

#--------------------------------------------------------------------------------------------------
class shclients (clients):
  r"""
Instances of this class are lists of clients obtained by the SH launcher.

:param _N: number of clients to launch
:param ka: additional keyword arguments passed to each invocation of :class:`Client`
  """
#--------------------------------------------------------------------------------------------------
  def __init__(self,_N,**ka):
    launchers = _N*(('sh','-c',shellscript()),)
    super(shclients,self).__init__(launchers,**ka)

#--------------------------------------------------------------------------------------------------
class sshclients (clients):
  r"""
Instances of this class are lists of clients obtained by the SSH launcher.

:param _hosts: available hosts for launching
:type _hosts: list(pair(:class:`str`,\ :class:`int`))
:param ka: additional keyword arguments passed to each invocation of :class:`Client`

Each item in *_hosts* specifies a hostname and
the maximum number of processes which can be started on that host.
  """
#--------------------------------------------------------------------------------------------------
  def __init__(self,_hosts,**ka):
    N,hosts = _hosts
    hosts = tuple(host for _,host in sorted((i,h) for h,n in hosts for i in range(n)))[:N]
    script = 'sh -c \'{}\''.format(shellscript())
    launchers = (('ssh','-T','-q',host,script) for host in hosts)
    super(sshclients,self).__init__(launchers,**ka)

#--------------------------------------------------------------------------------------------------
class Proxy (Pyro4.Proxy):
#--------------------------------------------------------------------------------------------------
  def __init__(self,uri):
    super(Proxy,self).__init__(uri)
    self._pyroHmacKey = HMAC_KEY
  def __del__(self):
    self.unregister__()
    super(Proxy,self).__del__()
class Server:
  getproxy = Proxy
  def __new__(cls,daemon,*a,**ka):
    if daemon is None: self._pyroHmacKey = a[0]
    else:
      self = super(Server,cls).__new__(cls,*a,**ka)
      uri = daemon.register(self)
      self.proxy = self.getproxy(uri)
    return self
  def unregistered__(self): self._pyroDaemon.unregister(self)
  def __getnewargs__(self): return None,self.proxy

#--------------------------------------------------------------------------------------------------
def resolve(x):
#--------------------------------------------------------------------------------------------------
  if isinstance(x,str):
    module,member = x.rsplit('.',1)
    x = getattr(__import__(module,fromlist=(member,)),member)
  return x

#--------------------------------------------------------------------------------------------------
class ClientException (Exception): pass
#--------------------------------------------------------------------------------------------------
