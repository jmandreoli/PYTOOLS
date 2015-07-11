# File:                 main.py
# Creation date:        2014-06-15
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              A thin layer around Pyro to launch remote daemons
#
# *** Copyright (c) 2014 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import sys, os, Pyro4, subprocess, random, pickle, logging, traceback, base64, functools

logger = logging.getLogger(__name__)

HMAC_KEY = bytes(random.randrange(256) for i in range(64))
def set_hmackey(w): w._pyroHmacKey = HMAC_KEY; return w
def init_hmackey(k): global HMAC_KEY; HMAC_KEY=k; return k

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
#Pyro4.config.SOCK_REUSE = True
Pyro4.config.METADATA = False

def resolve(x):
  if isinstance(x,str):
    module,member = x.rsplit('.',1)
    x = getattr(__import__(module,fromlist=(member,)),member)
  return x

if __name__=='__main__':
  ok = True
  try:
    os.setsid()
    init_hmackey,hmackey,factory,ka = pickle.load(sys.stdin.buffer)
    sys.stdin.close()
    daemon = Pyro4.Daemon(host=os.uname().nodename)
    daemon._pyroHmacKey = init_hmackey(hmackey)
    factory = resolve(factory)
    srv = factory(**ka)
    uri = daemon.register(srv)
    del srv
    sys.stdout.write(uri.asString())
  except:
    ok = False
    sys.stdout.write('\n')
    sys.stdout.write(base64.b64encode(traceback.format_exc().encode()).decode())
  sys.stdout.write('\n')
  sys.stdout.close()
  if ok:
    daemon.requestLoop()
    daemon.close()
    sys.exit(0)
  else:
    sys.exit(1)

def shellscript():
  r"""
Returns the shell script which launches a python process with the same PYTHONPATH, LD_LIBRARY_PATH and current directory, then runs this module using the -m option (hence with name __main__), which performs the following:

* reads from stdin a pickled triple ( *hmackey* , *factory* , *kwargs* ) and closes stdin
* sets the host (server) Pyro HMAC key to *hmackey* (a :const:`str`\ )
* creates an object *srv* by invoking *factory* (a callable) with keyword arguments *kwargs* (a :const:`dict`\ )
* starts a Pyro daemon and registers *srv* in it
* writes the Pyro uri of *srv* on stdout and closes it

If *factory* is a string, it must be a qualified function name. That function is loaded and used instead.
  """
  return 'cd {}; export PYTHONPATH={}; export LD_LIBRARY_PATH={}; exec {} -m {}'.format(os.path.abspath(os.getcwd()),os.environ.get('PYTHONPATH',''),os.environ.get('LD_LIBRARY_PATH',''),sys.executable,__name__)

#--------------------------------------------------------------------------------------------------
def server(exiting={},**body):
  r"""
The default factory. It creates a fully configurable server object.

:param body: a dictionary of method definitions
:param exiting: a dictionary giving the exit wrapper of each method (if any)

A method can have one of the following wrappers:

* :const:`None`: the result is returned verbatim unless the method is detected to be a generator function, in which case, same as 'multi'
* 'async': the call is returned immediately with no result
* 'shutdown': no result is returned and the server shuts down (implies async)
* 'proxy': the result is registered in the server daemon and a Proxy is returned
* 'iproxy': the result is expected to be iterable and an IterableProxy is returned
* a callable: the result of invoking that callable to the pair (server,result) is returned
  """
#--------------------------------------------------------------------------------------------------
  def wrap(F,exit=None):
    def multi(F):
      import inspect
      g = F
      while isinstance(g,functools.partial): g = g.func
      return inspect.isgeneratorfunction(g)
    if not callable(F): raise TypeError('Expected callable found {}'.format(type(F)))
    async = False
    if exit is None: exit = iterproxy if multi(F) else verbatim
    elif exit=='async': exit = verbatim; async = True
    elif exit=='shutdown': exit = shutdown
    elif exit=='proxy': exit = simpleproxy
    elif exit=='iproxy': exit = iterproxy
    elif callable(exit): pass
    else: raise ValueError('Expected \'async\'|\'shutdown\'|\'proxy\'|\'iproxy\'|None|callable')
    def F_(self,*a,**ka): return exit(self,F(*a,**ka))
    return Pyro4.core.oneway(F_) if async else F_
  def declare(self,f,att=None,**ka):
    F = resolve(f)
    if att is None: att = F.__name__
    else: raise TypeError('Expected {} found {}'.format(str,type(att)))
    setattr(self.__class__,att,wrap(F,**ka))
  iterproxy = lambda server,r: Iterable(server._pyroDaemon,r)
  simpleproxy = lambda server,r: Object(server._pyroDaemon,r)
  verbatim = lambda server,r: r
  shutdown = lambda server,r=None: server._pyroDaemon.shutdown()
  bbody = dict(declare=declare,shutdown=Pyro4.core.oneway(shutdown))
  bbody.update((att,wrap(resolve(f),exit=exiting.get(att))) for att,f in body.items())
  c = type('AutoServer',(),bbody)
  return c()

#--------------------------------------------------------------------------------------------------
class Client(Pyro4.Proxy):
  r"""
An instance of this class launches a server then acts as a Pyro proxy to that server.

:param launchcmd: a tuple passed to :class:`subprocess.Popen` to launch the server
:param factory: a function run on the server which creates an object of which this is a Pyro proxy
:type factory: function | :const:`str`
:param ka: a keyword argument dictionary passed to the factory
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,launchcmd,factory=server,**ka):
    subp = subprocess.Popen(launchcmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    try: pickle.dump((init_hmackey,HMAC_KEY,factory,ka),subp.stdin)
    finally: subp.stdin.close()
    uri = subp.stdout.readline().strip()
    if not uri:
      try: detail = base64.b64decode(subp.stdout.readline()).decode()
      except: detail = ''
      raise ClientException(launchcmd,subp.poll(),detail)
    uri = uri.decode()
    subp.stdout.close()
    super(Client,self).__init__(uri)
    self._pyroHmacKey = HMAC_KEY

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
class ServerObject:
#--------------------------------------------------------------------------------------------------
  getproxy = Proxy
  def __new__(cls,daemon,*a,**ka):
    if daemon is None: return set_hmackey(a[0])
    else: return super(ServerObject,cls).__new__(cls)
  def __init__(self,obj):
    uri = daemon.register(obj)
    self.mproxy = Pyro4.Proxy(daemon.register(self))
    self.proxy = self.getproxy(uri)
  def __getnewargs__(self): return None,self.proxy
class Proxy (Pyro4.Proxy):
  def __init__(self,uri):
    self._remoteProxy =
    super()
  def __del__(self):
    try: self._remoteProxy.close()
    except: pass
    super(Proxy,self).__del__()
#--------------------------------------------------------------------------------------------------
class ServerIterableObject (ServerObject):
#--------------------------------------------------------------------------------------------------
  getproxy = IterableProxy
  def __init__(self,iterator):
    self.iterator = iterator
    super(ServerIterableObject,self).__init__(obj=self)
  def next(self): return next(self.iterator)
class IterableProxy (Proxy):
  def __iter__(self): return self
  def __next__(self): return self.next()

#--------------------------------------------------------------------------------------------------
class ClientException (Exception): pass
#--------------------------------------------------------------------------------------------------

