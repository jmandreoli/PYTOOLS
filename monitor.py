# File:			monitor.py
# Creation date:	2014-12-01
# Contributors:		Jean-Marc Andreoli
# Language:		Python
# Purpose:		Generic loop control
#
r"""
:mod:`PYTOOLS.monitor` --- Generic loop monitoring
==================================================

This module provides basic functionalities to monitor loops.

An example
----------

The following piece of code illustrates the use of this module.

.. literalinclude:: ../demo/monitor.py
   :language: python
   :tab-width: 2

In :func:`demo1`, the monitored iterable is unbounded, but the monitor stops it by setting a threshold on the number of iterations and the total cpu time. Adjust *maxiter* and *maxcpu* to see how far your kernel can go.

In :func:`demo2`, the monitored iterable is also unbounded, and produces a regular sample of the coordinates of a point moving at constant speed on a regular curve (here a cycloid). A :func:`averaging_monitor` computes statistics on the first coordinate of the sample points. A :func:`buffer_monitor` is used to collect the sample points into a list buffer (add a *size* argument to bound the size of the buffer). Function :func:`display` displays it dynamically on a :mod:`matplotlib` figure.

Typical output:

.. literalinclude:: ../demo/monitor.out

.. |output| image:: ../demo/monitor.gif
   :width: 400px

|output|

Available types and functions
-----------------------------
"""

from functools import partial
from . import basic_stats

#==================================================================================================
class Monitor:
  r"""
Instances of this class provide generic loop monitoring functionality.

:param coroutines: a tuple of generator functions, to be invoked concurrently with the loop to monitor
:param cat: a tuple of labels (for printing purposes only)

Monitors can be composed using the multiplication operator (their *coroutines* are concatenated).

Methods:
  """
#==================================================================================================

  def __init__(self,cat,coroutines=()):
    self.category = cat
    self.coroutines = coroutines

  def __mul__(self,other):
    assert isinstance(other,Monitor)
    return Monitor(self.category+other.category,self.coroutines+other.coroutines)

  def __imul__(self,other):
    assert isinstance(other,Monitor)
    self.category += other.category
    self.coroutines += other.coroutines
    return self

  def __str__(self):
    return 'Monitor<{}>'.format('*'.join(self.category))

#--------------------------------------------------------------------------------------------------
  def run(self,loop,env=None,detach=None,**ka):
    r"""
Enumerates *loop* and monitors it.

:param loop: an iterable yielding arbitrary objects
:param env: an environment, i.e. an object which can be assigned arbitrary attributes
:type env: typically :class:`State`
:param detach: if not None, arguments (e.g. `daemon`) to create a thread on which to run the loop
:type detach: :class:`dict`
:param ka: initialisations of *env* (attribute :attr:`logger` is set as default to :const:`None` in *ka*)
:return: the environment *env* at the end of the loop, or immediately if *detach* is not :const:`None`

If *env* is :const:`None`, it is initialised to a new instance of :class:`State`. The items of *ka* are used to initialise *env*. Its attribute :attr:`stop` is assigned :const:`None`. Its attribute :attr:`thread` is assigned :const:`None` if *detach* is :const:`None`, otherwise a thread object on which the loop is run, initialised by *detach* (key ``daemon`` is by default set to :const:`True` and key ``delay`` is taken to be a time delay in sec before the computation on the thread is started). A list of coroutines is obtained by calling each element of :attr:`coroutines` with argument *env*, then *loop* is enumerated. At the end of each iteration, the following attributes are updated in *env*.

- :attr:`icputime`: cpu time of the last iteration
- :attr:`cputime`: cumulated cpu time of the loop iterations
- :attr:`walltime`: cumulated wall time of the loop iterations
- :attr:`value`: object yielded by the last iteration

Then each coroutine is advanced (using function :func:`next`), possibly updating *env*. The enumeration continues while the attribute :attr:`stop` of *env* remains :const:`None`.
    """
#--------------------------------------------------------------------------------------------------
    from threading import Thread
    from time import time, process_time, sleep
    def run_(loop,env,delay=None,**ka):
      env.stop = None
      env.cputime = 0.
      env.walltime = 0.
      for k,v in ka.items(): setattr(env,k,v)
      coroutines = [coroutine(env) for coroutine in self.coroutines]
      if delay is not None: sleep(delay)
      t_ = process_time(),time()
      for x in loop:
        t = process_time(),time()
        d = env.icputime = t[0]-t_[0]
        env.cputime += d
        env.walltime += t[1]-t_[1]
        env.value = x
        for c in coroutines: next(c)
        if env.stop is not None: break
        t_ = t
      for c in coroutines: c.close()
    ka.setdefault('logger',None)
    if env is None: env = State()
    if detach is None:
      env.thread = None
      run_(loop,env,**ka)
    elif isinstance(detach,dict):
      detach = detach.copy()
      detach.setdefault('daemon',True)
      delay = detach.pop('delay',None)
      if isinstance(delay,(int,float)):
        env.thread = w = Thread(target=run_,args=(loop,env,delay),kwargs=ka,**detach)
        w.start()
      else: raise TypeError('detach[\`delay\`] must be {}|{}, not {}'.format(int,float,type(delay)))
    else:
      raise TypeError('detach must be {}, not {}'.format(dict,type(detach)))
    return env

#==================================================================================================
# utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def monitor(f):
  r"""
Returns a monitor factory associated to function *f*. Meant to be used as a decorator.

:param f: a generator function expecting an environment as first argument
:return: a monitor factory associated to *f*
:rtype: a callable returning :class:`Monitor` instances

The returned factory, when invoked with some arguments, returns a monitor with *f*, bound to the given arguments, as sole coroutine. *f* may yield values. If one of the aguments has the special name ``label``, it is not passed to *f* and its value is taken to be an attribute name of the environment to which the value yielded by *f* is assigned after each iteration.
  """
#--------------------------------------------------------------------------------------------------
  import inspect
  def F(*a,label=None,**ka):
    if label is None:
      def coroutine(env,a=a,ka=ka): return f(env,*a,**ka)
    else:
      assert isinstance(label,str)
      def coroutine(env,a=a,ka=ka):
        for x in f(env,*a,**ka):
          setattr(env,label,x)
          yield
    return Monitor((f.__name__,),(coroutine,))
  sig = inspect.signature(f)
  parm = list(sig.parameters.values())
  del parm[0]
  lbl = inspect.Parameter('label',inspect.Parameter.POSITIONAL_OR_KEYWORD,default=None)
  if parm and parm[-1].kind == inspect.Parameter.VAR_KEYWORD: parm.insert(-1,lbl)
  else: parm.append(lbl)
  F.__signature__ = sig.replace(parameters=parm)
  F.__name__ = f.__name__
  F.__module__ = f.__module__
  F.__doc__ = f.__doc__
  return F

#--------------------------------------------------------------------------------------------------
@monitor
def iterc_monitor(env,maxiter=0,maxcpu=float('inf'),maxwall=float('inf'),show=None,fmt:callable=None):
  r"""
Returns a monitor managing the number of iterations and enabling basic logging.

:param maxiter: stops the loop at that number of iterations (if reached)
:type maxiter: :class:`int`
:param maxcpu: stops the loop after that amount of cpu time (if reached)
:type maxcpu: :class:`float`
:param maxwall: stops the loop after that amount of wall time (if reached)
:type maxwall: :class:`float`
:param show: controls the logging rate (see below)
:type show: :class:`int`
:param fmt: invoked to produce the log string, passed the current iteration count and environment
:type fmt: :class:`Callable[[int,object],str]`
  """
#--------------------------------------------------------------------------------------------------
  from itertools import count
  logger = env.logger
  if logger is None:
    for x in count(1):
      if x == maxiter: env.stop = 'maxiter'
      elif env.cputime>maxcpu: env.stop = 'maxcpu'
      elif env.walltime>maxwall: env.stop = 'maxwall'
      yield x
  elif show is None:
    s = ''
    for x in count(1):
      if x == maxiter: env.stop = 'maxiter'; s = ' !maxiter'
      elif env.cputime>maxcpu: env.stop = 'maxcpu'; s = ' !maxcpu'
      elif env.walltime>maxwall: env.stop = 'maxwall'; s = ' !maxwall'
      logger.info('%s%s',fmt(x,env),s)
      yield x
  else:
    waitshow = 0
    coeff = 1./show-1.
    s = ''
    for x in count(1):
      if x == maxiter: env.stop = 'maxiter'; s = ' !maxiter'
      elif env.cputime>maxcpu: env.stop = 'maxcpu'; s = ' !maxcpu'
      elif env.walltime>maxwall: env.stop = 'maxwall'; s=' !maxwall'
      elif waitshow==0: waitshow = int(x*coeff)
      else: waitshow -= 1; yield x; continue
      logger.info('%s%s',fmt(x,env),s)
      yield x

#--------------------------------------------------------------------------------------------------
@monitor
def accu_monitor(env,targetf=None,initf=None):
  r"""
Returns a monitor which accumulates values from the loop.

:param targetf: extracts from the environment the variable to accumulate (by incremental addition)
:type targetf: :class:`Callable[[object],T']`
:param initf: returns the initial value of the accumulator
:type initf: :class:`Callable[[],T]`

Here, type `T` is assumed to support incremental addition of type `T'`
  """
#--------------------------------------------------------------------------------------------------
  accu = initf()
  while True:
    accu += targetf(env)
    yield accu

#--------------------------------------------------------------------------------------------------
def buffer_monitor(targetf,size=0,val=(),**ka):
  r"""
Returns a monitor which buffers information collected from the loop.

:param targetf: extracts from the environment the variable to buffer
:type targetf: :class:`Callable[[object],Iterable[object]]`
:param size: size of the buffer (if null, the buffer is infinite, otherwise first in first out policy is applied)
:type size: :class:`int`
:param val: initial content of the buffer (truncated if greater than *size*)
:type val: :class:`Iterable[object]`

The buffer state consists of the last *size* results of applying callable *targetf* to the environment at each iteration.
  """
#--------------------------------------------------------------------------------------------------
  class boundedlist (list):
    def __init__(self,size=0,val=()): super().__init__(val); del self[:-size]; self.size = size
    def __iadd__(self,x): L = super().__iadd__(x); del self[:-self.size]; return L
  return accu_monitor(targetf,partial(boundedlist,size,val),**ka)

#--------------------------------------------------------------------------------------------------
def stats_monitor(targetf,**ka):
  r"""
Returns a monitor which computes some basic statistics about the loop.

:param targetf: extracts from the environment the variable on which to compute the statistics
:type targetf: :class:`Callable[[object],Union[int,float,complex,numpy.array]]`

The computed statistics consists of a named triple <count,mean,variance> of the list of results of applying callable *targetf* to the environment at each iteration.
  """
#--------------------------------------------------------------------------------------------------
  return accu_monitor(targetf,basic_stats,**ka)

#--------------------------------------------------------------------------------------------------
class State:
  def __init__(self,**ka): self.__dict__.update(ka)
#--------------------------------------------------------------------------------------------------
