# File:			monitor.py
# Creation date:	2014-12-01
# Contributors:		Jean-Marc Andreoli
# Language:		Python
# Purpose:		Generic loop control
#
# *** Copyright (c) 2013 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***

import threading
from time import clock, time
from collections import namedtuple
from functools import wraps, partial

#==================================================================================================
class Monitor:
  r"""
Instances of this class provide generic loop monitoring functionality.

:param coroutines: a tuple of generators, to be started concurrently with the loop to monitor
:param cat: a tuple of labels (for printing purposes only)

Monitors can be composed using the multiplication operator (their *coroutines* are concatenated).

.. |monitor| replace:: This is passed through the :func:`monitor` decorator, so the first argument *env* should be ignored in invocations, and a *label* argument can be specified. This generator yields the iteration count (assigned to the attribute named by *label* if present).

Methods:
  """
#==================================================================================================

  def __init__(self,cat,coroutines=()):
    self.category = cat
    self.coroutines = coroutines

  def __mul__(self,other):
    assert isinstance(other,Monitor)
    return Monitor(self.category+other.category,self.coroutines+other.coroutines)

  def __str__(self):
    return 'Monitor<{}>'.format('*'.join(self.category))

#--------------------------------------------------------------------------------------------------
  def run(self,loop,env=None,detach=False,daemon=True,**ka):
    r"""
Enumerates *loop* and monitors it.

:param loop: an iterable yielding arbitrary objects
:param env: an environment, i.e. an object which can be assigned arbitrary attributes
:type env: typically :class:`State`\|\ :class:`NoneType`
:param detach: whether to start the loop on a separate thread
:type detatch: :class:`bool`
:param daemon,ka: passed to the thread in detach-mode, otherwise ignored
:return: the environment *env* at the end of the loop, or immediately if *detatch* is :const:`True`

If *env* is :const:`None`, it is initialised to a new instance of :class:`State`. Its attribute :attr:`stop` is assigned :const:`None`. Its attribute :attr:`thread` is assigned :const:`None` if *detach* is :const:`False`, otherwise the thread object on which the loop is run. A list of coroutines is obtained by calling each element of :attr:`coroutines` with argument *env*, then *loop* is enumerated. At the end of each iteration, the following attributes are set in *env*.

- :attr:`cputime`: cumulated cpu time of the loop iterations
- :attr:`value`: object yielded by the last iteration

Then each coroutine is advanced (using function :func:`next`), possibly updating *env*. The enumeration continues while the attribute :attr:`stop` of *env* remains :const:`None`.
    """
#--------------------------------------------------------------------------------------------------
    if env is None: env = State()
    if detach:
      w = threading.Thread(target=partial(self.run0,loop,env),daemon=daemon,**ka)
      env.thread = w
      w.start()
    else:
      env.thread = None
      self.run0(loop,env)
    return env
  def run0(self,loop,env):
    coroutines = [coroutine(env) for coroutine in self.coroutines]
    env.stop = None
    env.cputime = 0.
    t0 = clock()
    for x in loop:
      t = clock()
      env.cputime += t-t0
      env.value = x
      for c in coroutines: next(c)
      if env.stop is not None: break
      t0 = t

#==================================================================================================
# utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
def monitor(f):
  r"""
Returns a monitor associated to function *f*. Meant to be used as a decorator.

:param f: a generator expecting an environment as first argument
:return: a monitor associated to *f*
:rtype: :class:`Monitor`

*f* is taken to be the sole coroutine of the returned monitor. *f* may yield values. In that case, if the monitor is called with an argument named *label*, its value is taken to be an attribute name of the environment to which the value yielded by *f* is assigned after each iteration.
  """
#--------------------------------------------------------------------------------------------------
  @wraps(f)
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
  return F

#--------------------------------------------------------------------------------------------------
@monitor
def iterc_monitor(env,maxiter=0,maxcpu=float('inf'),show=None,fmt=None,logger=None):
  r"""
Returns a monitor managing the number of iterations and making basic logging.

:param maxiter: stops the loop at that number of iterations (if reached)
:type maxiter: :class:`int`
:param maxcpu: stops the loop after that amount of cpu time (if reached)
:type maxcpu: :class:`float`
:param show: controls the logging rate (see below)
:type show: :class:`float`\|\ :class:`NoneType`
:param fmt: invoked to produce the log string, passed the current iteration count and environment
:type fmt: callable
:param logger: the logger to use
:type logger: :class:`logging.Logger`\|\ :class:`NoneType`

If *logger* is :const:`None`, no logging occurs (*fmt* and *show* are ignored). Otherwise, if *show* is :const:`None`, logging occurs at each iteration. Otherwise, logging occurs every period roughly equal to *show* times the iteration count (hence, the logging rate slows down with the number of iterations).

|monitor|
  """
#--------------------------------------------------------------------------------------------------
  x = 1
  if logger is None:
    while True:
      if x == maxiter: env.stop = 'maxiter'
      elif env.cputime>maxcpu: env.stop = 'maxcpu'
      yield x
      x += 1
  elif show is None:
    while True:
      if x == maxiter: env.stop = 'maxiter'
      elif env.cputime>maxcpu: env.stop = 'maxcpu'
      logger.info('%s %s',fmt(x,env),'' if env.stop is None else '!'+env.stop)
      yield x
      x += 1
  else:
    assert isinstance(show,float) and show<=1.
    lastshow = 0
    while True:
      if x == maxiter: env.stop = 'maxiter'
      elif env.cputime>maxcpu: env.stop = 'maxcpu'
      if env.stop or show*x>lastshow:
        logger.info('%s %s',fmt(x,env),'' if env.stop is None else '!'+env.stop)
        lastshow = x
      yield x
      x += 1

#--------------------------------------------------------------------------------------------------
@monitor
def averaging_monitor(env,targetf=None,rtype=namedtuple('stats',('count','mean','var'))):
  r"""
Returns a monitor which computes a triple <length,expectation,variance> of the list of results of applying callable *targetf* to the environment at each iteration.

|monitor|
  """
#--------------------------------------------------------------------------------------------------
  n,xmean,xvar = 0,0.,0.
  while True:
    x = targetf(env)
    n += 1
    d = x-xmean
    xmean += d/n
    xvar += ((1.-1./n)*d*d-xvar)/n
    yield rtype(n,xmean,xvar)

#--------------------------------------------------------------------------------------------------
@monitor
def buffer_monitor(env,buf,size=0,targetf=None):
  r"""
Returns a monitor which buffers the results of applying callable *targetf* to the environment at each iteration.

|monitor|
  """
#--------------------------------------------------------------------------------------------------
  while True:
    x = targetf(env)
    buf.append(x)
    del buf[:-size]
    yield x

#--------------------------------------------------------------------------------------------------
class State: pass
#--------------------------------------------------------------------------------------------------


