# File:			monitor.py
# Creation date:	2014-12-01
# Contributors:		Jean-Marc Andreoli
# Language:		Python
# Purpose:		Generic loop control
#
# *** Copyright (c) 2013 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***

import logging
from collections import namedtuple
from . import type_annotation_autocheck

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

  def __str__(self):
    return 'Monitor<{}>'.format('*'.join(self.category))

#--------------------------------------------------------------------------------------------------
  def run(self,loop,env=None,detach=None,**ka):
    r"""
Enumerates *loop* and monitors it.

:param loop: an iterable yielding arbitrary objects
:param env: an environment, i.e. an object which can be assigned arbitrary attributes
:type env: typically :class:`State`\|\ :class:`NoneType`
:param detach: if not None, time (in sec) after which the loop is started on a separate thread
:type detach: :class:`float`\|\ :class:`NoneType`
:param ka: passed to the thread in detach-mode, otherwise ignored (by default, key ``daemon`` in *ka* is set to :const:`True`)
:return: the environment *env* at the end of the loop, or immediately if *detach* is not :const:`None`

If *env* is :const:`None`, it is initialised to a new instance of :class:`State`. Its attribute :attr:`stop` is assigned :const:`None`. Its attribute :attr:`thread` is assigned :const:`None` if *detach* is :const:`None`, otherwise the thread object on which the loop is run. A list of coroutines is obtained by calling each element of :attr:`coroutines` with argument *env*, then *loop* is enumerated. At the end of each iteration, the following attributes are set in *env*.

- :attr:`icputime`: cpu time of the last iteration
- :attr:`cputime`: cumulated cpu time of the loop iterations
- :attr:`value`: object yielded by the last iteration

Then each coroutine is advanced (using function :func:`next`), possibly updating *env*. The enumeration continues while the attribute :attr:`stop` of *env* remains :const:`None`.
    """
#--------------------------------------------------------------------------------------------------
    from threading import Thread
    from time import process_time, sleep
    from functools import partial
    def run0(loop,env,delay=None):
      coroutines = [coroutine(env) for coroutine in self.coroutines]
      env.stop = None
      env.cputime = 0.
      if delay is not None: sleep(delay)
      t0 = process_time()
      for x in loop:
        t = process_time()
        d = env.icputime = t-t0
        env.cputime += d
        env.value = x
        for c in coroutines: next(c)
        if env.stop is not None: break
        t0 = t
    if env is None: env = State()
    if detach is None:
      env.thread = None
      run0(loop,env)
    elif not isinstance(detach,(int,float)):
      raise TypeError('Expected {}|{}, found {}'.format(int,float,type(detach)))
    else:
      ka.setdefault('daemon',True)
      w = Thread(target=partial(run0,loop,env,detach),**ka)
      env.thread = w
      w.start()
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
  parm.append(inspect.Parameter('label',inspect.Parameter.POSITIONAL_OR_KEYWORD,default=None))
  F.__signature__ = sig.replace(parameters=parm)
  F.__name__ = f.__name__
  F.__module__ = f.__module__
  F.__doc__ = f.__doc__
  return F

#--------------------------------------------------------------------------------------------------
@type_annotation_autocheck
@monitor
def iterc_monitor(env,maxiter:int=0,maxcpu:float=float('inf'),logger:logging.Logger=None,show:(float,type(None))=None,fmt:(callable,type(None))=None):
  r"""
Returns a monitor managing the number of iterations and enabling basic logging.

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
:param label: name of an attribute of the environment to which the iteration count is assigned at each iteration
:type label: :class:`str`\|\ :class:`NoneType`

If *logger* is :const:`None`, no logging occurs (*fmt* and *show* are ignored). Otherwise, if *show* is :const:`None`, logging occurs at each iteration. Otherwise, logging occurs every period roughly equal to *show* times the iteration count (hence, the logging rate slows down with the number of iterations).
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
    waitshow = 0
    coeff = 1/show-1
    while True:
      if x == maxiter: env.stop = 'maxiter'
      elif env.cputime>maxcpu: env.stop = 'maxcpu'
      if env.stop is not None: logger.info('%s !%s',fmt(x,env),env.stop)
      elif waitshow==0:
        logger.info('%s',fmt(x,env))
        waitshow = int(x*coeff)
      else: waitshow -= 1
      yield x
      x += 1

#--------------------------------------------------------------------------------------------------
@type_annotation_autocheck
@monitor
def averaging_monitor(env,targetf:callable=None,rtype=namedtuple('stats',('count','mean','var'))):
  r"""
Returns a monitor which computes some basic statistics about the loop.

:param targetf: extracts from the environment the variable on which to compute the statistics
:type targetf: callable
:param label: name of an attribute of the environment to which the computed stats is assigned at each iteration
:type label: :class:`str`\|\ :class:`NoneType`

The computed statistics consists of a named triple <count,mean,variance> of the list of results of applying callable *targetf* to the environment at each iteration.
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
@type_annotation_autocheck
@monitor
def buffer_monitor(env,size:int=0,targetf:callable=None):
  r"""
Returns a monitor which buffers information collected from the loop.

:param targetf: extracts from the environment the variable to buffer
:type targetf: callable
:param size: size of the buffer (if null, the buffer is infinite, otherwise first in first out policy is applied)
:type size: :class:`int`
:param label: name of an attribute of the environment to which the buffer is assigned at each iteration
:type label: :class:`str`\|\ :class:`NoneType`
  """
#--------------------------------------------------------------------------------------------------
  buf = []
  while True:
    x = targetf(env)
    buf.append(x)
    del buf[:-size]
    yield buf

#--------------------------------------------------------------------------------------------------
class State: pass
#--------------------------------------------------------------------------------------------------


