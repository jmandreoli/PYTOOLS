# File:                 tfutil.py
# Creation date:        2017-05-04
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities for tensorflow
#

import os, sys, subprocess
from pathlib import Path
from datetime import datetime
from functools import partial
from itertools import chain
from shutil import rmtree
from collections import OrderedDict, defaultdict, namedtuple
from numpy import zeros, sqrt, square
import tensorflow
from . import html_parlist,HtmlPlugin,time_fmt,unid,odict,html_table
from .monitor import Monitor

#==================================================================================================
def tf_initialise(run,s):
  r"""
Initialises the tensorflow session *s* in *run*.

* If the graph of *s* has an attribute ``saver``, its value must be :const:`None` or a tensorflow saver attached to that graph. The saver is activated to restore the values of the graph variables from the last checkpoint of this run. If the saver is :const:`None` or there is no available checkpoint, the variables are initialised by activating the initialisation ops of the graph.
* Otherwise, the last checkpoint of this run is used to both replace the session graph and initialise its variables. If there is no available checkpoint, an error is raised.
  """
#==================================================================================================
  if hasattr(s.graph,'saver'):
    saver = s.graph.saver
    ckp = None if saver is None else saver.last_checkpoints[-1] if saver.last_checkpoints else tensorflow.train.latest_checkpoint(str(run.path))
    if ckp is None: s.run(s.graph.get_collection_ref(tensorflow.GraphKeys.INIT_OP))
    else: saver.restore(s,ckp)
  else:
    ckp = tensorflow.train.latest_checkpoint(str(run.path))
    saver = tensorflow.train.import_meta_graph(ckp+'.meta')
    saver.restore(s,ckp)
    s.graph.saver = saver
  return s

#==================================================================================================
def tf_monitor(run,su_period=100,ck_period=2000):
  r"""
Returns a loop monitor for this run. When method :meth:`run` of the monitor is invoked with an :class:`Iterable`, it iterates over its elements and performs various operations after each iteration. The operations (executed in the current default session and based on this run) are:

* at specified intervals and at the end, dump a summary of the current state of the graph
* at specified intervals and at the end, save a checkpoint of the current state of the graph

The graph of the default session must have an attribute :attr:`saver` (possibly :const:`None`) used for checkpointing. The monitor is of class :class:`..Monitor` and can thus be combined with other monitors.

:param run: a run object
:type run: :class:`Run`
:param su_period: a tensorflow summary is dumped every *su_period* iterations
:type su_period: :class:`int`
:param ck_period: a tensorflow checkpoint is created every *ck_period* iterations
:type ck_period: :class:`int`
  """
#==================================================================================================
  from itertools import count
  if not hasattr(run,'summary_writer'):
    run.summary_writer = partial(tensorflow.summary.FileWriter,logdir=str(run.path))
  assert isinstance(su_period,int) and isinstance(ck_period,int)
  def cycler(n):
    if n>0:
      r = range(1,n)
      while True: yield True; yield from (False for i in r)
    else:
      while True: yield False
  def coroutine(env):
    s = tensorflow.get_default_session()
    g = s.graph
    summary_ops = g.get_collection_ref(tensorflow.GraphKeys.SUMMARY_OP)
    summary_writer = run.summary_writer(graph=g)
    gstep_ops = g.get_collection_ref(tensorflow.GraphKeys.GLOBAL_STEP)
    assert len(gstep_ops)<=1
    step_getter = (lambda s,step,gstep=gstep_ops[0]: s.run(gstep)) if gstep_ops else (lambda s,step: step)
    try:
      for step,do_su,do_ck in zip(count(1),cycler(su_period),cycler(ck_period)):
        if do_su is True or env.stop is not None:
          vstep = step_getter(s,step)
          for v in s.run(summary_ops):
            summary_writer.add_summary(v,global_step=vstep)
          summary_writer.flush()
        if do_ck is True or env.stop is not None:
          if g.saver is not None:
            vstep = step_getter(s,step)
            g.saver.save(s,save_path=str(run.path/'model.ckpt'),global_step=vstep,write_meta_graph=env.stop)
          if env.logger is not None: env.logger.info('[iterc=%s] checkpoint created',step)
        yield
    finally: summary_writer.close()
  return Monitor(('tfrun',),(coroutine,))

#==================================================================================================
def tf_supervisor(run,**ka):
  r"""
Returns an instance of :class:`tensorflow.train.Supervisor` with logdir pointing to this run, and which terminates each managed session by a checkpoint.
  """
#==================================================================================================
  from contextlib import contextmanager
  class Supervisor (tensorflow.train.Supervisor):
    @contextmanager
    def managed_session(sv,*a,**ka):
      from tensorflow.python.platform import tf_logging as logging
      with super().managed_session(*a,**ka) as s:
        yield s
        logging.info('Saving checkpoint to path %s',sv.save_path)
        if sv.saver is not None: sv.saver.save(s,sv.save_path,global_step=sv.global_step,write_meta_graph=True)
  return Supervisor(logdir=str(run.path),checkpoint_basename='model.ckpt',**ka)

#==================================================================================================
class TF_Model:
  r"""
This class is not meant to be instantiated, only refined. It defines a class of tensorflow graphs designed for a naive externally fed training loop.
  """
#==================================================================================================
#--------------------------------------------------------------------------------------------------
  @classmethod
  def graph(cls,g,*a,saver=None):
    r"""
The list of tensors *a* is meant to be sent to summary. The first one is optimised.
    """
#--------------------------------------------------------------------------------------------------
    with g.as_default():
      gstep = tensorflow.train.create_global_step()
      tensorflow.group(
        cls.optimiser().minimize(a[0]),
        tensorflow.assign_add(gstep,1),*(tf_freeze(t) for t in a),
        name='train')
      tensorflow.add_to_collection(tensorflow.GraphKeys.INIT_OP,tensorflow.global_variables_initializer())
      tensorflow.add_to_collection(tensorflow.GraphKeys.SUMMARY_OP,tensorflow.summary.merge_all())
      g.saver = None if saver is None else tensorflow.train.Saver(**saver)
    return g

#--------------------------------------------------------------------------------------------------
  @classmethod
  def feeder(cls,g,placeholders=None,formatter=(lambda b: b)):
    r"""
:param g: a tensorflow graph
:param placeholders: a list of names of placeholders in *g*
:type placeholders: :class:`List[str]`
:param formatter: a batch formatter function
:type formatter: :class:`Callable[[Any],List[numpy.array]]`

Returns the callable which, given a batch, returns the feed dictionary whose keys are the placeholders denoted by *placeholders* in *g* and values are the corresponding arrays in the list obtained by applying callable *formatter* to the batch.

Must be overloaded in subclasses to provide the values of *placeholders* and *formatter*.
    """
#--------------------------------------------------------------------------------------------------
    def f(batch,setup={},placeholders=[g.get_tensor_by_name(p+':0') for p in placeholders]):
      fd = dict(zip(placeholders,formatter(batch)))
      fd.update(setup)
      return fd
    return f

#--------------------------------------------------------------------------------------------------
  @classmethod
  def trainloop(cls,s,traindata,trainsetup={}):
    r"""
:param s: a tensorflow session
:param traindata: training data as a batch iterator
:type traindata: :class:`Iterable[Any]`
:param trainsetup: feed dictionary added to the feed dictionary of each training batch

Iterates over *traindata* and evaluates the op named ``train`` on each batch. Invokes (just once) method :meth:`feeder` with the graph of session *s* as unique argument to obtain the feeder, which is then applied to each batch to obtain the feed dictionary of its evaluation.
    """
#--------------------------------------------------------------------------------------------------
    feeder = cls.feeder(s.graph)
    for batch in traindata: yield s.run('train',feeder(batch,trainsetup))

#--------------------------------------------------------------------------------------------------
  @classmethod
  def testreport(cls,s,testdata,testsetup={}):
    r"""
:param s: a tensorflow session
:param testdata: test data as a batch iterator
:param testsetup: feed dictionary added to the feed dictionary of the test batch

Returns a status report containing the number of steps so far, and the accuracy computed on the test batch by evaluating the op named ``accuracy``. Invokes :meth:`feeder` with the graph of session *s* as unique argument to obtain the feeder, which is then applied to the test batches to obtain the feed dictionary of its evaluation.
    """
#--------------------------------------------------------------------------------------------------
    feeder = cls.feeder(s.graph)
    q = 0
    for n,batch in enumerate(testdata,1): q += (s.run('accuracy:0',feeder(batch,testsetup))-q)/n
    return {
      'global-step':    s.run(tensorflow.train.get_global_step(s.graph)),
      'test-accuracy':  q,
    }

#==================================================================================================
def tf_categorical_accuracy(y,y_,name='accuracy'):
  r"""
Returns the accuracy op of a batch prediction score tensor *y* to a reference tensor *y_*. Tensors *y,y_* are of shape *(n,d)* and *(n,)*, repectively, where *n* is the size of the current batch and *d* the dimension of the score vector and onehot representation.

:param y,y_: 2d tensorflow nodes
:type y,y_: :class:`tensorflow.Tensor`
:param name: name of the result op
  """
#==================================================================================================
  return tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(y,1,output_type=y_.dtype),y_),tensorflow.float32),name=name)

#==================================================================================================
def tf_categorical_loss(y,y_,name='loss'):
  r"""
Returns the cross-entropy loss op of a batch prediction score tensor *y* to a reference tensor *y_*. Tensors *y,y_* are of shape *(n,d)* and *(n,)*, repectively, where *n* is the size of the current batch and *d* the dimension of the score vector.

:param y,y_: tensorflow nodes
:type y,y_: :class:`tensorflow.Tensor`
:param name: name of the result op
  """
#==================================================================================================
  return tensorflow.reduce_mean(tensorflow.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,logits=y),name=name)

#==================================================================================================
def tf_freeze(t,name=None,summary_cmd=tensorflow.summary.scalar,**ka):
  r"""
Creates a tensorflow variable meant to hold a frozen value of op *t*, and returns the assignment op which updates the created variable with the current value of *t*. The name of the freeze variable, if not provided by *name*, is derived from that of *t* suffixed with an underscore. If *summary_cmd* is not :const:`None`, it is taken to be a summary creation command, invoked with the name of the freeze variable suffixed with 't' and the freeze variable as its first two arguments, and passed the remaining keyword arguments *ka*.

:param t: a tensorflow tensor or operation
:type t: :class:`Union[tensorflow.Tensor,tensorflow.Operation]`
:param name: name of the freeze variable
:type name: :class:`str`
:param summary_cmd: the summary command to include the freeze variable in the summary
:type summary_cmd: :class:`Callable[[str,tensorflow.Tensor,...],None]`
:param ka: passed to the summary command
  """
#==================================================================================================
  if isinstance(t,tensorflow.Operation): op = t; t = t.outputs[0]
  else: assert isinstance(t,tensorflow.Tensor); op = t.op
  if name is None: name = op.name+'_'
  t_ = tensorflow.get_variable(name,shape=t.shape,dtype=t.dtype,trainable=False)
  if summary_cmd is not None: summary_cmd(name+'t',t_,**ka)
  return tensorflow.assign(t_,t)
