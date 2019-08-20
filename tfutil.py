# File:                 tfutil.py
# Creation date:        2017-05-04
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities for tensorflow
#

import os, sys, subprocess
from contextlib import contextmanager

import tensorflow

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
  @contextmanager
  def session(cls,path,**ka):
    saver = tensorflow.train.Saver()
    with tensorflow.Session(**ka) as s:
      saver.restore(s,tensorflow.train.latest_checkpoint(str(path)))
      yield s

#--------------------------------------------------------------------------------------------------
  @classmethod
  def trainloop(cls,path,traindata,trainsetup={},**ka):
    r"""
:param path: Checkpointing directory path
:param traindata: train data as batch iterator
:param trainsetup: feed dictionary added to the feed dictionary of each training batch

Iterates over *traindata* and evaluates the op named ``train`` on each batch. Invokes (just once) method :meth:`feeder` with the session as unique argument to obtain the feeder, which is then applied to each batch to obtain the feed dictionary of its evaluation.
    """
#--------------------------------------------------------------------------------------------------
    with tensorflow.train.MonitoredTrainingSession(checkpoint_dir=str(path),**ka) as s:
      feeder = cls.feeder(s.graph)
      for batch in traindata:
        if s.should_stop(): return
        s.run('train',feeder(batch,trainsetup))

#--------------------------------------------------------------------------------------------------
  @classmethod
  def testreport(cls,path,testdata,testsetup={},**ka):
    r"""
:param path: Checkpointing directory path
:param testdata: test data as batch iterator
:param testsetup: feed dictionary added to the feed dictionary of the test batch

Returns a status report containing the number of steps so far, and the accuracy computed on the test batch by evaluating the op named ``accuracy``. Invokes :meth:`feeder` with the graph of session *s* as unique argument to obtain the feeder, which is then applied to the test batches to obtain the feed dictionary of its evaluation.
    """
#--------------------------------------------------------------------------------------------------
    with cls.session(path) as s:
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
