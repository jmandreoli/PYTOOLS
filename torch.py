# File:                 ptutil.py
# Creation date:        2020-01-15
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities for pytorch
#
r"""
:mod:`PYTOOLS.torch` --- Utilities for pytorch
==============================================

This module provides some utilities for pytorch development.

Available types and functions
-----------------------------
"""

from contextlib import contextmanager
from time import time, process_time
from pydispatch import Dispatcher, Property
import torch

#==================================================================================================
class ClassificationRun (Dispatcher):
  r"""
Instances of this class are classification runs. Two kinds of runs are supported:

* ``train`` runs  (method :meth:`exec_train`) are used to train a classification model,
* ``proto`` runs (method :meth:`exec_proto`) are used to discover prototypical instances of each class given a model.

Train runs emit messages during their execution, so can be controlled. Use method :meth:`Dispatcher.bind` (general) or :meth:`ClassificationRun.bind_listeners` (specific) to bind callbacks to the run.
  """
#==================================================================================================

  _events_ = ['open','batch','epoch','close']

  # for all runs:
  ## preset configuration
  lossF: 'Callable[[torch.tensor,torch.tensor],float]'
  r"""Classification loss (cross-entropy)"""
  accuracyF: 'Callable[[torch.tensor,torch.tensor],float]'
  r"""Classification accuracy"""
  ## set at instantiation
  net: 'torch.nn.Module'
  r"""The model"""
  optimiser: 'Callable[[List[torch.nn.Parameter]],torch.optim.Optimizer]'
  r"""The optimiser factory"""
  device: 'torch.device'
  r"""The device on which to execute the run"""
  params: 'Map[str,str]'
  r"""A dictionary for the parameters of the run (subset of attributes)"""

  # for train runs:
  ## set at instantiation
  train_data: 'Iterable[Tuple[torch.tensor,torch.tensor]]'
  r"""[``train``] Each batch is a pair of a tensor of inputs and a tensor of labels (first dim = size of batch)"""
  valid_data: 'Iterable[Tuple[torch.tensor,torch.tensor]]'
  r"""[``train``] Each batch is a pair of a tensor of inputs and a tensor of labels (first dim = size of batch)"""
  test_data: 'Iterable[Tuple[torch.tensor,torch.tensor]]'
  r"""[``train``] Each batch is a pair of a tensor of inputs and a tensor of labels (first dim = size of batch)"""
  ## set at execution
  progress: 'float'
  r"""[``train``] Between 0. and 1., stop when 1. is reached"""
  time: 'Tuple[float,float]'
  r"""[``train``\*] A pair of the walltime and processing time since beginning of run"""
  step: 'int'
  r"""[``train``\*] Number of completed global steps"""
  epoch: 'int'
  r"""[``train``\*] Number of completed epochs"""
  batch: 'int'
  r"""[``train``\*] Number of completed batches"""
  loss: 'float'
  r"""[``train``\*] Mean loss since beginning of current epoch"""
  tnet: 'torch.nn.Module'
  r"""[``train``\*] Copy of :attr:`net` located on :attr:`device`"""
  eval_valid: 'Callable[[],Tuple[float,float]]'
  r"""[``train``\*] Function returning the validation performance (loss,accuracy) at the end of the last completed step"""
  eval_test: 'Callable[[],Tuple[float,float]]'
  r"""[``train``\*] Function returning the test performance (loss,accuracy) at the end of the last completed step"""

  # for proto runs:
  ## set at instantiation
  nepoch: 'int'
  r"""[``proto``] Number of epochs to run"""
  labels: 'Union[int,Iterable[int]]'
  r"""[``proto``] List of labels to process (or range if integer type)"""
  init: 'torch.tensor'
  r"""[``proto``] Initial instance"""
  projection: 'Callable[[torch.tensor],NoneType]'
  r"""[``proto``] Called after each optimiser step to re-project instance within domain range"""

#--------------------------------------------------------------------------------------------------
  def __init__(self,**ka):
#--------------------------------------------------------------------------------------------------
    self.lossF = torch.nn.CrossEntropyLoss()
    self.accuracyF = lambda outputs,labels: torch.mean((torch.argmax(outputs,1)==labels).float())
    self.params = {}
    for k,v in ka.items():
      if k not in ('net','optimiser','device','train_data','valid_data'): self.params[k] = str(v)
      setattr(self,k,v)

#--------------------------------------------------------------------------------------------------
  def exec_train(self):
    r"""
Executes a train run. Parameters passed as attributes.
    """
#--------------------------------------------------------------------------------------------------
    def ondevice(data):
      for inputs,labels in data: yield inputs.to(self.device),labels.to(self.device)
    def eval_cache(data):
      current = None,-1
      def cache():
        nonlocal current
        if current[1] != self.step: current = self.eval(net,ondevice(data)),self.step
        return current[0]
      return cache
    self.tnet = net = self.net.to(self.device)
    self.eval_valid,self.eval_test = eval_cache(self.valid_data),eval_cache(self.test_data)
    optimiser = self.optimiser(net.parameters())
    self.progress = 0.
    self.step = self.epoch = 0
    self.time = 0.,0.
    with training(net,True):
      self.emit('open',self)
      try:
        start = process_time(),time()
        while self.progress<1.:
          self.batch = 0; self.loss = 0.
          for inputs,labels in ondevice(self.train_data):
            optimiser.zero_grad()
            loss = self.lossF(net(inputs),labels)
            loss.backward()
            optimiser.step()
            self.step += 1; self.batch += 1
            self.loss += (loss.item()-self.loss)/self.batch
            self.time = process_time()-start[0],time()-start[1]
            del inputs,labels,loss # free memory before emitting (not sure this works)
            self.emit('batch',self)
          self.epoch += 1
          self.emit('epoch',self)
      finally:
        self.emit('close',self)

#--------------------------------------------------------------------------------------------------
  def exec_proto(self):
    r"""
Executes a proto run. Parameters passed as attributes.
    """
#--------------------------------------------------------------------------------------------------
    def proto(label):
      param = torch.autograd.Variable(init.clone(),requires_grad=True).to(self.device)
      optimiser = self.optimiser([param])
      label = torch.tensor([label]).to(device=self.device)
      for n in range(self.nepoch):
        optimiser.zero_grad()
        loss = self.lossF(net(param),label)
        loss.backward()
        optimiser.step()
        project(param.data)
      return param.detach().numpy()
    project = (lambda a: None) if self.project is None else self.project
    labels = range(self.labels) if isinstance(self.labels,int) else self.labels
    self.tnet = net = self.net.to(self.device)
    init = self.init[None,...] # make a single instance batch
    with training(net,True): return tuple(proto(label)[0] for label in labels)

#--------------------------------------------------------------------------------------------------
  def eval(self,net,data):
    r"""
Evaluates a model on some data.

:param net: the model
:type net: :class:`torch.Module`
:param data: a list of batches
:type data: :class:`Iterable[Tuple[torch.tensor,torch.tensor]]`
    """
#--------------------------------------------------------------------------------------------------
    loss = 0.; accuracy = 0.
    with training(net,False),torch.no_grad():
      for n,(inputs,labels) in enumerate(data,1):
        outputs = net(inputs)
        loss += (self.lossF(outputs,labels).item()-loss)/n
        accuracy += (self.accuracyF(outputs,labels).item()-accuracy)/n
    return loss,accuracy

#--------------------------------------------------------------------------------------------------
  def bind_listeners(self,*listeners):
    r"""
Binds the set of *listeners* (objects) to the events. The list of listeners is processed sequentially. For each listener, the methods of that listener whose name match ``on_`` followed by the name of a supported event are bound to this dispatcher.

:param listeners: list of listener objects
:param type: :class:`List[object]`
    """
#--------------------------------------------------------------------------------------------------
    for x in listeners:
      self.bind(**dict((ev,t) for ev in self._events_ for t in (getattr(x,'on_'+ev,None),) if t is not None))
    return self

#==================================================================================================
class RunBaseListener:
  r"""
Instances of this class control basic classification runs.

:param max_epoch: maximum number of epochs to run
:type max_epoch: :class:`int`
:param max_time: maximum total wall time to run
:type max_time: :class:`int`
:param period: training metrics are logged after this number of batch iterations (repeatedly)
:type period: :class:`int`
:param vperiod: validation metrics are logged after this number of batch iterations (repeatedly)
:type vperiod: :class:`int`
:type vperiod: :class:`int`
:param logger: logger to use for logging information
:type logger: :class:`logging.Logger`
  """
#==================================================================================================

#--------------------------------------------------------------------------------------------------
  def __init__(self,max_epoch=int(1e9),max_time=float('inf'),period:'int'=None,vperiod:'int'=None,logger=None):
#--------------------------------------------------------------------------------------------------
    current = lambda run: (run.time[1],run.step,run.epoch,run.batch)
    header = 'TIME','STEP','EPO','BAT'
    current_fmt = '%6.1f %6d %3d/%4d '
    header_fmt =  '%6s %6s %3s/%4s '
    def header_info(run): logger.info(header_fmt,*header)
    def train_info(run):
      if run.batch%period==0:
        logger.info(current_fmt+'TRAIN loss: %.3f',*current(run),run.loss)
    def valid_info(run):
      if run.batch%vperiod==0:
        logger.info(current_fmt+'VALIDATION loss: %.3f, accuracy: %.3f',*current(run),*run.eval_valid())
    def test_info(run):
      logger.info(current_fmt+'TEST loss: %.3f, accuracy: %.3f',*current(run),*run.eval_test())
    def progress_info(run):
      logger.info(current_fmt+'PROGRESS: %s',*current(run),'{:.0%}'.format(run.progress))
    def set_progress(run): run.progress = min(max(run.epoch/max_epoch,run.time[1]/max_time),1.)
    on_open = None if logger is None else header_info
    on_batch = None if logger is None else compose((None if period is None else train_info),(None if vperiod is None else valid_info))
    on_epoch = compose(set_progress,(None if logger is None else progress_info))
    on_close = None if logger is None else test_info
    self.on_open,self.on_close,self.on_batch,self.on_epoch = on_open,on_close,on_batch,on_epoch

#==================================================================================================
class RunMlflowListener:
  r"""
Instances of this class log information about classification runs into mlflow.

:param period: training metrics are logged after this number of batch iterations (repeatedly)
:type period: :class:`int`
:param vperiod: validation metrics are logged after this number of batch iterations (repeatedly)
:type vperiod: :class:`int`
:param checkpoint_after: the model is logged at the first epoch end after this number of seconds (repeatedly)
:type checkpoint_after: :class:`float`
  """
#==================================================================================================

#--------------------------------------------------------------------------------------------------
  def __init__(self,uri:'str',exp:'str',period:'int'=None,vperiod:'int'=None,checkpoint_after:'float'=None):
#--------------------------------------------------------------------------------------------------
    import mlflow, mlflow.pytorch
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    def train_info(run):
      if run.batch%period==0:
        mlflow.log_metric('tloss',run.loss,run.step)
    def valid_info(run):
      if run.batch%vperiod==0:
        vloss,vaccu = run.eval_valid()
        mlflow.log_metric('vloss',vloss,run.step)
        mlflow.log_metric('vaccu',vaccu,run.step)
    def on_open(run):
      mlflow.start_run()
      mlflow.log_params(run.params)
      run.checkpointed = 0.
    def on_close(run):
      try:
        vloss,vaccu = run.eval_valid()
        mlflow.log_metric('vloss',vloss,run.step)
        mlflow.log_metric('vaccu',vaccu,run.step)
        loss,accu = run.eval_test()
        mlflow.set_tag('test-loss',loss)
        mlflow.set_tag('test-accu',accu)
        mlflow.pytorch.log_model(run.tnet,'model')
      finally:
        mlflow.end_run()
    on_batch = compose((None if period is None else train_info),(None if vperiod is None else valid_info))
    def on_epoch(run):
      t = run.time[1]
      if t-run.checkpointed>checkpoint_after:
        mlflow.pytorch.log_model(run.tnet,'model_{:03d}'.format(run.epoch))
        run.checkpointed = t
    if checkpoint_after is None: on_epoch = None
    self.on_open,self.on_close,self.on_batch,self.on_epoch = on_open,on_close,on_batch,on_epoch

  @staticmethod
  def load_model(uri:'str',exp:'str',run_id:'str'=None,**ka):
    import mlflow, mlflow.pytorch
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    if run_id is None: run_id = mlflow.search_runs().run_id[0] # most recent run
    return mlflow.pytorch.load_model('runs:/{}/model'.format(run_id),**ka)

#==================================================================================================
@contextmanager
def training(net,flag):
#==================================================================================================
  oflag = net.training
  net.train(flag)
  yield
  net.train(oflag)

#==================================================================================================
def compose(*L):
#==================================================================================================
  L = [f for f in L if f is not None]
  if not L: return None
  if len(L)==1: return L[0]
  return lambda run,L=L: tuple(f(run) for f in L)
