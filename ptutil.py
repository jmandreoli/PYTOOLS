# File:                 ptutil.py
# Creation date:        2020-01-15
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities for pytorch
#
r"""
:mod:`PYTOOLS.ptutil` --- Utilities for pytorch
===============================================

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
Instances of this class are classification runs.
  """
#==================================================================================================

  _events_ = ['open','batch','epoch','close']

  net = None
  optimiser = None
  device = None
  train_data = None
  valid_data = None

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
  def train(self):
#--------------------------------------------------------------------------------------------------
    self.tnet = net = self.net.to(self.device)
    optimiser = self.optimiser(net.parameters())
    with training(net,True):
      self.progress = 0.
      self.step = self.epoch = 0
      self.walltime = self.cputime = 0.
      self.valid_ = None,-1
      self.emit('open',self)
      starttimes = process_time(),time()
      try:
        while self.progress<1.:
          self.batch = 0; self.loss = 0.
          for inputs,labels in self.train_data:
            inputs,labels = inputs.to(self.device),labels.to(self.device)
            optimiser.zero_grad()
            loss = self.lossF(net(inputs),labels)
            loss.backward()
            optimiser.step()
            self.step += 1
            self.batch += 1; self.loss += (loss.item()-self.loss)/self.batch
            self.walltime,self.cputime = time()-starttimes[1],process_time()-starttimes[0]
            self.emit('batch',self)
          self.epoch += 1
          self.emit('epoch',self)
      finally:
        self.emit('close',self)

#--------------------------------------------------------------------------------------------------
  @property
  def valid(self):
#--------------------------------------------------------------------------------------------------
    v = self.valid_
    if v[1] != self.step: self.valid_ = v = self.predict(self.valid_data),self.step
    return v[0]

#--------------------------------------------------------------------------------------------------
  def predict(self,data):
#--------------------------------------------------------------------------------------------------
    net = self.net.to(self.device)
    loss = 0.; accuracy = 0.
    with training(net,False),torch.no_grad():
      for n,(inputs,labels) in enumerate(data,1):
        inputs,labels = inputs.to(self.device),labels.to(self.device)
        outputs = net(inputs)
        loss += (self.lossF(outputs,labels).item()-loss)/n
        accuracy += (self.accuracyF(outputs,labels).item()-accuracy)/n
    return loss,accuracy

#--------------------------------------------------------------------------------------------------
  def bind_listeners(self,*a):
#--------------------------------------------------------------------------------------------------
    for x in a:
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
  def __init__(self,max_epoch=int(1e9),max_time=float('inf'),period:int=None,vperiod:int=None,logger=None):
#--------------------------------------------------------------------------------------------------
    current = lambda run: (run.walltime,run.step,run.epoch,run.batch)
    header = 'TIME','STEP','EPO','BAT'
    current_fmt = '%6.1f %6d %3d/%4d '
    header_fmt =  '%6s %6s %3s/%4s '
    def header_info(run): logger.info(header_fmt,*header)
    def train_info(run):
      if run.batch%period==0:
        logger.info(current_fmt+'TRAIN loss: %.3f',*current(run),run.loss)
    def valid_info(run):
      if run.batch%vperiod==0:
        logger.info(current_fmt+'VALIDATION loss: %.3f, accuracy: %.3f',*current(run),*run.valid)
    def all_info(run): train_info(run); valid_info(run)
    on_open = None if logger is None else header_info
    on_batch = None if logger is None else (None if vperiod is None else valid_info) if period is None else (train_info if vperiod is None else all_info)
    def on_epoch(run):
      run.progress = p = min(max(run.epoch/max_epoch,run.walltime/max_time),1.)
      if logger is not None:
        p = '{:.0%}'.format(p)
        logger.info(current_fmt+'PROGRESS: %s',*current(run),p)
    self.on_open,self.on_batch,self.on_epoch = on_open,on_batch,on_epoch

#==================================================================================================
class RunMlflowListener:
  r"""
Instances of this class log information about classification runs into mlflow.

:param period: training metrics are logged after this number of batch iterations (repeatedly)
:type period: :class:`int`
:param vperiod: validation metrics are logged after this number of batch iterations (repeatedly)
:type vperiod: :class:`int`
:param checkpoint: the model is logged after this number of epochs (repeatedly)
:type checkpoint: :class:`int`
  """
#==================================================================================================

#--------------------------------------------------------------------------------------------------
  def __init__(self,uri:str,exp:str,period:int=None,vperiod:int=None,checkpoint:int=None):
#--------------------------------------------------------------------------------------------------
    import mlflow, mlflow.pytorch
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    def train_info(run):
      if run.batch%period==0:
        mlflow.log_metric('tloss',run.loss,run.step)
    def valid_info(run):
      if run.batch%vperiod==0:
        vloss,vaccu = run.valid
        mlflow.log_metric('vloss',vloss,run.step)
        mlflow.log_metric('vaccu',vaccu,run.step)
    def all_info(run): train_info(run); valid_info(run)
    def on_open(run):
      mlflow.start_run()
      mlflow.log_params(run.params)
    def on_close(run):
      vloss,vaccu = run.valid
      mlflow.log_metric('vloss',vloss,run.step)
      mlflow.log_metric('vaccu',vaccu,run.step)
      mlflow.pytorch.log_model(run.tnet,'model')
      mlflow.end_run()
    on_batch = (None if vperiod is None else valid_info) if period is None else (train_info if vperiod is None else all_info)
    def on_epoch(run):
      if run.epoch%checkpoint==0:
        mlflow.pytorch.log_model(run.tnet,'model_{:03d}'.format(run.epoch))
    if checkpoint is None: on_epoch = None
    self.on_open,self.on_close,self.on_batch,self.on_epoch = on_open,on_close,on_batch,on_epoch

  @staticmethod
  def restore(uri:str,exp:str,run_id:str=None):
    import mlflow, mlflow.pytorch
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    if run_id is None: run_id = mlflow.search_runs().run_id[0] # most recent run
    return mlflow.pytorch.load_model('runs:/{}/model'.format(run_id))

#==================================================================================================
@contextmanager
def training(net,flag):
#==================================================================================================
  oflag = net.training
  net.train(flag)
  yield
  net.train(oflag)
