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
  walltime: 'float'
  r"""[``train``\*] Wall time since beginning of run"""
  proctime: 'float'
  r"""[``train``\*] Process time since beginning of run"""
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
      if k not in ('net','optimiser','pin_memory'): self.params[k] = str(v)
      setattr(self,k,v)
    if hasattr(self,'data'):
      D = self.data
      dl = dict(batch_size=self.batch_size,drop_last=True,pin_memory=self.pin_memory)
      self.train_data,self.valid_data,self.test_data = (
        torch.utils.data.DataLoader(D.train,shuffle=True,**dl),
        torch.utils.data.DataLoader(D.valid,**dl),
        torch.utils.data.DataLoader(D.test,**dl),
      )

#--------------------------------------------------------------------------------------------------
  def exec_train(self):
    r"""
Executes a train run. Parameters passed as attributes. Emitted events: open, batch epoch, close.
    """
#--------------------------------------------------------------------------------------------------
    def eval_cache(data):
      current = None,-1
      def cache():
        nonlocal current
        if current[1] != self.step: current = self.eval(net,to(data,self.device)),self.step
        return current[0]
      return cache
    self.tnet = net = self.net.to(self.device)
    self.eval_valid,self.eval_test = eval_cache(self.valid_data),eval_cache(self.test_data)
    optimiser = self.optimiser(net.parameters())
    self.progress = 0.
    self.step = self.epoch = 0
    self.walltime = self.proctime = 0.
    with training(net,True):
      self.emit('open',self)
      try:
        start = time(), process_time()
        while self.progress<1.:
          self.batch = 0; self.loss = 0.
          for inputs,labels in to(self.train_data,self.device):
            optimiser.zero_grad()
            loss = self.lossF(net(inputs),labels)
            loss.backward()
            optimiser.step()
            self.step += 1; self.batch += 1
            self.loss += (loss.item()-self.loss)/self.batch
            self.walltime,self.proctime = time()-start[0], process_time()-start[1]
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
    project = (lambda a: None) if self.project is None else self.project
    self.protos = torch.repeat_interleave(self.init[None,...],len(self.labels),0)
    net = self.net.to(self.device)
    with training(net,True):
      for proto,label in zip(self.protos,self.labels):
        param = torch.nn.Parameter(proto[None,...]).to(self.device)
        optimiser = self.optimiser([param])
        label = torch.tensor([label]).to(device=self.device)
        for n in range(self.nepoch):
          optimiser.zero_grad()
          loss = self.lossF(net(param),label)
          loss.backward()
          optimiser.step()
          project(param.data[0])

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
    periodic = lambda p: None if p is None else (lambda run: run.batch%p==0)
    current = lambda run: (run.walltime,run.step,run.epoch,run.batch)
    header = 'TIME','STEP','EPO','BAT'
    current_fmt = '%6.1f %6d %3d/%4d '
    header_fmt =  '%6s %6s %3s/%4s '
    if logger is None:
      header_info = train_info = valid_info = test_info = progress_info = None
    else:
      def header_info(run):
        logger.info(header_fmt,*header)
      def train_info(run):
        logger.info(current_fmt+'TRAIN loss: %.3f',*current(run),run.loss)
      def valid_info(run):
        logger.info(current_fmt+'VALIDATION loss: %.3f, accuracy: %.3f',*current(run),*run.eval_valid())
      def test_info(run):
        logger.info(current_fmt+'TEST loss: %.3f, accuracy: %.3f',*current(run),*run.eval_test())
      def progress_info(run):
        logger.info(current_fmt+'PROGRESS: %s',*current(run),'{:.0%}'.format(run.progress))

    def set_progress(run): run.progress = min(max(run.epoch/max_epoch,run.walltime/max_time),1.)

    self.on_open = header_info
    self.on_batch = abs(PROC(train_info)%periodic(period)+PROC(valid_info)%periodic(vperiod))
    self.on_epoch = abs(PROC(set_progress)+PROC(progress_info))
    self.on_close = test_info

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
    from functools import partial
    periodic = lambda p: None if p is None else (lambda run: run.batch%p==0)
    def tperiodic_(run,p,a):
      t = run.walltime; r = t-getattr(run,a,0.)>p
      if r: setattr(run,a,t)
      return r
    tperiodic = lambda p,a: None if p is None else partial(tperiodic_,p=p,a=a)

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    def train_info(run):
      mlflow.log_metric('tloss',run.loss,run.step)
    def valid_info(run):
      vloss,vaccu = run.eval_valid()
      mlflow.log_metric('vloss',vloss,run.step)
      mlflow.log_metric('vaccu',vaccu,run.step)
    def test_info(run):
      loss,accu = run.eval_test()
      mlflow.set_tag('test-loss',loss)
      mlflow.set_tag('test-accu',accu)
    def checkpoint(run):
      mlflow.pytorch.log_model(run.tnet,'model_{:03d}'.format(run.epoch))
    def open_mlflow(run):
      mlflow.start_run()
      mlflow.log_params(run.params)
    def close_mlflow(run):
      try:
        valid_info(run)
        test_info(run)
        mlflow.pytorch.log_model(run.tnet,'model')
      finally:
        mlflow.end_run()

    self.on_open = open_mlflow
    self.on_batch = abs(PROC(train_info)%periodic(period)+PROC(valid_info)%periodic(vperiod))
    self.on_epoch = abs(PROC(checkpoint)%tperiodic(checkpoint_after,'checkpointed'))
    self.on_close = close_mlflow

  @staticmethod
  def load_model(uri:'str',exp:'str',run_id:'str'=None,**ka):
    import mlflow, mlflow.pytorch
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    if run_id is None: run_id = mlflow.search_runs().run_id[0] # most recent run
    return mlflow.pytorch.load_model('runs:/{}/model'.format(run_id),**ka)

#==================================================================================================
class ClassificationDataset:
  r"""
Instances of this class are classification datasets.
  """
#==================================================================================================
  name = None
  classes = None
  def raw(self,train=None): raise NotImplementedError()
  def mpl(self,ax): raise NotImplementedError()

  loaded = False
#--------------------------------------------------------------------------------------------------
  def load(self,pcval,**dl):
#--------------------------------------------------------------------------------------------------
    o = self.__class__()
    D = self.raw(train=True)
    n = len(D); nvalid = int(pcval*n); ntrain = n-nvalid
    o.train,o.valid = torch.utils.data.random_split(D,(ntrain,nvalid))
    o.test = self.raw(train=False)
    o.loaded = True
    return o

#--------------------------------------------------------------------------------------------------
  def display(self,rowspec,colspec,control=None,**ka):
#--------------------------------------------------------------------------------------------------
    from ipywidgets import IntSlider,Layout,Label,HBox
    from matplotlib.pyplot import subplots
    parsespec = lambda spec: (spec if len(spec)==3 else spec+(0.,)) if isinstance(spec,tuple) else (1,spec,0.)
    nrow,rowsz,rowp = parsespec(rowspec)
    ncol,colsz,colp = parsespec(colspec)
    fig,axes = subplots(nrow,ncol,squeeze=False,figsize=(ncol*colsz+rowp,nrow*rowsz+colp),**ka)
    updates = [(ax,self.mpl(ax)) for axr in axes for ax in axr]
    def disp(sample):
      for (value,label),(ax,update) in zip(sample,updates):
        update(value)
        ax.set_title(self.classes[label],fontsize='xx-small',pad=1)
    if control is None:
      dataset = self.raw(train=False)
      N = len(dataset)
      K = len(updates)
      w = IntSlider(value=1,min=0,step=K,max=(N//K-1)*K,layout=Layout(width='10cm'))
      w.observe((lambda ev: disp(dataset[ev.new+k] for k in range(K))),'value')
      w.value = 0
      return HBox((w,Label('+[0-{}]/ {}'.format(K-1,N),layout=Layout(align_self='center'))))
    else: control(disp)

#--------------------------------------------------------------------------------------------------
  def __repr__(self):
#--------------------------------------------------------------------------------------------------
    if self.loaded:
      r = ','.join(f'{k}:{len(getattr(self,k))}' for k in ('train','valid','test'))
      return f'{self.name}<{r}>'
    else:
      return self.name

#==================================================================================================
@contextmanager
def training(net,flag):
#==================================================================================================
  oflag = net.training
  net.train(flag)
  yield
  net.train(oflag)

#==================================================================================================
class PROC:
  r"""
Instances of this class are encapsulated callables or :const:`None` (void callable) which support operation `+` (sequential composition) and `%` with a callable (conditioning). All callables take a single argument.
  """
#==================================================================================================
  def __init__(self,f=None): self.func = f
  def __mod__(self,other):
    if self.func is None or other is None: return PROC()
    assert callable(other)
    return PROC(lambda u,f=self.func,c=other: f(u) if c(u) else None)
  def __add__(self,other):
    assert isinstance(other,PROC)
    return other if self.func is None else self if other.func is None else PROC(lambda u,f1=self.func,f2=other.func: (f1(u),f2(u)))
  def __abs__(self): return self.func

#==================================================================================================
def to(data,device):
#==================================================================================================
  for input,label in data: yield input.to(device),label.to(device)
