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
from functools import partial
from time import time, process_time
from pydispatch import Dispatcher, Property
import torch

#==================================================================================================
class Run (Dispatcher):
  r"""
Runs may emit messages during their execution. Each message is passed a single argument, which is the run itself, so runs can be controlled by message listeners. Use method :meth:`Dispatcher.bind` (general) or :meth:`Run.bind_listeners` (specific) to bind listeners to a run.
  """
#==================================================================================================

  optimiser: 'Callable[[List[torch.nn.Parameter]],torch.optim.Optimizer]'
  r"""The optimiser factory"""
  params: 'Map[str,str]'
  r"""A dictionary for the parameters of the run (subset of attributes)"""
  device: 'torch.device'
  r"""The device on which to execute the run"""
  net: 'torch.nn.Module'
  r"""The model"""
  tnet: 'torch.nn.Module'
  r"""The model located at :attr:`device`"""

#--------------------------------------------------------------------------------------------------
  def __init__(self,**ka):
#--------------------------------------------------------------------------------------------------
    self.listeners = []
    self.params = {}
    for k,v in ka.items():
      if k not in ('net','optimiser','pin_memory'): self.params[k] = str(v)
      setattr(self,k,v)
    self.tnet = self.net.to(self.device)

#--------------------------------------------------------------------------------------------------
  def __call__(self):
    r"""
Executes the run. Defined in subclasses.
    """
#--------------------------------------------------------------------------------------------------
    raise NotImplementedError()

#--------------------------------------------------------------------------------------------------
  def bind_listeners(self,*listeners):
    r"""
Binds the set of *listeners* (objects) to the events. The list of listeners is processed sequentially. For each listener, the methods of that listener whose name match ``on_`` followed by the name of a supported event are bound to this dispatcher.

:param listeners: list of listener objects
:param type: :class:`Iterable[Any]`
    """
#--------------------------------------------------------------------------------------------------
    for x in listeners:
      self.bind(**dict((ev,t) for ev in self._events_ for t in (getattr(x,'on_'+ev,None),) if t is not None))
    self.listeners.extend(listeners)
    return self

#--------------------------------------------------------------------------------------------------
  @classmethod
  def listenerFactory(cls,name):
    r"""Decorator to declare a listener factory attached to a subclass of :class:`Run`."""
#--------------------------------------------------------------------------------------------------
    def app(f):
      setattr(cls,'{}Listener'.format(name.title()),f)
      setattr(cls,'bind{}Listener'.format(name.title()),(lambda self,*a,**ka: self.bind_listeners(f(*a,**ka))))
      return f
    return app

#==================================================================================================
class SupervisedRun (Run):
  r"""
Supervised runs (by default classification runs) learn from data ni the form of instance-label pairs.
  """
#==================================================================================================

  lossF: 'Callable[[torch.tensor,torch.tensor],float]'
  r"""Loss function"""
  accuracyF: 'Callable[[torch.tensor,torch.tensor],float]'
  r"""Accuracy function"""

  # Default values for classification runs (override in subclasses)
  lossF = torch.nn.CrossEntropyLoss()
  accuracyF = staticmethod(lambda outputs,labels: torch.mean((torch.argmax(outputs,1)==labels).float()))

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

#==================================================================================================
class SupervisedTrainRun (SupervisedRun):
  r"""
Supervised train runs execute a simple training loop for supervised learning.
  """
#==================================================================================================

  _events_ = ['open','batch','epoch','close']

  ## set at instantiation
  train_data: 'Iterable[Tuple[torch.tensor,torch.tensor]]'
  r"""Each batch is a pair of a tensor of inputs and a tensor of labels (first dim = size of batch)"""
  valid_data: 'Iterable[Tuple[torch.tensor,torch.tensor]]'
  r"""Each batch is a pair of a tensor of inputs and a tensor of labels (first dim = size of batch)"""
  test_data: 'Iterable[Tuple[torch.tensor,torch.tensor]]'
  r"""Each batch is a pair of a tensor of inputs and a tensor of labels (first dim = size of batch)"""
  ## set at execution
  progress: 'float'
  r"""Between 0. and 1., stop when 1. is reached"""
  walltime: 'float'
  r"""Wall time since beginning of run"""
  proctime: 'float'
  r"""Process time since beginning of run"""
  step: 'int'
  r"""Number of completed global steps"""
  epoch: 'int'
  r"""Number of completed epochs"""
  batch: 'int'
  r"""Number of completed batches"""
  loss: 'float'
  r"""Mean loss since beginning of current epoch"""
  eval_valid: 'Callable[[],Tuple[float,float]]'
  r"""Function returning the validation performance (loss,accuracy) at the end of the last completed step"""
  eval_test: 'Callable[[],Tuple[float,float]]'
  r"""Function returning the test performance (loss,accuracy) at the end of the last completed step"""

#--------------------------------------------------------------------------------------------------
  def __init__(self,data=None,**ka):
#--------------------------------------------------------------------------------------------------
    super().__init__(**ka)
    if data is None:
      assert all(isinstance(getattr(self,c+'_data',None),torch.utils.data.DataLoader) for c in ('train','valid','test'))
    else:
      self.params['data'] = repr(data)
      for c in 'train','valid','test':
        setattr(self,c+'_data',torch.utils.data.DataLoader(getattr(data,c),batch_size=self.batch_size,drop_last=True,pin_memory=self.pin_memory,shuffle=c=='train'))

#--------------------------------------------------------------------------------------------------
  def __call__(self):
#--------------------------------------------------------------------------------------------------
    def eval_cache(data): # returns a cached version of self.eval
      current = None,-1
      def cache():
        nonlocal current
        if current[1] != self.step: current = self.eval(net,to(data,self.device)),self.step
        return current[0]
      return cache
    net = self.tnet
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

#==================================================================================================
class SupervisedInvRun (SupervisedRun):
  r"""
Supervised inv runs attempts to inverse the model at a sample of labels.
  """
#==================================================================================================
  nepoch: 'int'
  r"""Number of epochs to run"""
  labels: 'Union[int,Iterable[int]]'
  r"""List of labels to process (or range if integer type)"""
  init: 'torch.tensor'
  r"""Initial instance"""
  projection: 'Callable[[torch.tensor],NoneType]'
  r"""Called after each optimiser step to re-project instance within domain range"""

#--------------------------------------------------------------------------------------------------
  def __call__(self):
#--------------------------------------------------------------------------------------------------
    project = (lambda a: None) if self.project is None else self.project
    self.protos = torch.repeat_interleave(self.init[None,...],len(self.labels),0)
    net = self.tnet
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

#==================================================================================================
@SupervisedTrainRun.listenerFactory('Base')
class SupervisedTrainRunListener:
  r"""
Instances of this class control basic classification runs.

:param max_epoch: maximum number of epochs to run
:type max_epoch: :class:`int`
:param max_time: maximum total wall time to run
:type max_time: :class:`int`
:param logger: logger to use for logging information
:type logger: :class:`logging.Logger`
:param status: pair of a function returning the status of a run and corresponding headers
:type status: :class:`Tuple[Callable[[Any],Tuple],Tuple]`
:param status_fmt: pair of formats for the components of *status*
:type status_fmt: :class:`Tuple[str,str]`
:param itrain,ivalid,itest: function returning various info on a run
:type itrain,ivalid,itest: :class:`Callable[[Any],Tuple]`
:param itrain_fmt,ivalid_fmt,itest_fmt: formats for the results of the corresponding functions
:type itrain_fmt,ivalid_fmt,itest_fmt: :class:`str`
  """
#==================================================================================================

#--------------------------------------------------------------------------------------------------
  def __init__(self,max_epoch=int(1e9),max_time=float('inf'),logger=None,
    status=(
      ('TIME','STEP','EPO','BAT'),
      (lambda run: (run.walltime,run.step,run.epoch,run.batch))
    ),
    status_fmt:'Tuple[str,str]'=('%6s %6s %3s/%4s ','%6.1f %6d %3d/%4d '),
    itrain=(lambda run: (run.loss,)),
    ivalid=(lambda run: run.eval_valid()),
    itest= (lambda run: run.eval_test()),
    itrain_fmt:'str'='TRAIN loss: %.3f',
    ivalid_fmt:'str'='VALIDATION loss: %.3f, accuracy: %.3f',
    itest_fmt:'str'= 'TEST loss: %.3f, accuracy: %.3f',
    **config
  ):
#--------------------------------------------------------------------------------------------------
    if logger is None:
      self.itrain = self.ivalid = self.iprogress = i_test = i_header = None
    else:
      self.itrain = lambda run,fmt=status_fmt[1]+itrain_fmt,s=status[1]: logger.info(fmt,*s(run),*itrain(run))
      self.ivalid = lambda run,fmt=status_fmt[1]+ivalid_fmt,s=status[1]: logger.info(fmt,*s(run),*ivalid(run))
      self.iprogress = lambda run,fmt=status_fmt[1]+'PROGRESS: %s',s=status[1]: logger.info(fmt,*s(run),'{:.0%}'.format(run.progress))
      i_test = lambda run,fmt=status_fmt[1]+itest_fmt,s=status[1]: logger.info(fmt,*s(run),*itest(run))
      i_header = lambda run,fmt=status_fmt[0],s=status[0]: logger.info(fmt,*s)
    def set_progress(run): run.progress = min(max(run.epoch/max_epoch,run.walltime/max_time),1.)
    self.set_progress = set_progress
    self.on_open = i_header
    self.on_close = i_test
    self.on_batch = None
    self.on_epoch = abs(PROC(set_progress)+PROC(self.iprogress))
    self.configure(**config)

#--------------------------------------------------------------------------------------------------
  def configure(self,train_p=False,valid_p=False):
    r"""
:param train_p: run selector for train info logging
:type train_p: :class:`Union[Callable[[Any],bool],bool]`
:param valid_p: run selector for validation info logging
:type valid_p: :class:`Union[Callable[[Any],bool],bool]`

Standard definitions for on_batch and on_epoch (can be overridden in subclasses).
    """
#--------------------------------------------------------------------------------------------------
    self.on_batch = abs(PROC(self.on_batch)+PROC(self.itrain)%train_p)
    self.on_epoch = abs(PROC(self.on_epoch)+PROC(self.ivalid)%valid_p)

#==================================================================================================
@SupervisedTrainRun.listenerFactory('Mlflow')
class SupervisedTrainRunMlflowListener:
  r"""
Instances of this class log information about classification runs into mlflow.

:param uri: mlflow tracking uri
:type uri: :class:`str`
:param exp: experiment name
:type exp: :class:`str`
:param itrain,ivalid,itest: function returning various info on a run
:type itrain,ivalid,itest: :class:`Callable[[Any],Tuple]`
:param itrain_labels,ivalid_labels,itest_labels: labels for the results of the corresponding functions
:type itrain_labels,ivalid_labels,itest_labels: :class:`Iterable[str]`
  """
#==================================================================================================

#--------------------------------------------------------------------------------------------------
  def __init__(self,uri:'str',exp:'str',
    itrain=(lambda run: (run.loss,)),
    ivalid=(lambda run: run.eval_valid()),
    itest =(lambda run: run.eval_test()),
    itrain_labels = ('tloss',),
    ivalid_labels = ('vloss','vaccu'),
    itest_labels = ('loss','accu'),
    **config
  ):
#--------------------------------------------------------------------------------------------------
    import mlflow, mlflow.pytorch

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    def i_train(run):
      for key,val in zip(itrain_labels,itrain(run)): mlflow.log_metric(key,val,run.step)
    self.itrain = i_train
    def i_valid(run):
      for key,val in zip(ivalid_labels,ivalid(run)): mlflow.log_metric(key,val,run.step)
    self.ivalid = i_valid
    def i_progress(run): mlflow.set_tag('progress',run.progress)
    self.iprogress = i_progress
    def i_test(run):
      for key,val in zip(itest_labels,itest(run)): mlflow.set_tag(key,val)
    self.checkpoint = lambda run: mlflow.pytorch.log_model(run.tnet,'model_{:03d}'.format(run.epoch))
    def open_mlflow(run):
      mlflow.start_run(); mlflow.log_params(run.params)
    def close_mlflow(run):
      try:
        i_valid(run); i_progress(run); i_test(run)
        mlflow.pytorch.log_model(run.tnet,'model')
      finally: mlflow.end_run()

    self.on_open = open_mlflow
    self.on_close = close_mlflow
    self.on_batch = None
    self.on_epoch = i_progress
    self.configure(**config)

#--------------------------------------------------------------------------------------------------
  def configure(self,train_p=False,valid_p=False,checkpoint_p=False):
    r"""
:param train_p: run selector for train info logging
:type train_p: :class:`Callable[[Any],bool]`
:param valid_p: run selector for validation info logging
:type valid_p: :class:`Union[Callable[[Any],bool],bool]`
:param checkpoint_p: run selector for checkpointing
:type checkpoint_p: :class:`Union[Callable[[Any],bool],bool]`

Standard definitions for on_batch and on_epoch (can be overridden in subclasses).
    """
#--------------------------------------------------------------------------------------------------
    self.on_batch = abs(PROC(self.on_batch)+PROC(self.itrain)%train_p)
    self.on_epoch = abs(PROC(self.on_epoch)+PROC(self.ivalid)%valid_p+PROC(self.checkpoint)%checkpoint_p)

#--------------------------------------------------------------------------------------------------
  @staticmethod
  def load_model(uri:'str',exp:'str',run_id:'str'=None,epoch=None,**ka):
    r"""
Returns a model saved in a run.
    """
#--------------------------------------------------------------------------------------------------
    import mlflow, mlflow.pytorch
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    if run_id is None: run_id = mlflow.search_runs().run_id[0] # most recent run
    return mlflow.pytorch.load_model('runs:/{}/model{}'.format(run_id,('' if epoch is None else '_{:03d}'.format(epoch))),**ka)

#==================================================================================================
class ClassificationDatasource:
  r"""
Instances of this class are datasources for classification training.
  """
#==================================================================================================
  name = None
  classes = None
  def raw(self,train=None): raise NotImplementedError()
  def mpl(self,ax): raise NotImplementedError()

  loaded = False
#--------------------------------------------------------------------------------------------------
  def load(self,pcval):
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
Instances of this class are encapsulated callables or :const:`None` (void callable) which support operation `+` (sequential invocation) and `%` with a callable (conditioning). All callables take a single argument.
  """
#==================================================================================================
  def __init__(self,f=None): self.func = f
  def __mod__(self,other):
    if self.func is None or other is False: return PROC()
    if other is True: return self
    assert callable(other)
    return PROC(lambda u,f=self.func,c=other: f(u) if c(u) else None)
  def __add__(self,other):
    assert isinstance(other,PROC)
    return other if self.func is None else self if other.func is None else PROC(lambda u,f1=self.func,f2=other.func: f1(u) is not None or f2(u) is not None or None)
  def __abs__(self): return self.func

#==================================================================================================
def periodic(p,counter=None):
  r"""
:param p: periodicity (in counter value)
:type p: :class:`Union[NoneType,int,float]`
:param counter: run attribute used as counter
:type counter: :class:`str`
:rtype: :class:`Callable[[Any],bool]`

Returns a run selector, i.e. a callable which takes a run as input and returns a boolean.

* If *p* is of type :class:`int`, a run is selected if the counter value is a multiple of *p*. Default counter: :attr:`batch`
* If *p* is of type :class:`float`, a run is selected unless the increase in the counter value since the last successful selection is lower than *p*. Default counter: :attr:`walltime` (in secs)
  """
#==================================================================================================
  class F:
    def __init__(self,counter): self.current,self.counter = 0.,counter
    def __call__(self,run):
      t = getattr(run,self.counter); r = t-self.current>p
      if r: self.current = t
      return r
  counter_ = ('batch' if isinstance(p,int) else 'walltime') if counter is None else counter
  return None if p is None else (lambda run: getattr(run,counter_)%p == 0) if isinstance(p,int) else F(counter_)

#==================================================================================================
def to(data,device):
#==================================================================================================
  for input,label in data: yield input.to(device),label.to(device)
