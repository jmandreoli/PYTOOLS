# File:                 torch.py
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

from __future__ import annotations
from typing import Sequence, Any, Union, Callable, Iterable, Mapping, Tuple
import logging; logger = logging.getLogger(__name__)

from contextlib import contextmanager
from functools import partial
from time import time, ctime, process_time
from socket import getfqdn
from pydispatch import Dispatcher, Property
import torch

#==================================================================================================
class Run (Dispatcher):
  r"""
Instances of this class are callables, taking no argument and returning no value. The execution of a run only changes its attribute values. At the creation of the run, all the key-value pairs in *ka* are turned into attributes of the run. If the key starts with `_`, that character is dropped from the added attribute name, which is then marked as "non visible". The remaining attributes are gathered in a dictionary :attr:`visible`, meant to provide a human readable description of the run.

Runs may emit messages during their execution. Each message is passed a single argument, which is the run itself, so runs can be controlled by message listeners. Methods :meth:`Dispatcher.bind` (inherited) and :meth:`bind_listeners` can be used to bind callbacks to a run. Some methods, defined in subclasses, of the form :meth:`bind<name>Listener` are also available to bind specific callbacks named by `<name>`.

Attributes (\*) must be instantiated at creation time (with attribute :attr:`device`, if present, at the end).
  """
#==================================================================================================

  net: torch.nn.Module
  r"""(\*)The model"""
  measures: Sequence[Callable[[torch.tensor,...],torch.tensor]]
  r"""(\*)List of measures, each returning a scalar (0-D) tensor"""
  device: torch.device
  r"""The device on which to execute the run"""
  stepper: Stepper
  r"""Time and step keeper"""
  walltime: float
  r"""Wall time elapsed (in sec) between last invocation of :meth:`reset` and last invocation of :meth:`tick` on :attr:`stepper`"""
  proctime: float
  r"""Process time elapsed (in sec) between last invocation of :meth:`reset` and last invocation of :meth:`tick` on :attr:`stepper`"""
  step: int
  r"""Number of invocations of :meth:`tick` since last invocation of :meth:`reset` on :attr:`stepper`"""

#--------------------------------------------------------------------------------------------------
  def __init__(self,**ka):
#--------------------------------------------------------------------------------------------------
    self.visible = dict(device='default')
    self.device_ = None
    for k,v in ka.items():
      if k.startswith('_'): k = k[1:]
      else: r = ' '.join(repr(v).split()); self.visible[k] = r if len(r)<80 else r[:77]+'...'
      setattr(self,k,v)
    self.listeners = [] # to hold the list of listeners, otherwise they may be garbage collected
    self.stepper = Stepper(self)

  @property
  def device(self): return self.device_
  @device.setter
  def device(self,v): self.visible['device'] = self.device_ = v; self.to(v)

#--------------------------------------------------------------------------------------------------
  def to(self,device):
    r"""
Moves run to *device*. This implementation moves :attr:`net` (the model) to *device*. May be refined in subclasses.
    """
#--------------------------------------------------------------------------------------------------
    self.net = self.net.to(device)

#--------------------------------------------------------------------------------------------------
  def __call__(self):
    r"""
Executes the run. This implementation raises a :class:`NotImplementedError`.
    """
#--------------------------------------------------------------------------------------------------
    raise NotImplementedError()

#--------------------------------------------------------------------------------------------------
  def eval(self,data:Iterable[Tuple[float,Tuple[torch.tensor,...],Tuple[torch.tensor,...]]])->Tuple[float,...]:
    r"""
Evaluates model :attr:`net` on *data* according to :attr:`measures`. Each entry in *data* is a tuple whose first item is a weight, second item is a tuple of input tensors on which the model is executed, followed by any number of extra parameters for the measures. For each data entry, an evaluation consists in applying the model with the first item as list of arguments, then estimating each measure on the resulting tensor with the remaining items as list of arguments. The returned value is a tuple of length equal to the number of measures, holding weighted averages (as floats) over the whole *data* sequence.

:param data: a list of data entries
    """
#--------------------------------------------------------------------------------------------------
    avg = 0.
    weight = 0.
    with training(self.net,False),torch.no_grad():
      for w,inputs,args in data:
        outputs = self.net(*to(inputs,self.device))
        weight += w
        avg += (torch.stack([m(outputs,*to(args,self.device)) for m in self.measures])-avg)*(w/weight)
    return tuple(avg.to('cpu').numpy())

#--------------------------------------------------------------------------------------------------
  def eval_cache(self,data):
    r"""
Returns a cached version of :meth:`eval` on *data*. The cache is cleared at each tick of the :attr:`stepper`.
    """
#--------------------------------------------------------------------------------------------------
    current = None,-1
    def cache():
      nonlocal current
      if current[1] != self.step: current = self.eval(data),self.step
      return current[0]
    return cache

#--------------------------------------------------------------------------------------------------
  def bind_listeners(self,*listeners):
    r"""
Binds the set of *listeners* (any objects) to the events of this run. For each listener, the methods of that listener whose name match ``on_`` followed by the name of a supported event are bound as callback to this run. These methods are usually specified as attributes of type :class:`PROC`, so as to easily be configurable.

:param listeners: list of listener objects
    """
#--------------------------------------------------------------------------------------------------
    for x in listeners:
      self.bind(**dict((ev,t) for ev in self._events_ for t in (getattr(x,'on_'+ev,None),) if t is not None))
    self.listeners.extend(listeners)
    return self

#--------------------------------------------------------------------------------------------------
  @classmethod
  def listenerFactory(cls,name:str)->Callable[[Callable],Callable]:
    r"""
Used as decorator to attach a listener factory to a subclass *cls* of this class. The factory is available as method :meth:`bind<name>Listener` using the provided *name*.

:param name: short name for the factory
    """
#--------------------------------------------------------------------------------------------------
    def app(f):
      setattr(cls,f'{name}Listener',f)
      F = lambda self,*a,**ka: self.bind_listeners(f(*a,**ka))
      F.__doc__ = f'Binds an event listener obtained by invoking factory :func:`{name}Listener`'
      setattr(cls,f'bind{name}Listener',F)
      return f
    return app

  def __repr__(self): return f'{self.__class__.__name__}<{self.created[0]}|{self.created[1]}>'

#==================================================================================================
class SupervisedRun (Run):
  r"""
Supervised runs operate on data in the form of instance-label pairs. Attributes (\*) must be instantiated at creation time. Attribute :attr:`measures` is also set to a reasonable default for classification (the pair of the loss and accuracy functions).
  """
#==================================================================================================

  lossF: Callable[[torch.tensor,torch.tensor],float]
  r"""(\*)Loss function (defaults to cross-entropy loss for classification)"""
  @property
  def measures(self): return (self.lossF,)+self.omeasures

  def eval(self,data:Iterable[Tuple[torch.tensor,torch.tensor]]):
    r"""Adapts the data format"""
    return super().eval((1.,(inputs,),label) for inputs,label in data)

  # Default values for classification runs (override in subclasses)
  lossF = torch.nn.CrossEntropyLoss()
  omeasures = (lambda outputs,labels: torch.mean((torch.argmax(outputs,1)==labels).float())),

#==================================================================================================
class SupervisedTrainRun (SupervisedRun):
  r"""
Runs of this type execute a simple supervised training loop. Attributes (\*) must be instantiated at creation time. All the other attributes are initialised and updated by the run execution, except :attr:`progress` which is initialised (to 0.) by the run execution, but needs to be updated by some listener.
  """
#==================================================================================================

  _events_ = ['open','batch','epoch','close']

  ## set at instantiation
  train_data: Iterable[Tuple[torch.tensor,torch.tensor]]
  r"""(\*)Iterable of batches. Each batch is a pair of a tensor of inputs and a tensor of labels (first dim = size of batch)"""
  valid_data: Iterable[Tuple[torch.tensor,torch.tensor]]
  r"""(\*)Same format as :attr:`train_data`"""
  test_data: Iterable[Tuple[torch.tensor,torch.tensor]]
  r"""(\*)Same format as :attr:`train_data`"""
  optimiser: Callable[[Sequence[torch.nn.Parameter]],torch.optim.Optimizer]
  r"""(\*)The optimiser factory"""
  ## set at execution
  progress: float
  r"""Between 0. and 1., stops the run when 1. is reached (must be updated by a listener)"""
  epoch: int
  r"""Number of completed epochs"""
  batch: int
  r"""Number of completed batches within current epoch"""
  loss: float
  r"""Average loss since beginning of current epoch"""
  eval_valid: Callable[[],Tuple[float,float]]
  r"""Function returning the validation performance at the end of the last completed step (cached)"""
  eval_test: Callable[[],Tuple[float,float]]
  r"""Function returning the test performance at the end of the last completed step (cached)"""

#--------------------------------------------------------------------------------------------------
  def __call__(self):
#--------------------------------------------------------------------------------------------------
    net = self.net
    self.eval_valid,self.eval_test = map(self.eval_cache,(self.valid_data,self.test_data))
    optimiser = self.optimiser(net.parameters())
    with training(net,True):
      self.stepper.reset()
      self.progress = 0.; self.epoch = 0
      self.emit('open',self)
      try:
        while self.progress<1.:
          self.batch = 0; self.loss = 0.
          for inputs,labels in to(self.train_data,self.device):
            optimiser.zero_grad()
            loss = self.lossF(net(inputs),labels)
            loss.backward()
            optimiser.step()
            self.batch += 1
            self.loss += (loss.item()-self.loss)/self.batch
            del inputs,labels,loss # free memory before emitting (not sure this works)
            self.stepper.tick()
            self.emit('batch',self)
          self.epoch += 1
          self.emit('epoch',self)
      finally:
        self.emit('close',self)

#==================================================================================================
class SupervisedInvRun (SupervisedRun):
  r"""
Runs of this type attempt to inverse the model at a sample of labels. Attributes (\*) must be instantiated at creation time. All the other attributes are initialised and updated by the run execution.
  """
#==================================================================================================
  nepoch: int
  r"""(\*)Number of epochs to run"""
  labels: Sequence[int]
  r"""(\*)List of labels (pure value, not tensor) to process (or range if integer type)"""
  init: torch.tensor
  r"""(\*)Initial instance"""
  projection: Callable[[torch.tensor],NoneType]
  r"""(\*)Called after each optimiser step to re-project instances within domain range"""
  optimiser: Callable[[List[torch.nn.Parameter]],torch.optim.Optimizer]
  r"""(\*)The optimiser factory"""
  protos: Sequence[torch.tensor]
  r"""The estimated inverse image of :attr:`labels`"""

#--------------------------------------------------------------------------------------------------
  def __call__(self):
#--------------------------------------------------------------------------------------------------
    projection = (lambda a: None) if self.projection is None else self.projection
    self.protos = torch.repeat_interleave(self.init[None,...],len(self.labels),0)
    net = self.net
    with training(net,True):
      self.stepper.reset()
      for proto,label in zip(self.protos,self.labels):
        param = torch.nn.Parameter(proto[None,...]).to(self.device)
        optimiser = self.optimiser([param])
        label = torch.tensor([label]).to(device=self.device)
        for n in range(self.nepoch):
          optimiser.zero_grad()
          loss = self.lossF(net(param),label)
          loss.backward()
          optimiser.step()
          projection(param.data[0])
          self.stepper.tick()

#==================================================================================================
@SupervisedTrainRun.listenerFactory('Base')
class SupervisedTrainRunListener:
  r"""
Instances of this class monitor basic supervised training runs. By default, train, validation and test info are logged, respectively, on batch end, epoch end and run end. At least one of *max_epoch* or *max_time* must be set, or the run never ends.

:param max_epoch: maximum number of epochs to run; default: no limit
:param max_time: maximum total wall time to run; default: no limit
:param logger: logger to use for logging information
:param status: pair of a tuple of headers and a function returning the status of a run as a tuple matching those headers
:param status_fmt: pair of format strings for the components of *status*
:param itrain,ivalid,itest: function returning various info on a run
:param itrain_fmt,ivalid_fmt,itest_fmt: format strings for the results of the corresponding functions
:param config: passed as keyword arguments to method :meth:`configure`
  """
#==================================================================================================

#--------------------------------------------------------------------------------------------------
  def __init__(self,max_epoch:int=None,max_time:float=None,logger:logging.Logger=None,
    status:Tuple[Tuple[str,...],Callable[[Run],Tuple[Any,...]]]=(
      ('TIME','STEP','EPO','BAT'),
      (lambda run: (run.walltime,run.step,run.epoch,run.batch))
    ),
    status_fmt:Tuple[str,str]=('%6s %6s %3s/%4s ','%6.1f %6d %3d/%4d '),
    itrain:Callable[[Run],Tuple[Any,...]]=(lambda run: (run.loss,)),
    ivalid:Callable[[Run],Tuple[Any,...]]=(lambda run: run.eval_valid()),
    itest:Callable[[Run],Tuple[Any,...]] =(lambda run: run.eval_test()),
    itrain_fmt:str='TRAIN loss: %.3f',
    ivalid_fmt:str='VALIDATION loss: %.3f, accuracy: %.3f',
    itest_fmt:str= 'TEST loss: %.3f, accuracy: %.3f',
    **config
  ):
#--------------------------------------------------------------------------------------------------
    assert max_epoch is not None or max_time is not None
    if logger is None:
      self.itrain = self.ivalid = self.iprogress = self.itest = self.iheader = None
    else:
      self.itrain = lambda run,fmt=status_fmt[1]+itrain_fmt,s=status[1]: logger.info(fmt,*s(run),*itrain(run))
      self.ivalid = lambda run,fmt=status_fmt[1]+ivalid_fmt,s=status[1]: logger.info(fmt,*s(run),*ivalid(run))
      self.iprogress = lambda run,fmt=status_fmt[1]+'PROGRESS: %s',s=status[1]: logger.info(fmt,*s(run),'{:.0%}'.format(run.progress))
      self.itest = lambda run,fmt=status_fmt[1]+itest_fmt,s=status[1]: logger.info(fmt,*s(run),*itest(run))
      def i_header(run,fmt=status_fmt[0],s=status[0]):
        logger.info('RUN %s',run)
        fmtp = 'PARAM %{}s= %s'.format(max(list(map(len,run.visible.keys()))))
        for k,v in run.visible.items(): logger.info(fmtp,k,v)
        logger.info(fmt,*s)
      self.iheader = i_header
    if max_epoch is None: max_epoch = float('inf')
    if max_time is None: max_time = float('inf')
    def set_progress(run): run.progress = min(max(run.epoch/max_epoch,run.walltime/max_time),1.)
    self.set_progress = set_progress
    self.configure(**config)

#--------------------------------------------------------------------------------------------------
  def configure(self,
    train_p:Union[Callable[[Run],bool],bool]=False,
    valid_p:Union[Callable[[Run],bool],bool]=False
  ):
    r"""
This method, called at the end of the constructor, configures the callback methods of this listener. This implementation assigns reasonable defaults, using the run selectors passed in arguments, and can be refined in subclasses.

:param train_p: run selector for train info logging at each batch
:param valid_p: run selector for validation info logging at each epoch
    """
#--------------------------------------------------------------------------------------------------
    self.on_open = self.iheader
    self.on_close = self.itest
    self.on_batch = abs(PROC(self.itrain)%train_p)
    self.on_epoch = abs(PROC(self.set_progress)+PROC(self.iprogress)+PROC(self.ivalid)%valid_p)

#==================================================================================================
@SupervisedTrainRun.listenerFactory('Mlflow')
class SupervisedTrainRunMlflowListener:
  r"""
Instances of this class provide mlflow logging of supervised training runs. By default, train, validation and test info are logged, respectively, on batch end, epoch end and run end.

:param uri: mlflow tracking uri
:param exp: experiment name
:param itrain,ivalid,itest: function returning various info on a run
:param itrain_labels,ivalid_labels,itest_labels: labels for the results of the corresponding functions
:param config: passed as keyword arguments to method :meth:`configure`
  """
#==================================================================================================

#--------------------------------------------------------------------------------------------------
  def __init__(self,uri:str,exp:str,
    itrain:Callable[[Run],Tuple[Any,...]]=(lambda run: (run.loss,)),
    ivalid:Callable[[Run],Tuple[Any,...]]=(lambda run: run.eval_valid()),
    itest:Callable[[Run],Tuple[Any,...]] =(lambda run: run.eval_test()),
    itrain_labels:Tuple[str,...] = ('tloss',),
    ivalid_labels:Tuple[str,...] = ('vloss','vaccu'),
    itest_labels:Tuple[str,...] = ('loss','accu'),
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
    self.checkpoint = lambda run: mlflow.pytorch.log_model(run.net,f'model_{run.epoch:03d}')
    def open_mlflow(run): mlflow.start_run(); mlflow.log_params(run.visible)
    self.open_mlflow = open_mlflow
    def close_mlflow(run):
      try:
        i_valid(run); i_progress(run); i_test(run)
        mlflow.pytorch.log_model(run.net,'model')
      finally: mlflow.end_run()
    self.close_mlflow = close_mlflow
    self.configure(**config)

#--------------------------------------------------------------------------------------------------
  def configure(self,
    train_p:Union[Callable[[Run],bool],bool]=False,
    valid_p:Union[Callable[[Run],bool],bool]=False,
    checkpoint_p:Union[Callable[[Run],bool],bool]=False,
  ):
    r"""
This method, called at the end of the constructor, configures the callback methods of this listener. This implementation assigns reasonable defaults, using the run selectors passed in arguments, and can be refined in subclasses.

:param train_p: run selector for train info logging at each batch
:param valid_p: run selector for validation info logging at each epoch
:param checkpoint_p: run selector for checkpointing at each epoch
    """
#--------------------------------------------------------------------------------------------------
    self.on_open = self.open_mlflow
    self.on_close = self.close_mlflow
    self.on_batch = abs(PROC(self.itrain)%train_p)
    self.on_epoch = abs(PROC(self.iprogress)+PROC(self.ivalid)%valid_p+PROC(self.checkpoint)%checkpoint_p)

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
    return mlflow.pytorch.load_model('runs:/{}/model{}'.format(run_id,('' if epoch is None else f'_{epoch:03d}')),**ka)

#==================================================================================================
class ClassificationDatasource:
  r"""
Instances of this class are data sources for classification training. Attributes (\*) must be instantiated at creation time.
  """
#==================================================================================================
  name: str
  r"""(\*)Descriptive name of the datasource"""
  classes: Iterable[str]
  r"""(\*)Descriptive names for the classes"""
  train: torch.utils.data.Dataset
  r"""(\*)The train split of the data source"""
  test: torch.utils.data.Dataset
  r"""(\*)The test split of the data source"""

#--------------------------------------------------------------------------------------------------
  def mpl(self,ax:matplotlib.Axes)->Callable[[torch.tensor],None]:
    r"""
Returns a callable, which, when passed a data instance, displays it on *ax* (its label is also used as title of *ax*). This implementation raises a :class:`NotImplementedError`.

:param ax: a display area for a data instance
    """
#--------------------------------------------------------------------------------------------------
    raise NotImplementedError()

#--------------------------------------------------------------------------------------------------
  def loaders(self,pcval:float,**ka):
    r"""
Returns a dictionary which can be passed as keyword arguments to the :class:`SupervisedTrainRun` constructor to initialise its required attributes:

* invisible attributes :attr:`train_data`, :attr:`valid_data`, :attr:`test_data`, obtained by calling :class:`torch.utils.data.DataLoader` on the corresponding split with key-word parameters *ka* (with prefix `_` removed from keys);
* a visible attribute :attr:`data` holding a description of the datasource;
* all the attributes (visible and invisible) in *ka*, so they appear as attributes of the run.

:param pcval: proportion (between 0. and 1.) of instances from the train split to use for validation.
    """
#--------------------------------------------------------------------------------------------------
    D = self.train
    n = len(D); nvalid = int(pcval*n); ntrain = n-nvalid
    train,valid = torch.utils.data.random_split(D,(ntrain,nvalid))
    test = self.test
    r = ','.join(f'{c}:{len(split)}' for c,split in zip(('train','valid','test'),(train,valid,test)))
    r = f'{self.name}<{r}>'
    params = dict(data=r,**ka)
    ka = dict(((k[1:] if k.startswith('_') else k),v) for k,v in ka.items())
    for c,split in zip(('train','valid','test'),(train,valid,test)):
      params['_'+c+'_data'] = torch.utils.data.DataLoader(split,shuffle=c=='train',**ka)
    return params

#--------------------------------------------------------------------------------------------------
  def display(self,
    rowspec:Union[float,Tuple[float,int],Tuple[float,int,float]],
    colspec:Union[float,Tuple[float,int],Tuple[float,int,float]],
    dataset:Iterable[Tuple[Any,int]]=None,
    **ka
    ):
    r"""
Displays a set of labelled data instances *dataset* in a grid. The specification of the rows (resp. cols) of the grid consists of three numbers:

* height (resp. width) of the cells; required
* number of rows (resp. cols); if -1 it is adjusted so the the whole dataset fits in the grid; default: 1
* additional space in the row (resp. col) dimension; default: 0.

At most one of the number of rows or columns can be -1. When both are positive and the dataset does not fit in the grid, a slider is created to allow page browsing through the dataset.

:param rowspec,colspec: specification of the rows/cols of the grid
:param dataset: the data to display
:param ka: keyword arguments passed to :func:`matplotlib.pyplot.subplots` to create the grid
    """
#--------------------------------------------------------------------------------------------------
    def disp(sample):
      for (value,label),(ax,update) in zip(pad(sample),updates):
        visible = value is not None
        ax.set(visible=visible)
        if visible:
          update(value)
          ax.set_title(self.classes[label],fontsize='xx-small',pad=1)
    def widget_control(dataset,K,disp):
      from ipywidgets import IntSlider,Label,HBox,Button
      disp_ = lambda: disp([dataset[k] for k in range(w_sel.value,min(w_sel.value+K,N))])
      N = len(dataset)
      w_closeb = Button(icon='close',tooltip='Close browser',layout=dict(width='.5cm',padding='0'))
      w_sel = IntSlider(value=0,min=0,step=K,max=(adj(N,K)-1)*K,layout=dict(width='10cm'))
      w_main = HBox((w_sel,Label(f'+[0-{K-1}]/ {N}',layout=dict(align_self='center')),w_closeb))
      w_closeb.on_click(lambda ev: (close(fig),w_main.close()))
      w_sel.observe((lambda ev: disp_()),'value')
      disp_()
      return w_main
    def pad(it):
      for x in it: yield x
      while True: yield None,None
    def adj(N,K):
      from math import ceil
      return int(ceil(N/K))
    def parsespec(spec):
      def fmt(sz,n=None,pad=None):
        assert isinstance(sz,(int,float)) and sz>0.
        if n is None: n = 1
        else: assert isinstance(n,int) and (n==-1 or n>=1)
        if pad is None: pad = 0.
        else: assert isinstance(pad,(int,float)) and pad>=0.
        return sz,n,pad
      return (spec,1,0.) if isinstance(spec,(int,float)) else fmt(*spec)
    from matplotlib.pyplot import subplots, close
    if dataset is None: dataset = self.test
    try: N = len(dataset)
    except: dataset = list(dataset); N = len(dataset)
    rowsz,nrow,rowp = parsespec(rowspec)
    colsz,ncol,colp = parsespec(colspec)
    if ncol == -1: assert nrow>0; ncol = adj(N,nrow)
    elif nrow == -1: assert ncol>0; nrow = adj(N,ncol)
    K = nrow*ncol
    fig,axes = subplots(nrow,ncol,squeeze=False,figsize=(ncol*colsz+colp,nrow*rowsz+rowp),**ka)
    updates = [(ax,self.mpl(ax)) for axr in axes for ax in axr]
    if N<=K: disp(dataset)
    else: return widget_control(dataset,K,disp)

#==================================================================================================
# Miscellaneous utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
@contextmanager
def training(net,flag):
#--------------------------------------------------------------------------------------------------
  oflag = net.training
  net.train(flag)
  yield
  net.train(oflag)

#--------------------------------------------------------------------------------------------------
class PROC:
  r"""
This class is meant to facilitate writing flexible pipelines of callable invocations. An instance of this class encapsulates a callable or :const:`None` (void instance). All callables take a single argument (typically a run). Generic function :func:`abs` (absolute value) returns the encapsulated callable. This class supports two operations.

* Operation `+` (sequential invocation). The two operands must be instances of :class:`PROC` and so is the result. If any of the two operands is void, the result is the other operand, otherwise it is defined by::

   abs(x+y) == lambda u,X=abs(x),Y=abs(y): X(u) is not None or Y(u) is not None or None

* Operation `%` (conditioning). The first operand must be an instance of :class:`PROC` and so is the result. The second operand must be a selector (a function with one input and a boolean output), or a boolean value (equivalent to the constant selector returning that value). If the first operand is void, so is the result; if the second operand is :const:`False`, the result is also void; if the second operand is :const:`True`, the result is the first operand; otherwise it is defined by::

   abs(x%y) == lambda u,X=abs(x): X(u) if y(u) else None
  """
#--------------------------------------------------------------------------------------------------
  def __init__(self,f=None): self.func = f
  def __mod__(self,other):
    if self.func is None or other is False: return PROC()
    if other is True: return self
    return PROC(lambda u,f=self.func,c=other: f(u) if c(u) else None)
  def __add__(self,other):
    if not isinstance(other,PROC): return NotImplemented
    return other if self.func is None else self if other.func is None else PROC(lambda u,f1=self.func,f2=other.func: f1(u) is not None or f2(u) is not None or None)
  def __abs__(self): return self.func

#--------------------------------------------------------------------------------------------------
def periodic(p:Union[int,float],counter:str=None)->Union[Callable[[Run],bool],None]:
  r"""
Returns a run selector, i.e. a callable which takes a run as input and returns a boolean. The selection is based on the value of a counter held by attribute *counter*.

* If *p* is of type :class:`int`, a run is selected if the counter value is a multiple of *p*. Default counter: :attr:`Run.step`.
* If *p* is of type :class:`float`, a run is selected if the increase in the counter value since the last successful selection is greater than *p*. Default counter: :attr:`Run.walltime`.

:param p: periodicity (in counter value)
:param counter: run attribute to use as counter
  """
#--------------------------------------------------------------------------------------------------
  if p is None: return None
  if isinstance(p,int):
    if counter is None: counter = 'step'
    return lambda run,a=counter: getattr(run,a)%p == 0
  elif isinstance(p,float):
    if counter is None: counter = 'walltime'
    current = 0.
    def F(run,a=counter):
      nonlocal current
      c = getattr(run,a); r = c-current>p
      if r: current = c
      return r
    return F
  raise TypeError(f'p[expected: int|float|NoneType; found: {type(p)}]')

#--------------------------------------------------------------------------------------------------
class Stepper:
#--------------------------------------------------------------------------------------------------
  def __init__(self,run):
    started = None
    run.created = getfqdn(),ctime()
    def reset():
      nonlocal started
      started = time(),process_time()
      run.walltime = run.proctime = 0.
      run.step = 0
    def tick():
      run.walltime,run.proctime = time()-started[0],process_time()-started[1]
      run.step += 1
    self.reset,self.tick = reset,tick

#--------------------------------------------------------------------------------------------------
def to(data,device):
#--------------------------------------------------------------------------------------------------
  for x in data: yield ((t.to(device) if isinstance(t,torch.Tensor) else t) for t in x)
