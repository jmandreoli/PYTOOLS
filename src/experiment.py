# File:                 experiment.py
# Creation date:        2020-01-15
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities for pytorch experiments
#

from __future__ import annotations
from typing import Sequence, Any, Callable, Iterable, Mapping, Tuple, Optional, TypeVar, Generic
import logging; logger = logging.getLogger(__name__)

from collections import namedtuple
from contextlib import contextmanager
from functools import partial
from itertools import count, zip_longest
from copy import deepcopy
from pathlib import Path
from time import time, ctime
from pydispatch import Dispatcher
import torch
from . import fmtwrap, time_fmt

#--------------------------------------------------------------------------------------------------
class udict (dict): # a utility class reflecting a dictionary as an object (attributes=keys); must be at the top
#--------------------------------------------------------------------------------------------------
  def __getattr__(self,a): return self[a]
  def __delattr__(self,a): del self[a]
  def __setattr__(self,a,v): self[a] = v
  def __getstate__(self): return dict(self)
  def __setstate__(self,d): self.update(d)

ProcessInfo = namedtuple('ProcessInfo','init host pid started')
DataInfo = namedtuple('DataInfo','source train valid test')
class RunStop (Exception): pass

#==================================================================================================
class Run (Dispatcher):
  r"""
Instances of this class are callables, taking no argument and returning no value. The execution of a run only changes its attribute values. All the keyword arguments of the constructor are passed to method :meth:`set`.

Runs may emit messages during their execution. Each message is passed a single argument, which is the run itself, so runs can be controlled by callbacks attached to messages. Listeners (instances of class :class:`RunListener`) are special cases of callbacks. A run can invoke a method :meth:`add<name>Listener` to attach a listener of type `<name>` to itself. Some such methods are defined in class :class:`Run` and its subclasses (see their documentation for details).

A typical event callback stores the result of a campaign of measures on the run. Measures (instances of class :class:`Measure`) are in charge of reporting individual measures. A run can invoke a method :meth:`add<name>Measure` to attach a measure of type `<name>` to itself. Some such methods are defined in class :class:`Run` and its subclasses (see their documentation for details). In particular, all the pytorch losses can be added in this way.

Attributes (\*) must be explicitly instantiated at creation time.
  """
#==================================================================================================

  _events_ = 'open','close','start_batch','end_batch','start_epoch','end_epoch','checkpoint'

  ## set at instantiation
  device: Optional[torch.device|str] = None
  r"""(\*)The device on which to run the model"""
  net: torch.nn.Module
  r"""(\*)The model (initially on cpu, but loaded on :attr:`device` at the beginning of the run and restored to cpu at the end)"""
  train_split: Iterable[Tuple[torch.Tensor,torch.Tensor]]
  r"""(\*)Iterable of batches. Each batch is a pair of a tensor of inputs and a tensor of labels (first dim = size of batch)"""
  valid_split: Iterable[Tuple[torch.Tensor,torch.Tensor]]
  r"""(\*)Same format as :attr:`train_data`"""
  test_split: Iterable[Tuple[torch.Tensor,torch.Tensor]]
  r"""(\*)Same format as :attr:`train_data`"""
  optimiser_factory: Callable[[Sequence[torch.nn.Parameter]],torch.optim.Optimizer]
  r"""(\*)The optimiser factory"""
  listeners: Mapping[str,RunListener]
  r"""Named listener objects"""
  measures: Mapping[str,Accumulator]
  r"""Named measure objects"""
  ## set at execution
  optimiser: torch.optim.Optimizer
  r"""The optimiser"""
  checkpoint: Checkpoint
  r"""The checkpointer"""
  walltime: Callable[[],float]
  r"""Wall time elapsed (in sec) since start of run"""
  step: int
  r"""Number of optimiser updates"""
  epoch: int
  r"""Number passes through the data"""
  loss: float = float('inf')
  r"""Last step loss"""
  r_step: int = 0
  r"""Number of optimiser updates in current data pass"""
  r_loss: float = float('inf')
  r"""Running loss average in current data pass"""
  progress: float = 0.
  r"""Completion between 0. and 1., set by listeners only"""
  process_info: ProcessInfo
  r"""Info about the process"""
  eval_valid: Callable[[],dict[str,float]]
  r"""Function returning the validation performance at the end of the last completed step (cached)"""
  eval_test: Callable[[],dict[str,float]]
  r"""Function returning the test performance at the end of the last completed step (cached)"""

#--------------------------------------------------------------------------------------------------
  def __init__(self,**ka):
#--------------------------------------------------------------------------------------------------
    self._config = fmtwrap(self.__dict__)
    self.listeners = udict()
    self.measures = udict()
    self.config(**ka)
  def config(self,**ka): self._config.update(**ka)

#--------------------------------------------------------------------------------------------------
  def main(self):
    r"""
Executes the run. Must be defined in subclasses. This implementation raises an error.
    """
#--------------------------------------------------------------------------------------------------
    net = self.net
    self.optimiser = optimiser = self.optimiser_factory(net.parameters())
    self.eval_valid,self.eval_test = map(self.eval_cache,(self.valid_split,self.test_split))
    net.to(self.device)
    self.initial()
    with training(net,True):
      try:
        self.r_loss = 0.; self.r_step = 0
        self.emit('open', self)
        while True: # iteration = epoch
          self.emit('start_epoch',self)
          for inputs in self.train_split: # iteration = batch
            self.emit('start_batch',self)
            inputs = to(inputs,self.device)
            optimiser.zero_grad()
            loss = net(*inputs)
            loss.backward()
            optimiser.step()
            del inputs # free memory before emitting
            self.loss = loss.item()
            self.step += 1; self.r_step += 1
            self.r_loss += (self.loss-self.r_loss)/self.r_step
            self.emit('end_batch',self)
          self.epoch += 1; self.r_loss = 0.; self.r_step = 0
          self.emit('end_epoch',self)
          self.checkpoint()
      except RunStop: self.checkpoint()
      finally: self.emit('close',self)

  def initial(self):
    from socket import getfqdn
    from os import getpid
    from datetime import datetime
    if (ini:=self.checkpoint.initial(self)) is None: ini=''; self.step = self.epoch = 0; self.walltime = (lambda start=time(): time()-start)
    else: ini = f'{ini}[{ctime(ini.stat().st_mtime)}][{time_fmt(self.checkpoint.last,1)}]'
    self.process_info = ProcessInfo(init=ini,host=getfqdn(),pid=getpid(),started=datetime.now())

  def state_dict(self):
    return {
      'net': self.net.state_dict(),
      'optimiser': self.optimiser.state_dict(),
      'step': self.step,
      'epoch': self.epoch,
      'listeners': {k:v.state_dict() for k,v in self.listeners.items()},
      'walltime': self.walltime(),
    }

  def load_state_dict(self,D:dict[str,Any]):
    self.net.load_state_dict(D['net'])
    self.optimiser.load_state_dict(D['optimiser'])
    self.step = D['step']
    self.epoch = D['epoch']
    for k,v in self.listeners.items(): v.load_state_dict(D['listeners'][k])
    self.walltime = (lambda start=time(),offset=D['walltime']: time()-start+offset)

  #--------------------------------------------------------------------------------------------------
  def eval(self,data:Iterable[Tuple[float,Tuple[torch.Tensor,...],Tuple[torch.Tensor,...]]])->dict[str,float]:
    r"""
Evaluates the embedding part of model :attr:`net` on *data* according to :attr:`measures`. The batches are not assumed constant sized.

:param data: a list of data entries
    """
#--------------------------------------------------------------------------------------------------
    acs = self.measures.values()
    net = self.net
    for ac in acs: ac.ini()
    with training(net,False),torch.no_grad():
      for inputs in data:
        inputs = to(inputs,self.device)
        x = net.embedding(*inputs)
        for ac in acs: ac.inc(x)
    return dict(zip(self.measures,(ac.val() for ac in acs)))

#--------------------------------------------------------------------------------------------------
  def eval_cache(self,data):
    r"""
Returns a cached version of :meth:`eval` on *data*. The cache is cleared at each step.
    """
#--------------------------------------------------------------------------------------------------
    current = None,-1
    def cache()->dict[str,float]:
      nonlocal current
      if current[1] != self.step: current = self.eval(data),self.step
      return current[0]
    return cache

#--------------------------------------------------------------------------------------------------
  @classmethod
  def measure_factory(cls,name:str,norm:Callable[[Any],float]=None,doc:bool=True)->Callable[[Callable],Callable]:
    r"""
Used as decorator to attach a measure factory (the decorated object, assumed callable) to a subclass *cls* of this class. The factory is then available as method :meth:`add<name>Measure` using the provided *name*. Its invocation accepts a parameter :attr:`label` under which its result in stored in the measures registry (by default, the label is the lowercase version of *name*).

:param name: short name for the factory
    """
#--------------------------------------------------------------------------------------------------
    def app(f):
      def F(self,label=name.lower(),*a,**ka):
        m = f(*a,**ka)
        self.measures[label] = acc = {
          'mean':partial(AvgAccumulator,feed=lambda x,m=m: (len(x[0]),m(*x).item())),
          'sum':partial(SumAccumulator,feed=lambda x,m=m: m(*x).item()),
          'none':partial(ListAccumulator,feed=lambda x,m=m: m(*x).cpu().detach().numpy()),
        }[m.reduction](descr=repr(m),norm=norm)
        return acc
      setattr(cls,f'add{name}Measure',F)
      if doc: F.__doc__ = f'Measure factory for :class:`{f.__qualname__}`. The result is assigned as attribute *label* in the measures register.'
      return f
    return app

#--------------------------------------------------------------------------------------------------
  @classmethod
  def listener_factory(cls,name:str,doc:bool=True)->Callable[[Callable],Callable]:
    r"""
Used as decorator to attach a listener factory (the decorated object, assumed callable) to a subclass *cls* of this class. The factory is then available as method :meth:`add<name>Listener` using the provided *name*. Its invocation accepts a parameter :attr:`label` under which its result in stored in the listeners registry (by default, the label is the lowercase version of *name*). The decorated object may have an attribute :attr:`tools` listing the name of tool attributes, which are passed into as attribute :attr:`<name>_tools` of *cls*.

:param name: short name for the factory
    """
#--------------------------------------------------------------------------------------------------
    def app(f):
      def F(self,label=name.lower(),*a,**ka): self.listeners[label] = x = f(*a,**ka); return x
      if doc: F.__doc__ = f'Listener factory for :class:`{f.__qualname__}`. The result is assigned as attribute *label* in the listeners register.'
      setattr(cls,f'add{name}Listener',F)
      tools = getattr(f,'tools',None)
      if tools is not None: setattr(cls,f'{name}_tools',udict((t,getattr(f,t)) for t in set(tools)))
      return f
    return app

  @contextmanager
  def activate_listeners(self):
    callbacks = lambda x,evs=self._events_: ((ev,t) for ev in evs if (t:=getattr(x,'on_'+ev,None)) is not None)
    L = tuple(tuple(callbacks(x)) for x in self.listeners.values() if x.active)
    for c in L: self.bind(**dict(c))
    yield
    for c in L: self.unbind(t for ev,t in c)

#--------------------------------------------------------------------------------------------------
  def describe(self,tab:str='',file=None,**ka):
    r"""
Prints out the list of configuration attributes (with their values) as well as the list of measures and listeners.

:param tab: an indentation string
:param file: passed to the print function
    """
#--------------------------------------------------------------------------------------------------
    self._config.describe(tab=tab,file=file,**ka)
    for k,x in self.measures.items():
      print(tab,f'measures.{k}:',sep='',file=file)
      print(tab+'  ',x,sep='',file=file)
    for k,x in self.listeners.items():
      print(tab,f'listeners.{k}:',sep='',file=file)
      x._config.describe(tab=tab+'  ',file=file,**ka)

  def __call__(self):
    with self.activate_listeners(): self.main()
  def __repr__(self):
    i = self.process_info
    return f'{self.__class__.__name__}({i.host}:{i.pid}|{i.init}|{i.started.ctime()})'

#==================================================================================================
class Net (torch.nn.Module):
  r"""
Instances of this class are nets with an embedding phase immediately followed by a loss.

:param loss: a loss function, or layer it is is parametrized
  """
#==================================================================================================
  loss: Callable
  loss_factory: Callable[[Mapping[str,Any]],Callable]
  def __init__(self,loss:Optional[Callable]=None):
    super().__init__()
    self.loss = self.loss_factory(**(loss or {})) if loss is None or isinstance(loss,dict) else loss
  def embedding(self,*inputs,**ka):
    r"""Returns a (seq of) embedding representation for each (seq of) input feature in the batch"""
    raise NotImplementedError()
  def forward(self,*inputs):
    r"""Sequentially composes embedding and loss."""
    return self.loss(*self.embedding(*inputs))

#==================================================================================================
class SupervisedNet (Net):
  r"""
Instances of this class are supervised task nets, where each component of a batch consists of one or more feature vector(s) and a label. The :meth:`embedding` method returns the result of applying a scoring net to the features and leaves the label unchanged.

:param net: The scoring net
  """
#==================================================================================================
  loss_factory = torch.nn.CrossEntropyLoss
  def __init__(self,net:torch.nn.Module=None,**ka):
    super().__init__(**ka)
    self.score = net
  def embedding(self,*inputs):
    *feats,labels = inputs
    scores = self.score(*feats)
    return scores,labels

#==================================================================================================
class Measure:
  r"""Base class for measures."""
#==================================================================================================
  Agg = {'mean':torch.mean,'sum':torch.sum,'none':(lambda x:x)}
  reduction = 'mean'
  def __init__(self,reduction=None):
    if reduction is not None: self.reduction = reduction
    self.agg = self.Agg[self.reduction]
  def __repr__(self): return f'{self.__class__.__name__}(reduction={self.reduction})'

def predefine_measures_from_standard_losses(): # invoked only once below, then destroyed
  # adds one measure to class Run for each loss function defined in torch.nn
  for p in dir(torch.nn):
    if p.endswith('Loss'): Run.measure_factory(p[:-4],doc=False)(getattr(torch.nn,p))
predefine_measures_from_standard_losses()
del predefine_measures_from_standard_losses

#==================================================================================================
@Run.measure_factory('ZeroOne')
class ZeroOneMeasure (Measure):
  r"""
Instances of this class measure accuracy for classification runs.
  """
#==================================================================================================
  def __call__(self,scores,labels): return self.agg((torch.argmax(scores,1)==labels).float())

#==================================================================================================
class RunListener:
  r"""
Instances of this class are listener objects, which can be bound to a :class:`Run` instance. All parameters are processed by method :meth:`setup`.
  """

  schedule: Mapping[str,Schedule]
  r"""Action schedule per stage (typically: train,valid)"""
  info: Mapping[str,Callable[[Run],Any]]
  r"""Run information generator per stage, available to the event listeners"""
  active: bool
  r"""Whether this instance is actively listening to events"""

#==================================================================================================
  def __init__(self,**ka):
    self._config = fmtwrap({})
    self._config.update(**ka)
    self.setup(**self._config.ref)
#==================================================================================================
  def setup(
      self,
      itrain:Callable[[Run],dict[str,Any]]=(lambda run: {'tloss':run.r_loss}),
      ivalid:Callable[[Run],dict[str,Any]]=(lambda run: run.eval_valid()),
      itest:Callable[[Run],dict[str,Any]]=(lambda run: run.eval_test()),
      active=True,
  ):
    r"""
Processes the constructor parameters.

:param itrain: extracts train information about a run
:param ivalid: extracts valid information about a run
:param itest: extracts test information about a run
    """
#==================================================================================================
    self.info = udict(train=itrain,valid=ivalid,test=itest)
    self.active = active
  def describe(self,**ka): self._config.describe(**ka)
  def state_dict(self): return {'schedule':self.schedule}
  def load_state_dict(self,D): self.__dict__.update(D)

#==================================================================================================
@Run.listener_factory('Base')
class BaseRunListener (RunListener):
  r"""
A :class:`RunListener` subclass with three types of actions: checkpointing; monitoring (and eventually stopping) progress; logging information.
  """
#==================================================================================================
  stop: Mapping[str,int|float]
  r"""Stop thresholds for each counter"""
  action: Mapping[str,Callable[[Run],None]]
  r"""Logging action per stage"""

  def setup(
      self,
      logger:Optional[logging.Logger]=logger,
      path:Optional[str|Path]=None,
      max_epoch:int=1<<30,
      max_time:float=float('inf'),
      train_schedule:float|Iterable[float]=float('inf'),
      valid_schedule:int|Iterable[int]=1<<32,
      status: Tuple[Tuple[str,...],Callable[[Run],Tuple]] = (
          ('TIME','COMP','STEP','EPO','BAT'),
          (lambda run: (run.walltime(),100*(run.progress),run.step,run.epoch,run.r_step)),
      ),
      status_fmt: Tuple[str,str]=('%6s %5s%% %6s %4s/%5s','%6.1f %5.1f%% %6d %4d/%5d'),
      fmt: Optional[Mapping[str,Callable[[Any],str]]]=None,
      **ka
  ):
    r"""
:param logger: logger to use to report information
:param max_epoch: stop after this number of epochs
:param max_time: stop after this elapsed time (in seconds)
:param train_schedule: time schedule for logging train information
:param valid_schedule: epoch schedule for logging validation information
    """
    super().setup(**ka)
    self.stop = udict(epoch=max_epoch,time=max_time)
    self.schedule = udict(train=Schedule(train_schedule),valid=Schedule(valid_schedule))
    fmt_ = lambda x,fmt={'size':'d','err':'s'}|(fmt or {}),default='.3f': ' '.join(f'{k}={v.__format__(fmt.get(k,default))}' for k,v in x.items() if v is not None)
    self.info.update(
      checkpoint=(lambda run: {'size':(run.checkpoint.path.stat().st_size if run.checkpoint.path.exists() else None),'err':run.checkpoint.err}),
    )
    self.action = udict({
      k:(lambda run,a=a,fmt=f'{status_fmt[1]} {k.upper()}: %s',s=status[1]: logger.info(fmt,*s(run),fmt_(a(run))))
      for k,a in self.info.items()
    })
    def aheader(run,fmt=status_fmt[0],s=status[0]): logger.info('RUN %s',run); logger.info(fmt,*s)
    self.action.header = aheader

  def on_open(self,run): self.action.header(run); self.set_progress(run,True); self.action.valid(run)
  def on_close(self,run): self.action.test(run)
  def on_end_epoch(self,run):
    self.set_progress(run,True)
    if self.schedule.valid(run.epoch): self.action.valid(run)
  def on_end_batch(self,run):
    if self.schedule.train(run.walltime()): self.set_progress(run,False); self.action.train(run)
  def on_checkpoint(self,run): self.set_progress(run,False); self.action.checkpoint(run)
  def set_progress(self,run,check:bool):
    run.progress = x = max(run.walltime()/self.stop.time,run.epoch/self.stop.epoch)
    if check and x>=1.: self.action.valid(run); raise RunStop()

#==================================================================================================
@Run.listener_factory('Mlflow')
class RunMlflowListener (RunListener):
  r"""
Instances of this class provide mlflow logging.
  """
#==================================================================================================
  tools = 'load_model',
  run_id = None
#--------------------------------------------------------------------------------------------------
  def setup(
      self,
      uri:str=None,
      exp:str=None,
      train_schedule:int|Iterable[int]=1<<32,
      valid_schedule:int|Iterable[int]=1<<32,
      **ka
  ):
    r"""
:param uri: mlflow tracking uri
:param exp: experiment name
:param train_schedule: time schedule for logging train information
:param valid_schedule: epoch schedule for logging validation information
    """
#--------------------------------------------------------------------------------------------------
    import mlflow, mlflow.pytorch
    super().setup(**ka)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    self.schedule = udict(train=Schedule(train_schedule),valid=Schedule(valid_schedule))

  def on_open(self,run):
    import mlflow
    r = mlflow.start_run(run_id=self.run_id)
    if self.run_id is None:
      self.run_id = r.info.run_id
      try: mlflow.log_params(dict((k,v[:250]) for k,v in run._config.items()))
      except: mlflow.end_run(); mlflow.delete_run(r.info.run_id); raise
    mlflow.set_tag('progress',getattr(run,'progress',0))

  def on_close(self,run):
    import mlflow
    try: mlflow.pytorch.log_model(run.net,'model')
    finally: mlflow.end_run()

  def on_end_batch(self,run):
    import mlflow
    if self.schedule.train(run.step): mlflow.log_metrics(self.info.train(run),run.step)
  def on_end_epoch(self,run):
    import mlflow
    if self.schedule.valid(run.epoch): mlflow.log_metrics(self.info.valid(run),run.step); mlflow.set_tag('progress',getattr(run,'progress',0))

  def state_dict(self): return {'run_id':self.run_id,**super().state_dict()}

#--------------------------------------------------------------------------------------------------
  @classmethod
  def load_model(cls,uri:str,exp:str,run_id:str=None,epoch:int=None,**ka):
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
  classes: Iterable[str]
  r"""(\*)Descriptive names for the classes"""
  train: torch.utils.data.Dataset
  r"""(\*)The train split of the data source"""
  test: torch.utils.data.Dataset
  r"""(\*)The test split of the data source"""

#--------------------------------------------------------------------------------------------------
  def mpl(self,ax:'matplotlib.Axes')->Callable[['numpy.array'],None]:
    r"""
Returns a callable, which, when passed a data instance, displays it on *ax* (its label is also used as title of *ax*). This implementation raises a :class:`NotImplementedError`.

:param ax: a display area for a data instance
    """
#--------------------------------------------------------------------------------------------------
    raise NotImplementedError()

#--------------------------------------------------------------------------------------------------
  def loaders(self,pcval:float,**ka)->dict[str,Any]:
    r"""
Returns a dictionary which can be passed as keyword arguments to the :class:`Run` constructor to initialise its required attributes:

* non visible attributes :attr:`train_data`, :attr:`valid_data`, :attr:`test_data`, obtained by calling :class:`torch.utils.data.DataLoader` on the corresponding split with key-word parameters *ka* (with prefix `_` removed from keys);
* a visible attribute :attr:`data` holding a description of the datasource;
* all the attributes (visible or not) in *ka*, so they appear as attributes of the run.

:param pcval: proportion (between 0. and 1.) of instances from the train split to use for validation.
    """
#--------------------------------------------------------------------------------------------------
    D = self.train
    n = len(D); nvalid = int(pcval*n); ntrain = n-nvalid
    train,valid = torch.utils.data.random_split(D,(ntrain,nvalid))
    test = self.test
    params = dict(datainfo=DataInfo(self,len(train),len(valid),len(test)),**ka)
    ka = dict(((k[1:] if k.startswith('_') else k),v) for k,v in ka.items())
    for c,split in zip(('train','valid','test'),(train,valid,test)):
      params['_'+c+'_split'] = torch.utils.data.DataLoader(split,shuffle=c=='train',**ka)
    return params

#--------------------------------------------------------------------------------------------------
  def display(self,
    rowspec:float|Tuple[float,int]|Tuple[float,int,float],
    colspec:float|Tuple[float,int]|Tuple[float,int,float],
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
    from ipywidgets import IntSlider,Text,Label,HBox,VBox,Button,Output
    from matplotlib.pyplot import subplots, close
    def disp(sample):
      for (ax,update),(*feat,label) in zip_longest(updates,sample,fillvalue=(None,)):
        visible = label is not None
        ax.set(visible=visible)
        if visible:
          update(*feat)
          ax.set_title(self.classes[label],fontsize='xx-small',pad=1)
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

    if dataset is None: dataset = self.test
    try: N = len(dataset)
    except TypeError: dataset = list(dataset); N = len(dataset)
    rowsz,nrow,rowp = parsespec(rowspec)
    colsz,ncol,colp = parsespec(colspec)
    if ncol == -1: assert nrow>0; ncol = adj(N,nrow)
    elif nrow == -1: assert ncol>0; nrow = adj(N,ncol)
    K = nrow*ncol
    lastpage = adj(N,K)

    w_closeb = Button(icon='close',tooltip='Close browser',layout=dict(width='auto',padding='1px'))
    w_sel = IntSlider(value=1,min=1,max=lastpage,layout=dict(width='10cm'),readout=False)
    w_page = Text(disabled=True,layout=dict(width=f'{.3+.3*len(str(lastpage))}cm'))
    w_out = Output()
    w_main = VBox((HBox((w_closeb,Label(f'{self} page:'),w_page,w_sel)),w_out))
    with w_out:
      gs = {'left':.05,'right':.95}|ka.get('gridspec_kw',{})
      fig,axes = subplots(nrow,ncol,squeeze=False,figsize=(ncol*colsz+colp,nrow*rowsz+rowp),gridspec_kw=gs,**ka)
    def disp_(): w_page.value = str(w_sel.value); disp([dataset[k] for k in range((w_sel.value-1)*K,min(w_sel.value*K,N))])
    def close_(): close(fig); w_main.close()
    w_closeb.on_click(lambda _: close_())
    w_sel.observe((lambda _: disp_()),'value')
    updates = [(ax,self.mpl(ax)) for row in axes for ax in row]
    disp_()
    return w_main

#==================================================================================================
# Miscellaneous utilities
#==================================================================================================

#--------------------------------------------------------------------------------------------------
class Checkpoint:
#--------------------------------------------------------------------------------------------------
  path: Optional[Path]
  rate: float
  run: Run
  last: float
  err: Optional[Exception]
  def __init__(self,path:Optional[str|Path]=None,rate:float=float('inf')):
    self.path = None if path is None else Path(path); self.rate = rate; self.last = 0.
  def __call__(self):
    from dill import dump
    from tempfile import NamedTemporaryFile
    path = self.path
    if path is None: return
    run = self.run
    t = run.walltime()
    if t-self.last >= self.rate:
      self.err = None
      try:
        with NamedTemporaryFile(dir=path.parent) as v:
          dump((t,run.state_dict()),v)
          v.flush()
          try: path.unlink()
          except: pass
          path.hardlink_to(v.name)
      except Exception as exc: self.err = exc
      else: self.last = t
      run.emit('checkpoint',run)
  def initial(self,run:Run):
    from dill import load
    path = self.path; self.run = run
    if path is None or not path.exists(): return None
    with path.open('rb') as u: self.last,D = load(u); run.load_state_dict(D)
    return path

Run.checkpoint = Checkpoint()

#--------------------------------------------------------------------------------------------------
class Schedule:
#--------------------------------------------------------------------------------------------------
  __slots__ = 'it','goal'
  def __init__(self,p:int|float|Iterable):
    self.it = count(p,p) if isinstance(p,(int,float)) else iter(p)
    self.goal = next(self.it)
  def __call__(self,v):
    flag = False
    while self.goal <= v: flag = True; self.goal = next(self.it)
    return flag

#--------------------------------------------------------------------------------------------------
def clone_module(m,out=None):
  r"""
Clones a :class:`torch.Module` *m* into another one *out* or a new one if *out* is :const:`None`.
  """
#--------------------------------------------------------------------------------------------------
  if out is None: out = deepcopy(m)
  out.load_state_dict(m.state_dict())
  return out

#--------------------------------------------------------------------------------------------------
@contextmanager
def training(obj,v):
  r"""
Returns a context which saves the value of :attr:`training` in *obj* on enter and restores it on exit.
  """
#--------------------------------------------------------------------------------------------------
  oldv = obj.training; obj.train(v); yield; obj.train(oldv)

#--------------------------------------------------------------------------------------------------
def to(L,device): return ((t.to(device) if isinstance(t,torch.Tensor) else t) for t in L)
#--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
class Accumulator:
#--------------------------------------------------------------------------------------------------
  def __init__(self,feed:Callable=None,norm:Callable=None,descr:str=None):
    self.feed = (lambda x:x) if feed is None else feed
    self.norm = (lambda x:x) if norm is None else norm
    self.repr = f'{self.__class__.__name__}({self.feed.__name__},{self.norm.__name__})' if descr is None else descr
  def ini(self): self._val = self._ini()
  def inc(self,x): self._inc(self.feed(x))
  def val(self): return self.norm(self._val)
  def __repr__(self): return self.repr
  def _ini(self): raise NotImplementedError()
  def _inc(self,v): raise NotImplementedError()

#--------------------------------------------------------------------------------------------------
class AvgAccumulator (Accumulator):
#--------------------------------------------------------------------------------------------------
  # input as pair of scalars (weight,value)
  def _ini(self): self._w = 0.; return 0.
  def _inc(self,x): w,v = x; self._w += w; self._val += (v-self._val)*(w/self._w)

#--------------------------------------------------------------------------------------------------
class MeanAccumulator (Accumulator):
#--------------------------------------------------------------------------------------------------
  def _ini(self): self._n = 0.; return 0.
  def _inc(self,x): self._n += 1.; self._val += (x-self._val)/self._n

#--------------------------------------------------------------------------------------------------
class SumAccumulator (Accumulator):
#--------------------------------------------------------------------------------------------------
  def _ini(self): return 0.
  def _inc(self,x): self._val += x

#--------------------------------------------------------------------------------------------------
class ListAccumulator (Accumulator):
#--------------------------------------------------------------------------------------------------
  def _ini(self): return []
  def _inc(self,x): self._val.append(x)

# #--------------------------------------------------------------------------------------------------
# class PROC:
#   r"""
# This class is meant to facilitate writing flexible pipelines of callable invocations. An instance of this class encapsulates a callable or :const:`None` (void instance). All callables take a single argument (typically a run). Generic function :func:`abs` (absolute value) returns the encapsulated callable. This class supports two operations.
#
# * Operation `+` (sequential invocation). The two operands must be instances of :class:`PROC` and so is the result. If any of the two operands is void, the result is the other operand, otherwise it is defined by::
#
#    abs(x+y) == lambda u,X=abs(x),Y=abs(y): X(u) is not None or Y(u) is not None or None
#
# * Operation `%` (conditioning). The first operand must be an instance of :class:`PROC` and so is the result. The second operand must be a selector (a function with one input and a boolean output), or a boolean value (equivalent to the constant selector returning that value). If the first operand is void, so is the result; if the second operand is :const:`False`, the result is also void; if the second operand is :const:`True`, the result is the first operand; otherwise it is defined by::
#
#    abs(x%y) == lambda u,X=abs(x): X(u) if y(u) else None
#   """
# #--------------------------------------------------------------------------------------------------
#   def __init__(self,f=None): self.func = f
#   def __mod__(self,other):
#     if self.func is None or other is False: return PROC()
#     if other is True: return self
#     return PROC(lambda u,f=self.func,c=other: f(u) if c(u) else None)
#   def __add__(self,other):
#     if not isinstance(other,PROC): return NotImplemented
#     return other if self.func is None else self if other.func is None else PROC(lambda u,f1=self.func,f2=other.func: f1(u) is not None or f2(u) is not None or None)
#   def __abs__(self): return self.func
#
# #==================================================================================================
# class SupervisedInvRun (Run):
#   r"""
# Instances of this class are runs in which the role of data and parameters are inverted.
#   """
# #==================================================================================================
#   protos:Tuple[torch.Tensor,...]
#   projection:Callable[[torch.Tensor],None]
#   train_data = ()
#   valid_data = ()
#   test_data = ()
#   class InvNet (torch.nn.Module):
#     r"""A variant of *net* where the parameters are given by *protos*."""
#     def __init__(self,net,protos):
#       super().__init__()
#       *feats,labels = protos
#       self.params = torch.nn.ParameterList(torch.nn.Parameter(x) for x in feats)
#       self.labels = labels
#       super().__setattr__('net',net) # so the net is not added as a submodule of the inv net
#     def forward(self):
#       r"""Simply invokes *net* with the appropriate arguments."""
#       return self.net(*self.params,self.labels)
#   def main(self):
#     net = self.net
#     try:
#       self.net = self.InvNet(net,self.protos)
#       super().main()
#     finally: self.net = net
#   @staticmethod
#   def on_post_optim(run): tuple(run.projection(p.data) for p in run.net.params)
#
# #==================================================================================================
# @Run.listener_factory('Tensorboard')
# class RunTensorboardListener (RunListener):
#   r"""
# Instances of this class provide tensorboard logging for runs. Limited functionality if standalone tb is installed (without tf).
#   """
# #==================================================================================================
# #--------------------------------------------------------------------------------------------------
#   def set(self,
#     itrain:Callable[[Run],Tuple[Any,...]]=(lambda run: {'loss/train':run.loss.val()}),
#     ivalid:Callable[[Run],Tuple[Any,...]]=(lambda run: dict((k+'/valid',v) for k,v in run.eval_valid().items())),
#     itest:Callable[[Run],Tuple[Any,...]] =(lambda run: dict((k+'/test',v) for k,v in run.eval_test().items())),
#     **ka
#   ):
#     r"""
# :param uri: mlflow tracking uri
# :param exp: experiment name
# :param itrain: function returning train info on a run
# :param ivalid: function returning validation info on a run
# :param itest: function returning test info on a run
#
# Computes a number of information loggers, stored in attributes :attr:`itrain`, :attr:`ivalid`, :attr:`itest`, :attr:`iprogress`, :attr:`open_tb`, :attr:`close_tb`. The *uri* must refer to a valid tensorboard experiment storage.
#     """
# #--------------------------------------------------------------------------------------------------
#     log_metrics = lambda f,run: tuple(run.tbwriter.add_scalar(k,v,run.step) for k,v in f(run).items())
#     self.itrain = None if itrain is None else partial(log_metrics,itrain)
#     self.ivalid = None if ivalid is None else partial(log_metrics,ivalid)
#     self.itest = None if itest is None else partial(log_metrics,itest)
#     self.iprogress = lambda run: run.tbwriter.add_scalar('progress',run.progress,run.step)
#     self.checkpoint = None # MISSING: checkpoint
#     self.open_tb = lambda run: tuple(run.tbwriter.add_text(k,'<pre>{}</pre>'.format(v.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')),0) for k,v in run._config.items())
#     self.close_tb = None # MISSING: save model
#     super().set(**ka)
#
# #--------------------------------------------------------------------------------------------------
#   def set2(self,
#     train_p:Callable[[Run],bool]|bool=False,
#     valid_p:Callable[[Run],bool]|bool=False,
#     checkpoint_p:Callable[[Run],bool]|bool=False,
#     **ka
#   ):
#     r"""
# :param train_p: run selector for train info logging at each batch
# :param valid_p: run selector for validation info logging at each epoch
# :param checkpoint_p: run selector for checkpointing at each epoch
#
# Configures the callbacks of this listener, stored as attributes :attr:`on_open`, :attr:`on_close`, :attr:`on_batch`, :attr:`on_epoch`, using components defined by method :meth:`set`.
#     """
# #--------------------------------------------------------------------------------------------------
#     self.on_open = abs(PROC(self.open_tb))
#     self.on_close = abs(PROC(self.ivalid)+PROC(self.itest)+PROC(self.iprogress)+PROC(self.close_tb))
#     self.on_batch = abs(PROC(self.itrain)%train_p)
#     self.on_epoch = abs(PROC(self.iprogress)+PROC(self.ivalid)%valid_p+PROC(self.checkpoint)%checkpoint_p)
#     super().set2(**ka)
#
# #--------------------------------------------------------------------------------------------------
#   def set3(self,root=None,exp=None,**ka):
#     r"""
# Protects callbacks :attr:`on_open` and :attr:`on_close` from exceptions to make sure the tensorboard run is not left in a corrupted state.
#     """
# #--------------------------------------------------------------------------------------------------
#     from torch.utils.tensorboard import SummaryWriter
#     from pathlib import Path
#     from datetime import datetime
#     import shutil
#     root = Path(root).resolve()
#     assert root.is_dir()
#     exp = root/exp
#     exp.mkdir(exist_ok=True)
#     def on_open(run,f=self.on_open):
#       path = exp/'{0}_{1.host}_{1.pid}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'),run.process_info)
#       run.tbwriter = SummaryWriter(log_dir=str(path))
#       try: f(run)
#       except:
#         run.tbwriter.close()
#         if path.exists(): shutil.rmtree(path)
#         raise
#     def on_close(run,f=self.on_close):
#       try: f(run)
#       finally: run.tbwriter.close()
#     self.on_open,self.on_close = on_open,on_close
#     super().set3(**ka)
