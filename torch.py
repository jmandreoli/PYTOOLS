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

from collections import namedtuple
from contextlib import contextmanager
from functools import partial, update_wrapper
from copy import deepcopy
from pydispatch import Dispatcher, Property
import torch
from . import fmtwrap

#--------------------------------------------------------------------------------------------------
class udict (dict): # a utility class; must be att the top
#--------------------------------------------------------------------------------------------------
  def __getattr__(self,a): return self[a]
  def __delattr__(self,a): del self[a]
  def __setattr__(self,a,v): self[a] = v

ProcessInfo = namedtuple('ProcessInfo','host pid started')
DataInfo = namedtuple('DataInfo','source train valid test')

#==================================================================================================
class Run (Dispatcher):
  r"""
Instances of this class are callables, taking no argument and returning no value. The execution of a run only changes its attribute values. All the keyword arguments of the constructor are passed to method :meth:`set`.

Runs may emit messages during their execution. Each message is passed a single argument, which is the run itself, so runs can be controlled by message listeners. Listeners (instances of class :class:`RunListener`) are special cases of listeners. A run can invoke a method :meth:`add<name>Listener` to attach a listener of type `<name>` to itself. Some such methods are defined in subclasses of :class:`RunListener` (see their documentation for details).

Attributes (\*) must be instantiated at creation time.
  """
#==================================================================================================

  _events_ = 'open','batch','epoch','close', 'pre_loss','post_loss','pre_optim','post_optim'

  ## set at instantiation
  device: Union[torch.device,str] = None
  r"""(\*)The device on which to run the model"""
  net: torch.nn.Module
  r"""(\*)The model (initially on cpu, but loaded on :attr:`device` at the beginning of the run and restored to cpu at the end)"""
  train_data: Iterable[Tuple[torch.Tensor,torch.Tensor]]
  r"""(\*)Iterable of batches. Each batch is a pair of a tensor of inputs and a tensor of labels (first dim = size of batch)"""
  valid_data: Iterable[Tuple[torch.Tensor,torch.Tensor]]
  r"""(\*)Same format as :attr:`train_data`"""
  test_data: Iterable[Tuple[torch.Tensor,torch.Tensor]]
  r"""(\*)Same format as :attr:`train_data`"""
  optimiser_factory: Callable[[Sequence[torch.nn.Parameter]],torch.optim.Optimizer]
  r"""(\*)The optimiser factory"""
  stepper: Stepper
  r"""Time and step keeper"""
  walltime: float
  r"""Wall time elapsed (in sec) between last invocations of :meth:`reset` and :meth:`tick` on :attr:`stepper`"""
  proctime: float
  r"""Process time elapsed (in sec) between last invocations of :meth:`reset` and :meth:`tick` on :attr:`stepper`"""
  step: int
  r"""Number of invocations of :meth:`tick` since last invocation of :meth:`reset` on :attr:`stepper`"""
  eval_valid: Callable[[],Tuple[float,float]]
  r"""Function returning the validation performance at the end of the last completed step (cached)"""
  eval_test: Callable[[],Tuple[float,float]]
  r"""Function returning the test performance at the end of the last completed step (cached)"""
  ## set at execution
  progress: float
  r"""Between 0. and 1., stops the run when 1. is reached (must be updated by a listener)"""
  epoch: int
  r"""Number of completed epochs"""
  batch: int
  r"""Number of completed batches within current epoch"""
  loss: float
  r"""Average loss since beginning of current epoch"""

#--------------------------------------------------------------------------------------------------
  def __init__(self,**ka):
#--------------------------------------------------------------------------------------------------
    self._config = fmtwrap(self.__dict__)
    self.stepper = Stepper(self)
    self.listeners = udict() # listeners register
    self.measures = udict()
    self.config(**ka)
    self.bind(**dict((ev,t) for ev in self._events_ if (t:=getattr(self,'on_'+ev,None)) is not None))
  def config(self,**ka): self._config.update(**ka)

#--------------------------------------------------------------------------------------------------
  def main(self):
    r"""
Executes the run. Must be defined in subclasses. This implementation raises an error.
    """
#--------------------------------------------------------------------------------------------------
    net = self.net
    net.to(self.device)
    optimiser = self.optimiser_factory(net.parameters())
    self.eval_valid,self.eval_test = map(self.eval_cache,(self.valid_data,self.test_data))
    with training(net,True):
      self.stepper.reset()
      self.progress = 0.; self.epoch = 0
      self.batch = 0; self.loss = 0.
      self.emit('open',self)
      try:
        while self.progress<1.:
          for inputs in self.train_data:
            inputs = to(inputs,self.device)
            optimiser.zero_grad()
            self.emit('pre_loss',self)
            self.loss_ = loss = net(*inputs)
            self.emit('post_loss',self)
            loss.backward()
            self.emit('pre_optim',self)
            optimiser.step()
            self.emit('post_optim',self)
            self.batch += 1
            self.loss += (loss.item()-self.loss)/self.batch
            del inputs,loss # free memory before emitting (not sure this works)
            self.stepper.tick()
            self.emit('batch',self)
            if self.progress>=1.: return
          self.epoch += 1
          self.batch = 0; self.loss = 0.
          self.emit('epoch',self)
      finally:
        self.emit('close',self)

#--------------------------------------------------------------------------------------------------
  def eval(self,data:Iterable[Tuple[float,Tuple[torch.Tensor,...],Tuple[torch.Tensor,...]]])->Tuple[float,...]:
    r"""
Evaluates the policy part of model :attr:`net` on *data* according to :attr:`measures`. The training batches are not assumed constant sized.

:param data: a list of data entries
    """
#--------------------------------------------------------------------------------------------------
    avg_measures = 0.
    net = self.net
    with training(net,False),torch.no_grad():
      W = 0.
      for inputs in data:
        inputs = to(inputs,self.device)
        x = net.embedding(*inputs)
        measures = tuple(m(*x) for m in self.measures.values())
        w = len(x[0]); W += w
        avg_measures += (torch.stack(measures)-avg_measures)*(w/W)
    return dict(zip(self.measures,avg_measures.cpu().numpy()))

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
  def predef_measures(self,**ka):
    r"""
A helper to attach measures to this run. *ka* maps names to strings denoting predefined measures.
    """
#--------------------------------------------------------------------------------------------------
    def lookup(n):
      ka = {}
      if isinstance(n,tuple): n,ka = n
      return self.predef_measures_lookup[n](**ka)
    self.measures.update((name,lookup(n)) for name,n in ka.items())
  predef_measures_lookup = {
    'ZeroOne': (lambda reduction='mean',Agg={'mean':torch.mean,'sum':torch.sum,'none':(lambda x:x)}: lambda scores,labels,agg=Agg[reduction]: agg((torch.argmax(scores,1)==labels).float())),
    **dict((p[:-4],getattr(torch.nn,p)) for p in dir(torch.nn) if p.endswith('Loss'))
  }

#--------------------------------------------------------------------------------------------------
  @classmethod
  def listener_factory(cls,name:str)->Callable[[Callable],Callable]:
    r"""
Used as decorator to attach a listener factory (the decorated object, assumed callable) to a subclass *cls* of this class. The factory is then available as method :meth:`add<name>Listener` using the provided *name*. Its invocation accepts a parameter :attr:`label` under which its result in stored in the listeners registry (by default, the label is the lowercase version of *name*). The decorated object may have an attribute :attr:`tools` listing the name of tool attributes, which are passed into as attribute :attr:`<name>_tools` of *cls*.

:param name: short name for the factory
    """
#--------------------------------------------------------------------------------------------------
    def app(f):
      def F(self,label=name.lower(),*a,**ka): self.listeners[label] = x = f(*a,**ka); return x
      F.__doc__ = f'Listener factory for {f}. The result is assigned as attribute *label* in the listeners register.'
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
Prints out the list of configuration attributes (with their values) as well as the list of listeners.

:param tab: an indentation string
:param file: passed to the print function
    """
#--------------------------------------------------------------------------------------------------
    self._config.describe(tab=tab,file=file,**ka)
    for k,x in self.listeners.items():
      print(tab,f'listener.{k}:',sep='',file=file)
      x.describe(tab=tab+'  ',file=file,**ka)

  def __call__(self):
    with self.activate_listeners(): self.main()
  def __repr__(self):
    i = self.process_info
    return f'{self.__class__.__name__}({i.host}:{i.pid}|{i.started.ctime()})'

#==================================================================================================
class SupervisedNet (torch.nn.Module):
  r"""
Instances of this class are supervised task nets, composed of a main module computing an encoding of the features followed by a loss module measuring the computed encoding against ground truth labels. In training mode, a single scalar is returned, which is the average of the loss measures over the batch. In evaluation mode, a tuple of scalars is returned, the first one being the average loss, the others being averages of other measures (passed in forward).
  """
#==================================================================================================
  def __init__(self,policy,loss=None):
    super().__init__()
    self.policy = policy
    if loss is None: loss = torch.nn.CrossEntropyLoss()
    self.loss = loss
  def embedding(self,*inputs):
    *feats,labels = inputs
    scores = self.policy(*feats)
    return scores,labels
  def forward(self,*inputs):
    return self.loss(*self.embedding(*inputs))

#==================================================================================================
class SupervisedInvRun (Run):
  r"""
Instances of this class are runs in which the role of data and parameters are inverted.
  """
#==================================================================================================
  protos:Tuple[torch.Tensor,...]
  projection:Callable[[torch.Tensor,...],None]
  train_data = (),
  valid_data = ()
  test_data = ()
  class InvNet (torch.nn.Module):
    def __init__(self,net,protos):
      super().__init__()
      *feats,labels = protos
      self.params = torch.nn.ParameterList(torch.nn.Parameter(x) for x in feats)
      self.labels = labels
      self.__dict__.update(net=net) # so the net is not added as a submodule of the inv net
    def forward(self):
      return self.net(*self.params,self.labels)
  def main(self):
    net = self.net
    try:
      self.net = self.InvNet(self.net,self.protos)
      super().main()
    finally: self.net = net
  @staticmethod
  def on_post_optim(run): tuple(run.projection(p.data) for p in run.net.params)

#==================================================================================================
class RunListener:
  r"""Base class for run listeners"""
#==================================================================================================
  def __init__(self,**ka): self._config = fmtwrap({}); self.config(**ka)
  def config(self,**ka): self._config.update(**ka); self.set(**self._config.proxy)
  def set(self,**ka): self.set2(**ka)
  def set2(self,**ka): self.set3(**ka)
  def set3(self,active=True): self.active = active
  def describe(self,**ka): self._config.describe(**ka)

#==================================================================================================
@Run.listener_factory('Base')
class RunBaseListener (RunListener):
  r"""
Instances of this class provide basic logging/monitoring for supervised training runs.
  """
#==================================================================================================
#--------------------------------------------------------------------------------------------------
  def set(self,max_epoch:int=None,max_time:float=None,logger:logging.Logger=None,
    status:Tuple[Tuple[str,...],Callable[[Run],Tuple[Any,...]]]=(
      ('TIME','STEP','EPO','BAT'),
      (lambda run: (run.walltime,run.step,run.epoch,run.batch))
    ),
    status_fmt:Tuple[str,str]=('%6s %6s %4s/%5s','%6.1f %6d %4d/%5d'),
    itrain:Callable[[Run],Mapping[str,Any]]=(lambda run: dict(tloss=run.loss)),
    ivalid:Callable[[Run],Mapping[str,Any]]=(lambda run: run.eval_valid()),
    itest:Callable[[Run],Mapping[Any]]=(lambda run: run.eval_test()),
    fmt:Mapping[str,Callable[[Any],str]]={},
    **ka
  ):
    r"""
:param max_epoch: maximum number of epochs to run; default: no limit
:param max_time: maximum total wall time to run; default: no limit
:param logger: logger to use for logging information
:param status: pair of a tuple of headers and a function returning the status of a run as a tuple matching those headers
:param status_fmt: pair of format strings for the components of *status*
:param itrain,ivalid,itest: function returning various info on a run
:param fmt: dictionary associating measure names to their formats

Computes a number of information loggers, stored in attributes :attr:`itrain`, :attr:`ivalid`, :attr:`itest`, :attr:`iprogress`, :attr:`iheader`, as well as the main listenering function, stored as attribute :attr:`progress`. At least one of *max_epoch* or *max_time* must be set, or the run never ends.
    """
#--------------------------------------------------------------------------------------------------
    assert max_epoch is not None or max_time is not None
    self.itrain = self.ivalid = self.itest = self.iheader = self.iprogress = None
    fmt_ = lambda x,fmt=fmt,default='{:.3f}'.format: ' '.join(f'{k}={fmt.get(k,default)(v)}' for k,v in x.items())
    if logger is not None:
      if itrain is not None:
        self.itrain = lambda run,fmt=f'{status_fmt[1]} TRAIN: %s',s=status[1]: logger.info(fmt,*s(run),fmt_(itrain(run)))
      if ivalid is not None:
        self.ivalid = lambda run,fmt=f'{status_fmt[1]} VALID: %s',s=status[1]: logger.info(fmt,*s(run),fmt_(ivalid(run)))
      if itest is not None:
        self.itest = lambda run,fmt=f'{status_fmt[1]} TEST: %s',s=status[1]: logger.info(fmt,*s(run),fmt_(itest(run)))
      self.iprogress = lambda run,fmt=f'{status_fmt[1]} PROGRESS: %s',s=status[1]: logger.info(fmt,*s(run),f'{run.progress:.0%}')
      self.iheader = lambda run,fmt=status_fmt[0],s=status[0]: logger.info(fmt,*s) if logger.info('RUN %s',run) or True else None
    if max_epoch is None: max_epoch = float('inf')
    if max_time is None: max_time = float('inf')
    def set_progress(run): run.progress = min(max(run.epoch/max_epoch,run.walltime/max_time),1.)
    self.set_progress = set_progress
    super().set(**ka)

#--------------------------------------------------------------------------------------------------
  def set2(self,
    train_p:Union[Callable[[Run],bool],bool]=False,
    valid_p:Union[Callable[[Run],bool],bool]=False,
    **ka
  ):
    r"""
:param train_p: run selector for train info logging at each batch
:param valid_p: run selector for validation info logging at each epoch

Configures the callbacks of this listener, stored as attributes :attr:`on_open`, :attr:`on_close`, :attr:`on_batch`, :attr:`on_epoch`, using components defined by method :meth:`set`.
    """
#--------------------------------------------------------------------------------------------------
    self.on_open = abs(PROC(self.iheader))
    self.on_close = abs(PROC(self.itest))
    self.on_batch = abs(PROC(self.itrain)%train_p)
    self.on_epoch = abs(PROC(self.set_progress)+PROC(self.iprogress)+PROC(self.ivalid)%valid_p)
    super().set2(**ka)

#==================================================================================================
@Run.listener_factory('Mlflow')
class RunMlflowListener (RunListener):
  r"""
Instances of this class provide mlflow logging.
  """
#==================================================================================================
  tools = 'load_model',
#--------------------------------------------------------------------------------------------------
  def set(self,uri:str,exp:str,
    itrain:Callable[[Run],Tuple[Any,...]]=(lambda run: dict(tloss=run.loss)),
    ivalid:Callable[[Run],Tuple[Any,...]]=(lambda run: run.eval_valid()),
    itest:Callable[[Run],Tuple[Any,...]] =(lambda run: run.eval_test()),
    **ka
  ):
    r"""
:param uri: mlflow tracking uri
:param exp: experiment name
:param itrain,ivalid,itest: function returning various info on a run

Computes a number of information loggers, stored in attributes :attr:`itrain`, :attr:`ivalid`, :attr:`itest`, :attr:`iprogress`, :attr:`open_mlflow`, :attr:`close_mlflow`. The *uri* must refer to a valid mlflow experiment storage.
    """
#--------------------------------------------------------------------------------------------------
    import mlflow, mlflow.pytorch
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    log_metrics = lambda f,run: mlflow.log_metrics(f(run),run.step)
    self.itrain = None if itrain is None else partial(log_metrics,itrain)
    self.ivalid = None if ivalid is None else partial(log_metrics,ivalid)
    self.itest = None if itest is None else partial(log_metrics,itest)
    self.iprogress = lambda run: mlflow.set_tag('progress',run.progress)
    self.checkpoint = lambda run: mlflow.pytorch.log_model(run.net,f'model_{run.epoch:03d}')
    self.open_mlflow = lambda run: mlflow.log_params(dict((k,v[:250]) for k,v in run._config.items()))
    self.close_mlflow = lambda run: mlflow.pytorch.log_model(run.net,'model')
    super().set(**ka)

#--------------------------------------------------------------------------------------------------
  def set2(self,
    train_p:Union[Callable[[Run],bool],bool]=False,
    valid_p:Union[Callable[[Run],bool],bool]=False,
    checkpoint_p:Union[Callable[[Run],bool],bool]=False,
    **ka
  ):
    r"""
:param train_p: run selector for train info logging at each batch
:param valid_p: run selector for validation info logging at each epoch
:param checkpoint_p: run selector for checkpointing at each epoch

Configures the callbacks of this listener, stored as attributes :attr:`on_open`, :attr:`on_close`, :attr:`on_batch`, :attr:`on_epoch`, using components defined by method :meth:`set`.
    """
#--------------------------------------------------------------------------------------------------
    self.on_open = abs(PROC(self.open_mlflow))
    self.on_close = abs(PROC(self.ivalid)+PROC(self.itest)+PROC(self.iprogress)+PROC(self.close_mlflow))
    self.on_batch = abs(PROC(self.itrain)%train_p)
    self.on_epoch = abs(PROC(self.iprogress)+PROC(self.ivalid)%valid_p+PROC(self.checkpoint)%checkpoint_p)
    super().set2(**ka)

#--------------------------------------------------------------------------------------------------
  def set3(self,**ka):
    r"""
Protects callbacks :attr:`on_open` and :attr:`on_close` from exceptions to make sure the mlflow run is not left in a corrupted state.
    """
#--------------------------------------------------------------------------------------------------
    import mlflow
    def on_open(run,f=self.on_open):
      r = mlflow.start_run()
      try: f(run)
      except: mlflow.end_run(); mlflow.delete_run(r.info.run_id); raise
    def on_close(run,f=self.on_close):
      try: f(run)
      finally: mlflow.end_run()
    self.on_open,self.on_close = on_open,on_close
    super().set3(**ka)

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
@Run.listener_factory('Tensorboard')
class RunTensorboardListener (RunListener):
  r"""
Instances of this class provide tensorboard logging for runs. Limited functionality if standalone tb is installed (without tf).
  """
#==================================================================================================
#--------------------------------------------------------------------------------------------------
  def set(self,
    itrain:Callable[[Run],Tuple[Any,...]]=(lambda run: {'loss/train':run.loss}),
    ivalid:Callable[[Run],Tuple[Any,...]]=(lambda run: dict((k+'/valid',v) for k,v in run.eval_valid().items())),
    itest:Callable[[Run],Tuple[Any,...]] =(lambda run: dict((k+'/test',v) for k,v in run.eval_test().items())),
    **ka
  ):
    r"""
:param uri: mlflow tracking uri
:param exp: experiment name
:param itrain,ivalid,itest: function returning various info on a run

Computes a number of information loggers, stored in attributes :attr:`itrain`, :attr:`ivalid`, :attr:`itest`, :attr:`iprogress`, :attr:`open_mlflow`, :attr:`close_mlflow`. The *uri* must refer to a valid mlflow experiment storage.
    """
#--------------------------------------------------------------------------------------------------
    log_metrics = lambda f,run: tuple(run.tbwriter.add_scalar(k,v,run.step) for k,v in f(run).items())
    self.itrain = None if itrain is None else partial(log_metrics,itrain)
    self.ivalid = None if ivalid is None else partial(log_metrics,ivalid)
    self.itest = None if itest is None else partial(log_metrics,itest)
    self.iprogress = lambda run: run.tbwriter.add_scalar('progress',run.progress,run.step)
    self.checkpoint = None # MISSING: checkpoint
    self.open_tb = lambda run: tuple(run.tbwriter.add_text(k,'<pre>{}</pre>'.format(v.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')),0) for k,v in run._config.items())
    self.close_tb = None # MISSING: save model
    super().set(**ka)

#--------------------------------------------------------------------------------------------------
  def set2(self,
    train_p:Union[Callable[[Run],bool],bool]=False,
    valid_p:Union[Callable[[Run],bool],bool]=False,
    checkpoint_p:Union[Callable[[Run],bool],bool]=False,
    **ka
  ):
    r"""
:param train_p: run selector for train info logging at each batch
:param valid_p: run selector for validation info logging at each epoch
:param checkpoint_p: run selector for checkpointing at each epoch

Configures the callbacks of this listener, stored as attributes :attr:`on_open`, :attr:`on_close`, :attr:`on_batch`, :attr:`on_epoch`, using components defined by method :meth:`set`.
    """
#--------------------------------------------------------------------------------------------------
    self.on_open = abs(PROC(self.open_tb))
    self.on_close = abs(PROC(self.ivalid)+PROC(self.itest)+PROC(self.iprogress)+PROC(self.close_tb))
    self.on_batch = abs(PROC(self.itrain)%train_p)
    self.on_epoch = abs(PROC(self.iprogress)+PROC(self.ivalid)%valid_p+PROC(self.checkpoint)%checkpoint_p)
    super().set2(**ka)

#--------------------------------------------------------------------------------------------------
  def set3(self,root=None,exp=None,**ka):
    r"""
Protects callbacks :attr:`on_open` and :attr:`on_close` from exceptions to make sure the mlflow run is not left in a corrupted state.
    """
#--------------------------------------------------------------------------------------------------
    from torch.utils.tensorboard import SummaryWriter
    from pathlib import Path
    from datetime import datetime
    import shutil
    root = Path(root).resolve()
    assert root.is_dir()
    exp = root/exp
    exp.mkdir(exist_ok=True)
    def on_open(run,f=self.on_open):
      path = exp/'{0}_{1.host}_{1.pid}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'),run.process_info)
      run.tbwriter = SummaryWriter(log_dir=str(path))
      try: f(run)
      except:
        run.tbwriter.close()
        if path.exists(): shutil.rmtree(path)
        raise
    def on_close(run,f=self.on_close):
      try: f(run)
      finally: run.tbwriter.close()
    self.on_open,self.on_close = on_open,on_close
    super().set3(**ka)

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
  def mpl(self,ax:matplotlib.Axes)->Callable[[torch.Tensor],None]:
    r"""
Returns a callable, which, when passed a data instance, displays it on *ax* (its label is also used as title of *ax*). This implementation raises a :class:`NotImplementedError`.

:param ax: a display area for a data instance
    """
#--------------------------------------------------------------------------------------------------
    raise NotImplementedError()

#--------------------------------------------------------------------------------------------------
  def loaders(self,pcval:float,**ka):
    r"""
Returns a dictionary which can be passed as keyword arguments to the :class:`Run` constructor to initialise its required attributes:

* non config attributes :attr:`train_data`, :attr:`valid_data`, :attr:`test_data`, obtained by calling :class:`torch.utils.data.DataLoader` on the corresponding split with key-word parameters *ka* (with prefix `_` removed from keys);
* a config attribute :attr:`data` holding a description of the datasource;
* all the attributes (config or not) in *ka*, so they appear as attributes of the run.

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
      w_main = HBox((Label(str(self)),w_sel,Label(f'+[0-{K-1}]/ {N}',layout=dict(align_self='center')),w_closeb))
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
def periodic(p:Union[int,float],counter:str=None)->Callable[[Run],bool]:
  r"""
Returns a run selector, i.e. a callable which takes a run as input and returns a boolean. The selection is based on the value of a counter held by attribute *counter* of the run.

:param p: periodicity (in counter value)
:param counter: run attribute to use as counter

* If *p* is of type :class:`int`, a run is selected if the counter value is a multiple of *p*. Default counter: :attr:`Run.step`.
* If *p* is of type :class:`float`, a run is selected if the increase in the counter value since the last successful selection is greater than *p*. Default counter: :attr:`Run.walltime`.
  """
#--------------------------------------------------------------------------------------------------
  class Periodic:
    def __init__(self,p,counter):
      if isinstance(p,int):
        if counter is None: counter = 'step'
        call = lambda run,a=counter: getattr(run,a)%p == 0
        R = f'{counter}≡0[{p}]'
      elif isinstance(p,float):
        if counter is None: counter = 'walltime'
        def call(run,a=counter,s=self):
          marks = run.stepper.marks
          last = marks.get(s,None)
          if last is None: last = marks[s] = 0.
          current = getattr(run,a); r = current-last>p
          if r: marks[s] = current
          return r
        R = f'Δ{counter}>{p}'
      else: raise TypeError(f'p[expected: int|float|NoneType; found: {type(p)}]')
      self.call,self.repr = call,f'periodic({R})'
    def __call__(self,run): return self.call(run)
    def __repr__(self): return self.repr
  return None if p is None else Periodic(p,counter)

#--------------------------------------------------------------------------------------------------
class Stepper:
#--------------------------------------------------------------------------------------------------
  def __init__(self,run):
    from socket import getfqdn
    from os import getpid
    from datetime import datetime
    from time import time, process_time
    started = None
    self.marks = marks = {}
    run.process_info = ProcessInfo(host=getfqdn(),pid=getpid(),started=datetime.now())
    def reset():
      nonlocal started
      marks.clear()
      started = time(),process_time()
      run.walltime = run.proctime = 0.
      run.step = 0
    def tick():
      run.walltime,run.proctime = time()-started[0],process_time()-started[1]
      run.step += 1
    self.reset,self.tick = reset,tick

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