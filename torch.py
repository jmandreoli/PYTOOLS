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
from time import time, ctime, process_time
from pydispatch import Dispatcher, Property
import torch

#==================================================================================================
class Run (Dispatcher):
  r"""
Instances of this class are callables, taking no argument and returning no value. The execution of a run only changes its attribute values. At the creation of the run, all the key-value pairs in *ka* are turned into attributes of the run. If the key starts with `_`, that character is dropped from the added attribute name, which is then marked as "non visible". The remaining attributes are gathered in a dictionary :attr:`visible`, meant to provide a human readable description of the run.

Runs may emit messages during their execution. Each message is passed a single argument, which is the run itself, so runs can be controlled by message listeners. Methods :meth:`Dispatcher.bind` (inherited) and :meth:`bind_listeners` can be used to bind callbacks to a run. Some methods, defined in subclasses, of the form :meth:`bind<name>Listener` are also available to bind specific callbacks named by `<name>`.

Attributes (\*) must be instantiated at creation time.
  """
#==================================================================================================

  measures: 'Sequence[Callable[[torch.tensor,...],torch.tensor]]'
  r"""(\*)List of measures, each returning a scalar (0-D) tensor"""
  device: 'torch.device'
  r"""(\*)The device on which to execute the run"""
  net: 'torch.nn.Module'
  r"""(\*)The model"""
  tnet: 'torch.nn.Module'
  r"""The same model, located at :attr:`device`"""
  clock: 'Clock'
  r"""A clock to record walltime and process time"""

#--------------------------------------------------------------------------------------------------
  def __init__(self,**ka):
#--------------------------------------------------------------------------------------------------
    self.listeners = []
    self.visible = {}
    for k,v in ka.items():
      if k.startswith('_'): k = k[1:]
      else: r = ' '.join(repr(v).split()); self.visible[k] = r if len(r)<80 else r[:77]+'...'
      setattr(self,k,v)
    self.tnet = self.net.to(self.device)
    self.clock = Clock()

#--------------------------------------------------------------------------------------------------
  def __call__(self):
    r"""
Executes the run. This implementation raises a :class:`NotImplementedError`.
    """
#--------------------------------------------------------------------------------------------------
    raise NotImplementedError()

#--------------------------------------------------------------------------------------------------
  def eval(self,data):
    r"""
Evaluates model :attr:`tnet` on *data* according to :attr:`measures`. Each input item in *data* consists of a tuple of an input tensor on which the model is executed, followed by any number of extra parameters for the measures. For each input item, an evaluation consists in applying the model to the first tensor, then estimating each measure on the output tensor with the extra parameters. The returned value is a tuple of length equal to the number of measures, holding averages (as floats) over the whole *data* sequence.

:param data: a list of input items
:type data: :class:`Iterable[Tuple[torch.tensor,Any,...]]`
:rtype: :class:`Tuple[float,...]`
    """
#--------------------------------------------------------------------------------------------------
    avg = 0.
    with training(self.tnet,False),torch.no_grad():
      for n,(inputs,*args) in enumerate(data,1):
        outputs = self.tnet(inputs)
        avg += (torch.stack([m(outputs,*args) for m in self.measures])-avg)/n
    return tuple(avg.to('cpu').numpy())

#--------------------------------------------------------------------------------------------------
  def bind_listeners(self,*listeners):
    r"""
Binds the set of *listeners* (any objects) to the events of this run. For each listener, the methods of that listener whose name match ``on_`` followed by the name of a supported event are bound as callback to this run. These methods are usually specified as attributes of type :class:`PROC`, so as to easily be configurable.

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
    r"""
Used as decorator to attach a listener factory to a subclass *cls* of :class:`Run`, available as method :meth:`bind<name>Listener` using the provided *name*.

:param name: short name for the factory
:type name: :class:`str`
    """
#--------------------------------------------------------------------------------------------------
    def app(f):
      setattr(cls,'{}Listener'.format(name),f)
      F = lambda self,*a,**ka: self.bind_listeners(f(*a,**ka))
      F.__doc__ = r"""Binds an event listener obtained by invoking factory :func:`{}Listener`""".format(name)
      setattr(cls,'bind{}Listener'.format(name),F)
      return f
    return app

  def __repr__(self): return f'{self.__class__.__name__}<{self.clock.created}>'

#==================================================================================================
class SupervisedRun (Run):
  r"""
Supervised runs operate on data in the form of instance-label pairs. Attributes (\*) must be instantiated at creation time. Attribute :attr:`measures` is also set to a reasonable default for classification (the pair of the loss and accuracy functions).
  """
#==================================================================================================

  lossF: 'Callable[[torch.tensor,torch.tensor],float]'
  r"""(\*)Loss function (defaults to cross-entropy loss for classification)"""

  # Default values for classification runs (override in subclasses)
  lossF = torch.nn.CrossEntropyLoss()
  measures = lossF,(lambda outputs,labels: torch.mean((torch.argmax(outputs,1)==labels).float()))

#==================================================================================================
class SupervisedTrainRun (SupervisedRun):
  r"""
Runs of this type execute a simple supervised training loop. Attributes (\*) must be instantiated at creation time. All the other attributes are initialised and updated by the run execution, except :attr:`progress` which is initialised (to 0.) by the run execution, but needs to be updated by some listener.
  """
#==================================================================================================

  _events_ = ['open','batch','epoch','close']

  ## set at instantiation
  train_data: 'Iterable[Tuple[Tensor,Tensor]]'
  r"""(\*)Iterable of batches. Each batch is a pair of a tensor of inputs and a tensor of labels (first dim = size of batch)"""
  valid_data: 'Iterable[Tuple[Tensor,Tensor]]'
  r"""(\*)Same format as :attr:`train_data`"""
  test_data: 'Iterable[Tuple[Tensor,Tensor]]'
  r"""(\*)Same format as :attr:`train_data`"""
  optimiser: 'Callable[[List[torch.nn.Parameter]],torch.optim.Optimizer]'
  r"""(\*)The optimiser factory"""
  ## set at execution
  progress: 'float'
  r"""Between 0. and 1., stops the run when 1. is reached (must be updated by a listener)"""
  step: 'int'
  r"""Number of completed batches overall"""
  epoch: 'int'
  r"""Number of completed epochs"""
  batch: 'int'
  r"""Number of completed batches within current epoch"""
  loss: 'float'
  r"""Average loss since beginning of current epoch"""
  eval_valid: 'Callable[[],Tuple[float,float]]'
  r"""Function returning the validation performance at the end of the last completed step (cached)"""
  eval_test: 'Callable[[],Tuple[float,float]]'
  r"""Function returning the test performance at the end of the last completed step (cached)"""

#--------------------------------------------------------------------------------------------------
  def __call__(self):
#--------------------------------------------------------------------------------------------------
    def eval_cache(data): # returns a cached version of self.eval
      current = None,-1
      def cache():
        nonlocal current
        if current[1] != self.step: current = self.eval(to(data,self.device)),self.step
        return current[0]
      return cache
    net = self.tnet
    self.eval_valid,self.eval_test = eval_cache(self.valid_data),eval_cache(self.test_data)
    optimiser = self.optimiser(net.parameters())
    self.progress = 0.
    self.step = self.epoch = 0
    with training(net,True):
      self.clock.set()
      self.emit('open',self)
      try:
        while self.progress<1.:
          self.batch = 0; self.loss = 0.
          for inputs,labels in to(self.train_data,self.device):
            optimiser.zero_grad()
            loss = self.lossF(net(inputs),labels)
            loss.backward()
            optimiser.step()
            self.step += 1; self.batch += 1
            self.loss += (loss.item()-self.loss)/self.batch
            del inputs,labels,loss # free memory before emitting (not sure this works)
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
  nepoch: 'int'
  r"""(\*)Number of epochs to run"""
  labels: 'Sequence[int]'
  r"""(\*)List of labels (pure value, not tensor) to process (or range if integer type)"""
  init: 'torch.tensor'
  r"""(\*)Initial instance"""
  projection: 'Callable[[torch.tensor],NoneType]'
  r"""(\*)Called after each optimiser step to re-project instances within domain range"""
  optimiser: 'Callable[[List[torch.nn.Parameter]],torch.optim.Optimizer]'
  r"""(\*)The optimiser factory"""
  protos: 'Sequence[Tensor]'
  r"""The estimated inverse image of :attr:`labels`"""

#--------------------------------------------------------------------------------------------------
  def __call__(self):
#--------------------------------------------------------------------------------------------------
    projection = (lambda a: None) if self.projection is None else self.projection
    self.protos = torch.repeat_interleave(self.init[None,...],len(self.labels),0)
    net = self.tnet
    with training(net,True):
      self.clock.set()
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

#==================================================================================================
@SupervisedTrainRun.listenerFactory('Base')
class SupervisedTrainRunListener:
  r"""
Instances of this class monitor basic supervised training runs. All the parameters below are initialised with reasonable values, except *config* passed to :meth:`configure`. By default, train, validation and test info are logged, respectively, on batch end, epoch end and run end.

:param max_epoch: maximum number of epochs to run
:type max_epoch: :class:`int`
:param max_time: maximum total wall time to run
:type max_time: :class:`int`
:param logger: logger to use for logging information
:type logger: :class:`logging.Logger`
:param status: pair of a list of headers and a function returning the status of a run as a tuple matching those headers
:type status: :class:`Tuple[Callable[[Run],Tuple],Tuple]`
:param status_fmt: pair of format strings for the components of *status*
:type status_fmt: :class:`Tuple[str,str]`
:param itrain,ivalid,itest: function returning various info on a run
:type itrain,ivalid,itest: :class:`Callable[[Run],Tuple]`
:param itrain_fmt,ivalid_fmt,itest_fmt: format strings for the results of the corresponding functions
:type itrain_fmt,ivalid_fmt,itest_fmt: :class:`str`
:param config: passed as keyword arguments to method :meth:`configure`
  """
#==================================================================================================

#--------------------------------------------------------------------------------------------------
  def __init__(self,max_epoch=int(1e9),max_time=float('inf'),logger=None,
    status=(
      ('TIME','STEP','EPO','BAT'),
      (lambda run: (run.clock.walltime,run.step,run.epoch,run.batch))
    ),
    status_fmt:'Tuple[str,str]'=('%6s %6s %3s/%4s ','%6.1f %6d %3d/%4d '),
    itrain=(lambda run: (run.loss,)),
    ivalid=(lambda run: run.eval_valid()),
    itest =(lambda run: run.eval_test()),
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
      def i_header(run,fmt=status_fmt[0],s=status[0]):
        logger.info('RUN %s',run)
        for k,v in run.visible.items(): logger.info('PARAM %10s= %s',k,v)
        logger.info(fmt,*s)
    def set_progress(run): run.progress = min(max(run.epoch/max_epoch,run.clock.walltime/max_time),1.)
    self.set_progress = set_progress
    self.on_open = i_header
    self.on_close = i_test
    self.on_batch = None
    self.on_epoch = abs(PROC(set_progress)+PROC(self.iprogress))
    self.configure(**config)

#--------------------------------------------------------------------------------------------------
  def configure(self,train_p=False,valid_p=False):
    r"""
This method is meant to initialise attributes :attr:`on_batch` and :attr:`on_epoch` of this listener. This implementation assigns reasonable defaults, using the following run selectors.

:param train_p: run selector for train info logging at each batch
:type train_p: :class:`Union[Callable[[Run],bool],bool]`
:param valid_p: run selector for validation info logging at each epoch
:type valid_p: :class:`Union[Callable[[Run],bool],bool]`
    """
#--------------------------------------------------------------------------------------------------
    self.on_batch = abs(PROC(self.on_batch)+PROC(self.itrain)%train_p)
    self.on_epoch = abs(PROC(self.on_epoch)+PROC(self.ivalid)%valid_p)

#==================================================================================================
@SupervisedTrainRun.listenerFactory('Mlflow')
class SupervisedTrainRunMlflowListener:
  r"""
Instances of this class provide mlflow logging of supervised training runs. All the parameters below are initialised with reasonable values, except *config* passed to :meth:`configure`. By default, train, validation and test info are logged, respectively, on batch end, epoch end and run end.

:param uri: mlflow tracking uri
:type uri: :class:`str`
:param exp: experiment name
:type exp: :class:`str`
:param itrain,ivalid,itest: function returning various info on a run
:type itrain,ivalid,itest: :class:`Callable[[Run],Tuple]`
:param itrain_labels,ivalid_labels,itest_labels: labels for the results of the corresponding functions
:type itrain_labels,ivalid_labels,itest_labels: :class:`Sequence[str]`
:param config: passed as keyword arguments to method :meth:`configure`
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
      mlflow.start_run(); mlflow.log_params(run.visible)
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
This method is meant to initialise attributes :attr:`on_batch` and :attr:`on_epoch` of this listener. This implementation assigns reasonable defaults, using the following run selectors.

:param train_p: run selector for train info logging at each batch
:type train_p: :class:`Union[Callable[[Run],bool],bool]`
:param valid_p: run selector for validation info logging at each epoch
:type valid_p: :class:`Union[Callable[[Run],bool],bool]`
:param checkpoint_p: run selector for checkpointing at each epoch
:type checkpoint_p: :class:`Union[Callable[[Run],bool],bool]`
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
Instances of this class are data sources for classification training. Attributes (\*) must be instantiated at creation time.
  """
#==================================================================================================
  name: 'str'
  r"""(\*)Descriptive name of the datasource"""
  classes: 'Iterable[str]'
  r"""(\*)Descriptive names for the classes"""
  train: 'torch.utils.data.Dataset'
  r"""(\*)The train split of the data source"""
  test: 'torch.utils.data.Dataset'
  r"""(\*)The test split of the data source"""

#--------------------------------------------------------------------------------------------------
  def mpl(self,ax):
    r"""
Returns a callable, which, when passed a data instance, displays it on *ax* (its label is also used as title of *ax*). This implementation raises a :class:`NotImplementedError`.

:param ax: a display area for a data instance
:type ax: :class:`matplotlib.Axes`
:rtype: :class:`Callable[[Tensor],NoneType]`
    """
#--------------------------------------------------------------------------------------------------
    raise NotImplementedError()

#--------------------------------------------------------------------------------------------------
  def loaders(self,pcval,**ka):
    r"""
Returns a dictionary which can be passed as keyword arguments to the :class:`SupervisedTrainRun` constructor to initialise its required attributes:

* the invisible attributes :attr:`train_data`, :attr:`valid_data`, :attr:`test_data`, obtained by calling :class:`torch.utils.data.DataLoader` on the corresponding split and key-word parameters *ka* (with prefix `_` removed from keys);
* visible attributes :attr:`data` holding a description of the datasource and :attr:`pcval` referring to the portion of the train split used for validation;
* all the attributes (visible or invisible) defined in *ka*.

:param pcval: proportion (between 0. and 1.) of instances from the train split to use for validation.
:type pcval: :class:`float`
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
  def display(self,rowspec,colspec,dataset=None,**ka):
    r"""
Displays a set of labelled data instances *dataset* in a grid. The specification of the rows (resp. cols) of the grid consists of three numbers:

* height (resp. width) of the cells; required
* number of rows (resp. cols); if -1 it is adjusted so the the whole dataset fits in the grid; default: 1
* additional space in the row (resp. col) dimension; default: 0.

At most one of the number of rows or columns can be -1. When both are positive and the dataset does not fit in the grid, a slider is created to allow page browsing through the dataset.

:param rowspec,colspec: specification of the rows/cols of the grid
:type rowspec,colspec: :class:`Union[float,Tuple[int,float],Tuple[int,float,float]]`
:param dataset: the data to display
:type dataset: :class:`Iterable[Tuple[Any,int]]`
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
      from ipywidgets import IntSlider,Layout,Label,HBox
      N = len(dataset)
      w = IntSlider(value=1,min=0,step=K,max=(adj(N,K)-1)*K,layout=Layout(width='10cm'))
      w.observe((lambda ev: disp([dataset[k] for k in range(ev.new,min(ev.new+K,N))])),'value')
      w.value = 0
      return HBox((w,Label('+[0-{}]/ {}'.format(K-1,N),layout=Layout(align_self='center'))))
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
    from matplotlib.pyplot import subplots
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
This class is meant to facilitate writing flexible pipelines of callable invocations. An instance of this class encapsulates a callable or :const:`None` (void callable). All callables take a single argument (typically a run). Generic function :func:`abs` (absolute value) returns the encapsulated callable. This class supports two operations.

* Operation `+` (sequential invocation). The two operands must be instances of :class:`PROC` and so is the result. If any of the two operands is void, the result is the other operand, otherwise it is defined by::

   abs(x+y) == lambda u,X=abs(x),Y=abs(y): X(u) is not None or Y(u) is not None or None

* Operation `%` (conditioning). The first operand must be an instance of :class:`PROC` and so is the result. The second operand must be a selector (a function with one input and a boolean output), or a boolean value. If the first operand is void, so is the result; if the second operand is :const:`False`, the result is also void; if the second operand is :const:`True`, the result is the first operand; otherwise it is defined by::

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
def periodic(p,counter=None):
  r"""
Returns a run selector, i.e. a callable which takes a run as input and returns a boolean. The selection is based on the value of a counter specified by *counter*, which can be a string (name of attribute of the run used as counter) or an explicit callable which takes as input the run and returns the value of the counter.

* If *p* is of type :class:`int`, a run is selected if the counter value is a multiple of *p*. Default counter: batch number in current epoch
* If *p* is of type :class:`float`, a run is selected unless the increase in the counter value since the last successful selection is lower than *p*. Default counter: walltime since run start (in secs)

:param p: periodicity (in counter value)
:type p: :class:`Union[NoneType,int,float]`
:param counter: run feature to use as counter
:type counter: :class:`Union[Callable[[Run],int],Callable[[Run],float],str,NoneType]`
:rtype: :class:`Union[Callable[[Run],bool],NoneType]`
  """
#--------------------------------------------------------------------------------------------------
  if p is None: return None
  if isinstance(p,int):
    if counter is None: return lambda run: run.batch%p == 0
    if isinstance(counter,str): return lambda run,a=counter: getattr(run,a)%p == 0
    if not callable(counter): raise TypeError('counter[expected: callable|str|NoneType; found: {}]'.format(type(counter)))
    return lambda run: counter(run)%p == 0
  if not isinstance(p,float): raise TypeError('p[expected: int|float|NoneType; found: {}]'.format(type(p)))
  if counter is None: counter = lambda run: run.clock.walltime
  elif isinstance(counter,str): counter = lambda run,a=counter: getattr(run,a)
  elif not callable(counter): raise TypeError('counter[expected: callable|str|NoneType; found: {}]'.format(type(counter)))
  current = 0.
  def F(run):
    nonlocal current
    c = counter(run); r = c-current>p
    if r: current = c
    return r
  return F

#--------------------------------------------------------------------------------------------------
class Clock:
#--------------------------------------------------------------------------------------------------
  def __init__(self): self.created = ctime(); self.start = None
  def set(self): self.start = time(),process_time()
  @property
  def walltime(self): return time()-self.start[0]
  @property
  def proctime(self): return process_time()-self.start[1]

#--------------------------------------------------------------------------------------------------
def to(data,device):
#--------------------------------------------------------------------------------------------------
  for input,label in data: yield input.to(device),label.to(device)
