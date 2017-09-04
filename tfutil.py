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
from collections import OrderedDict
from numpy import zeros, sqrt, square
import tensorflow
from . import basic_stats,html_stack,html_table,HtmlPlugin
from .monitor import Monitor

#==================================================================================================
class TFTrace (HtmlPlugin):
  r"""
Instances of this class represent Tensorfow traces. A TF trace is composed of runs, each run generates a set of event files. A TF trace is attached to a single directory. There is one sub-directory per run, with a name composed by concatenating ``run-`` with the timestamp of the start of the run (secs, formatted in hexa).
  """
#==================================================================================================
#--------------------------------------------------------------------------------------------------
  def __new__(cls,spec,cache={}):
    r"""
:param spec: the path of the target directory
:type spec: :class:`Union[str,pathlib.Path,TFTrace]`
    """
#--------------------------------------------------------------------------------------------------
    if isinstance(spec,cls): return spec
    elif isinstance(spec,str): root = Path(spec)
    else: root = spec; assert isinstance(root,Path)
    root = root.resolve()
    self = cache.get(root)
    if self is None:
      self = super().__new__(cls)
      root.mkdir(exist_ok=True)
      self.root = root; self.runs = []; self.runs_ = {}; self.loaded = False
      cache[root] = self
    return self

#--------------------------------------------------------------------------------------------------
  @property
  def most_recent(self):
    r"""
Returns the most recent run of this TF trace.
    """
#--------------------------------------------------------------------------------------------------
    self.load()
    return self.runs[0]

#--------------------------------------------------------------------------------------------------
  def refresh(self):
    r"""
Refreshes this TF trace.
    """
#--------------------------------------------------------------------------------------------------
    self.loaded = False
    for run in self.runs: run.refresh()

#--------------------------------------------------------------------------------------------------
  def load(self):
    r"""
Loads this TF trace from its directory.
    """
#--------------------------------------------------------------------------------------------------
    if self.loaded: return
    self.runs.clear(); runs_ = {}
    for d in sorted(self.root.iterdir(),key=(lambda d: d.stat().st_ctime)):
      if not d.name.startswith('run-'): continue
      run = self.runs_.get(d)
      if run is None: run = Run(d)
      self.runs.insert(0,run); runs_[d] = run
    self.runs_ = runs_
    self.loaded = True

#--------------------------------------------------------------------------------------------------
  def create(self):
    r"""
Creates a new (empty) run and adds it to this TF trace.
    """
#--------------------------------------------------------------------------------------------------
    d = self.root/'run-{:x}'.format(int(datetime.now().timestamp())); d.mkdir()
    run = Run(d); self.runs.insert(0,run); self.runs_[d] = run
    return run

#--------------------------------------------------------------------------------------------------
# Methods defining the Mapping behaviour
#--------------------------------------------------------------------------------------------------

  def __getitem__(self,k): return self.runs[k]
  def __delitem__(self,k):
    if isinstance(k,int): self.runs.pop(k).destroy()
    elif isinstance(k,slice):
      for run in self.runs[k]: run.destroy()
      del self.runs[k]
    else: raise TypeError('List indices must be integers or slices, not {}'.format(type(k)))
  def __setitem__(self,k,v): raise Exception('Direct create/update not permitted on TFTrace')
  def __iter__(self): return iter(self.runs)

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  def as_html(self,incontext): self.load(); return html_table(((i,run.as_html(incontext)) for i,run in enumerate(self.runs)),((lambda x:x),))

#==================================================================================================
class Run (HtmlPlugin):
  r"""
Instances of this class represent tensorflow runs.
  """
#==================================================================================================

#--------------------------------------------------------------------------------------------------
  def __init__(self,path,title=None):
#--------------------------------------------------------------------------------------------------
    self.path = path
    self.title = str(datetime.fromtimestamp(path.stat().st_ctime)) if title is None else title
    d_mod = str(path/'mod'); d_log = str(path/'log'); d_ckp = str(path/'ckp')
    self.model_builder = partial(tensorflow.saved_model.builder.SavedModelBuilder,export_dir=d_mod)
    self.model_load = partial(tensorflow.saved_model.loader.load,export_dir=d_mod,tags=['model'])
    self.summary_writer = partial(tensorflow.summary.FileWriter,logdir=d_log)
    def checkpoint_saver(*a,**ka):
      x = tensorflow.train.Saver(*a,**ka)
      x.save = partial(x.save,save_path=d_ckp); x.restore = partial(x.restore,save_path=d_ckp)
      return x
    self.checkpoint_saver = checkpoint_saver
    self.destroy = partial(rmtree,str(path))
    self.evfs = {}
    self.loaded = False

  def __hash__(self): return hash(self.path)
  def __eq__(self,other): return isinstance(other,Run) and self.path == other.path

#--------------------------------------------------------------------------------------------------
  def monitor(self,g,period=100,ckperiod=2000,summary_fd=None):
    r"""
Returns a loop monitor for this run. The monitor has a method :meth:`run`. When that method is invoked with an iterable object of type :class:`Iterable[Dict[tensorflow.Variable,object]]` (e.g. returned by function :func:`tf_main`), it iterates over that object, and performs various operations on the items. The monitor is of class :class:`..Monitor` and can thus be combined with other monitors.

:param g: the tensorflow graph for this run
:type g: :class:`tensorflow.Graph`
:param period: a tensorflow summary is dumped every *period* iterations
:type period: :class:`int`
:param ckperiod: a tensorflow checkpoint is created every *period* iterations
:type ckperiod: :class:`int`
:param summary_fd: the feed dictionary for summaries
:type summary_fd: :class:`Dict[tensorflow.Variable,object]`
:rtype: :class:`..Monitor`
    """
#--------------------------------------------------------------------------------------------------
    from itertools import cycle, count
    summary = g.get_tensor_by_name('Merge/MergeSummary:0')
    with g.as_default():
      summary_writer = self.summary_writer()
      checkpoint_saver = self.checkpoint_saver()
    model_builder = self.model_builder()
    def coroutine(env):
      s = tensorflow.get_default_session()
      for step,n_su,n_ck in zip(count(1),cycle(range(period-1,-1,-1)),cycle(range(ckperiod-1,-1,-1))):
        if n_su==0:
          v = s.run(summary,feed_dict=summary_fd)
          summary_writer.add_summary(v,step)
          summary_writer.flush()
        if n_ck==0: env.lastcheckpoint = checkpoint_saver.save(s)
        if env.stop:
          model_builder.add_meta_graph_and_variables(s,['model'])
          model_builder.save()
        yield
    return Monitor(('tfrun',),(coroutine,))

#--------------------------------------------------------------------------------------------------
  def clip(self,n):
    r"""
Truncates all the event files of thus run to *n* entries.
    """
#--------------------------------------------------------------------------------------------------
    for evf in self.evfs.values(): evf.clip = n

#--------------------------------------------------------------------------------------------------
  def refresh(self):
#--------------------------------------------------------------------------------------------------
    self.loaded = False
    for evf in self.evfs.values(): evf.refresh()

#--------------------------------------------------------------------------------------------------
  def load(self):
#--------------------------------------------------------------------------------------------------
    if self.loaded: return
    d_log = self.path/'log'
    if d_log.is_dir():
      evfs = OrderedDict()
      for n,f in enumerate(f for f in sorted(d_log.iterdir(),key=(lambda f: f.stat().st_mtime))  if f.name.startswith('events.out.tfevents')):
        evf = self.evfs.get(f)
        if evf is None: evf = EVFile(f)
        evf.title = '{}{{{}}}'.format(self.title,n)
        evfs[f] = evf
    else: evfs = {}
    self.evfs = evfs
    self.loaded = True

#--------------------------------------------------------------------------------------------------
  def tensorboard(self,hostname=None):
    r"""
Returns a tensorboard url.
    """
#--------------------------------------------------------------------------------------------------
    import ipywidgets
    from socket import getfqdn
    from threading import Thread
    from random import randint
    from IPython.display import clear_output, display
    def dump():
      nonlocal sub
      with wout:
        print('Server launched')
        try:
          for x in sub.stdout:
            if x: sys.stdout.write(x)
            else: break
        except Exception as e: status = e
        else: status = 'OK'
        print('Server terminated, status:',status)
      sub = None
      wtoggle.description = 'start'
      wlink.value = ''
      wout.layout.border = 'thin solid black'
    def toggle(server_launchcmd=str(Path(sys.executable).parent/'tensorboard')):
      nonlocal sub
      if sub is None:
        port = str(randint(10000,20000))
        wout.clear_output()
        sub = subprocess.Popen((server_launchcmd,'--host',hostname,'--port',port,'--logdir',str(self.path)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
        wtoggle.description = 'stop'
        wlink.value = '<a href="http://{}:{}" target="_blank">view</a>'.format(hostname,port)
        wout.layout.border = 'thin solid blue'
        Thread(target=dump,daemon=True).start()
      else: sub.terminate()
    if hostname is None: hostname = getfqdn()
    sub = None
    wtoggle = ipywidgets.Button(description='start')
    wlink = ipywidgets.HTML('')
    wout = ipywidgets.Output(layout=dict(overflow_y='scroll',height='5cm',border='thin solid black'))
    wtoggle.on_click(lambda b: toggle())
    return ipywidgets.VBox(children=(ipywidgets.HBox(children=(wtoggle,wlink)),wout))

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  def as_html(self,incontext): self.load(); return html_stack(*(evf.as_html(incontext) for evf in self.evfs.values()))

#==================================================================================================
class EVFile (HtmlPlugin):
#==================================================================================================
  TYPE = dict((v,i) for i,v in enumerate('simple_value image histo audio tensor ?'.split()))
  style = 'table th,td {text-align:left; background-color: white; border: thin solid black}'
#--------------------------------------------------------------------------------------------------
  def __init__(self,path,clip=1000,title=None):
#--------------------------------------------------------------------------------------------------
    self.path = path; self.e_count = 0; self.clip_ = clip; self.timestamp = 0
    self.title = str(self.path) if title is None else title

  def __hash__(self): return hash(self.path)
  def __eq__(self,other): return isinstance(other,EVFile) and self.path == other.path

#--------------------------------------------------------------------------------------------------
  @property
  def clip(self):
#--------------------------------------------------------------------------------------------------
    return self.clip_
#--------------------------------------------------------------------------------------------------
  @clip.setter
  def clip(self,n):
#--------------------------------------------------------------------------------------------------
    self.clip_ = n
    if n>self.e_count: self.timestamp = 0

#--------------------------------------------------------------------------------------------------
  def refresh(self):
#--------------------------------------------------------------------------------------------------
    self.timestamp = 0

#--------------------------------------------------------------------------------------------------
  def load(self):
#--------------------------------------------------------------------------------------------------
    t = self.path.stat().st_mtime
    if t>self.timestamp:
      self.timestamp = t
      clipped = False
      e_count = g_count = s_count = 0; tags = {}; last = None
      s_bnd = [0,0]; sd_stats = basic_stats(); sdt_stats = basic_stats()
      for e in tensorflow.train.summary_iterator(str(self.path)):
        # e is an Event protocol buffer object
        what = e.WhichOneof('what')
        if what == 'summary':
          # e.summary is a Summary protocol buffer object
          if last is None: s_bnd[0] = e.step
          else: s_bnd[1] = e.step; d = e.step-last.step; sd_stats += d; sdt_stats += (e.wall_time-last.wall_time)/d
          last = e; s_count += 1
          for v in e.summary.value:
            D = tags.get(v.tag)
            if D is None: tags[v.tag] = D = zeros(len(self.TYPE))
            D[self.TYPE.get(v.WhichOneof('value'),-1)] += 1
        elif what == 'graph_def': g_count += 1
        e_count += 1
        if e_count>=self.clip_: clipped = True; break
      self.s_bnd,self.s_count,self.sd_stats,self.sdt_stats,self.g_count,self.tags,self.e_count,self.clipped = s_bnd,s_count,sd_stats,sdt_stats,g_count,tags,e_count,clipped

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  def as_html(self,_):
    from lxml.html.builder import E
    self.load()
    thead = E.tr(E.td(E.b(self.title),E.span(' [{} events{}]'.format(self.e_count,'!' if self.clipped else '')),colspan='3',style='background-color:gray; color:white; text-align:center'))
    def tbody():
      if self.g_count: yield E.th('graph'),E.td(E.b('count'),': {}'.format(self.g_count),colspan='2')
      if self.s_count:
        yield E.th('steps'),E.td(E.b('count'),': {} '.format(self.s_count),E.b('first'),': {} '.format(self.s_bnd[0]),E.b('last'),': {} '.format(self.s_bnd[1]),colspan='2')
        yield E.th('inter-step'),E.td(E.b('avg'),': {:.1f} '.format(self.sd_stats.avg),E.b('std'),': {:.1f} '.format(sqrt(self.sd_stats.var)),colspan='2')
        yield E.th('step-ms'),E.td(E.b('avg'),': {:.3g} '.format(1000*self.sdt_stats.avg),E.b('std'),': {:.3g} '.format(1000*sqrt(self.sdt_stats.var)),colspan='2')
        firstrow = True
        for tag,v in sorted(self.tags.items()):
          c = [E.td(tag),E.td(*(e for typ,x in zip(self.TYPE,v/self.s_count) if x for e in (E.b(typ),(' ({:.1%}) '.format(x) if 1.-x>1e-10 else ''))))]
          if firstrow: firstrow = False; c.insert(0,E.th('tags',rowspan=str(len(self.tags))))
          yield c
    return E.table(E.style(self.style,scoped="scoped"),E.thead(thead),E.tbody(*(E.tr(*cells) for cells in tbody())))

#==================================================================================================
def tf_accuracy(y,y_,name='accuracy',sname='train-accuracy'):
  r"""
Returns the accuracy op of a batch prediction score tensor *y* to a reference tensor *y_* in onehot representation. Both tensors *y,y_* are of shape *(n,d)* where *n* is the size of the current batch and *d* the dimension of the score vector and onehot representation. The accuracy tensor is also added to the summary.

:param y,y_: 2d tensorflow node
:type y,y_: :class:`tensorflow.Tensor`
:param name: name of the accuracy op
:param sname: name of the summary item for the accuracy op
  """
#==================================================================================================
  x = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(y,1),tensorflow.argmax(y_,1)),tensorflow.float32),name=name)
  tensorflow.summary.scalar(sname,x)
  return x

#==================================================================================================
def tf_loss(y,y_,name='loss',sname='loss'):
  r"""
Returns the cross-entropy loss op of a batch prediction score tensor *y* to a reference tensor *y_* in onehot representation. Both tensors *y,y_* are of shape *(n,d)* where *n* is the size of the current batch and *d* the dimension of the score vector and onehot representation. The loss tensor is also added to the summary.

:param y,y_: 2d tensorflow node
:type y,y_: :class:`tensorflow.Tensor`
:param name: name of the loss op
:param sname: name of the summary item for the loss op
  """
#==================================================================================================
  x = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y),name=name)
  tensorflow.summary.scalar(sname,x)
  return x

#==================================================================================================
def tf_run(data={},target='accuracy',s=None):
  r"""
Computes the result of a tensorflow tensor on some data.

:param data: mapping from variables (or their names) to values
:type data: :class:`Dict[tensorflow.Variable,object]`
:param target: the tensor to compute or its name
:type target: :class:`Union[str,tensorflow.Tensor]`
:param s: the session to use for the computation (defaults to the tensorflow default session)
:type s: :class:`tensorflow.Session`
:rtype: :class:`object`
  """
#==================================================================================================
  if s is None: s = tensorflow.get_default_session()
  if isinstance(target,str): target = s.graph.get_tensor_by_name(target+':0')
  return s.run(target,feed_dict=data)

#==================================================================================================
def tf_iter(batches,init_step='init',iter_step='train',s=None):
  r"""
Iterates over *batches* and yields the result of a tensorflow operation, paired with the current session.

:param batches: an iterable of mappings from variables (or their names) to values
:type batches: :class:`Iterable[Dict[Union[str,tensorflow.Variable],object]]`
:param init_step: the operation to compute at initialisation or its name
:type init_step: :class:`Union[str,tensorflow.Operation]`
:param iter_step: the operation to compute at each iteration or its name
:type iter_step: :class:`Union[str,tensorflow.Operation]`
:param s: the session to use for the computation (defaults to the tensorflow default session)
:type s: :class:`tensorflow.Session`
:rtype: :class:`Iterable[Tuple[tensorflow.Session,Dict[tensorflow.Variable,object]]]`
  """
#==================================================================================================
  if s is None: s = tensorflow.get_default_session()
  init_step,iter_step = ((s.graph.get_operation_by_name(op) if isinstance(op,str) else op) for op in (init_step,iter_step))
  s.run(init_step)
  for fd in batches:
    s.run(iter_step,feed_dict=fd)
    yield fd

#==================================================================================================
def tf_config(**ka):
#==================================================================================================
  ka.setdefault('log_device_placement',True)
  d = ka.setdefault('device_count',{})
  n = d.setdefault('GPU',0)
  d = ka.setdefault('gpu_options',{})
  d.setdefault('visible_device_list',','.join(str(x) for x in gpus(-n)) if n else '')
  ka['gpu_options'] = tensorflow.GPUOptions(**d)
  return tensorflow.ConfigProto(**ka)

#==================================================================================================
def manage(*paths,ivname='tr'):
#==================================================================================================
  import ipywidgets
  from IPython.display import clear_output, display
  from IPython.core.getipython import get_ipython
  def showtr():
    if tr is not None: tr.refresh()
    with wout: clear_output(); display(tr)
  def settr(c): nonlocal tr; tr = c.new; interpreter.push({ivname:tr}); showtr()
  interpreter = get_ipython()
  tr = ()
  interpreter.push({ivname:tr})
  wtr = ipywidgets.Dropdown(description=ivname,options=OrderedDict(chain((('...',()),),((p,TFTrace(p)) for p in paths))),style={'description_width':'initial'})
  wshow = ipywidgets.Button(icon='fa-refresh',tooltip='refresh',layout=ipywidgets.Layout(width='.4cm',padding='0cm'))
  wout = ipywidgets.Output()
  wtr.observe(settr,'value')
  wshow.on_click(lambda b: showtr())
  return ipywidgets.VBox(children=(ipywidgets.HBox(children=(wtr,wshow),layout={'border-bottom':'thin solid black'}),wout))

#==================================================================================================
def gpus(n=0):
#==================================================================================================
  L = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],stdout=subprocess.PIPE,check=True,universal_newlines=True)
  L = [x.strip().split(', ') for x in L.stdout.strip().split('\n')]
  D = dict((gpu_uuid,int(pid)) for gpu_uuid,pid in L)
  L = subprocess.run(['nvidia-smi','--query-gpu=index,memory.total,memory.free,gpu_uuid','--format=csv,noheader,nounits'],stdout=subprocess.PIPE,check=True,universal_newlines=True)
  L = [x.strip().split(', ') for x in L.stdout.strip().split('\n')]
  if n:
    L = [int(index) for index,memory_total,memory_free,gpu_uuid in L if D.get(gpu_uuid) is None]
    if n<0: n = -n; assert len(L)>=n,'Insufficient GPU resources'
    return L[:n]
  else:
    return [(int(index),int(memory_free)/int(memory_total),D.get(gpu_uuid)) for index,memory_total,memory_free,gpu_uuid in L]
