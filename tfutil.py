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
from collections import OrderedDict, defaultdict
from numpy import zeros, sqrt, square
import tensorflow
from . import basic_stats,html_stack,html_table,HtmlPlugin,unid
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
    def ckptsave(ckpt,*a,**ka): ckpt.save(*a,save_path=d_ckp,**ka)
    self.ckptsave = ckptsave
    def ckptrestore(ckpt,*a,**ka): ckpt.restore(*a,save_path=d_ckp,**ka)
    self.ckptrestore = ckptrestore
    self.destroy = partial(rmtree,str(path))
    self.evfs = {}
    self.loaded = False

  def __hash__(self): return hash(self.path)
  def __eq__(self,other): return isinstance(other,Run) and self.path == other.path

#--------------------------------------------------------------------------------------------------
  def monitor(self,period=100,ckperiod=2000,summary_fd=None,ckpt=None):
    r"""
Returns a loop monitor for this run. The monitor has a method :meth:`run`. When that method is invoked with an iterable of feed dictionaries (type: :class:`Iterable[Dict[tensorflow.Variable,object]]`), it performs various operations on its items. The operations (executed in the current default session) are:

* at specified intervals, dump a summary based on the current state of the graph
* at specified intervals, save a checkpoint of the current state of the graph
* at the end, save the graph and its current state

The monitor is of class :class:`..Monitor` and can thus be combined with other monitors.

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
    def coroutine(env):
      model_builder = self.model_builder()
      s = tensorflow.get_default_session()
      g = s.graph
      summary = g.get_tensor_by_name('Merge/MergeSummary:0')
      summary_writer = self.summary_writer(graph=g)
      for step,n_su,n_ck in zip(count(1),cycle(range(period-1,-1,-1)),cycle(range(ckperiod-1,-1,-1))):
        if n_su==0:
          v = s.run(summary,feed_dict=summary_fd)
          summary_writer.add_summary(v,global_step=step)
          summary_writer.flush()
        if n_ck==0 and ckpt is not None: self.ckptsave(ckpt,s,global_step=step)
        if env.stop:
          summary_writer.close()
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
  style = '''
#toplevel > tbody th, #toplevel > tbody td {text-align:left; background-color: white; border: thin solid black}
#toplevel > thead > tr > td { background-color:gray; color:white; text-align:center }
  '''
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
      clipped = False; e_count = 0; info = {}
      cat = dict(summary=EVF_summary,log_message=EVF_log_message)
      for n,evt in enumerate(tensorflow.train.summary_iterator(str(self.path))):
        if n == self.clip_: clipped = True; break
        # evt is an Event protobuf object
        e_count += 1; what = evt.WhichOneof('what')
        p = info.get(what)
        if p is None: info[what] = p = cat.get(what,EVF_base)(what)
        p.add(evt,getattr(evt,what))
      info = sorted(info.values(),key=(lambda p: (p.weight,p.label)))
      self.clipped,self.e_count,self.info = clipped,e_count,info

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  def as_html(self,_):
    from lxml.html.builder import E
    self.load()
    thead = E.tr(E.td(E.b(self.title),E.span(' [{}{} events]'.format(self.e_count,('+' if self.clipped else ''))),colspan='4'))
    tbody = (e for p in self.info for e in html_table_prefix_rows(list(p.html()),E.th(p.label)))
    idn = unid()
    return E.div(E.style(self.style.replace('#toplevel','#'+idn),scoped="scoped"),E.table(E.thead(thead),E.tbody(*tbody),id=idn))

#==================================================================================================
# Parsing/representation classes for EVFile
#==================================================================================================

#--------------------------------------------------------------------------------------------------
class EVF_base:
#--------------------------------------------------------------------------------------------------
  weight = 0
  def __init__(self,label):
    self.count,self.label = 0,label
  def add(self,evt,x): self.count += 1
  def html(self):
    from lxml.html.builder import E
    yield E.tr(E.td(E.b('count'),': {}'.format(self.count),colspan='3'))
#--------------------------------------------------------------------------------------------------
class EVF_summary (EVF_base):
#--------------------------------------------------------------------------------------------------
  weight = -1
  TYPE = 'simple_value image histo audio tensor ?'.split()
  def __init__(self,label):
    super().__init__(label)
    # initialise step statistics (only for summary steps)
    self.last,self.bnd = None,[0,0]
    self.d_stats,self.t_stats = basic_stats(),basic_stats()
    # initialise tag type counts, one per tag
    self.tagcount = defaultdict(lambda n=len(self.TYPE): zeros(n))
  def add(self,evt,x):
    super().add(evt,x)
    # compute step statistics
    if self.last is None: self.bnd[0] = evt.step
    else: self.bnd[1] = evt.step; d = evt.step-self.last.step; self.d_stats += d; self.t_stats += (evt.wall_time-self.last.wall_time)/d
    self.last = evt
    # x is a Summary protobuf object
    for v in x.value:
      try: i = self.TYPE.index(v.WhichOneof('value'))
      except ValueError: i = -1
      self.tagcount[v.tag][i] += 1.
  def html(self):
    from lxml.html.builder import E
    # display step statistics
    yield E.tr(E.th('steps'),E.td(E.b('count'),': {} '.format(self.count),E.b('first'),': {} '.format(self.bnd[0]),E.b('last'),': {} '.format(self.bnd[1]),colspan='2'))
    yield E.tr(E.th('inter-step'),E.td(E.b('avg'),': {:.1f} '.format(self.d_stats.avg),E.b('std'),': {:.1f} '.format(sqrt(self.d_stats.var)),colspan='2'))
    yield E.tr(E.th('step-ms'),E.td(E.b('avg'),': {:.3g} '.format(1000*self.t_stats.avg),E.b('std'),': {:.3g} '.format(1000*sqrt(self.t_stats.var)),colspan='2'))
    # display Summary details
    yield from html_table_prefix_rows([E.tr(E.td(tag),E.td(*(e for typ,x in zip(self.TYPE,v/self.count) if x for e in (E.b(typ),(' ({:.1%}) '.format(x) if 1.-x>1e-10 else ''))))) for tag,v in sorted(self.tagcount.items())],E.th('tags'))

#--------------------------------------------------------------------------------------------------
class EVF_log_message (EVF_base):
#--------------------------------------------------------------------------------------------------
  weight = -2
  TYPE = [10,20,30,40,50,0]
  NAME = 'DEBUGGING INFO WARN ERROR FATAL UNKNOWN'.split()
  def __init__(self,label):
    super().__init__(label); self.perlevel = zeros(len(self.TYPE),dtype=int)
  def add(self,evt,x):
    super().add(evt,x)
    # x in a LogMessage protobuf object
    try: i = self.TYPE.index(x.log_message.value.level)
    except ValueError: i = -1
    self.perlevel[i] += 1
  def html(self):
    from lxml.html.builder import E
    # display LogMessage details
    yield from (E.tr(E.td(lvl),E.td(E.b('count'),': {} '.format(n),colspan='2')) for lvl,n in zip(self.NAME,self.perlevel) if n != 0)

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
def tf_run(op,data={},target=None,s=None):
  r"""
Computes the result of a tensorflow tensor on some data.

:param data: a feed dictionary
:type data: :class:`Dict[tensorflow.Variable,object]`
:param op: the operation to compute or its name
:type op: :class:`Union[str,tensorflow.Operation]`
:param target: the index of the output of *op* to return (if :const:`None` *op* itself is run and :const:`None` is returned)
:type target: :class:`Union[int,slice]`
:param s: the session to use for the computation (defaults to the tensorflow default session)
:type s: :class:`tensorflow.Session`
:rtype: :class:`object`
  """
#==================================================================================================
  if s is None: s = tensorflow.get_default_session()
  if isinstance(op,str): op = s.graph.get_operation_by_name(op)
  target = op if target is None else op.outputs[target]
  return s.run(target,feed_dict=data)

#==================================================================================================
def tf_iter(op,batches,target=None,s=None):
  r"""
Iterates over *batches*, performs a tensorflow op on each item and yields the items.

:param batches: an iterable of feed dictionaries
:type batches: :class:`Iterable[Dict[Union[str,tensorflow.Variable],object]]`
:param op: the operation to compute at each iteration or its name
:type op: :class:`Union[str,tensorflow.Operation]`
:param target: the index of the output of *op* to yield (if :const:`None` *op* itself is run and :const:`None` is returned)
:type target: :class:`Union[int,slice]`
:param s: the session to use for the computation (defaults to the tensorflow default session)
:type s: :class:`tensorflow.Session`
:rtype: :class:`Iterable[Tuple[tensorflow.Session,Dict[tensorflow.Variable,object]]]`
  """
#==================================================================================================
  if s is None: s = tensorflow.get_default_session()
  if isinstance(op,str): op = s.graph.get_operation_by_name(op)
  target = op if target is None else op.outputs[target]
  for fd in batches: yield s.run(target,feed_dict=fd)

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

#==================================================================================================
def html_table_prefix_rows(L,e):
#==================================================================================================
  e.set('rowspan',str(len(L)))
  L[0].insert(0,e)
  return L
