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
from collections import OrderedDict, defaultdict, namedtuple
from numpy import zeros, sqrt, square
import tensorflow
from . import html_parlist,HtmlPlugin,time_fmt,unid,odict
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
    self.root.mkdir(exist_ok=True)
    self.runs.clear(); runs_ = {}
    for d in sorted(self.root.glob('run-*'),reverse=True):
      run = self.runs_.get(d)
      if run is None: run = Run(d,'Run[{}]'.format(datetime.fromtimestamp(int(d.name[4:],16))))
      self.runs.append(run); runs_[d] = run
    self.runs_ = runs_
    self.loaded = True

#--------------------------------------------------------------------------------------------------
  def create(self):
    r"""
Creates a new (empty) run and adds it to this TF trace.
    """
#--------------------------------------------------------------------------------------------------
    d = self.root/'run-{:x}'.format(int(datetime.now().timestamp()))
    run = Run(d); self.runs.insert(0,run); self.runs_[d] = run
    return run

#--------------------------------------------------------------------------------------------------
# Methods defining the Mapping behaviour
#--------------------------------------------------------------------------------------------------

  def __getitem__(self,k): self.load(); return self.runs[k]
  def __delitem__(self,k):
    self.load()
    if isinstance(k,int): run = self.runs.pop(k); L = [run]
    elif isinstance(k,slice): L = self.runs[k]; del self.runs[k]
    else: raise TypeError('List indices must be integers or slices, not {}'.format(type(k)))
    for run in L: del self.runs_[run.path]; run.destroy()
  def __setitem__(self,k,v): raise Exception('Direct create/update not permitted on {}'.format(type(self)))
  def __iter__(self): self.load(); return iter(self.runs)

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  def as_html(self,_):
    self.load()
    return html_parlist(_,self.runs,(),opening=('Trace {',),closing=('}',))

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
    self.title = str(path) if title is None else title
    def checkpoint(s,saver,**ka):
      if saver is None: return
      return saver.save(s,save_path=str(path/'model.ckpt'),**ka)
    self.checkpoint = checkpoint
    def initialise(s,saver=None):
      if saver is None:
        saver = tensorflow.train.import_meta_graph(tensorflow.train.latest_checkpoint(str(path))+'.meta')
        if saver is None: return
      ckp = saver.last_checkpoints[-1] if saver.last_checkpoints else tensorflow.train.latest_checkpoint(str(path))
      if ckp is None: s.run(s.graph.get_collection_ref(tensorflow.GraphKeys.INIT_OP))
      else: saver.restore(s,ckp)
      return saver
    self.initialise = initialise
    self.summary_writer = partial(tensorflow.summary.FileWriter,logdir=str(path))
    self.destroy = partial(rmtree,str(path))
    self.evfs = {}
    self.loaded = False

  def __hash__(self): return hash(self.path)
  def __eq__(self,other): return isinstance(other,Run) and self.path == other.path

#--------------------------------------------------------------------------------------------------
  def monitor(self,period=100,ckperiod=2000,saver=None,global_step=None):
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
:rtype: :class:`..Monitor`
    """
#--------------------------------------------------------------------------------------------------
    from itertools import cycle, count
    def coroutine(env):
      s = tensorflow.get_default_session()
      stepv = (lambda s,step: step) if global_step is None else (lambda s,step,gstep=global_step: s.run(gstep))
      g = s.graph
      summary_ops = g.get_collection_ref(tensorflow.GraphKeys.SUMMARY_OP)
      summary_writer = self.summary_writer(graph=g)
      try:
        for step,n_su,n_ck in zip(count(1),cycle(range(period-1,-1,-1)),cycle(range(ckperiod-1,-1,-1))):
          if n_su==0:
            vstep = stepv(s,step)
            for v in s.run(summary_ops):
              summary_writer.add_summary(v,global_step=vstep)
            summary_writer.flush()
            if env.logger is not None: env.logger.info('[iterc=%s] summary dumped',step)
          if n_ck==0 or env.stop:
            self.checkpoint(s,env.saver,global_step=stepv(s,step),write_meta_graph=env.stop)
            if env.logger is not None: env.logger.info('[iterc=%s] checkpoint created',step)
          yield
      finally: summary_writer.close()
    return Monitor(('tfrun',),(coroutine,))

#--------------------------------------------------------------------------------------------------
  def supervisor(self,**ka):
#--------------------------------------------------------------------------------------------------
    return tensorflow.train.Supervisor(logdir=str(self.path),checkpoint_basename='model.ckpt',**ka)

#--------------------------------------------------------------------------------------------------
  def tensorboard(self,**ka):
#--------------------------------------------------------------------------------------------------
    raise NotImplementedError()

#--------------------------------------------------------------------------------------------------
  def clip(self,n):
    r"""
Truncates all the event files of this run to *n* entries.
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
    self.path.mkdir(exist_ok=True)
    evfs = OrderedDict()
    for n,f in enumerate(sorted(self.path.rglob('events.out.tfevents*'),key=(lambda f: f.stat().st_mtime))):
      evf = self.evfs.get(f)
      if evf is None: evf = EVFile(f)
      evf.title = '{}{{{}}}'.format(self.title,n)
      evf.walltime = datetime.fromtimestamp(f.stat().st_mtime)
      evfs[f] = evf
    self.evfs = evfs
    self.loaded = True

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  def as_html(self,_):
    self.load()
    return html_parlist(_,self.evfs.values(),(),opening=('Run {',),closing=('}',))

#==================================================================================================
class EVFile (HtmlPlugin):
#==================================================================================================
  style = '''
#toplevel { margin-left: .2mm; margin-right: 4mm; }
#toplevel > tbody > tr > th, #toplevel > tbody > tr > td { text-align:left; background-color: white; border: thin solid black }
#toplevel > thead > tr > td { background-color:gray; color:white; text-align: center; }
  '''
#--------------------------------------------------------------------------------------------------
  def __init__(self,path,clip=1000,title=None):
#--------------------------------------------------------------------------------------------------
    self.path,self.timestamp = path,0
    self.clip_,self.clipped = clip,False
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
    if n>self.clip_ and self.clipped: self.timestamp = 0
    self.clip_ = n

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
      clipped = False; s = EVF_summariser()
      for n,evt in enumerate(tensorflow.train.summary_iterator(str(self.path))):
        if n == self.clip_: clipped = True; break
        s.add(evt)
      self.clipped,self.summariser = clipped,s

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  def as_html(self,_):
    from lxml.html.builder import E
    self.load()
    thead = E.tr(E.td(E.b(self.title),(' (clipped at {})'.format(self.clip_) if self.clipped else ''),colspan='4'))
    tbody = (tr for label,html in self.summariser.html() for tr in html_table_prefix_rows(html,E.th(label)))
    idn = unid('tfutil')
    return E.div(E.style(self.style.replace('#toplevel','#'+idn),scoped='scoped'),E.table(E.thead(thead),E.tbody(*tbody),id=idn))

#==================================================================================================
# Parsing/representation classes for EVFile
#==================================================================================================

#--------------------------------------------------------------------------------------------------
class EVF_summariser:
#--------------------------------------------------------------------------------------------------
  def __init__(self):
    self.hist = []
    self.detail = defaultdict(EVF_base,summary=EVF_summary(),log_message=EVF_log_message())
  def add(self,evt,evtype=namedtuple('evtype',('time','step'))):
    # evt is an Event protobuf object
    self.hist.append(evtype(evt.wall_time,evt.step))
    what = evt.WhichOneof('what')
    self.detail[what].add(getattr(evt,what))
  def html(self):
    from lxml.html.builder import E
    if not self.hist: return
    L = sorted(self.hist,key=(lambda e: e.time))
    tfirst,tspan = L[0].time, L[-1].time-L[0].time
    x = ' over {}, first'.format(time_fmt(tspan)) if tspan else ''
    eventdetail = [('count','{}{} on {}'.format(len(L),x,datetime.fromtimestamp(tfirst)))]
    if tspan:
      L = {}
      for e in self.hist:
        ee = L.get(e.step)
        if ee is None or e.time<ee.time: L[e.step] = e
      L = sorted(L.values(),key=(lambda e: e.step))
      if len(L)>1:
        sfirst,slast = L[0].step,L[-1].step; sspan = slast-sfirst
        x = '{} in {}-{}'.format(len(L),sfirst,slast)
        L = sorted((e2.time-e1.time)/(e2.step-e1.step) for e1,e2 in zip(L[:-1],L[1:]) if e2.time>e1.time)
        n = len(L)
        if n: x += ' (ca. {}/step)'.format(time_fmt((L[n//2]+L[n//2-1])/2 if n%2==0 else L[n//2]))
        eventdetail.append(('steps',x))
        def svg():
          smap = lambda e: '{:.1%}'.format((e.step-sfirst)/sspan)
          tmap = lambda e: '{:.1%}'.format((e.time-tfirst)/tspan)
          yield E.line(x1='0',y1='1mm',x2='100%',y2='1mm',style='stroke:black; stroke-width: 2')
          yield E.line(x1='0',y1='5mm',x2='100%',y2='5mm',style='stroke:black; stroke-width: 2')
          for e in self.hist:
            yield E.line(x1=tmap(e),y1='1mm',x2=smap(e),y2='5mm',style='stroke:gray; stroke-width:1')
        eventdetail.append(('t-line',E.div(E.svg(*svg(),width='80mm',height='6mm'))))
      else:
        eventdetail.append(('steps','singleton: {{{}}}'.format(L[0].step)))
    yield 'events', [E.tr(E.th(n),E.td(x,colspan='2')) for n,x in eventdetail]
    for label,p in self.detail.items():
      if p.count==0: continue
      yield label,list(p.html())

#--------------------------------------------------------------------------------------------------
class EVF_base:
#--------------------------------------------------------------------------------------------------
  def __init__(self): self.count = 0
  def add(self,x): self.count += 1
  def html(self):
    from lxml.html.builder import E
    yield E.tr(E.td(E.b('count'),': {}'.format(self.count),colspan='3'))
#--------------------------------------------------------------------------------------------------
class EVF_summary (EVF_base):
#--------------------------------------------------------------------------------------------------
  TYPE = 'simple_value image histo audio tensor ?'.split()
  def __init__(self):
    super().__init__()
    # initialise tag type counts, one per tag
    self.tagcount = defaultdict(lambda n=len(self.TYPE): zeros(n,dtype=int))
  def add(self,x):
    super().add(x)
    # x is a Summary protobuf object
    for v in x.value:
      try: i = self.TYPE.index(v.WhichOneof('value'))
      except ValueError: i = -1
      self.tagcount[v.tag][i] += 1.
  def html(self):
    from lxml.html.builder import E
    yield from super().html()
    L = ((tag,v,v.sum()) for tag,v in sorted(self.tagcount.items()))
    # normally, each tag should have the same TYPE in all summaries, but, for safety, we show the distribution of TYPEs
    dist = lambda v,vs: (e for typ,x in zip(self.TYPE,v/vs) if x for e in (E.b(typ),(' ({:.1%}) '.format(x) if 1.-x>1e-10 else '')))
    yield from html_table_prefix_rows([E.tr(E.td(tag,' ({})'.format(vs)),E.td(*dist(v,vs))) for tag,v,vs in L],E.th('tags'))

#--------------------------------------------------------------------------------------------------
class EVF_log_message (EVF_base):
#--------------------------------------------------------------------------------------------------
  TYPE = [10,20,30,40,50,0]
  NAME = 'DEBUGGING INFO WARN ERROR FATAL UNKNOWN'.split()
  def __init__(self):
    super().__init__()
    # initialise loglevel counts
    self.perlevel = zeros(len(self.TYPE),dtype=int)
  def add(self,x):
    super().add(x)
    # x in a LogMessage protobuf object
    try: i = self.TYPE.index(x.log_message.value.level)
    except ValueError: i = -1
    self.perlevel[i] += 1
  def html(self):
    from lxml.html.builder import E
    yield from super().html()
    yield from (E.tr(E.td(lvl),E.td(E.b('count'),': {} '.format(n),colspan='2')) for lvl,n in zip(self.NAME,self.perlevel) if n != 0)

#==================================================================================================
def tf_accuracy(y,y_,name='accuracy'):
  r"""
Returns the accuracy op of a batch prediction score tensor *y* to a reference tensor *y_* in onehot representation. Both tensors *y,y_* are of shape *(n,d)* where *n* is the size of the current batch and *d* the dimension of the score vector and onehot representation.

:param y,y_: 2d tensorflow node
:type y,y_: :class:`tensorflow.Tensor`
:param name: name of the result op
  """
#==================================================================================================
  return tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(y,1),tensorflow.argmax(y_,1)),tensorflow.float32),name=name)

#==================================================================================================
def tf_loss(y,y_,name='loss'):
  r"""
Returns the cross-entropy loss op of a batch prediction score tensor *y* to a reference tensor *y_* in onehot representation. Both tensors *y,y_* are of shape *(n,d)* where *n* is the size of the current batch and *d* the dimension of the score vector and onehot representation.

:param y,y_: 2d tensorflow node
:type y,y_: :class:`tensorflow.Tensor`
:param name: name of the result op
  """
#==================================================================================================
  return tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y),name=name)

#==================================================================================================
def tf_freeze(t,name=None,summary_cmd=tensorflow.summary.scalar,val=0,**ka):
  r"""
Returns a tensorflow variable initialised with constant *val* and meant to be updated by the value of operation or tensor *t*.

:param t: a tensorflow tensor or operation
:type t: :class:`Union[tensorflow.Tensor,tensorflow.Operation]`
:param name: name of the result variable
:type name: :class:`str`
:param summary_cmd: the summary command to include the result variable in the summary
:type summary_cmd: :calss:`Callable[...,None]`
:param val: the initial value of the result variable
:param ka: passed to the summary command
  """
#==================================================================================================
  if isinstance(t,tensorflow.Operation): op = t; t = t.outputs[0]
  else: assert isinstance(t,tensorflow.Tensor); op = t.op
  if name is None: name = op.name+'_'
  t_ = tensorflow.get_variable(name,initializer=tensorflow.constant(val,dtype=t.dtype))
  if summary_cmd is not None: summary_cmd(name+'t',t_,**ka)
  return t_

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
    if tr: tr.refresh()
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

def forsave():
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
