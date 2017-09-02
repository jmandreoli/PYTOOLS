import os, sys, subprocess
from pathlib import Path
from datetime import datetime
from functools import partial
from itertools import chain
from shutil import rmtree
from collections import OrderedDict
from numpy import zeros, sqrt, square
import tensorflow
from myutil import basic_stats,html_stack,html_table,HtmlPlugin

# Management of runs

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

  def __getitem__(self,k): return self.runs[k]
  def __delitem__(self,k):
    if isinstance(k,int): self.runs.pop(k).destroy()
    elif isinstance(k,slice):
      for run in self.runs[k]: run.destroy()
      del self.runs[k]
    else: raise TypeError('List indices must be integers or slices, not {}'.format(type(k)))
  def __setitem__(self,k,v): raise Exception('Direct create/update not permitted on TFTrace')
  def __iter__(self): return iter(self.runs)

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
    self.model_load = partial(tensorflow.saved_model.loader.load,export_dir=d_mod)
    self.summary_writer = partial(tensorflow.summary.FileWriter,logdir=d_log)
    def checkpoint_saver(*a,**ka):
      x = tensorflow.train.Saver(*a,**ka)
      x.save = partial(x.save,save_path=d_ckp); x.restore = partial(x.restore,save_path=d_ckp)
      return x
    self.checkpoint_saver = checkpoint_saver
    self.destroy = partial(rmtree,str(path))
    self.evfs = {}
    self.loaded = False

#--------------------------------------------------------------------------------------------------
  def monitor(self,period=100,ckperiod=2000,setup={},summary_first_batch=True):
    r"""
Returns a loop monitor for this run. Typical invocation::

   tf = TFTrace('/mypath')
   data = gatherdata()
   m = tf.monitor()
   with tensorflow.Session() as s:
     m.run(tf_main(s,data))

:param period: a tensoflow summary is dumped every *period* steps
:type period: :class:`int`
:param ckperiod: a tensorflow checkpoint is created every *period* steps
:type ckperiod: :class:`int`
:param setup: the feed dictionary for summaries is updated with *setup*
:type setup: :class:`Dict[str,object]`
:param summary_first_batch: 
:type summary_first_batch: :class:`bool`
    """
#--------------------------------------------------------------------------------------------------
    from ..monitor import Monitor
    from itertools import cycle, count
    curbatch = not summary_first_batch
    def coroutine(env):
      s,fd = env.value
      if not curbatch: fds = fd.copy(); fds.update(setup)
      summary = s.graph.get_tensor_by_name('Merge/MergeSummary:0')
      summary_writer = self.summary_writer(graph=s.graph)
      model_builder = self.model_builder()
      checkpoint_saver = self.checkpoint_saver()
      for step,n_su,n_ck in zip(count(1),cycle(range(period-1,-1,-1)),cycle(range(ckperiod-1,-1,-1))):
        if n_su==0:
          if curbatch: s,fd = env.value; fds = fd.copy(); fds.update(setup)
          v = s.run(summary,feed_dict=fds)
          summary_writer.add_summary(v,step)
          summary_writer.flush()
        if n_ck==0: env.lastcheckpoint = checkpoint_saver.save(s)
        if env.stop:
          model_builder.add_meta_graph_and_variables(s,['model'])
          model_builder.save()
        yield
    return Monitor(('tfrun',),(coroutine,))

  def clip(self,n):
    for evf in self.evfs.values(): evf.clip = n

  def refresh(self):
    self.loaded = False
    for evf in self.evfs.values(): evf.refresh()

  def load(self):
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

  def tensorboard(self,hostname=None):
    raise NotImplementedError()
    import ipywidgets
    from socket import getfqdn
    from IPython.display import clear_output, display
    if hostname is None: hostname = getfqdn()
    
    sub = subprocess.Popen((str(Path(sys.executable).parent/'tensorboard'),'--host',hostname,'--port',port,'--logdir',self.path),stdout=subprocess.PIPE,universal_newlines=True)

    def showtrace(): clear_output(); display(self.trace)
    def settrace(c): self.trace = c.new; showtrace()
    def refresh():
      if self.trace is not None: self.trace.refresh()
      showtrace()
    self.trace = None
    wtrace = ipywidgets.Dropdown(options=OrderedDict(chain((('!',None),),((p,TFTrace(p)) for p in paths))))
    wtrace.observe(settrace,'value')
    wrefresh = ipywidgets.Button(icon='fa-refresh',tooltip='refresh',layout=ipywidgets.Layout(width='.4cm'))
    wrefresh.on_click(lambda b: refresh())
    self.widget = ipywidgets.HBox(children=(wtrace,wrefresh))


  def __hash__(self): return hash(self.path)
  def __eq__(self,other): return isinstance(other,Run) and self.path == other.path
  def as_html(self,incontext): self.load(); return html_stack(*(evf.as_html(incontext) for evf in self.evfs.values()))

#==================================================================================================
class EVFile (HtmlPlugin):
#==================================================================================================
  TYPE = dict((v,i) for i,v in enumerate('simple_value image histo audio tensor ?'.split()))
  style = 'table th,td {text-align:left; background-color: white; border: thin solid black}'
  def __init__(self,path,clip=1000,title=None):
    self.path = path; self.e_count = 0; self.clip_ = clip; self.timestamp = 0
    self.title = str(self.path) if title is None else title
  @property
  def clip(self): return self.clip_
  @clip.setter
  def clip(self,n):
    self.clip_ = n
    if n>self.e_count: self.timestamp = 0

  def refresh(self): self.timestamp = 0

  def load(self):
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

  def __hash__(self): return hash(self.path)
  def __eq__(self,other): return isinstance(other,EVFile) and self.path == other.path
  def as_html(self,_):
    from lxml.builder import E
    self.load()
    thead = E.TR(E.TD(E.B(self.title),E.SPAN(' [{} events{}]'.format(self.e_count,'!' if self.clipped else '')),colspan='3',style='background-color:gray; color:white; text-align:center'))
    def tbody():
      if self.g_count: yield E.TH('graph'),E.TD(E.B('count'),': {}'.format(self.g_count),colspan='2')
      if self.s_count:
        yield E.TH('steps'),E.TD(E.B('count'),': {} '.format(self.s_count),E.B('first'),': {} '.format(self.s_bnd[0]),E.B('last'),': {} '.format(self.s_bnd[1]),colspan='2')
        yield E.TH('inter-step'),E.TD(E.B('avg'),': {:.1f} '.format(self.sd_stats.avg),E.B('std'),': {:.1f} '.format(sqrt(self.sd_stats.var)),colspan='2')
        yield E.TH('step-ms'),E.TD(E.B('avg'),': {:.3g} '.format(1000*self.sdt_stats.avg),E.B('std'),': {:.3g} '.format(1000*sqrt(self.sdt_stats.var)),colspan='2')
        firstrow = True
        for tag,v in sorted(self.tags.items()):
          c = [E.TD(tag),E.TD(*(e for typ,x in zip(self.TYPE,v/self.s_count) if x for e in (E.B(typ),(' ({:.1%}) '.format(x) if 1.-x>1e-10 else ''))))]
          if firstrow: firstrow = False; c.insert(0,E.TH('tags',rowspan=str(len(self.tags))))
          yield c
    return E.TABLE(E.STYLE(self.style,scoped="scoped"),E.THEAD(thead),E.TBODY(*(E.TR(*cells) for cells in tbody())))


#==================================================================================================
def tf_accuracy(y,y_,name='accuracy',sname='train-accuracy'):
#==================================================================================================
  x = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(y,1),tensorflow.argmax(y_,1)),tensorflow.float32),name=name)
  tensorflow.summary.scalar(sname,x)
  return x

#==================================================================================================
def tf_loss(y,y_,name='loss',sname='loss'):
#==================================================================================================
  x = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y),name=name)
  tensorflow.summary.scalar(sname,x)
  return x

#==================================================================================================
# test computation
def tf_test(s,data,setup={},op='accuracy',Lx=('x',),y='y_true'):
#==================================================================================================
  Lx = [s.graph.get_tensor_by_name(v+':0') for v in Lx]
  y = s.graph.get_tensor_by_name(y+':0')
  op = s.graph.get_tensor_by_name(op+':0')
  fd = {y:data[1]}
  fd.update(zip(Lx,data[0]))
  fd.update(setup)
  return s.run(op,feed_dict=fd)

#==================================================================================================
# main loop computation
def tf_main(s,data,setup={},Lx=('x',),y='y_true',init_step='init',train_step='train'):
#==================================================================================================
  init_step = s.graph.get_operation_by_name(init_step)
  train_step = s.graph.get_operation_by_name(train_step)
  Lx = [s.graph.get_tensor_by_name(v+':0') for v in Lx]
  y = s.graph.get_tensor_by_name(y+':0')
  s.run(init_step)
  for Lbatch_xs,batch_ys in data:
    fd = {y:batch_ys}
    fd.update(zip(Lx,Lbatch_xs))
    fd.update(setup)
    s.run(train_step,feed_dict=fd)
    yield s,fd

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
def trmanage(*paths,ivname='tr'):
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
