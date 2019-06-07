# File:                 project.py
# Creation date:        2019-06-07
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Some utilities for project management
#

import os, sys, subprocess
from pathlib import Path
from datetime import datetime
from functools import partial
from itertools import chain
from shutil import rmtree
from collections import OrderedDict, defaultdict, namedtuple
from numpy import zeros, sqrt, square
from . import html_parlist,HtmlPlugin,time_fmt,unid,odict,html_table
from .monitor import Monitor

#==================================================================================================
class Project (HtmlPlugin):
  r"""
Instances of this class represent projects. A Project is composed of runs, each run generates a set of event files. A Project is attached to a single directory. There is one sub-directory per run, with a name composed by concatenating ``run-`` with the timestamp of the start of the run (secs, formatted in hexa).
  """
#==================================================================================================
#--------------------------------------------------------------------------------------------------
  def __new__(cls,spec,cache={}):
    r"""
:param spec: the path of the target directory
:type spec: :class:`Union[str,pathlib.Path,Project]`
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
Returns the most recent run of this Project.
    """
#--------------------------------------------------------------------------------------------------
    self.load()
    return self.runs[0]

#--------------------------------------------------------------------------------------------------
  def refresh(self):
    r"""
Refreshes this Project.
    """
#--------------------------------------------------------------------------------------------------
    self.loaded = False
    for run in self.runs: run.refresh()

#--------------------------------------------------------------------------------------------------
  def load(self):
    r"""
Loads this Project from its directory.
    """
#--------------------------------------------------------------------------------------------------
    if self.loaded: return
    self.root.mkdir(exist_ok=True)
    self.runs.clear(); runs_ = {}
    for d in sorted(self.root.glob('run-*'),reverse=True):
      run = self.runs_.get(d)
      if run is None: run = Run(d,'Run<{}>'.format(datetime.fromtimestamp(int(d.name[4:],16))))
      self.runs.append(run); runs_[d] = run
    self.runs_ = runs_
    self.loaded = True

#--------------------------------------------------------------------------------------------------
  def create(self):
    r"""
Creates a new (empty) run and adds it to this Project.
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
    return html_table(((k,(v,)) for k,v in enumerate(self.runs)),fmts=((lambda x: x.as_html(_)),),opening=repr(self))
  def __repr__(self): return 'Project<{}>'.format(self.root)

#==================================================================================================
class Run (HtmlPlugin):
  r"""
Instances of this class represent Project runs.
  """
#==================================================================================================

#--------------------------------------------------------------------------------------------------
  def __init__(self,path,title=None):
#--------------------------------------------------------------------------------------------------
    self.path = path
    self.title = str(path) if title is None else title
    self.destroy = partial(rmtree,str(path))
    self.evfs = {}
    self.loaded = False

  def __hash__(self): return hash(self.path)
  def __eq__(self,other): return isinstance(other,Run) and self.path == other.path

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
    for n,f in enumerate(sorted(self.path.rglob('events.out.*'),key=(lambda f: f.stat().st_mtime))):
      evf = self.evfs.get(f)
      if evf is None: evf = EVFile(f)
      evfs[f] = evf
    self.evfs = evfs
    self.loaded = True

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  def as_html(self,_):
    self.load()
    return html_table(sorted((k,(v,)) for k,v in enumerate(self.evfs.values())),fmts=((lambda x: x.as_html(_)),),opening=repr(self))
  def __repr__(self): return self.title

#==================================================================================================
class EVFile (HtmlPlugin):
#==================================================================================================
  style = '''
#toplevel { margin-left: .2mm; margin-right: 4mm; }
#toplevel > tbody > tr > th, #toplevel > tbody > tr > td { text-align:left; background-color: white; border: thin solid black }
#toplevel > thead > tr > td { background-color:gray; color:white; text-align: center; }
  '''
#--------------------------------------------------------------------------------------------------
  def __init__(self,path,clip=1000):
#--------------------------------------------------------------------------------------------------
    self.path,self.timestamp = path,0
    self.clip_,self.clipped = clip,False

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
    from tensorflow.train import summary_iterator
    t = self.path.stat().st_mtime
    if t>self.timestamp:
      self.timestamp = t
      clipped = False; s = EVF_summariser()
      for n,evt in enumerate(summary_iterator(str(self.path))):
        if n == self.clip_: clipped = True; break
        s.add(evt)
      self.clipped,self.summariser = clipped,s

#--------------------------------------------------------------------------------------------------
# Representation methods
#--------------------------------------------------------------------------------------------------

  def as_html(self,_):
    from lxml.html.builder import E
    self.load()
    thead = E.tr(E.td(repr(self),(' clipped @ {}'.format(self.clip_) if self.clipped else ''),colspan='4'))
    tbody = (tr for label,html in self.summariser.html() for tr in html_table_prefix_rows(html,E.th(label)))
    idn = unid('project')
    return E.div(E.style(self.style.replace('#toplevel','#'+idn),scoped='scoped'),E.table(E.thead(thead),E.tbody(*tbody),id=idn))
  def __repr__(self): return 'EventFile<{}>'.format(self.path.name[20:])

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
def manage(*paths,ivname='project'):
#==================================================================================================
  import ipywidgets
  from IPython.display import clear_output, display
  from IPython.core.getipython import get_ipython
  def showproject():
    if project: project.refresh()
    with w_out: clear_output(); display(project)
  def setproject(c): nonlocal project; project = c.new; interpreter.push({ivname:project}); showproject()
  interpreter = get_ipython()
  project = ()
  interpreter.push({ivname:project})
  w_project = ipywidgets.Dropdown(description=ivname,options=OrderedDict(chain((('...',()),),((p,Project(p)) for p in paths))),style={'description_width':'initial'})
  w_out = ipywidgets.Output()
  b_refresh = ipywidgets.Button(icon='refresh',tooltip='refresh',layout=ipywidgets.Layout(width='.4cm',padding='0cm'))
  b_close = ipywidgets.Button(icon='close',tooltip='close',layout=ipywidgets.Layout(width='.4cm',padding='0cm'))
  w_main = ipywidgets.VBox(children=(ipywidgets.HBox(children=(w_project,b_refresh,b_close),layout={'border-bottom':'thin solid black'}),w_out))
  w_project.observe(setproject,'value')
  b_refresh.on_click(lambda b: showproject())
  b_close.on_click(lambda b: w_main.close())
  return w_main

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
  def tensorboard(path,hostname=None):
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
        sub = subprocess.Popen((server_launchcmd,'--host',hostname,'--port',port,'--logdir',str(path)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
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
