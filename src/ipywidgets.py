# File:                 ipywidgets.py
# Creation date:        2021-06-01
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Utilities for ipywidgets
#

from __future__ import annotations
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence
import logging; logger = logging.getLogger(__name__)

import pandas, traceback
import sqlalchemy.engine
from functools import wraps
from contextlib import contextmanager
from pathlib import Path
from collections import namedtuple
import traitlets
from ipywidgets import Widget, Label, IntSlider, FloatSlider, Text, IntText, FloatText, BoundedIntText, BoundedFloatText, Password, HTML, Checkbox, Dropdown, Select, SelectMultiple, Button, Output, Tab, Stack, VBox, HBox, Layout, Valid, Play, jslink, AppLayout

__all__ = 'app', 'seq_browser', 'file_browser', 'db_browser', 'hastrait_editor', 'PlayTracks', 'SelectMultipleOrdered', 'SimpleButton', 'setdefault_layout', 'setdefault_children_layout', 'AutoWidthStyle',

AutoWidthStyle = {'description_width':'auto'}

#==================================================================================================
class app (AppLayout):
  r"""
An instance of this class is an app based on module :mod:`ipywidgets`, consisting of a toolbar, a console and additional child widgets.

:param toolbar: widgets added at the end of the toolbar (in addition to the default toolbar widgets)
:param children: additional children widgets appended to the main widget
  """
#==================================================================================================

  toolbar: HBox
  r"""The toolbar widget (first child of :attr:`main`)"""
  console: Output
  r"""A console terminal (just below :attr:`toolbar`)"""

  def __init__(self,toolbar:Sequence[Widget]=(),children:Sequence[Widget]=(),**ka):
    w_closeb = SimpleButton(icon='close',tooltip='Close app')
    self.console = console = Output()
    w_clearb = SimpleButton(icon='trash',tooltip='Clear console')
    w_console = HBox([w_clearb,console],layout={'border': 'thin solid', 'display': 'none'})
    self.toolbar = HBox([w_closeb,*toolbar])
    super().__init__(header=VBox([self.toolbar,w_console]),center=VBox(children),**ka)
    self.layout.grid_template_rows = self.layout.grid_template_columns = None
    def _console_clear(b):
      from IPython.display import clear_output
      with console: clear_output()
    w_clearb.on_click(_console_clear)
    def _console_display(c): w_console.layout.display = '' if c.new else 'none'
    console.observe(_console_display,'outputs')
    _close_callbacks = [self.close_all]
    self.on_close = _close_callbacks.append
    w_closeb.on_click(lambda b: [f() for f in _close_callbacks])
  def __enter__(self): return self.console.__enter__()
  def __exit__(self,*a): self.console.__exit__(*a)

  def mpl_figure(self,*a,**ka):
    r"""
Adds a :mod:`matplotlib` figure to this app.
    """
    from matplotlib.pyplot import close
    fig,w = mpl_figure(*a,asapp=False,**ka)
    self.center.children += (w,)
    self.on_close(lambda:close(fig))
    return fig

  def protect(self,f):
    r"""A decorator to redirect output and errors to the console."""
    @wraps(f)
    def F(*a,**ka):
      with self:
        try: return f(*a,**ka)
        except: traceback.print_exc(); raise
    return F

#==================================================================================================
class seq_browser (app):
  r"""
An instance of this class is an app to display a (long) sequence object in pagination mode.

:param D: a sequence object
:param start: the index of the initial page
:param pgsize: the size of the pages
:param ka: dictionary of layout values for the main widget
  """
#==================================================================================================
  def __init__(self,D:Sequence,start:int=1,pgsize:int=10):
    from IPython.display import display, clear_output
    # content initialisation
    P = (len(D)-1)//pgsize + 1
    # widget creation
    self.w_out = w_out = Output()
    if P == 1: w_pager = Label('1')
    else: w_pager = IntSlider(start,min=1,max=P,description='page:',layout={'width':'10cm'},readout=False,style=AutoWidthStyle)
    self.w_pager = w_pager
    w_pos = Label(f'',layout={'border':'thin solid','padding':'0mm 1mm 0mm 1mm'},style=AutoWidthStyle)
    super().__init__(toolbar=[w_pager,w_pos],children=[w_out])
    # event handling
    def show(n):
      with w_out:
        clear_output(wait=True)
        display(D[(n-1)*pgsize:n*pgsize])
      w_pos.value = f'{n}/{P}'
    # event binding
    w_pager.observe((lambda c: show(c.new)),'value')
    # widget initialisation
    show(start)

#==================================================================================================
class file_browser (app):
  r"""
An instance of this class is an app to browse the file at *path*, possibly while it expands. If *start* is :const:`None`, the start position is end-of-file. If *start* is of type :class:`int`, it denotes the exact start position in bytes. If *start* is of type :class:`float`, it must be between :const:`0.` and :const:`1.`, and the start position is set (approximatively) at that position relative to the whole file.

:param path: a path to an existing file
:param start: index of start pointed
:param step: step size in bytes
:param track: whether to track changes in the file
:param context: pair of number of lines before and after to display around current position
  """
#==================================================================================================
  def __init__(self,path:str|Path,start:int|float|None=None,step:int=50,track:bool=True,context:tuple[int,int]=(10,5)):
    # content initialisation
    path = Path(path).absolute()
    file = path.open('rb')
    fsize = path.stat().st_size or 1
    if start is None: start = fsize
    elif isinstance(start,float):
      assert 0 <= start <= 1
      start = int(start*fsize)
    else:
      assert isinstance(start,int)
      if start<0:
        start += fsize
        if start<0: start = 0
    # widget creation
    self.w_win = w_win = HTML(layout={'width':'15cm','border':'thin solid','overflow':'auto hidden'})
    self.w_ctrl = w_ctrl = IntSlider(start,min=0,max=fsize,step=step,readout=False,layout={'width':w_win.layout.width})
    w_toend = SimpleButton(icon='fast-forward',tooltip='Jump to end of file')
    w_tobeg = SimpleButton(icon='fast-backward',tooltip='Jump to beginning of file')
    w_pos = Label('')
    super().__init__(toolbar=[w_ctrl,w_pos,w_tobeg,w_toend],children=[w_win])
    self.on_close(file.close)
    # event handling
    def setpos(n,nbefore=context[0]+1,nafter=context[1]+1):
      def readlinesFwd(u,n,k):
        u.seek(n)
        for i in range(k):
          x = u.readline()
          if x: yield x[:-1]
          else: return
      def readlinesBwd(u,n,k,D=256):
        c = n; t = b''
        while True:
          if c==0:
            if t: yield t
            return
          d = D; c -= d
          if c<0: c,d = 0,d+c
          u.seek(c)
          L = u.read(d).rsplit(b'\n',k)
          L[-1] += t
          for x in L[-1:0:-1]:
            yield x
            k -= 1
            if k==0: return
          t = L[0]
      lines = (nbefore+nafter-1)*[b'']
      c = nbefore
      x = list(readlinesBwd(file,n,nbefore))
      lines[c] += b'\n'.join(x[:1]); lines[c-1:c-len(x):-1] = x[1:]
      x = list(readlinesFwd(file,n,nafter))
      lines[c] += b'\n'.join(x[:1]); lines[c+1:c+len(x)] = x[1:]
      lines = [x.decode().replace('&','&amp;').replace('<','&lt;').replace('>','&gt;') for x in lines]
      lines[c] = f'<span style="background-color: gray; color: white; border: thin solid gray;">{lines[c]}</span>'
      w_win.value = '<div style="white-space: pre; font-family: monospace; line-height:130%;">{}</div>'.format('\n'.join(lines))
      w_pos.value = f'{n}'
    def toend(): w_ctrl.value = fsize
    def tobeg(): w_ctrl.value = 0
    def resetsize():
      nonlocal fsize
      atend = w_ctrl.value == fsize
      w_ctrl.max = fsize = path.stat().st_size
      if atend: w_ctrl.value = fsize
      else: setpos(w_ctrl.value)
    # event binding
    w_ctrl.observe((lambda c: setpos(c.new)),'value')
    w_tobeg.on_click(lambda b: tobeg())
    w_toend.on_click(lambda b: toend())
    # initialisation
    setpos(start)
    if track:
      from watchdog.observers import Observer
      from watchdog.events import FileSystemEventHandler
      class MyHandler (FileSystemEventHandler):
        def __init__(self,p,f): super().__init__(); self.on_modified = (lambda evt: (f() if evt.src_path==p else None))
      observer = Observer()
      observer.schedule(MyHandler(str(path),resetsize),str(path.parent))
      observer.start()
      def observer_close(): observer.stop();observer.join()
      self.on_close(observer_close)

#==================================================================================================
class db_browser (app):
  r"""
An instance of this class is an app to explore a database specified by *spec*. If a metadata structure is specified, it must be bound to an existing engine and reflected.

:param spec: an sqlalchemy url or engine or metadata structure, defining the database to explore
  """
#==================================================================================================
  style = 'background-color:gray; color:white; font-weight:bold; padding:.2cm'
  def __init__(self,spec:str|sqlalchemy.engine.Engine):
    from pandas import read_sql_query
    from sqlalchemy import select, func, MetaData, create_engine
    from sqlalchemy.engine import Engine
    from IPython.display import display, clear_output
    # content initialisation
    if isinstance(spec,str): engine = create_engine(spec)
    elif isinstance(spec,Engine): engine = spec
    else: raise TypeError(f'Expected {str}|{Engine}; Found {type(spec)}')
    meta = MetaData()
    meta.reflect(bind=engine,views=True)
    if not meta.tables: self.main = Label(value='Database is empty'); return
    config = {name:db_browser_table(t) for name,t in meta.tables.items()}
    config_header = db_browser_table.header()
    # widget creation
    w_title = HTML(f'<div style="{self.style}">{engine}</div>')
    w_table = activable(Select(options=sorted(meta.tables.items()),rows=min(len(meta.tables),10),layout={'width':'10cm'}))
    w_schema = VBox([config_header,*(cfg.widget for cfg in config.values())])
    w_scol = SelectMultipleOrdered(layout={'width':'6cm'})
    w_ordr = SelectMultipleOrdered(layout={'width':'6cm'})
    d_titles,d_children = zip(('Column definitions',w_schema),('Column selection',w_scol),('Display ordering',w_ordr))
    w_detail = Tab(titles=d_titles,children=d_children,layout={'display':'none'})
    w_detailb = SimpleButton(tooltip='toggle detail display (red border means some columns are hidden)',icon='info-circle')
    w_reloadb = SimpleButton(tooltip='reload table',icon='refresh')
    w_nsample = BoundedIntText(description='sample:',min=1,max=50,step=1,layout={'width':'2.5cm'},style=AutoWidthStyle)
    w_size = Text('',disabled=True,tooltip='Number of rows',layout={'width':'4cm'},style=AutoWidthStyle)
    w_offset = IntSlider(description='offset',min=0,step=1,layout={'width':'10cm'},readout=False,style=AutoWidthStyle)
    w_out = Output()
    super().__init__(toolbar=[w_title],children=[HBox([w_table,w_detailb]),w_detail,HBox([w_nsample,Label('/',style=AutoWidthStyle),w_size,w_offset,w_reloadb]),w_out])
    # event handling
    def size(table): # returns the size of a table
      try: return read_sql_query(select(func.count()).select_from(table),engine).to_numpy().item()
      except: return -1
    def sample(columns,nsample,offset,order=None)->pandas.DataFrame: # returns nsample row samples of columns ordered by order
      sql = select(*columns).limit(nsample).offset(offset).order_by(*(order or columns))
      r = read_sql_query(sql,engine)
      r.index = list(range(offset,offset+min(nsample,len(r))))
      return r
    @self.protect
    def show():
      with w_out:
        clear_output(wait=True)
        display(sample(cfg.selected,cfg.nsample,cfg.offset,cfg.order))
    cfg = None
    def set_table(c=None):
      nonlocal cfg
      if c is None: new = w_table.value
      else:
        new = c.new
        config[c.old.name].widget.layout.display = 'none'
      cfg = config[new.name]
      sz = size(w_table.value)
      w_scol.rows = w_ordr.rows = min(len(cfg.options),10)
      w_size.value = str(sz)
      cfg.widget.layout.display = 'inline'
      with deactivate(w_table):
        w_scol.options = w_ordr.options = cfg.options
        w_scol.value, w_ordr.value = cfg.selected, cfg.order
        w_offset.max = max(sz-1,0)
        w_offset.value = cfg.offset
        w_nsample.value = cfg.nsample
      show()
    def set_scol(c):
      w_detailb.layout.border = 'none' if len(c.new) == len(cfg.options) else 'thin solid red'
      if w_table.active: cfg.selected = c.new; show()
    def set_ordr(c):
      if w_table.active: cfg.order = c.new; show()
    def set_offset(c):
      if w_table.active: cfg.offset = c.new; show()
    def set_nsample(c):
      if w_table.active: cfg.nsample = c.new; show()
    def toggledetail(inv={'inline':'none','none':'inline'}):
      w_detail.layout.display = inv[w_detail.layout.display]
    # event binding
    w_detailb.on_click(lambda b: toggledetail())
    w_reloadb.on_click(lambda b: show())
    w_table.observe(set_table,'value')
    w_offset.observe(set_offset,'value')
    w_nsample.observe(set_nsample,'value')
    w_scol.observe(set_scol,'value')
    w_ordr.observe(set_ordr,'value')
    # initialisation
    set_table()

#==================================================================================================
class db_browser_table:
  """An instance of this data class holds the configuration and widget of a table."""
#==================================================================================================
  style = 'background-color: navy; color: white; font-size: x-small; border: thin solid; text-align: center;'
  schema = ( # this is the schema of schemas!
    ('name',str,'name',Layout(width='4cm')),
    ('type',object,'type',Layout(width='4cm')),
    ('primary_key',bool,'P',Layout(width='.9cm',margin='0cm',top='.5mm')),
    ('nullable',bool,'N',Layout(width='.9cm',margin='0cm',top='.5mm')),
    ('unique',bool,'U',Layout(width='.9cm',margin='0cm',top='.5mm')),
    ('default',object,'default',Layout(width='4cm')),
    ('constraints',object,'constraints',Layout(width='4cm')),
    ('foreign_keys',object,'foreign',Layout(width='4cm')),
  )
  widget_factory = {
    str:(lambda x: Text(value=x)),
    bool:(lambda x: Checkbox(value=x,style=dict(description_width='initial'))),
    object:(lambda x: Text(value=str(x))),
  }
  offset = 0
  nsample = 5
  @classmethod
  def header(cls):
    return HBox([HTML(value=f'<div style="{cls.style}" title="{x[0]}">{x[2]}</div>',layout=x[3]) for x in cls.schema])
  def __init__(self,table):
    def widget(x,c):
      try: a = getattr(c,x[0])
      except AttributeError: w = Valid(value=False)
      else: w = self.widget_factory[x[1]](a)
      w.layout = x[3]
      w.disabled = True
      return w
    self.options = [(c.name,c) for c in table.columns]
    self.widget = VBox([HBox([widget(x,c) for x in self.schema]) for c in table.columns])
    self.widget.layout.display = 'none'
    self.selected = tuple(table.columns)
    self.order = tuple(table.primary_key)

#==================================================================================================
class hastrait_editor (app):
#==================================================================================================
  r"""
An instance of this class is an app to edit a traitlets structure. The construction of the widget is directed by metadata which must be associated to each editable trait in *target* under the key ``widget``. This is either a callable, which produces a widget for that trait, or a :class:`dict`, passed as keyword arguments to a widget guesser. Guessing is based on the trait class. Each property in *default_trait_layout* is applied to all the trait widget layouts which assign a :const:`None` value to that property. The overall editor widget can be further customised using the following attributes:

.. attribute:: header,footer

   They can be modified freely. The :attr:`header` and :attr:`footer` widgets (:class:`VBox` instances) appear immediately before and after, respectively, the trait widgets. A reset buttons is present in the toolbar by default.

.. attribute:: label_layout

   A shared :class:`Layout` instance formatting the label of all the traits. Modifications of this layout therefore applies to all trait labels.

:param target: a structure with traits
  """
  def __init__(self,target:traitlets.HasTraits,default_widget_layout=None,label_layout=None):
    # content initialisation
    initial = {}
    # widget creation
    layouts = default_widget_layout, label_layout
    config = dict((name,hastrait_editor_trait(name,t,*layouts)) for name,t in target.traits().items() if t.metadata.get('widget') is not None)
    w_resetb = SimpleButton(icon='undo',description='reset')
    self.header = VBox()
    self.footer = VBox()
    super().__init__(toolbar=[w_resetb],children=[self.header,*(cfg.widget for cfg in config.values()),self.footer])
    # event handling
    def updw(w,x): w.value = x
    def upda(name,w,x):
      try: setattr(target,name,x)
      except: w.value = getattr(target,name)
    def reset(data):
      for k,v in data.items(): setattr(target,k,v)
    # event binding
    for name,cfg in config.items():
      cfg.interact.value = initial[name] = getattr(target,name)
      cfg.resetb.on_click(lambda b,w=cfg.interact,x=initial[name]: updw(w,x))
      cfg.interact.observe((lambda c,name=name,w=cfg.interact: upda(name,w,c.new)),'value')
      target.observe((lambda c,w=cfg.interact: updw(w,c.new)),name)
    w_resetb.on_click(lambda b: reset(initial))

#==================================================================================================
class hastrait_editor_trait:
  """An instance of this data class holds the configuration and widget of a trait."""
#==================================================================================================
  default_label_layout = dict(width='2cm',padding='0 0 0 .2cm',align_self='flex-start')
  default_widget_layout = dict(width='15cm')
  def __init__(self,name,t,default_widget_layout,label_layout):
    def g_numeric(t,ws,typ):
      slider = {int:IntSlider,float:FloatSlider}
      btext = {int:BoundedIntText,float:BoundedFloatText}
      text = {int:IntText,float:FloatText}
      from math import isfinite
      vmin = ws.pop('min',None)
      if vmin is None and (t.min is not None and isfinite(t.min)): vmin = t.min
      vmax = ws.pop('max',None)
      if vmax is None and (t.max is not None and isfinite(t.max)): vmax = t.max
      m = ws.pop('mode','slider')
      if vmin is None or vmax is None:
        return text[typ](**ws)
      else:
        return dict(slider=slider, text=btext)[m][typ](min=vmin, max=vmax, **ws)
    def g_selector(t,ws):
      mode = dict(select=Select, dropdown=Dropdown)
      opt = ws.get('options')
      ws['options'] = [(str(v),v) for v in t.values] if opt is None else list(zip(opt,t.values))
      m = ws.pop('mode', 'select')
      return mode[m](**ws)
    def g_mselector(t,ws):
      opt = ws.get('options')
      ws['options'] = [(str(v),v) for v in t._trait.values] if opt is None else list(zip(opt,t._trait.values))
      ordr = ws.pop('ordered', False)
      return (SelectMultipleOrdered if ordr else SelectMultiple)(**ws)
    ws = t.metadata.get('widget')
    if callable(ws): w = ws()
    else: # guessing
      ws = dict(ws)
      if isinstance(t,traitlets.Integer):
        w = g_numeric(t,ws,int)
      elif isinstance(t,traitlets.Float):
        w = g_numeric(t,ws,float)
      elif isinstance(t,traitlets.Bool):
        w = Checkbox(**ws)
      elif isinstance(t,traitlets.Enum):
        w = g_selector(t,ws)
      elif isinstance(t,traitlets.List) and isinstance(t._trait,traitlets.Enum):
        w = g_mselector(t,ws)
      else:
        raise Exception('Cannot guess widget')
    setdefault_layout(w,**dict(self.default_widget_layout,**(default_widget_layout or {})))
    self.interact = w
    label = HTML(f'<span title="{t.help}">{name}</span>',layout=dict(self.default_label_layout,**(label_layout or {})))
    self.resetb = resetb = SimpleButton(icon='undo',tooltip='Reset to default')
    self.widget = HBox([resetb,label,w])

#==================================================================================================
class PlayTracks (HBox):
  r"""
Widgets of this class are similar to :class:`Play` widgets, with more functionality to navigate its value domain (interval of whole numbers). The domain is split into a sequence of contiguous intervals called tracks. Parameter *track* is a function which returns, for each number in the domain, the bounds of its track interval (closed on the left, open on the right), and :const:`None` outside the domain.

* If *track* is an :class:`int` instance, the intervals are of constant length *track*
* If *track* is a sequence of :class:`int` instances, possibly ended by the ellipsis `...`, it gives the length of the intervals (and repeat if ellipsis)
* Otherwise, *track* must be a function of one :class:`int` input returning two :class:`int` outputs

The current position in the domain is reported on a clock, showing the value of the position divided by a constant rate.

:param track_spec: track function specification
:param rate: position to clock-time ratio (default: clock-time ~ real time in sec)
:param value: initial clock-time (same as in :class:`Play` widgets, divided by *rate*)
:param playing: whether the widget is initially playing (same as in :class:`Play` widgets)
:param ka: passed to the :class:`Play` constructor
  """
#==================================================================================================
  value = traitlets.Float(min=0.)
  r"""Current clock-time (always rounded to an integer multiple of *rate*)"""
  playing = traitlets.Bool()
  r"""Whether the widget is currently playing"""
  def __init__(self,track_spec:int|Sequence[int]|Callable[[int],tuple[int,int]],rate:float|None=None,value:float=0.,playing:bool=False,**ka):
    def track_func_()->Callable[[int],tuple[int, int]]:
      if callable(track_spec):
        n = 0
        for _ in range(5): # check the first 5 values
          track_ = track_spec(n)
          if track_ is None: assert n>0; break
          else:
            p,q = track_
            assert isinstance(p,int) and isinstance(q,int) and p==n<q and all(track_spec(m) == track_ for m in range(n+1,q))
            n = q
        track_func = track_spec
      elif isinstance(track_spec,int):
        assert track_spec>0
        def track_func(x,T=track_spec): x -= x%T; return x,x+T
      else:
        Ld = list(track_spec)
        if (rep:=Ld[-1]==...): del Ld[-1]
        assert len(Ld)>0 and all(isinstance(x,int) and x>0 for x in Ld)
        L = [n:=0]
        for d in Ld: L.append(n:=n+d)
        from bisect import bisect
        N = len(L)
        if rep:
          def track_func(x,T=L[-1]):
            if x<0: return None
            r = x%T; x -= r; i = bisect(L,r); return x+L[i-1],x+L[i]
        else:
          def track_func(x): return (L[i-1],L[i]) if (i:=bisect(L,x))>0 and i<N else None
      return track_func
    track_func = track_func_()
    #
    style = ka.pop('style',{}); layout = ka.pop('layout',{})
    p,q = track_func(0)
    w_play = Play(p,min=p,max=q,show_repeat=False,layout={'width':'1.9cm'},**ka)
    w_ind = IntSlider(p,min=p,max=q,readout=False,continuous_update=False)
    w_clock_edit = FloatText(layout={'width':'1.6cm','padding':'0cm'})
    w_clock = Stable(w_clock_edit,'{:.2f}')
    jslink((w_play,'value'),(w_ind,'value'))
    jslink((w_play,'min'),(w_ind,'min'))
    jslink((w_play,'max'),(w_ind,'max'))
    if rate is None: rate = 1000./w_play.interval
    w_play.observe((lambda c: setattr(self,'value',(c.new+1.e-6)/rate) if self.active else None),'value')
    w_clock.observe((lambda c: setattr(self,'value',c.new) if self.active else None),'value')
    def from_self(v):
      with deactivate(self):
        assert v>=0.; n = int(v*rate); w_clock.value = (n+1.e-6)/rate
        if n >= w_play.max:
          if (tr:=track_func(n)) is not None: w_play.max = tr[1]; w_play.value,w_play.min = n,tr[0]
        elif n < w_play.min:
          if (tr:=track_func(n)) is not None: w_play.min = tr[0]; w_play.value,w_play.max = n,tr[1]
        else: w_play.value = n
    self.observe((lambda c: from_self(c.new)),'value')
    self.observe((lambda c: setattr(w_play,'playing',c.new)),'playing')
    w_play.observe((lambda c: setattr(self,'playing',c.new)),'playing')
    self.active = True; self.value = value; self.playing = playing
    super().__init__([w_play,w_ind,w_clock],style=style,layout=layout)

#==================================================================================================
class Login (HBox):
  r"""
A widget holding a server,username,password. The passwords are protected and cached in memory.
  """
#==================================================================================================
  _cache:dict[tuple[str,str],str] = {}
  def __init__(self,name,host='',user=''):
    w_host = Text(description=name,value='',style=AutoWidthStyle,layout={'width':'8cm'})
    w_user = Text(description='user',value='',style=AutoWidthStyle,layout={'width':'3cm'})
    w_password = Password(description='password',value='',style=AutoWidthStyle,layout={'width':'5cm'})
    self.active = True
    def from_hw():
      with deactivate(self):
        w_password.value = self._cache.get((w_host.value,w_user.value),'')
    def from_pw(v): self._cache[w_host.value,w_user.value] = v
    for w in (w_host,w_user): w.observe((lambda c: from_hw()),'value')
    w_password.observe((lambda c: from_pw(c.new)),'value')
    def get_v(): return w_host.value,w_user.value,w_password.value
    def set_v(v):
      w_host.value,w_user.value = v[:2]
      if len(v)>2: w_password.value, = v[2:]
    self.get_v,self.set_v = get_v,set_v
    set_v((host,user))
    super().__init__((w_host,w_user,w_password))
  @property
  def value(self): return self.get_v()
  @value.setter
  def value(self,v): self.set_v(v)

#==================================================================================================
class SelectMultipleOrdered (VBox):
  r"""
Essentially like :class:`SelectMultiple` but preserves order of selection.
  """
#==================================================================================================
  options = traitlets.Any()
  rows = traitlets.Integer(min=0)
  value = traitlets.Tuple()
  def __init__(self,value=(),**ka):
    w_sel = SelectMultiple(**ka)
    w_echo = Text(disabled=True)
    w_echo.layout.width = w_sel.layout.width
    def reorder(L):
      L = set(L)
      for x in self.value:
        try: L.remove(x)
        except KeyError: continue
        else: yield x
      yield from L
    def update_sel(c): self.value = tuple(reorder(c.new))
    w_sel.observe(update_sel,'value')
    def update(c):
      if set(c.new)!= set(w_sel.value): w_sel.value = c.new
      w_echo.value = ';'.join(map(str,c.new))
    self.observe(update,'value')
    for name in 'options','rows':
      self.observe((lambda c,name=name: setattr(w_sel,name,c.new)),name)
    self.value = tuple(value)
    super().__init__((w_sel,w_echo))

#==================================================================================================
class Stable (Stack):
  r"""
Essentially a wrapper around widget *main* allowing it to be safely interacted with even when it is updated concurrently. Widget *main* should not have callbacks bound to its value, nor should have its value updated externally: binding callbacks to or modifying the value should be done on the wrapper. Widget *main* is then used only to edit a frozen value of the wrapper, which becomes its current value when submitted. Note: unfortunately, there is no way in ipywidget to cancel an edition once it has started (e.g. using the escape key).

:param main: the widget to wrap
:param fmt: a callable which returns a :class:`str` representation of any value of *main*
  """
#==================================================================================================
  def __init__(self,main:Widget,fmt:Callable[[Any],str]|str,**ka):
    self.add_traits(value=type(main).value)
    self.active = True
    fmt_ = fmt.format if isinstance(fmt,str) else fmt
    layout = {k:x for k in main.layout.keys if not k.startswith('_') and (x:=getattr(main.layout,k)) is not None}
    w_display = SimpleButton((lambda: show_editor()),description=fmt_(main.value),tooltip='Click to edit',layout=layout)
    w_editor = HBox(children=[main,SimpleButton((lambda: hide_editor()),icon='close')],layout={'border':'thin solid blue'})
    def show_editor():
      with deactivate(self): main.value = self.value; self.selected_index = 1
    def hide_editor(): self.selected_index = 0
    self.observe((lambda c: setattr(w_display,'description',fmt_(c.new))),'value')
    def from_main(v):
      if self.active: self.value = v; hide_editor()
    main.observe((lambda c: from_main(c.new)),'value')
    super().__init__([w_display,w_editor],selected_index=0,**ka)

#==================================================================================================
class SimpleButton (Button):
  r"""
Helper class to define buttons with a single callback and predefined default layout components.

:param callback: the callback for the button (optional)
:param ka: override the default layout components
  """
#==================================================================================================
  default_layout = {'width':'auto','padding':'0 1mm 0 1mm'}
  def __init__(self,callback:Callable[[],None]|None=None,layout=None,**ka):
    super().__init__(layout=(self.default_layout|(layout or {})),**ka)
    if callback is not None: self.on_click(lambda _: callback())

#==================================================================================================
class RoundRobinButton (Button):
  r"""
An instance of this class is a button with states which form a cycle. Each click of the button advances to the next state and triggers the action associated with that state.

:param colours: an assignment of a colour to each state
:param size: the width and height of the button in mm, either as a pair of :class:`int` or a :class:`str`, e.g. (5,2) or '5x2'
  """
#==================================================================================================
  on_click_: Callable[[str|int,Callable[[],None]],None]
  r"""Assigns a callback to a state (specified by its colour or index)"""
  def __init__(self,*colours:Sequence[str],layout:Mapping[str,str]={},shape:tuple[int,int]|str=(5,2),**ka):
    def step(b):
      nonlocal state
      state += 1; state %= N
      self.style.button_color = colours[state]
      for callback in callbacks[state]: callback()
    if isinstance(shape,str): w,h = (2,5) if shape=='T' else map(int,shape.split('x',1))
    else: assert all(isinstance(x,int) for x in shape); w,h = shape
    layout = {'width':f'{w}mm','height':f'{h}mm','padding':'0','border':'0'}|layout
    self.on_click_ = lambda colour,callback: callbacks[colour_[colour] if isinstance(colour,str) else colour].append(callback)
    self.style.button_color = colours[-1]
    state:int = -1; N = len(colours)
    colour_ = {c:k for k,c in enumerate(colours)}
    callbacks:list[list[Callable[[],None]]] = [[] for _ in colours]
    super().__init__(layout=layout,**ka)
    self.on_click(step)

#==================================================================================================
def activable(w:Widget,ini=True):
  r"""
A helper to add an attribute :attr:`active` to a :class:`Widget` instance *w*. Call function :func:`deactivate` to enter a context where :attr:`active` is :const:`False` for *w*. Then other widgets can condition their callbacks on whether *w* is active.
  """
#==================================================================================================
  assert isinstance(w,Widget)
  w.active = ini
  return w
@contextmanager
def deactivate(w):
  a = w.active
  w.active = False
  try: yield
  finally: w.active = a

#==================================================================================================
def setdefault_layout(*a,**ka):
#==================================================================================================
  if not ka: return
  for w in a:
    for k,v in ka.items():
      if getattr(w.layout,k) is None: setattr(w.layout,k,v)

#==================================================================================================
def setdefault_children_layout(w,f=(lambda w: True),**ka):
#==================================================================================================
  if not ka: return
  setdefault_layout(*filter(f,w.children),**ka)
  w.observe((lambda c: setdefault_layout(*filter(f,c.new),**ka)),'children')
  return w

#==================================================================================================
def mpl_figure(*a,asapp=True,**ka):
#==================================================================================================
  from matplotlib.pyplot import figure, close
  w = Output()
  with w: fig = figure(*a, **ka)
  if asapp:
    from IPython.display import display
    fig.app = a = app([w])
    a.on_close(lambda: close(fig))
    display(a)
    return fig
  else:
    return fig,w
