# File:                 ipywidgets.py
# Creation date:        2021-06-01
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Utilities for ipywidgets
#

from __future__ import annotations
from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple
import logging; logger = logging.getLogger(__name__)

import pandas, traceback
import sqlalchemy.engine
from functools import wraps
from pathlib import Path
import traitlets
from ipywidgets import Widget, Label, IntSlider, FloatSlider, Text, IntText, FloatText, BoundedIntText, BoundedFloatText, Password, HTML, Checkbox, Dropdown, Select, SelectMultiple, Button, Output, Tab, VBox, HBox, Box, Layout, Valid, Play, jslink

__all__ = 'app', 'seq_browser', 'file_browser', 'db_browser', 'hastrait_editor', 'animator', 'SelectMultipleOrdered', 'SimpleButton', 'setdefault_layout', 'setdefault_children_layout', 'AutoWidthStyle',

AutoWidthStyle = dict(description_width='auto')

#==================================================================================================
class app:
  r"""
An instance of this class is an app based on module :mod:`ipywidgets`, consisting of a toolbar, a console and additional child widgets.

:param toolbar: widgets added at the end of the toolbar (in addition to the default toolbar widgets)
:param children: additional children widgets appended to the main widget
  """
#==================================================================================================

  main: VBox
  r"""The main widget"""
  toolbar: HBox
  r"""The toolbar widget (first child of :attr:`main`)"""
  console: Output
  r"""A console terminal (just below :attr:`toolbar`)"""

  def __init__(self,children:Sequence[Widget]=(),toolbar:Sequence[Widget]=()):
    w_closeb = SimpleButton(icon='close',tooltip='Close app')
    self.console = console = Output()
    w_clearb = SimpleButton(icon='trash',tooltip='Clear console')
    w_console = HBox([w_clearb,console],layout={'border': 'thin solid', 'display': 'none'})
    self.toolbar = HBox([w_closeb,*toolbar])
    self.main = main = VBox([self.toolbar,w_console,*children])
    def _console_clear(b):
      from IPython.display import clear_output
      with console: clear_output()
    w_clearb.on_click(_console_clear)
    def _console_display(c): w_console.layout.display = '' if c.new else 'none'
    console.observe(_console_display,'outputs')
    _close_callbacks = [main.close,console.close]
    self.on_close = _close_callbacks.append
    def _closeall():
      for f in _close_callbacks: f()
    self._closeall = _closeall
    w_closeb.on_click(lambda b: _closeall())
    self._repr_mimebundle_ = main._repr_mimebundle_
  def __del__(self): self._closeall()

  def __enter__(self): return self.console.__enter__()
  def __exit__(self,*a): self.console.__exit__(*a)

  def mpl_figure(self,*a,**ka):
    r"""
Adds a :mod:`matplotlib` figure to this app.
    """
    from matplotlib.pyplot import close
    fig,w = mpl_figure(*a,asapp=False,**ka)
    self.main.children += (w,)
    self.on_close(lambda:close(fig))
    return fig

  def protect(self,f):
    r"""A decorator to redirect output and errors to the console."""
    @wraps(f)
    def F(*a,**ka):
      with self:
        try: f(*a,**ka)
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
    else: w_pager = IntSlider(start,min=1,max=P,description='page:',layout=dict(width='10cm'),readout=False,style=AutoWidthStyle)
    self.w_pager = w_pager
    w_pos = Label(f'',layout=dict(border='thin solid',padding='0mm 1mm 0mm 1mm'),style=AutoWidthStyle)
    super().__init__([w_out],toolbar=[w_pager,w_pos])
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
  def __init__(self,path:str|Path,start:int|float=None,step:int=50,track:bool=True,context:Tuple[int,int]=(10,5)):
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
    self.w_win = w_win = HTML(layout=dict(width='15cm',border='thin solid',overflow_x='auto',overflow_y='hidden'))
    self.w_ctrl = w_ctrl = IntSlider(start,min=0,max=fsize,step=step,readout=False,layout=dict(width=w_win.layout.width))
    w_toend = SimpleButton(icon='fast-forward',tooltip='Jump to end of file')
    w_tobeg = SimpleButton(icon='fast-backward',tooltip='Jump to beginning of file')
    w_pos = Label('')
    super().__init__([w_win],toolbar=[w_ctrl,w_pos,w_tobeg,w_toend])
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
      w_win.value = '<div style="white-space: pre; font-family: monospace; line-height:130%">{}</div>'.format('\n'.join(lines))
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
  def __init__(self,spec:str|sqlalchemy.engine.Engine|sqlalchemy.MetaData):
    from pandas import read_sql_query
    from sqlalchemy import select, func, MetaData, create_engine
    from sqlalchemy.engine import Engine
    from IPython.display import display, clear_output
    # content initialisation
    if isinstance(spec,MetaData):
      meta = spec
      if not meta.is_bound(): raise ValueError(f'Argument of type {MetaData} must be bound to an existing engine')
      no_table_msg = f'{MetaData} object has no table (perhaps it was not reflected)'
    else:
      if isinstance(spec,str): spec = create_engine(spec)
      elif not isinstance(spec,Engine):
        raise TypeError(f'Expected {str}|{Engine}|{MetaData}; Found {type(spec)}')
      meta = MetaData(bind=spec)
      meta.reflect(views=True)
      no_table_msg = 'Database is empty'
    if not meta.tables:
      self.main = Label(value=no_table_msg)
      return
    config = dict((name,db_browser_table(t)) for name,t in meta.tables.items())
    config_header = db_browser_table.header()
    engine = meta.bind
    # widget creation
    w_title = HTML('<div style="{}">{}</div>'.format(self.style,engine))
    w_table = Select(options=sorted(meta.tables.items()),rows=min(len(meta.tables),10),layout=dict(width='10cm'))
    w_schema = VBox([config_header,*(cfg.widget for cfg in config.values())])
    w_scol = SelectMultipleOrdered(layout=dict(width='6cm'))
    w_ordr = SelectMultipleOrdered(layout=dict(width='6cm'))
    w_detail = Tab((w_schema,w_scol,w_ordr),layout=dict(display='none'))
    for i,label in enumerate(('Column definitions','Column selection','Display ordering')): w_detail.set_title(i,label)
    w_detailb = SimpleButton(tooltip='toggle detail display (red border means some columns are hidden)',icon='info-circle')
    w_reloadb = SimpleButton(tooltip='reload table',icon='refresh')
    w_nsample = BoundedIntText(description='sample:',min=1,max=50,step=1,layout=dict(width='2.5cm'),style=AutoWidthStyle)
    w_size = Text('',disabled=True,tooltip='Number of rows',layout=dict(width='4cm'),style=AutoWidthStyle)
    w_offset = IntSlider(description='offset',min=0,step=1,layout=dict(width='10cm'),readout=False,style=AutoWidthStyle)
    w_out = Output()
    super().__init__([HBox([w_table,w_detailb]),w_detail,HBox([w_nsample,Label('/',style=AutoWidthStyle),w_size,w_offset,w_reloadb]),w_out],toolbar=[w_title])
    # event handling
    def size(table): # returns the size of a table
      try: return engine.execute(select([func.count()]).select_from(table)).fetchone()[0]
      except: return -1
    def sample(columns,nsample,offset,order=None)->pandas.DataFrame: # returns nsample row samples of columns ordered by order
      sql = select(columns,limit=nsample,offset=offset,order_by=(order or columns))
      r = read_sql_query(sql,engine)
      r.index = list(range(offset,offset+min(nsample,len(r))))
      return r
    def show():
      with w_out:
        clear_output(wait=True)
        display(sample(cfg.selected,cfg.nsample,cfg.offset,cfg.order))
    active = True
    cfg = None
    def set_table(c=None):
      nonlocal active,cfg
      if c is None: new = w_table.value
      else:
        new = c.new
        config[c.old.name].widget.layout.display = 'none'
      cfg = config[new.name]
      sz = size(w_table.value)
      w_scol.rows = w_ordr.rows = min(len(cfg.options),10)
      w_size.value = str(sz)
      cfg.widget.layout.display = 'inline'
      active = False
      try:
        w_scol.options = w_ordr.options = cfg.options
        w_scol.value, w_ordr.value = cfg.selected, cfg.order
        w_offset.max = max(sz-1,0)
        w_offset.value = cfg.offset
        w_nsample.value = cfg.nsample
      finally: active = True
      show()
    def set_scol(c):
      w_detailb.layout.border = 'none' if len(c.new) == len(cfg.options) else 'thin solid red'
      if active: cfg.selected = c.new; show()
    def set_ordr(c):
      if active: cfg.order = c.new; show()
    def set_offset(c):
      if active: cfg.offset = c.new; show()
    def set_nsample(c):
      if active: cfg.nsample = c.new; show()
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
    super().__init__([self.header,*(cfg.widget for cfg in config.values()),self.footer],toolbar=[w_resetb])
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
class animator (app):
  r"""
An instance of this class is an app to control abstract animations. An animation is any callback which can be passed a frame number, enumerated by the animation widgets. The set of frame numbers is split into a sequence of contiguous intervals called tracks. Parameter *track* is a function which returns, for each valid frame number, the bounds of its track interval, and :const:`None` for invalid frame numbers.

* If *track* is an :class:`int` instance, the intervals are of constant length *track*
* If *track* is an increasing sequence of :class:`int` instances, the intervals are the consecutive pairs in the sequence
* Otherwise, *track* must be a function of one :class:`int` input returning two :class:`int` outputs

A number of other widgets are accessible in addition to :attr:`main`: :attr:`toolbar`, :attr:`w_console`

:param track: track function
:param children: additional children widgets
:param toolbar: additional toolbar widgets
:param continuity: if :const:`True` (default), proceeds to the next track at the end of each track
:param ka: passed to the :class:`Play` constructor
  """
#==================================================================================================

  def __init__(self,children=(),toolbar=(),track:int|Sequence[int]|Callable[[int],Tuple[int,int]]=None,continuity:bool=True,**ka):
    from ipywidgets import Play, IntSlider, jslink
    track_func: Callable[[int],Tuple[int, int]]
    if callable(track):
      track_func = track
      track_ = track_func(0)
      assert track_ is not None and len(track_) == 2 and isinstance((t0:=track_[0]),int) and isinstance((t1:=track_[1]),int) and t0<=0<t1 and track_func(t0)==track_ and ((t:=track(t1)[0]) is None or t==t1)
    elif isinstance(track,int):
      assert track>0
      def track_func(x,T=track): x -= x%T; return x,x+T
    else:
      L = tuple((0,*track))
      assert len(L)>1 and all(x<x_ for (x,x_) in zip(L[:-1],L[1:]))
      from bisect import bisect
      def track_func(x,L=L,imax=len(L)): i = bisect(L,x); return (L[i-1],L[i]) if i<imax else None
    self.track = track_func
    self.w_play = w_play = Play(0,min=0,max=track_func(0)[1],show_repeat=False,**ka)
    w_ind = IntSlider(0,readout=False)
    super().__init__(children,toolbar=[w_play,w_ind,*toolbar])
    self.on_close(w_play.close)
    jslink((w_play,'value'),(w_ind,'value'))
    jslink((w_play,'min'),(w_ind,'min'))
    jslink((w_play,'max'),(w_ind,'max'))
    if continuity:
      def _continuity(c):
        if c.new == w_play.max and (tr:=track_func(c.new)) is not None:
          w_play.max = tr[1]; w_play.min = tr[0]
      w_play.observe(_continuity,'value')

  def add_clock(self,rate:float=None):
    r"""Adds a clock to the player, displaying (and allowing edition of) an index quantity proportional to the frame number. *rate* is the proportion in index per frame."""
    from ipywidgets import Text, FloatText
    if rate is None: rate = 1000/self.w_play.interval # frame/sec
    w_clockb = SimpleButton(icon='stopwatch',tooltip='manually reset clock')
    w_clock = Text('',layout=dict(width='1.6cm',padding='0cm'),disabled=True)
    w_clock2 = FloatText(0,min=0,layout=dict(width='1.6cm',padding='0cm',display='none'))
    w_clock2.active = False
    def tick(n): w_clock.value = f'{n/rate:.2f}'
    def set_clock():
      self.pause()
      w_clock2.value = self.value/rate
      w_clockb.layout.visibility,w_clock.layout.display,w_clock2.layout.display,w_clock2.active = 'hidden','none','',True
    w_clockb.on_click(lambda b: set_clock())
    def clock_set():
      if not w_clock2.active: return
      w_clockb.layout.visibility,w_clock.layout.display,w_clock2.layout.display,w_clock2.active = 'visible','','none',False
      self.value = int(w_clock2.value*rate)
    w_clock2.observe((lambda c: clock_set()),'value')
    self.toolbar.children += (w_clockb, w_clock, w_clock2)
    self.bind(tick)
    return self

  def bind(self,f:Callable[[int],None],trait:str='value'):
    r"""Assigns a callback *f* to a trait on the :attr:`w_play` widget."""
    @wraps(f)
    def F(c):
      try: f(c.new)
      except:
        with self: traceback.print_exc() # using the implicit console
        self.pause()
        raise
    self.w_play.observe(F,trait)
    return F
  def unbind(self,F,trait='value'):
    r"""Unassigns a callback assigned by method :meth:`bind`."""
    self.w_play.unobserve(F,trait)

  def pause(self):
    r"""Pauses the animation"""
    self.w_play._playing = False

  @property
  def value(self):
    r"""Value of the current frame number; uses the track function to redefine the current track interval when set"""
    return self.w_play.value
  @value.setter
  def value(self,n:int):
    tmin,tmax = self.track(n)
    w = self.w_play
    # by construction, the intervals (tmin,tmax) and (w.min,w.max) are either disjoint or identical
    if w.max > tmax: w.min = tmin; w.value = n; w.max = tmax
    else: w.max = tmax; w.value = n; w.min = tmin

#==================================================================================================
class Login (HBox):
  r"""
A widget holding a server,username,password. The passwords are protected and cached in memory.
  """
#==================================================================================================
  _cache:dict[Tuple[str,str],str] = {}
  def __init__(self,name,host='',user=''):
    self._host = Text(description=name,value=host,style={'description_width':'auto'},layout={'width':'8cm'})
    self._user = Text(description='user',value=user,style={'description_width':'auto'},layout={'width':'3cm'})
    self._password = Password(description='password',value='',style={'description_width':'auto'},layout={'width':'5cm'})
    super().__init__(children=(self._host,self._user,self._password))
  @property
  def host(self): return self._host.value
  @property
  def user(self): return self._user.value
  @property
  def password(self):
    host_user = self.host,self.user
    v = self._password.value
    if v: self._cache[host_user] = v
    else: v = self._password.value = self._cache.get(host_user,'')
    return v

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
    super().__init__((w_sel,w_echo))
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

#==================================================================================================
class SimpleButton (Button):
  r"""
Helper class to define buttons with a single callback and predefined default layout components.

:param callback: the callback for the button (optional)
:param ka: override the default layout components
  """
#==================================================================================================
  default_layout = dict(width='auto',padding='0 .1cm 0 .1cm')
  def __init__(self,callback:Callable[[],None]=None,**ka):
    layout = ka.pop('layout',{})
    super().__init__(layout=dict(self.default_layout,**layout),**ka)
    if callback is not None: self.on_click(lambda b: callback())

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
