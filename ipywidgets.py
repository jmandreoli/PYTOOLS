# File:                 ipywidgets.py
# Creation date:        2018-02-01
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Utilities for the ipywidgets package
#
r"""
:mod:`PYTOOLS.ipywidgets` --- IPython widgets utilities
=======================================================

This module provides basic utilities for IPython widgets (from module :mod:`ipywidgets`).

Available types and functions
-----------------------------
"""

import traitlets
from ipywidgets import Label, IntSlider, FloatSlider, Text, HTML, IntText, FloatText, BoundedIntText, BoundedFloatText, Checkbox, Dropdown, Select, SelectMultiple, FloatProgress, Button, Output, Tab, Accordion, VBox, HBox, Box, Layout, Valid
from . import ondemand, unid

#==================================================================================================
def seq_browser(D,start=1,pgsize=10):
  r"""
:param D: a sequence object
:type D: :class:`Sequence`
:param start: the index of the initial page
:type start: :class:`int`
:param pgsize: the size of the pages
:type pgsize: :class:`int`

Returns an :class:`ipywidgets.Widget` to browse a sequence object *D* page per page in IPython.
  """
#==================================================================================================
  from IPython.display import display, clear_output
  P = (len(D)-1)//pgsize + 1
  if P==1: return D
  def show(n):
    with w_out:
      clear_output(wait=True)
      display(D[(n-1)*pgsize:n*pgsize])
  w_out = Output()
  w_closeb = Button(icon='close',tooltip='Close browser',layout=dict(width='.5cm',padding='0cm'))
  w_pager = IntSlider(description='page',value=start,min=1,max=P,layout=dict(width='20cm'))
  w_main = VBox(children=(HBox(children=(w_closeb,w_pager)),w_out))
  w_pager.observe((lambda c: show(c.new)),'value')
  w_closeb.on_click(lambda b: w_main.close())
  show(w_pager.value)
  return w_main

#==================================================================================================
def file_browser(path,start=None,step=50,track=True,context=(10,5),**ka):
  r"""
:param path: a path to an existing file
:type path: :class:`Union[str,pathlib.Path]`
:param start: index of start pointed
:type start: :class:`Union[int,float]`
:param step: step size in bytes
:type step: :class:`int`
:param track: whether to track changes in the file
:type track: :class:`bool`
:param context: pair of number of lines before and after to display around current position
:type context: :class:`Tuple[int,int]`

Returns an :class:`ipywidgets.Widget` to browse the file at *path*, possibly while it expands. If *start* is :const:`None`, the start position is end-of-file. If *start* is of type :class:`int`, it denotes the exact start position in bytes. If *start* is of type :class:`float`, it must be between :const:`0.` and :const:`1.`, and the start position is set (approximatively) at that position relative to the whole file.
  """
#==================================================================================================
  from pathlib import Path
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
    lines[c] = '<span style="background-color: gray; color: white; border: thin solid gray;">{}</span>'.format(lines[c])
    w_win.value = '<div style="white-space: pre; font-family: monospace; line-height:130%">{}</div>'.format('\n'.join(lines))
  def toend(): w_ctrl.value = fsize
  def tobeg(): w_ctrl.value = 0
  def close():
    w_main.close()
    file.close()
    if observer is not None: observer.stop(); observer.join()
  def resetsize():
    nonlocal fsize
    atend = w_ctrl.value == fsize
    w_ctrl.max = fsize = path.stat().st_size
    if atend: w_ctrl.value = fsize
    else: setpos(w_ctrl.value)
  if isinstance(path,str): path = Path(path)
  else: assert isinstance(path,Path)
  path = path.absolute()
  file = path.open('rb')
  fsize = path.stat().st_size or 1
  if start is None: start = fsize
  elif isinstance(start,float):
    assert 0<=start and start<=1
    start = int(start*fsize)
  else:
    assert isinstance(start,int)
    if start<0:
      start += fsize
      if start<0: start = 0
  ka.setdefault('width','15cm')
  ka.setdefault('border','thin solid black')
  # widget creation
  w_win = HTML(layout=dict(overflow_x='auto',overflow_y='hidden',**ka))
  w_ctrl = IntSlider(min=0,max=fsize,step=step,value=start,layout=dict(width=w_win.layout.width))
  w_toend = Button(icon='angle-double-right',tooltip='Jump to end of file',layout=dict(width='.5cm',padding='0cm'))
  w_tobeg = Button(icon='angle-double-left',tooltip='Jump to beginning of file',layout=dict(width='.5cm',padding='0cm'))
  w_closeb = Button(icon='close',tooltip='Close browser',layout=dict(width='.5cm',padding='0cm'))
  w_main = VBox(children=(HBox(children=(w_ctrl,w_tobeg,w_toend,w_closeb)),w_win))
  # widget updaters
  w_ctrl.observe((lambda c: setpos(c.new)),'value')
  w_closeb.on_click(lambda b: close())
  w_tobeg.on_click(lambda b: tobeg())
  w_toend.on_click(lambda b: toend())
  setpos(start)
  if track:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    class MyHandler (FileSystemEventHandler):
      def __init__(s,p,f): super().__init__(); s.on_modified = (lambda evt: (f() if evt.src_path==p else None))
    observer = Observer()
    observer.schedule(MyHandler(str(path),resetsize),str(path.parent))
    observer.start()
  else: observer=None
  return w_main

#==================================================================================================
def db_browser(spec):
  r"""
:param spec: an sqlalchemy url or engine or metadata structure, defining the database to explore
:type spec: :class:`Union[str,sqlalchemy.engine.Engine,sqlalchemy.sql.schema.MetaData]`

Returns an :class:`ipywidgets.Widget` to explore a database specified by *spec*. If a metadata structure is specified, it must be bound to an existing engine and reflected.
  """
#==================================================================================================
  from functools import lru_cache
  from pandas import read_sql_query
  from sqlalchemy import select, func, MetaData, create_engine
  from sqlalchemy.engine import Engine
  from IPython.display import display, clear_output
  def size(table): # returns the size of .table.
    try: return engine.execute(select([func.count()]).select_from(table)).fetchone()[0]
    except: return -1
  def sample(columns,nsample,offset,order=None): # returns .nsample. row samples of .columns. ordered by .order.
    sql = select(columns,limit=nsample,offset=offset,order_by=(order or columns))
    r = read_sql_query(sql,engine)
    r.index = list(range(offset,offset+min(nsample,len(r))))
    return r
  if isinstance(spec,MetaData):
    meta = spec
    if not meta.is_bound(): raise ValueError('Argument of type {} must be bound to an existing engine'.format(MetaData))
    no_table_msg = '{} object has no table (perhaps it was not reflected)'.format(MetaData)
  else:
    if isinstance(spec,str): spec = create_engine(spec)
    elif not isinstance(spec,Engine):
      raise TypeError('Expected {}|{}|{}; Found {}'.format(str,Engine,MetaData,type(spec)))
    meta = MetaData(bind=spec)
    meta.reflect(views=True)
    no_table_msg = 'Database is empty'
  if not meta.tables: return no_table_msg
  config = db_browser_initconfig(meta.tables)
  engine = meta.bind
  # widget creation
  w_title = HTML('<div style="{}">{}</div>'.format(db_browser.style['title'],engine))
  w_table = Select(options=sorted(meta.tables.items()),layout=dict(width='10cm'))
  w_size = Text(value='',tooltip='Number of rows',disabled=True,layout=dict(width='2cm'))
  w_closeb = Button(icon='close',tooltip='Close browser',layout=dict(width='.5cm',padding='0cm'))
  w_schema = VBox()
  w_scol = SelectMultipleOrdered(layout=dict(flex_flow='column'))
  w_ordr = SelectMultipleOrdered(layout=dict(flex_flow='column'))
  w_detail = Tab(children=(w_schema,w_scol,w_ordr),layout=dict(display='none'))
  for i,label in enumerate(('Column definitions','Column selection','Column ordering')): w_detail.set_title(i,label)
  w_detailb = Button(tooltip='toggle detail display (red border means some columns are hidden)',icon='info-circle',layout=dict(width='.4cm',padding='0'))
  w_reloadb = Button(tooltip='reload table',icon='refresh',layout=dict(width='.4cm',padding='0'))
  w_offset = IntSlider(description='offset',min=0,step=1,layout=dict(width='12cm'))
  w_nsample = IntSlider(description='nsample',min=1,max=50,step=1,layout=dict(width='10cm'))
  w_out = Output()
  w_main = VBox(children=(w_title,HBox(children=(w_table,w_size,w_detailb,w_reloadb,w_closeb)),w_detail,HBox(children=(w_offset,w_nsample,)),w_out))
  # widget updaters
  active = True
  cfg = None
  def show():
    with w_out:
      clear_output(wait=True)
      display(sample(cfg.selected,cfg.nsample,cfg.offset,cfg.order))
  def set_table(c=None):
    nonlocal active,cfg
    cfg = config[(w_table.value if c is None else c.new).name]
    sz = size(w_table.value)
    w_scol.rows = w_ordr.rows = min(len(cfg.options),20)
    w_size.value = str(sz)
    w_schema.children = cfg.schema
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
  def toggledetail(inv={'inline':'none','none':'inline'}): w_detail.layout.display = inv[w_detail.layout.display]
  # callback attachments
  w_detailb.on_click(lambda b: toggledetail())
  w_reloadb.on_click(lambda b: show())
  w_closeb.on_click(lambda b: w_main.close())
  w_table.observe(set_table,'value')
  w_offset.observe(set_offset,'value')
  w_nsample.observe(set_nsample,'value')
  w_scol.observe(set_scol,'value')
  w_ordr.observe(set_ordr,'value')
  # initialisation
  set_table()
  return w_main

db_browser.style = dict(
  schema='background-color: navy; color: white; font-size: x-small; border: thin solid white; text-align: center;',
  title='background-color:gray; color:white; font-weight:bold; padding:.2cm',
)

#--------------------------------------------------------------------------------------------------
def db_browser_initconfig(tables):
#--------------------------------------------------------------------------------------------------
  class Tconf:
    __slots__ = 'options','schema','selected','order','offset','nsample'
    def __init__(s,table,schemag=None,offset=0,nsample=5):
      s.options = [(c.name,c) for c in table.columns]
      s.schema = list(schemag(table.columns))
      s.selected = tuple(table.columns)
      s.order = tuple(table.primary_key)
      s.offset = offset
      s.nsample = nsample
  wstr = (lambda x: Text(value=x))
  wbool = (lambda x: Checkbox(value=x,style=dict(description_width='initial')))
  wany = (lambda x: Text(value=str(x)))
  schema = ( # this is the schema of schemas!
    ('name',wstr,'name',Layout(width='4cm')),
    ('type',wany,'type',Layout(width='4cm')),
    ('primary_key',wbool,'P',Layout(width='.9cm',margin='0cm',top='.5mm')),
    ('nullable',wbool,'N',Layout(width='.9cm',margin='0cm',top='.5mm')),
    ('unique',wbool,'U',Layout(width='.9cm',margin='0cm',top='.5mm')),
    ('default',wany,'default',Layout(width='4cm')),
    ('constraints',wany,'constraints',Layout(width='4cm')),
    ('foreign_keys',wany,'foreign',Layout(width='4cm')),
  )
  def schema_rows(cols):
    def widget(c,x):
      try: w = x[1](getattr(c,x[0]))
      except: w =  Valid(value=False)
      w.layout = x[3]
      w.disabled = True
      return w
    yield schema_hrow
    for c in cols:
      yield HBox(children=[widget(c,x) for x in schema])
  style = db_browser.style['schema']
  schema_hrow = HBox(children=[HTML(value='<div style="{}" title="{}">{}</div>'.format(style,x[0],x[2]),layout=x[3]) for x in schema])
  return dict((name,Tconf(t,schemag=schema_rows)) for name,t in tables.items())

#==================================================================================================
def hastrait_editor(target,default_trait_layout=dict(width='15cm'),**ka):
#==================================================================================================
  r"""
:param target: a structure with traits
:type target: :class:`traitlets.HasTraits`
:param default_trait_layout: a dictionary of layout items
:type default_trait_layout: :class:`Dict[str,str]`

Returns an :class:`ipywidgets.Widget` to edit a traitlets structure. The construction of the widget is directed by metadata which must be associated to each editable trait in *target* under the key ``widget``. This is either a callable, which produces a widget for that trait, or a :class:`dict`, passed as keyword arguments to a widget guesser. Guessing is based on the trait class. Each property in *default_trait_layout* is applied to all the trait widget layouts which assign a :const:`None` value to that property. The overall editor widget can be further customised using the following attributes:

.. attribute:: toolbar,header,footer

   They can be modified freely. The :attr:`toolbar` widget (a :class:`Toolbar` instance) appears immediately after the trait widgets, followed by the :attr:`footer` widget (a :class:`ipywidgets.VBox` instance). The :attr:`header` widget (a :class:`ipywidgets.VBox` instance) appears immediately before the trait widgets. A close and reset buttons are present in the toolbar by default.

.. attribute:: label_layout

   A shared :class:`ipywidgets.Layout` instance formatting the label of all the traits. Modifications of this layout therefore applies to all trait labels.
  """
  from functools import partial
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
    return dict(slider=slider,text=btext)[m][typ](min=vmin,max=vmax,**ws) if vmin is not None and vmax is not None else text[typ](**ws)
  def g_selector(t,ws):
    mode = dict(select=Select,dropdown=Dropdown)
    opt = ws.get('options')
    ws['options'] = [(str(v),v) for v in t.values] if opt is None else list(zip(opt,t.values))
    m = ws.pop('mode','select')
    return mode[m](**ws)
  def g_mselector(t,ws):
    opt = ws.get('options')
    ws['options'] = [(str(v),v) for v in t._trait.values] if opt is None else list(zip(opt,t._trait.values))
    ordr = ws.pop('ordered',False)
    return (SelectMultipleOrdered if ordr else SelectMultiple)(**ws)
  def updw(w,x): w.value = x
  def upda(name,w,x):
    try: setattr(target,name,x)
    except: w.value = getattr(target,name)
  def rows(D):
    for name,t in target.traits().items():
      val = getattr(target,name)
      ws = t.metadata.get('widget')
      if ws is None: continue
      if isinstance(ws,dict):
        ws = dict(ws)
        if isinstance(t,traitlets.Integer): w = g_numeric(t,ws,int)
        elif isinstance(t,traitlets.Float): w = g_numeric(t,ws,float)
        elif isinstance(t,traitlets.Bool): w = Checkbox(**ws)
        elif isinstance(t,traitlets.Enum): w = g_selector(t,ws)
        elif isinstance(t,traitlets.List) and isinstance(t._trait,traitlets.Enum): w = g_mselector(t,ws)
        else: raise Exception('Cannot guess widget')
      else: w = ws()
      apply_default_layout(w,**default_trait_layout)
      rbutton = Button(icon='undo',tooltip='Reset to default',layout=dict(width='0.5cm',padding='0cm'))
      label = HTML('<span title="{}">{}</span>'.format(str(t.help),name),layout=label_layout)
      rbutton.on_click(lambda but,w=w,x=val: updw(w,x))
      w.observe((lambda c,name=name,w=w: upda(name,w,c.new)),'value')
      target.observe((lambda c,w=w: updw(w,c.new)),name)
      w.value = D[name] = val
      yield HBox(children=(rbutton,label,w))
  def reset(data):
    for k,v in data.items(): setattr(target,k,v)
  initial = {}
  w_main = VBox(**ka)
  w_main.label_layout = label_layout = Layout(width='2cm',padding='0cm',align_self='flex-start')
  w_main.toolbar = tb = Toolbar()
  tb.add(w_main.close,icon='close',tooltip='Close browser',layout=dict(width='.5cm',padding='0cm'))
  tb.add(partial(reset,initial),icon='undo',description='reset',layout=dict(width='1.8cm'))
  w_main.header = VBox()
  w_main.footer = VBox()
  w_main.children = (w_main.header,)+tuple(rows(initial))+(w_main.toolbar,w_main.footer)
  return w_main

#==================================================================================================
def progress_reporter(progress,interval=None,maxerror=3):
#==================================================================================================
  """
:param progress: returns the progress value (between 0. and 1.) when invoked
:type progress: :class:`Callable[[],float]`
:param interval: duration in seconds between two progress reports
:type interval: :class:`float`

Returns a :class:`ipywidgets.Widget` reporting on the progress of some activity. The *progress* callable can, in addition to returning the current progress value, display information about the progress, using functions :func:`print` and :func:`IPython.display.display`.
  """
  import traceback
  from threading import Timer
  from IPython.display import clear_output
  error = 0
  stopped = False
  def update(repeat=True):
    nonlocal error
    if stopped: return
    with w_out:
      clear_output(wait=True)
      try:
        p = progress()
        w_progressbar.value = p
        w_progress.value = '{:.1%}'.format(p)
        error = 0
      except:
        error += 1
        print(traceback.format_exc())
    w_out.layout.border = 'thin solid red' if error else 'thin solid blue'
    if error > maxerror: w_out.layout.border = 'thick solid red'
    elif repeat: Timer((1. if error else interval),update).start()
  def close():
    nonlocal stopped
    stopped = True
    w_main.close()
  w_closeb = Button(icon='close',layout=dict(width='.5cm',padding='0cm'),tooltip='Close progress report')
  w_refreshb = Button(icon='refresh',layout=dict(width='.5cm',padding='0cm'),tooltip='Refresh progress report')
  w_progress = Label()
  w_progressbar = FloatProgress(value=0.,min=0.,max=1.,step=.01)
  w_out = Output()
  w_main = VBox((HBox((w_closeb,w_progressbar,w_progress,w_refreshb)),w_out))
  w_closeb.on_click(lambda b: close())
  w_refreshb.on_click(lambda b: update(False))
  update(interval is not None)
  return w_main

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
    w_display = Text(disabled=True)
    w_display.layout.width = w_sel.layout.width
    super().__init__(children=(w_sel,w_display))
    def g(L):
      L = set(L)
      for x in self.value:
        try: L.remove(x)
        except KeyError: continue
        else: yield x
      yield from L
    def update_sel(c): self.value = tuple(g(c.new))
    w_sel.observe(update_sel,'value')
    def update(c):
      if set(c.new)!= set(w_sel.value): w_sel.value = c.new
      w_display.value = ';'.join(map(str,c.new))
    self.observe(update,'value')
    for name in 'options','rows':
      self.observe((lambda c,name=name: setattr(w_sel,name,c.new)),name)
    self.value = tuple(value)

#==================================================================================================
class Toolbar(HBox):
  r"""
An instance of this class is a widget toolbar of buttons.
  """
#==================================================================================================
  def __init__(self,**ka):
    self.b_layout = {}
    super().__init__(**ka)

  def add(self,callback,**ka):
    r"""
:param callback: a function to call when the button is clicked
:type callback: :class:`Callable[Tuple[],Any]`

Adds a new button to the toolbar. If keyword arguments are present, they are passed to the button constructor. The button widget is returned.
    """
    from collections import ChainMap
    b = Button(**ka)
    apply_default_layout(b,**self.b_layout)
    b.on_click(lambda b: callback())
    self.children += (b,)
    return b

  def common_button_layout(self,**ka):
    r"""
Each property in *ka* is applied to all the button layouts of this toolbar (past and future), which assign a :const:`None` value to that property.
    """
    apply_default_layout(*self.children,**ka)
    self.b_layout.update(ka)

#==================================================================================================
def apply_default_layout(*a,**ka):
#==================================================================================================
  for w in a:
    for k,v in ka.items():
      if getattr(w.layout,k) is None: setattr(w.layout,k,v)
