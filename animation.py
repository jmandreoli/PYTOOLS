# File:                 animation.py
# Creation date:        2021-06-01
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Utilities for animations
#
r"""
:mod:`PYTOOLS.animation` --- Animation utilities
================================================

This module provides basic utilities for animations (in matplotlib), controlled either through :mod:`ipywidgets` widgets or home grown :mod:`matplotlib` widgets.

Available types and functions
-----------------------------
"""

from __future__ import annotations

from typing import Any, Union, Callable, Iterable, Mapping, Sequence, Tuple
import logging; logger = logging.getLogger(__name__)

import traitlets, traceback
from functools import wraps, partial
from matplotlib import rcParams
from matplotlib.pyplot import figure
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
try: from .ipywidgets import app, SimpleButton # so this owrks even if ipywidgets is not available
except: app = object

#==================================================================================================
class widget_animation_player (app):
  r"""
An instance of this class holds a widget :attr:`main` to control abstract animations. An animation is any callback which can be passed a frame number, enumerated by the animation widgets. The set of frame numbers is split into a sequence of contiguous intervals called tracks. Parameter *track* is a function which returns, for each valid frame number, the bounds of its track interval, and :const:`None` for invalid frame numbers.

* If *track* is an :class:`int` instance, the intervals are of constant length *track*
* If *track* is an increasing sequence of :class:`int` instances, the intervals are the consecutive pairs in the sequence
* Otherwise, *track* must be a function of one :class:`int` input returning two :class:`int` outputs

A number of other widgets are accessible in addition to :attr:`main`: :attr:`toolbar`, :attr:`w_console`

:param track: track function
:param children: passed as children of the :attr:`main` widget, which is a :class:`VBox`
:param continuity: if :const:`True` (default), proceeds to the next track at the end of each track
:param ka: passed to the :class:`Play` constructor
  """
#==================================================================================================
  def __init__(self,children=(),toolbar=(),track:Union[int,Sequence[int],Callable[[int],Tuple[int,int]]]=None,continuity:bool=True,**ka):
    from ipywidgets import Play, IntSlider, jslink
    if not callable(track):
      if isinstance(track,int):
        assert track>0
        def track(n,N=track): m = n-n%N; return m,m+N
      else:
        L = tuple((0,*track))
        assert len(L)>1 and all(isinstance(n,int) for n in L) and all(n<n_ for (n,n_) in zip(L[:-1],L[1:]))
        from bisect import bisect
        def track(n,L=L,imax=len(L)): i = bisect(L,n); return (L[i-1],L[i]) if i<imax else None
    self.track = track
    self.w_play = w_play = Play(0,min=0,max=track(0)[1],show_repeat=False,**ka)
    w_ind = IntSlider(0,readout=False)
    super().__init__(children,toolbar=[w_play,w_ind,*toolbar])
    self.on_close(w_play.close)
    jslink((w_play,'value'),(w_ind,'value'))
    jslink((w_play,'min'),(w_ind,'min'))
    jslink((w_play,'max'),(w_ind,'max'))
    if continuity:
      def _continuity(c):
        if c.new == w_play.max and (tr:=track(c.new)) is not None:
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

class Button:
  r"""
Instances of this class are press buttons in :mod:`matplotlib`.
  """
  def __init__(self,ax:Axes,label:str):
    self.enabled = True; self.ax = ax; ax.set(xticks=(),yticks=())
    ax.text(.5,.5,label,ha='center',va='center',transform=ax.transAxes)
    def clicked(f,ev): return f(ev) if self.enabled and ev.inaxes is ax else None
    self.on_clicked = lambda f: ax.figure.canvas.mpl_connect('button_press_event',partial(clicked,f))
  def set_enabled(self,b:bool=True): self.enabled = b

class Slider:
  r"""
Instances of this class are normalised sliders in :mod:`matplotlib`. Range is always 0-1.
  """
  def __init__(self,ax:Axes,**ka):
    self.ax = ax
    self.enabled = True; ax.set(xticks=(),yticks=())
    canvas = ax.figure.canvas
    pix2pc = ax.transAxes.inverted().transform
    rect = ax.add_patch(Rectangle((0.,0.),0.,1.,fill=True,transform=ax.transAxes,**ka))
    def set_val(v): rect.set(width=v); canvas.draw_idle()
    self.set_val = set_val
    def set_val_from_click(ev):
      if ev.button == MouseButton.LEFT and ev.key is None and not ev.dblclick and self.enabled and ev.inaxes is ax:
        set_val(pix2pc((ev.x,ev.y))[0])
    canvas.mpl_connect('button_press_event',set_val_from_click)
    def clicked(f,ev): return f(ev,pix2pc((ev.x,ev.y))[0]) if self.enabled and ev.inaxes is ax else None
    self.on_clicked = lambda f: canvas.mpl_connect('button_press_event', partial(clicked,f))
  def set_enabled(self,b:bool=True): self.enabled = b

class TextBox:
  r"""
Instances of this class are basic text boxes in :mod:`matplotlib`.

:param v: initial value
:param format: from value to string representation (can be a format string)
:param parse: from string representation to value
  """
  def __init__(self,ax,v,format:Union[str,Callable[[Any],str]]=(lambda v:v),parse:Callable[[str],Any]=(lambda v:v),**ka):
    self.ax = ax; self.value = v
    format = format.format if isinstance(format,str) else format
    self.enabled = True; self.active = False; ax.set(xticks=(),yticks=())
    submit_callbacks = []
    canvas = ax.figure.canvas
    self.content = content = ax.text(.5,.5,format(v),ha='center',va='center',transform=ax.transAxes,**ka)
    content.colour = content.get_color()
    def set_val(v):
      self.value = v
      if not self.active or not current: content.set(text=format(v)); canvas.draw_idle()
    self.set_val = set_val
    current = ''
    def keypress(ev):
      nonlocal current
      if self.enabled and self.active and ev.inaxes == ax:
        key = ev.key
        if (enter:=(key == 'enter')) or key == 'escape':
          if enter:
            try:
              self.value = v = parse(current)
              for f in submit_callbacks: f(v)
            except: return
          content.set_text(format(v))
          current = ''
          self.active = False; content.set(color=content.colour)
        else:
          if key == 'backspace': new = current[:-1]
          else: new = current+key
          current = new
          content.set_text(current or format(self.value))
        canvas.draw_idle()
    canvas.mpl_connect('key_press_event',keypress)
    def activate_from_click(ev):
      if ev.button == MouseButton.LEFT and ev.key is None and not ev.dblclick and self.enabled and ev.inaxes is ax:
        self.active = True; content.set(color='r'); canvas.draw_idle()
    canvas.mpl_connect('button_press_event',activate_from_click)
    self.on_submit = submit_callbacks.append
  def set_enabled(self,b:bool=True): self.enabled = b

class mpl_animation_player (traitlets.HasTraits):
  r"""
Instances of this class are animation players for :mod:`matplotlib`.

:param figsize: size of the content figure (excluding the toolbar)
:param tbsize: size of the toolbar
:param interval: passed to the animation
:param track: track generator
:param rate: coefficient to multiply by frame number for clock display (default: sec per frames)
:param ka: passed to the :func:`matplotlib.pyplot.figure` constructor
  """
  value_ = traitlets.Integer()
  def __init__(self,figsize=None,interval=None,track=None,rate=None,tbsize=((.15,1.,.8),.15),**ka):
    if rate is None: rate = 1000./interval
    if not callable(track):
      if isinstance(track,int):
        assert track>0
        def track(n,N=track): m = n-n%N; return m,m+N
      else:
        L = tuple((0,*track))
        assert len(L)>1 and all(isinstance(n,int) for n in L) and all(n<n_ for (n,n_) in zip(L[:-1],L[1:]))
        from bisect import bisect
        def track(n,L=L,imax=len(L)): i = bisect(L,n); return (L[i-1],L[i]) if i<imax else None
    self.track = track
    tbsize_ = sum(tbsize[0])
    if figsize is None: figsize = rcParams['figure.figsize']
    figsize_ = list(figsize); figsize_[1] += tbsize[1]; figsize_[0] = max(figsize_[0],tbsize_)
    self.main = main = figure(figsize=figsize_,**ka)
    r = tbsize[1],figsize[1]
    g = main.add_gridspec(nrows=2,ncols=1,height_ratios=r,bottom=0.,top=1.,left=0.,right=1.)
    toolbar = main.add_subfigure(g[0])
    for hid in (toolbar.canvas.manager.key_press_handler_id,toolbar.canvas.manager.button_press_handler_id):
      if hid is not None: toolbar.canvas.mpl_disconnect(hid)
    self.board = main.add_subfigure(g[1])
    r = list(tbsize[0])
    r[1] += figsize_[0]-tbsize_
    g = toolbar.add_gridspec(nrows=1,ncols=3,width_ratios=r,wspace=0.,bottom=0.,top=1.,left=0.,right=1.)
    # widget definition
    self.playb = playb = Button(toolbar.add_subplot(g[0]),'>')
    self.slider = slider = Slider(toolbar.add_subplot(g[1]))
    self.clock = clock = TextBox(toolbar.add_subplot(g[2]),0.,format='{:.02f}',parse=float)
    # main callback
    self.offset = 0; self.running = True; tmin = tmax = tspan = 0
    def reset_track(v):
      nonlocal tmin,tmax,tspan
      tr = track(v)
      if tr is None: anim.pause(); self.running = False; return
      if tr[1] != tmax: tmin,tmax = tr; tspan = tmax-tmin
      return v
    self.reset_track = reset_track
    def advance(n):
      v = n+self.offset
      if v>=tmax: reset_track(v)
      self.value_ = v
    self.anim = anim = FuncAnimation(main,advance,init_func=(lambda:None),repeat=False,interval=interval)
    #anim.pause(); self.running = False; # ignored!
    # callbacks
    def set_val(v): self.offset += v-self.value_; self.value_ = v
    def tick(c): slider.set_val((c.new-tmin)/tspan); clock.set_val(c.new/rate)
    def slider_set(ev,v):
      if ev.button == MouseButton.LEFT and ev.key is None and not ev.dblclick: set_val(int(tmin+tspan*v))
    def clock_set(v):
      v = int(v*rate)
      if reset_track(v) is None: return
      set_val(v)
    def playb_clicked(ev):
      if ev.button == MouseButton.LEFT and ev.key is None and not ev.dblclick:
        if self.running: anim.pause(); self.running = False
        else: anim.resume(); self.running = True
    self.observe(tick,'value_')
    playb.on_clicked(playb_clicked)
    slider.on_clicked(slider_set)
    clock.on_submit(clock_set)
  def set_enabled(self,b:bool=True):
    for a in (self.playb,self.slider,self.clock): a.set_enabled(b)
  def on_close(self,f):
    self.main.canvas.mpl_connect('close_event',(lambda ev: f()))
  def bind(self,f):
    @wraps(f)
    def F(c):
      try: f(c.new)
      except: self.pause(); raise
    self.observe(F,'value_')
    return F
  def unbind(self,f):
    self.unobserve(f,'value_')
  def pause(self):
    self.anim.pause(); self.running = False
  @property
  def value(self):
    r"""Value of the current frame number; uses the track function to redefine the current track interval when set"""
    return self.value_
  @value.setter
  def value(self,v:int):
    if self.reset_track(v) is None: raise ValueError(v)
    self.offset += v-self.value_; self.value_ = v
  def __enter__(self): pass
  def __exit__(self,*a): pass
  def _ipython_display_(self): return repr(self)
