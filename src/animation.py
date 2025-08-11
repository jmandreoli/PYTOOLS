# File:                 animation.py
# Creation date:        2021-06-01
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Utilities for animations
#

from __future__ import annotations
from matplotlib import rcParams, get_backend
from matplotlib.pyplot import figure
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import MouseButton
try: from .ipywidgets import app, SimpleButton # so this works even if ipywidgets is not available
except: app = object
from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple, Literal
import logging; logger = logging.getLogger(__name__)

#==================================================================================================
def track_function(track:float|Sequence[float]|Callable[[float],Tuple[float,float]],stype:type=float)->Callable[[float],Tuple[float,float]]:
  r"""
Builds a track map from a specification *track*. A track map is a callable of one scalar input returning two scalar outputs (bounds of its track interval), or :const:`None` (when input is out of domain).

* Its domain must be an interval (not necessarily bounded) containing 0
* The mapping of a scalar to the lower bound of its track interval should be non decreasing and right-continuous.

The specification *track* can be the track map itself, returned as such after minimal checks. As a helper, *track* can also be

* a positive scalar, in which case the track map is based on intervals of constant length equal to that scalar, starting at 0 and never ending
* an increasing sequence of positive scalars, in which case the track map is based on intervals which are the consecutive pairs in that sequence (prefixed with 0)

:param track: the track map specification
:param stype: the type of scalars passed to the track map
  """
# ==================================================================================================
  assert stype in (int,float)
  if callable(track):
    track_func = track
    track_ = track(stype(0))
    assert track_ is not None and len(track_)==2 and isinstance((t0:=track_[0]),stype) and isinstance((t1:=track_[1]),stype) and t0<=0<t1 and track(t0) == track_ and ((t:=track(t1)[0]) is None or t == t1)
  elif isinstance(track,(int,float)):
    T = stype(track)
    assert T>0
    def track_func(x,T=T): x -= x%T; return x,x+T
  else:
    L = tuple(map(stype,[0,*track]))
    assert len(L)>1 and all(x<x_ for (x,x_) in zip(L[:-1],L[1:]))
    from bisect import bisect
    def track_func(x,L=L,imax=len(L)): i = bisect(L,x); return (L[i-1],L[i]) if i<imax else None
  return track_func

#==================================================================================================
class BaseControlledAnimation (FuncAnimation):
  r"""
An instance of this class is a controllable :mod:`matplotlib` animation, created by this constructor and attached to a :class:`Figure` instance stored as attribute :attr:`board`. The board creation is not performed in this class, so it must be performed in the constructor of a subclass, prior to invoking this constructor.

The speed of the animation is controlled by two parameters whose product determines the number of real milliseconds per simulation time unit (stu):

* parameter *frame_per_stu*: the number of frames per stu
* parameter *interval*: the number of real milliseconds per frame

So long as each frame takes less than *interval* to be constructed and displayed, it remains displayed until the end of the interval, and the next frame is constructed and displayed starting at the beginning of the next interval. A reasonable value for *interval* is ca. 40 ms/frame (flicker fusion threshold).

:param display: the function which takes a simulation time and displays the corresponding (closest) frame
:param track: track map decomposing the simulation domain into tracks (typically obtained by :func:`track_function`)
:param frame_per_stu: frame rate, in frames per simulation time (if :const:`None` use animation real time rate)
:param ka: passed to the :class:`FuncAnimation` constructor
  """
#==================================================================================================
  board: Figure
  r"""The figure on which to start the animation"""
  running: bool
  r"""The running state of the animation"""
  frame: int
  r"""The current frame displayed by the animation"""
  frame_per_stu: float
  r"""The frame rate, in frames per simulation time units"""
  track: Callable[[float],Tuple[float,float]]
  r"""The track map"""

  def __init__(self,display:Callable[[float],None],frame_per_stu:float=None,track=None,**ka):
    def frames():
      self.setval(0.)
      while True:
        self.set_running(False); yield self.frame
        while self.setval() is None: yield self.frame
    self.track = track_function(track,float)
    super().__init__(self.board,(lambda n: display(n/frame_per_stu)),frames,init_func=(lambda:None),repeat=False,cache_frame_data=False,**ka)
    if frame_per_stu is None: frame_per_stu = 1000./self._interval
    else: frame_per_stu = float(frame_per_stu); assert frame_per_stu > 0
    self.frame_per_stu = frame_per_stu

  def set_running(self,b:bool):
    r"""Sets the running state of the animation to *b*."""
    self.running = b
    (self.resume if b else self.pause)()
    self.show_running(b)
  def toggle_running(self): self.set_running(not self.running)

  def show_running(self,b:bool):
    raise NotImplementedError()

  def setval(self,v:float|None=None):
    r"""Sets the current frame to be displayed by the animation so that it represents simulation time *v*. Return a non :const:`None` value if *v* is out of the frame range. This implementation raises an error."""
    raise NotImplementedError()

  def panes(self,nrows:int=1,ncols:int=1,sharex:str|bool=False,sharey:str|bool=False,gridspec_kw:Mapping|None=None,gridlines:bool=True,aspect:str='equal',**ka):
    r"""
Generator of panes on the board.

:param ka: a dictionary of keyword arguments passed to the :meth:`matplotlib.figure.add_subplot` method of each part (key ``gridlines`` is also allowed and denotes whether gridlines should be displayed)
    """
    from numpy import zeros
    share:dict[str|bool,Callable[[int,int],tuple[int,int]]] = {
      'all': (lambda row,col: (0,0)),
      True:  (lambda row,col: (0,0)),   # alias
      'row': (lambda row,col: (row,0)),
      'col': (lambda row,col: (0,col)),
      'none':(lambda row,col: (-1,-1)),
      False: (lambda row,col: (-1,-1)), # alias
    }
    share_ = tuple((dim,share[s]) for dim,s in (('sharex',sharex),('sharey',sharey)))
    gridspec = self.board.add_gridspec(nrows=nrows,ncols=ncols,**(gridspec_kw or {}))
    axes = zeros((nrows,ncols),dtype=object); axes[...] = None
    for row in range(nrows):
      for col in range(ncols):
        ax = self.board.add_subplot(gridspec[row,col],**dict((dim,axes[s(row,col)]) for dim,s in share_),aspect=aspect,**ka)
        ax.grid(gridlines)
        axes[row,col] = ax
        yield ax
    raise Exception(f'Insufficient number of parts on this board: {nrows*ncols}')

#==================================================================================================
class IPYControlledAnimation (BaseControlledAnimation,app):
  r"""
A instance of this class is a player for :mod:`matplotlib` animations controlled from `mod:ipywidgets` widgets. The animation board is created by invoking :func:`figure` with arguments *fig_kw*.

:param fig_kw: configuration of the animation figure
:param ka: passed to the superclass accepting the corresponding keys
  """
#==================================================================================================
  def __init__(self,display:Callable[[float],None],fig_kw={},toolbar=(),**ka):
    from ipywidgets import Text, FloatText, IntSlider
    ctrack = [-1,0,1] # first (included), last (excluded), length (in frames) of current track
    def setval(v=None,d=None,submit=False):
      if v is None:
        if d is None: n = self.frame+1; d = n-ctrack[0]
        else: n = ctrack[0]+d
        v = n/self.frame_per_stu
      else: n = int(v*self.frame_per_stu); d = n-ctrack[0]
      w_clock.value = f'{v:.02f}'
      w_track_manager.active = False
      if d<0 or d>=ctrack[2]:
        track_ = self.track(v)
        if track_ is None: w_track_manager.active = True; return True
        ctrack[:2] = (int(v_*self.frame_per_stu) for v_ in track_)
        w_track_manager.max = ctrack[2] = ctrack[1]-ctrack[0]
        d = n-ctrack[0]
      w_track_manager.value = d
      w_track_manager.active = True
      self.frame = n
      if submit: display(v); board.canvas.draw_idle()
    def show_running(b): w_play_toggler.icon = 'pause' if b else 'play'
    self.show_running = show_running
    self.setval = setval
    # global design and widget definitions
    w_play_toggler = SimpleButton(icon='')
    w_track_manager = IntSlider(0,min=0,readout=False)
    w_track_manager.active = True
    w_clockb = SimpleButton(icon='stopwatch',tooltip='manually reset clock')
    w_clock = Text('',layout={'width':'1.6cm','padding':'0cm'},disabled=True)
    w_clock2 = FloatText(0,min=0,layout={'width':'1.6cm','padding':'0cm','display':'none'})
    w_clock2.active = False
    app.__init__(self,toolbar=(w_play_toggler,w_track_manager,w_clockb,w_clock,w_clock2,*toolbar))
    self.board = board = self.mpl_figure(**fig_kw)
    # callbacks
    w_play_toggler.on_click(lambda b: self.toggle_running())
    def on_clockb_clicked():
      if w_clock2.active: z = '','none',False
      else: w_clock2.value = self.frame/self.frame_per_stu; z = 'none','',True
      w_clock.layout.display,w_clock2.layout.display,w_clock2.active = z
    w_clockb.on_click(lambda b: on_clockb_clicked())
    def on_clock2_changed(v):
      if w_clock2.active:
        w_clock.layout.display,w_clock2.layout.display,w_clock2.active = '','none',False
        setval(v,submit=True)
    w_clock2.observe((lambda c: on_clock2_changed(c.new)),'value')
    w_track_manager.observe((lambda c: setval(d=c.new,submit=True) if w_track_manager.active else None),'value')
    super().__init__(display,**ka)

#==================================================================================================
class MPLControlledAnimation (BaseControlledAnimation):
  r"""
Instances of this class are players for :mod:`matplotlib` animations, controlled within matplotlib. The animation board is created by invoking :func:`figure` with arguments *fig_kw*.

:param fig_kw: configuration of the animation figure (excluding the toolbar)
:param tbsize: size (inches) of the toolbar as ((hsize(play-button),hsize(track-manager),hsize(clock)),vsize(toolbar))
:param ka: passed to the superclass
  """
# ==================================================================================================
  def __init__(self,display:Callable[[float],None],fig_kw={},tbsize=((.15,1.,.8),.15),**ka):
    ctrack = [-1.,0.,1.]
    def setval(v=None,d=+1,submit=False):
      if v is None: n = self.frame+d; v = n/self.frame_per_stu
      else: n = int(v*self.frame_per_stu)
      x = (v-ctrack[0])/ctrack[2]
      if x<0 or x>=1:
        track_ = self.track(v)
        if track_ is None: return True
        ctrack[:2] = track_; ctrack[2] = track_[1]-track_[0]
        x = (v-ctrack[0])/ctrack[2]
      if not edit_value: clock.set(text=f'{v:.02f}',color='k')
      track_manager.set(width=x)
      self.frame = n
      if submit: display(v); main.canvas.draw_idle()
    def show_running(b): play_toggler.set(text='II' if b else r'$\blacktriangleright$'); toolbar.canvas.draw_idle() # '⏸︎︎' if b else '⏵'
    self.setval = setval
    self.show_running = show_running
    # global design
    tbsize_ = sum(tbsize[0])
    figsize = fig_kw.pop('figsize',None)
    if figsize is None: figsize = rcParams['figure.figsize']
    figsize_ = list(figsize); figsize_[1] += tbsize[1]; figsize_[0] = max(figsize_[0],tbsize_)
    self.main = main = figure(figsize=figsize_,**fig_kw)
    r = tbsize[1],figsize[1]
    toolbar,self.board = main.subfigures(nrows=2,height_ratios=r)
    r = list(tbsize[0])
    r[1] += figsize_[0]-tbsize_
    g = {'width_ratios':r,'wspace':0.,'bottom':0.,'top':1.,'left':0.,'right':1.}
    axes = toolbar.subplots(ncols=3,subplot_kw={'xticks':(),'yticks':(),'navigate':False},gridspec_kw=g)
    # widget definitions
    ax = axes[0]
    play_toggler = ax.text(.5,.5,'',ha='center',va='center',transform=ax.transAxes)
    ax = axes[1]
    ax.set(xlim=(0,1),ylim=(0,1))
    track_manager = ax.add_patch(Rectangle((0.,0.),0.,1.,fill=True))
    ax = axes[2]
    clock = ax.text(.5,.5,'',ha='center',va='center',transform=ax.transAxes)
    # callbacks
    def on_button_press(ev):
      if ev.button == MouseButton.LEFT and ev.key is None:
        if ev.inaxes is play_toggler.axes: self.toggle_running()
        elif ev.inaxes is track_manager.axes: setval(ctrack[0]+ev.xdata*ctrack[2],submit=True)
    edit_value = ''
    def on_key_press(ev):
      nonlocal edit_value
      key = ev.key
      if key == 'left' or key == 'right':
        if ev.inaxes is not None: setval(d=(+1 if key=='right' else -1),submit=True)
      elif ev.inaxes is clock.axes:
        v,c = f'{self.frame/self.frame_per_stu:.02f}','k'
        if key=='enter':
          try: v_ = float(edit_value)
          except: return
          edit_value = ''
          if setval(v_,submit=True) is None: return
        else:
          if key=='escape': edit_value = ''
          elif key=='backspace':
            edit_value = edit_value[:-1]
            if edit_value: v,c = edit_value,'b'
          elif key in '0123456789' or (key=='.' and '.' not in edit_value):
            edit_value += key; v,c = edit_value,'b'
          else: return
        clock.set(text=v,color=c)
        toolbar.canvas.draw_idle()
    toolbar.canvas.mpl_connect('button_press_event',on_button_press)
    toolbar.canvas.mpl_connect('key_press_event',on_key_press)
    super().__init__(display,**ka)

ControlledAnimation = IPYControlledAnimation if get_backend()=='widget' else MPLControlledAnimation
