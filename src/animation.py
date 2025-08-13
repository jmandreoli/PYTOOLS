# File:                 animation.py
# Creation date:        2021-06-01
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Utilities for animations
#

from __future__ import annotations

import matplotlib.axes
from matplotlib import rcParams, get_backend
from matplotlib.pyplot import figure
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
try: from .ipywidgets import app # so this works even if ipywidgets is not available
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
  interrupt: int|None
  r"""The current frame displayed by the animation"""
  frame_per_stu: float
  r"""The frame rate, in frames per simulation time units"""
  track: Callable[[float],Tuple[float,float]]
  r"""The track map"""
  show_running:Callable[[bool],None]
  r"""Show the running state of the animation"""
  show_control:Callable[[int,bool],None]
  r"""Show the control state of the animation (current frame index,whether current track was modified)"""
  pause_after_interrupt: bool

  def __init__(self,display:Callable[[float],None],frame_per_stu:float|None=None,track=None,**ka):
    def frames():
      n = 0; self.interrupt = None
      while True:
        self.pause(); yield None
        while True:
          if self.interrupt is None:
            if self.accept(n_:=n+1): n = n_
            else: break
          else:
            n = self.interrupt; self.interrupt = None
            if self.pause_after_interrupt: self.pause()
          yield n
    self.ctrack = [-1,0,1] # first frame (included), last frame (excluded), length (in frames) of current track
    self.track = track_function(track,float)
    if frame_per_stu is None: frame_per_stu = 1000./self._interval
    else: frame_per_stu = float(frame_per_stu); assert frame_per_stu > 0
    self.frame_per_stu = frame_per_stu
    super().__init__(
      self.board,
      (lambda n: (None if n is None else display(n/frame_per_stu))),
      frames,init_func=(lambda:display(0.)),
      repeat=False,cache_frame_data=False,**ka
    )

  def accept(self,n:int)->bool:
    d = n-self.ctrack[0]; new_track = d<0 or d>=self.ctrack[2]
    if new_track:
      if (tr:=self.track(n/self.frame_per_stu)) is None: return False
      self.ctrack[:2] = (int(v_*self.frame_per_stu) for v_ in tr)
      self.ctrack[2] = self.ctrack[1]-self.ctrack[0]
    self.show_control(n,new_track)
    return True

  def resume(self): super().resume(); self.pause_after_interrupt = False; self.show_running(True)
  def resume_at(self,n:int): super().resume(); self.interrupt = n
  def pause(self): super().pause(); self.pause_after_interrupt = True; self.show_running(False)

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
  active:bool
  def __init__(self,*args,fig_kw={},toolbar=(),**kwargs):
    from .ipywidgets import deactivate, SimpleButton, Stable
    from ipywidgets import FloatText, IntSlider, Label
    def show_control(n:int,new_track:bool=False):
      with deactivate(self):
        if new_track:
          for w,v in zip(w_clock_bounds,self.ctrack[:2]): w.value = f'{v/self.frame_per_stu:.2f}'
          w_track_manager.max = self.ctrack[2]
        w_track_manager.value,w_clock.value = n-self.ctrack[0],n/self.frame_per_stu
    self.show_control = show_control
    def show_running(b,D={True:'pause',False:'play'}): w_play_toggler.icon = D[b]
    self.show_running = show_running
    # global design and widget definitions
    w_play_toggler = SimpleButton((lambda D={'pause':self.pause,'play':self.resume}:D[w_play_toggler.icon]()),icon='')
    w_track_manager = IntSlider(0,min=0,readout=False)
    w_clock_bounds = [Label('',style={'font_size':'xx-small'}) for _ in range(2)]
    w_clock_edit = FloatText(0.,min=0.,layout={'width':'1.6cm','padding':'0cm'})
    w_clock = Stable(w_clock_edit,'{:.2f}'.format)
    app.__init__(self,toolbar=(w_play_toggler,w_clock_bounds[0],w_track_manager,w_clock_bounds[1],w_clock,*toolbar))
    self.board = self.mpl_figure(**fig_kw)
    self.active = True
    # callbacks
    def from_track(d):
      if self.active: n = self.ctrack[0]+d; self.resume_at(n); show_control(n)
    w_track_manager.observe((lambda c:from_track(c.new)),'value')
    def from_clock(v):
      if self.active:
        if self.accept(n:=int(v*self.frame_per_stu)): self.resume_at(n)
        else: show_control(w_track_manager.value)
    w_clock.observe((lambda c:from_clock(c.new)),'value')
    super().__init__(*args,**kwargs)

#==================================================================================================
class MPLControlledAnimation (BaseControlledAnimation):
  r"""
Instances of this class are players for :mod:`matplotlib` animations, controlled within matplotlib. The animation board is created by invoking :func:`figure` with arguments *fig_kw*.

:param fig_kw: configuration of the animation figure (excluding the toolbar)
:param tbsize: size (inches) of the toolbar as ((hsize(play-button),hsize(track-manager),hsize(clock)),vsize(toolbar))
:param ka: passed to the superclass
  """
# ==================================================================================================
  def __init__(self,*args,fig_kw={},tbsize=({'play_toggler':.2,'track_bound_beg':.4,'track_manager':2.,'track_bound_end':.4,'clock':.4},.15),**kwargs):
    # global design
    tbsize_ = sum(tbsize[0].values())
    figsize = fig_kw.pop('figsize',None)
    if figsize is None: figsize = rcParams['figure.figsize']
    figsize_ = list(figsize); figsize_[1] += tbsize[1]; figsize_[0] = max(figsize_[0],tbsize_)
    self.main = main = figure(figsize=figsize_,**fig_kw)
    r = tbsize[1],figsize[1]
    toolbar,self.board = main.subfigures(nrows=2,height_ratios=r)
    r = tbsize[0]
    r['track_manager'] += figsize_[0]-tbsize_
    g = {'width_ratios':r.values(),'wspace':0.,'bottom':0.,'top':1.,'left':0.,'right':1.}
    axes = toolbar.subplots(ncols=len(r),subplot_kw={'xticks':(),'yticks':(),'navigate':False},gridspec_kw=g)
    # required by super
    def show_control(n:int,new_track:bool=False):
      track_manager.val = n
      if not edit_value: clock.set(text=f'{n/self.frame_per_stu:.02f}',color='k')
      track_manager.set(width=(n-self.ctrack[0])/self.ctrack[2])
      if new_track:
        track_bound_beg.set(text=f'{self.ctrack[0]/self.frame_per_stu:.2f}')
        track_bound_end.set(text=f'{self.ctrack[1]/self.frame_per_stu:.2f}')
    self.show_control = show_control
    def show_running(b,D={True:(self.pause,'II'),False:(self.resume,r'$\blacktriangleright$')}): # '⏸︎︎', '⏵'
      onclick,icon = D[b]
      play_toggler.onclick = onclick; play_toggler.set(text=icon)
    self.show_running = show_running
    # widget definitions
    axes_:dict[str,matplotlib.axes.Axes] = dict(zip(r,axes))
    def widget(f:Callable[[matplotlib.axes.Axes],None])->matplotlib.axes.Axes: return f(axes_[f.__name__])
    @widget
    def play_toggler(ax): return ax.text(.5,.5,'',ha='center',va='center',transform=ax.transAxes)
    @widget
    def track_manager(ax): ax.set(xlim=(0,1),ylim=(0,1)); return ax.add_patch(Rectangle((0.,0.),0.,1.,fill=True))
    @widget
    def clock(ax): return ax.text(.5,.5,'',ha='center',va='center',transform=ax.transAxes)
    @widget
    def track_bound_beg(ax): return ax.text(.5,.5,'',ha='center',va='center',fontsize='xx-small',transform=ax.transAxes)
    @widget
    def track_bound_end(ax): return ax.text(.5,.5,'',ha='center',va='center',fontsize='xx-small',transform=ax.transAxes)
    # callbacks
    def on_button_press(ev):
      if ev.button == ev.button.LEFT and ev.key is None:
        if ev.inaxes is play_toggler.axes:
          play_toggler.onclick(); toolbar.canvas.draw_idle()
        elif ev.inaxes is track_manager.axes:
          d = int(ev.xdata*self.ctrack[2]); n = self.ctrack[0]+d
          self.resume_at(n); show_control(n)
    edit_value:str = ''
    def on_key_press(ev):
      nonlocal edit_value
      key = ev.key
      if key == 'left' or key == 'right':
        if ev.inaxes is not None:
          n = track_manager.val+{'left':-1,'right':1}[key]; n = min(max(n,self.ctrack[0]),self.ctrack[1]-1)
          self.show_control(n); self.resume_at(n); return
      elif ev.inaxes is clock.axes:
        if key=='enter':
          try: v_ = float(edit_value)
          except: return
          ev_ = edit_value; edit_value = ''
          if self.accept(n:=int(v_*self.frame_per_stu)): self.resume_at(n)
          else: edit_value = ev_
          return
        elif key=='escape': edit_value = ''; v,c = f'{track_manager.val/self.frame_per_stu:.2f}','k'
        elif key=='backspace' and len(edit_value)>1:
          edit_value = edit_value[:-1]; v,c = edit_value,'b'
        elif key in '0123456789' or (key=='.' and '.' not in edit_value):
          edit_value += key; v,c = edit_value,'b'
        else: return
        clock.set(text=v,color=c)
        toolbar.canvas.draw_idle()
    toolbar.canvas.mpl_connect('button_press_event',on_button_press)
    toolbar.canvas.mpl_connect('key_press_event',on_key_press)
    super().__init__(*args,**kwargs)

ControlledAnimation = IPYControlledAnimation if get_backend()=='widget' else MPLControlledAnimation
