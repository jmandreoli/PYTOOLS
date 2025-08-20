# File:                 animation.py
# Creation date:        2021-06-01
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Utilities for animations
#

from __future__ import annotations
import logging; logger = logging.getLogger(__name__)
from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple, Literal

from enum import Enum
from functools import partial
import matplotlib.axes
from matplotlib import rcParams, get_backend
from matplotlib.pyplot import figure
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
try: from .ipywidgets import app # so this works even if ipywidgets is not available
except: app = object

#==================================================================================================
AnimationStatus = Enum('AnimationStatus','playing paused')
class BaseControlledAnimation (FuncAnimation):
  r"""
An instance of this class is a controllable :mod:`matplotlib` animation, created by this constructor and attached to a :class:`Figure` instance stored as attribute :attr:`board`. The board creation is not performed in this class, so it must be performed in the constructor of a subclass, prior to invoking this constructor.

The speed of the animation is controlled by two parameters whose product determines the number of real milliseconds per simulation time unit (stu):

:param frame_per_stu: the number of frames per stu
:param interval: the number of real milliseconds per frame

At least one of these parameters must be provided. If one is missing, the other is imputed so that the simulation unit is 1 real second. So long as each frame takes less than *interval* to be constructed and displayed, it remains displayed until the end of the interval, and the next frame is constructed and displayed starting at the beginning of the next interval. A reasonable value for *interval* is ca. 40 ms/frame (flicker fusion threshold).

:param display: the function which takes a simulation time and displays it
:param track_spec: specification of a track map, passed to method :meth:`track_function`
:param ka: passed to the :class:`FuncAnimation` constructor
  """
#==================================================================================================
  board: Figure
  r"""The figure on which to start the animation"""
  frame_per_stu: float
  r"""The frame rate, in frames per simulation time units"""
  track_func: Callable[[float],Tuple[float,float]]
  r"""The track map"""
  show_status:Callable[[Enum],None]
  r"""Show the running state of the animation"""
  show_control:Callable[[int,bool],None]
  r"""Show the control state of the animation (current frame index, whether current track was modified)"""
  jump_to: Callable[[int,bool],None]
  r"""Interrupt the normal sequence of the animation (frame to jump to, whether current track may require update)"""
  track:Sequence[int]
  r"""Current track (start frame included, end frame excluded, length)"""

  #--------------------------------------------------------------------------------------------------
  def __init__(self,display:Callable[[Any],None],frame_per_stu:float|None=None,interval:float|None=None,track_spec=None,**ka):
  #--------------------------------------------------------------------------------------------------
    def set_status(f:Callable[[],None],s:AnimationStatus): f(); self.show_status(s)
    self.resume = partial(set_status,self.resume,AnimationStatus.playing)
    self.pause  = partial(set_status,self.pause,AnimationStatus.paused)
    def jump_to(n:int,check:bool=False):
      v = n/self.frame_per_stu; new_track = False
      if check is True:
        d = n-track[0]; new_track = d<0 or d>=track[2]
        if new_track:
          if (tr:=track_func(v)) is None: return None
          track[:2] = (int(v_*self.frame_per_stu) for v_ in tr)
          track[2] = x = track[1]-track[0]
          if x==0: track[1] += 1; track[2] += 1
      self.show_control(n,new_track)
      return v
    def frames():
      n:int = 0
      def jump_to_(n_:int,check:bool=False)->bool:
        nonlocal n
        if (v:=jump_to(n_,check)) is None: return False
        n = n_; display(v); self.board.canvas.draw_idle(); return True
      self.jump_to = jump_to_
      while True:
        self.pause(); yield None
        while (v:=jump_to((n_:=n+1),check=True)) is not None: n = n_; yield v
    self.track = track = [-1,0,1]
    track_func = self.track_function(track_spec)
    def positive(v): v = float(v); assert v>0; return v
    if frame_per_stu is None:
      assert interval is not None, 'At least one of *frame_per_stu* or *interval* must be provided'
      interval = positive(interval); frame_per_stu = 1000./interval
    else:
      frame_per_stu = positive(frame_per_stu)
      interval = 1000./frame_per_stu if interval is None else positive(interval)
    self.frame_per_stu = frame_per_stu
    super().__init__(self.board,(lambda v: None if v is None else display(v)),frames,init_func=(lambda:display(0.)),repeat=False,cache_frame_data=False,interval=interval,**ka)

#--------------------------------------------------------------------------------------------------
  @staticmethod
  def track_function(spec:int|float|Sequence[int|float]|Callable[[float],Tuple[float,float]])->Callable[[float],Tuple[float,float]]:
    r"""
Builds a track map from a specification *spec*. A track map is a callable of one scalar input returning two scalar outputs (bounds of its track interval), or :const:`None` (when input is out of domain).

* Its domain must be an interval (not necessarily bounded) containing 0
* The mapping of a scalar to the lower bound of its track interval should be non-decreasing and right-continuous.

The specification *spec* can be the track map itself, returned as such after minimal checks. As a helper, *spec* can also be

* a positive scalar, in which case the track map is based on intervals of constant length equal to that scalar, starting at 0. and never ending
* an increasing sequence of positive scalars, in which case the track map is based on intervals which are the consecutive pairs in that sequence (prefixed with 0.)

:param spec: the track map specification
  """
#--------------------------------------------------------------------------------------------------
    if callable(spec):
      track_func = spec
      track_ = spec(0.)
      assert track_ is not None and len(track_)==2 and isinstance((t0:=track_[0]),float) and isinstance((t1:=track_[1]),float) and t0<=0.<t1 and spec(t0) == track_ and ((t:=spec(t1)[0]) is None or t == t1)
    elif isinstance(spec,(int,float)):
      T = float(spec)
      assert T>0
      def track_func(x,T=T):
        if x<0.: return None
        x -= x%T; return x,x+T
    else:
      L = (0.,*map(float,spec))
      assert len(L)>1 and all(x<x_ for (x,x_) in zip(L[:-1],L[1:]))
      from bisect import bisect
      def track_func(x,L=L,imax=len(L)):
        if x<0.: return None
        i = bisect(L,x); return (L[i-1],L[i]) if i<imax else None
    return track_func

#--------------------------------------------------------------------------------------------------
  def panes(self,nrows:int=1,ncols:int=1,sharex:str|bool=False,sharey:str|bool=False,gridspec_kw:Mapping|None=None,gridlines:bool=True,aspect:str='equal',**ka):
    r"""
Generator of panes on the board.

:param ka: a dictionary of keyword arguments passed to the :meth:`matplotlib.figure.add_subplot` method of each part (key ``gridlines`` is also allowed and denotes whether gridlines should be displayed)
    """
#--------------------------------------------------------------------------------------------------
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
          for w,n_ in zip(w_clock_bounds,self.track[:2]): w.value = f'{n_/self.frame_per_stu:.2f}'
          w_track_manager.max = self.track[2]
        w_track_manager.value,w_clock.value = n-self.track[0],n/self.frame_per_stu
    self.show_control = show_control
    onclick = lambda: None
    def show_status(b,D={AnimationStatus.playing:('pause',(lambda: self.pause())),AnimationStatus.paused:('play',(lambda: self.resume()))}):
      nonlocal onclick; w_play_toggler.icon,onclick = D[b]
    self.show_status = show_status
    # global design and widget definitions
    w_play_toggler = SimpleButton((lambda: onclick()),icon='')
    w_track_manager = IntSlider(0,min=0,step=1,readout=False)
    w_clock_bounds = [Label('',style={'font_size':'xx-small'}) for _ in range(2)]
    w_clock_edit = FloatText(0.,min=0.,step=1e-10,style={'font_size':'xx-small'},layout={'width':'1.6cm','padding':'0cm'})
    w_clock = Stable(w_clock_edit,'{:.2f}'.format)
    app.__init__(self,toolbar=(w_play_toggler,w_clock_bounds[0],w_track_manager,w_clock_bounds[1],w_clock,*toolbar))
    self.board = self.mpl_figure(**fig_kw)
    self.active = True
    # callbacks
    def from_track(d):
      if self.active: self.jump_to(self.track[0]+d)
    w_track_manager.observe((lambda c:from_track(c.new)),'value')
    def from_clock(v,v_old):
      if self.active and not self.jump_to(int(v*self.frame_per_stu),check=True):
        with deactivate(self): w_clock.value = v_old
    w_clock.observe((lambda c:from_clock(c.new,c.old)),'value')
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
      track_manager.set(width=(n-self.track[0])/self.track[2])
      if new_track:
        track_bound_beg.set(text=f'{self.track[0]/self.frame_per_stu:.2f}')
        track_bound_end.set(text=f'{self.track[1]/self.frame_per_stu:.2f}')
    self.show_control = show_control
    def show_status(b,D={AnimationStatus.playing:('II',(lambda: self.pause())),AnimationStatus.paused:(r'$\blacktriangleright$',(lambda: self.resume()))}): # '⏸︎︎', '⏵'
      icon,onclick = D[b]; play_toggler.onclick = onclick; play_toggler.set(text=icon)
    self.show_status = show_status
    # widget definitions
    axes_:dict[str,matplotlib.axes.Axes] = dict(zip(r,axes))
    def widget(f:Callable[[matplotlib.axes.Axes],None])->matplotlib.axes.Axes: return f(axes_[f.__name__])
    @widget
    def play_toggler(ax): return ax.text(.5,.5,'',ha='center',va='center',transform=ax.transAxes)
    @widget
    def track_manager(ax): ax.set(xlim=(0,1),ylim=(0,1)); return ax.add_patch(Rectangle((0.,0.),0.,1.,fill=True))
    @widget
    def clock(ax): return ax.text(.5,.5,'',ha='center',va='center',fontsize='xx-small',transform=ax.transAxes)
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
          self.jump_to(self.track[0]+int(ev.xdata*self.track[2]))
    edit_value:str = ''
    def on_key_press(ev):
      nonlocal edit_value
      key = ev.key
      if key == 'left' or key == 'right':
        if ev.inaxes is not None:
          n = track_manager.val+{'left':-1,'right':1}[key]
          if n>=self.track[0] and n<self.track[1]: self.jump_to(n)
          return
      elif ev.inaxes is clock.axes:
        if key=='enter':
          try: v_ = float(edit_value)
          except: return
          ev_ = edit_value; edit_value = ''
          if not self.jump_to(int(v_*self.frame_per_stu),check=True): edit_value = ev_
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

def ControlledAnimation(*a,**ka):
  return IPYControlledAnimation(*a,**ka) if get_backend() == 'widget' else MPLControlledAnimation(*a,**ka)
