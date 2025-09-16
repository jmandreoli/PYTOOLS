# File:                 animation.py
# Creation date:        2021-06-01
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Utilities for animations
#

from __future__ import annotations
import logging;logger = logging.getLogger(__name__)
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from enum import Enum
from functools import partial
from collections import defaultdict
from matplotlib import rcParams, get_backend
from matplotlib.pyplot import figure
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from .ipywidgets import app

__all__ = 'AnimationStatus', 'BoardDisplayer',

#==================================================================================================
AnimationStatus = Enum('AnimationStatus','playing paused')
class BaseControlledAnimation (FuncAnimation):
  r"""
An instance of this class is a controllable animation (of type :class:`FuncAnimation`), attached to a board (of type :class:`Figure`) stored as attribute :attr:`board`. The board creation is not performed in this class, so it must be performed in the constructor of a subclass, prior to invoking this constructor.

The speed of the animation is controlled by two parameters whose product determines the number of real milliseconds per simulation time unit (stu):

:param frame_per_stu: the number of frames per stu
:param interval: the number of real milliseconds per frame

At least one of these parameters must be provided. If only one is present, the other is imputed so that 1 stu = 1 sec. So long as each frame takes less than *interval* to be constructed and displayed, it remains displayed until the end of the interval, and the next frame is constructed and displayed starting at the beginning of the next interval. A reasonable lower bound for *interval* is 40 ms/frame (flicker fusion threshold), i.e. 25 frame/s, which is also the value of *frame_per_stu* achieving 1 stu = 1 sec.

:param displayer: callable which, given a board, returns a callable which, given a frame, paints it on the board
:param track_spec: specification of a track map, passed to method :meth:`track_function`
:param ka: passed to the :class:`FuncAnimation` constructor
  """
#==================================================================================================
  board: Figure
  r"""The figure on which to start the animation"""
  frame_per_stu: float
  r"""The frame rate, in frames per simulation time unit"""
  track_func: Callable[[float],tuple[float,float]|None]
  r"""The track map"""
  show_status:Callable[[AnimationStatus],None]
  r"""Show the running state of the animation (playing or paused)"""
  show_control:Callable[[int,bool],None]
  r"""Show the control state of the animation (current frame index, whether current track is different from that of previous frame)"""
  jump_to: Callable[[int,bool],bool]
  r"""Interrupt the normal sequence of the animation (frame to jump to, whether current track may require update)"""
  track:Sequence[int]
  r"""Current track (start frame included, end frame excluded, length)"""

  #--------------------------------------------------------------------------------------------------
  def __init__(self,displayer:Callable[[Any],Callable[[float],None]],frame_per_stu:float|None=None,interval:float|None=None,track_spec=None,**ka):
  #--------------------------------------------------------------------------------------------------
    def set_status(f:Callable[[],None],s:AnimationStatus): f(); self.show_status(s)
    self.resume = partial(set_status,self.resume,AnimationStatus.playing)
    self.pause  = partial(set_status,self.pause,AnimationStatus.paused)
    def jump_to_(n:int,check:bool)->float|None:
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
      def jump_to(n_:int,check:bool)->bool:
        nonlocal n
        if (v:=jump_to_(n_,check)) is None: return False
        n = n_; display(v); self.board.canvas.draw_idle(); return True
      self.jump_to = jump_to
      while True:
        self.pause(); yield None
        while (v:=jump_to_((n_:=n+1),True)) is not None: n = n_; yield v
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
    display = displayer(self.board)
    super().__init__(self.board,(lambda v: None if v is None else display(v)),frames,init_func=(lambda:display(0.)),repeat=False,cache_frame_data=False,interval=interval,**ka)

#--------------------------------------------------------------------------------------------------
  @staticmethod
  def track_function(spec:int|float|Sequence[int|float]|Callable[[float],tuple[float,float]])->Callable[[float],tuple[float,float]|None]:
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

  def panes(self,**ka): return Panes(self.board,**ka)

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
      if self.active: self.jump_to(self.track[0]+d,False)
    w_track_manager.observe((lambda c:from_track(c.new)),'value')
    def from_clock(v,v_old):
      if self.active and not self.jump_to(int(v*self.frame_per_stu),True):
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
    axes_:dict[str,Axes] = dict(zip(r,axes))
    def widget(f:Callable[[Axes],Any])->Any: return f(axes_[f.__name__])
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
          self.jump_to(self.track[0]+int(ev.xdata*self.track[2]),False)
    edit_value:str = ''
    def on_key_press(ev):
      nonlocal edit_value
      key = ev.key
      if key == 'left' or key == 'right':
        if ev.inaxes is not None:
          n = track_manager.val+{'left':-1,'right':1}[key]
          if n>=self.track[0] and n<self.track[1]: self.jump_to(n,False)
          return
      elif ev.inaxes is clock.axes:
        if key=='enter':
          try: v_ = float(edit_value)
          except: return
          ev_ = edit_value; edit_value = ''
          if not self.jump_to(int(v_*self.frame_per_stu),True): edit_value = ev_
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

#==================================================================================================
class BoardDisplayer:
  r"""
An instance of this class is a callable which takes as input a board, prepares it for display, and returns a callable which, given a frame, paints it on the board. Typically used as a first argument (*displayer* callable) in a :class:`BaseControlledAnimation` constructor. The board (here an instance of class :class:`Figure`) is divided into panes (here instances of class :class:`Axes`). Each pane can be assigned a list of named pane displayers, corresponding to different views of the displayed phenomenon. The name of each view can be used to configure the view using parameter *view_cfg* in method :meth:`__call__` (invocation of this displayer).
  """
#==================================================================================================
  setup:Callable[[Any],None]
  r"""Invoked before invoking the pane displayers on a frame"""
  displayers:dict[tuple[int,int],list[tuple[str,Callable[[Any],Callable[[Any],None]]]]]
  def add_displayer(self,pos:tuple[int,int]|None=None,**ka:Callable[[Any],Callable[[Any],None]]):
    r"""
Adds a named pane displayer to this board displayer. A pane displayer is a callable which takes as input a pane, prepares it for display, and returns a callable which, given a frame, paints it on the pane according to the view name.

:param pos: the position of the pane on the board
:param ka: the keys are view names and the values are pane displayers
    """
    if pos is None: pos = (0,0)
    self.displayers[pos].extend((view,D) for view,D in ka.items())
    return self # so it can be chained
  def __init__(self): self.displayers = defaultdict(list)
#--------------------------------------------------------------------------------------------------
  def with_simpy_setup(self):
    r"""
Initialises attribute :attr:`setup` for a :mod:`simpy` simulation.
    """
#--------------------------------------------------------------------------------------------------
    from . import ResettableSimpyEnvironment
    env = ResettableSimpyEnvironment(0.)
    self.setup = lambda v: env.run(v)
    self.env = env
    return self
#--------------------------------------------------------------------------------------------------
  call_defaults = {'aspect':'equal'}
  def __call__(self,fig:Figure|SubFigure,nrows:int=1,ncols:int=1,sharex:str|bool=False,sharey:str|bool=False,gridspec_kw:Mapping|None=None,view_cfg:Mapping[str,dict]|None=None,gridlines:bool=True,**ka)->Callable[[Any],None]:
    r"""
Prepares the board for display, and returns a callable which takes as input a frame and actually displays it on the board.

:param fig: the board
:param nrows: the number of rows
:param ncols: the number of columns
:param sharex: sharing specification for the x-axis
:param sharey: sharing specification for the y-axis
:param gridspec_kw: a grid specification for the board
:param view_cfg: a dictionary of view configurations (each configuration is a :class:`dict`)
:param gridlines: whether to display gridlines on the axes
:param ka: passed on as subplot keywords to construct the axes
    """
#--------------------------------------------------------------------------------------------------
    from numpy import array
    assert isinstance(fig,(Figure,SubFigure))
    axes = array(nrows*[ncols*[None]],dtype=object)
    share:dict[str|bool,Callable[[int,int],Axes|None]] = {
      'all': (lambda row,col: None if row==col==0 else axes[0,0]),
      True:  (lambda row,col: None if row==col==0 else axes[0,0]),   # alias
      'row': (lambda row,col: None if row==0 else axes[row,0]),
      'col': (lambda row,col: None if col==0 else axes[0,col]),
      'none':(lambda row,col: None),
      False: (lambda row,col: None), # alias
    }
    share_ = tuple((dim,share[s]) for dim,s in (('sharex',sharex),('sharey',sharey)))
    gridspec = fig.add_gridspec(nrows=nrows,ncols=ncols,**(gridspec_kw or {}))
    ka = self.call_defaults|ka
    def _get(row,col):
      ax = axes[row,col]
      if ax is None:
        ax = axes[row,col] = fig.add_subplot(gridspec[row,col],**dict((dim,s(row,col)) for dim,s in share_),**ka)
        ax.grid(gridlines)
      return ax
    get_view_cfg:Callable[[str],dict] = (lambda view: {}) if view_cfg is None else view_cfg.get
    display_list = [self.setup,*(D(_get(*pos),**kw) for pos,L in self.displayers.items() for view,D in L if (kw:=get_view_cfg(view)) is not None)]
    def display(frm):
      for f in display_list: f(frm)
    return display
#--------------------------------------------------------------------------------------------------
  play_defaults = {'interval':40}
  r"""The default arguments passed to method :meth:`play`"""
  def play(self,displayer_kw=None,**ka):
    r"""
Returns an animation based on this displayer.

:param displayer_kw: a dictionary of keyword arguments passed when invoking this displayer
:param ka: passed to the animation constructor
    """
#--------------------------------------------------------------------------------------------------
    animation_factory = IPYControlledAnimation if get_backend() == 'widget' else MPLControlledAnimation
    ka = self.play_defaults|ka
    return animation_factory((self if displayer_kw is None else partial(self,**displayer_kw)),**ka)
