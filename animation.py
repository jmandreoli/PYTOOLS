# File:                 animation.py
# Creation date:        2021-06-01
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Utilities for animations
#
r"""
:mod:`PYTOOLS.animation` --- Animation utilities
================================================

This module provides basic utilities to play animations (in matplotlib). An animation is any callback which can be passed a frame number, enumerated by the player. The set of frame numbers is split into a sequence of contiguous intervals called tracks. A track function is a function of frame numbers which returns, for each valid frame number, the bounds of its track interval, and :const:`None` for invalid frame numbers.

Available types and functions
-----------------------------
"""

from __future__ import annotations

from typing import Any, Union, Callable, Iterable, Mapping, Sequence, Tuple
import logging; logger = logging.getLogger(__name__)

from matplotlib import rcParams
from matplotlib.pyplot import figure, Figure
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import MouseButton
try: from .ipywidgets import app, SimpleButton # so this works even if ipywidgets is not available
except: app = object

#==================================================================================================
def track_function(track:Union[float,Sequence[float],Callable[[float],Tuple[float,float]]],stype:type=float)->Callable[[float],Tuple[float,float]]:
  r"""
A track map decomposes an interval of scalar (containing 0) into contiguous intervals. It can be specified either as a callable of one scalar input returning two scalar outputs (bounds of its track interval), or as

* a scalar, in which case the intervals are of constant length equal to that scalar, starting at 0
* or an increasing sequence of scalars, in which case the intervals are the consecutive pairs in the sequence

:param track: the track map specification
:param stype: the type of scalars passed to the track map
  """
# ==================================================================================================
  assert stype is int or stype is float
  if callable(track):
    track_ = track(stype(0))
    assert track_ is not None and len(track_)==2 and track_[0]<=0<=track_[1]
  else:
    try: L = tuple(map(stype,(0,*track)))
    except:
      T = stype(track)
      assert T>0
      def track(x,T=T): x -= x%T; return x,x+T
    else:
      assert len(L)>1 and all(x<x_ for (x,x_) in zip(L[:-1],L[1:]))
      from bisect import bisect
      def track(x,L=L,imax=len(L)): i = bisect(L,x); return (L[i-1],L[i]) if i<imax else None
  return track

#==================================================================================================
class animation_player_base:
  r"""
Instances of this class are players for :mod:`matplotlib` animations. The speed of the animation is controlled by two parameters whose product determines the number of real milli-seconds per simulation time unit (stu):

* parameter *frame_per_stu*: the number of frames per stu
* parameter *interval*: the number of real milli-seconds per frame

Note that the *interval* is bounded below by the real time needed to construct and display the frames. Furthermore, below 40ms per frame, the human eye tends to loose vision persistency.

:param display: the function which takes a simulation time and displays the corresponding (closest) frame
:param track: track map decomposing the simulation interval into contiguous simulation time periods (tracks)
:param frame_per_stu: frame rate, in frames per simulation time (in :const:`None` use animation real time rate)
:param ka: passed to the :func:`matplotlib.animation.FuncAnimation` constructor
  """
#==================================================================================================
  board: Figure
  r"""The figure on which to start the animation"""
  running: bool
  r"""The running state of the animation"""
  setrunning: Callable[[bool],None]
  r"""The function to set the running state of the animation"""
  frame: int
  r"""The current frame displayed by the animation"""
  setval: Callable[[float],None]
  r"""The function to set the current frame to be displayed by the animation"""
  anim: FuncAnimation
  r"""The animation"""
  frame_per_stu: float
  r"""The frame rate, in frames per simulation time units"""
  track: Callable[[float],Tuple[float,float]]
  r"""The track map"""

  def __init__(self,display:Callable[[float],None],frame_per_stu:float=None,track=None,**ka):
    def frames():
      self.setval(0.)
      while True:
        self.setrunning(False); yield self.frame
        while self.setval() is None: yield self.frame
    self.track = track_function(track,float)
    self.anim = anim = FuncAnimation(self.board,(lambda n: display(n/frame_per_stu)),frames,init_func=(lambda:None),repeat=False,**ka)
    if frame_per_stu is None: frame_per_stu = 1000./anim._interval
    else: frame_per_stu = float(frame_per_stu); assert frame_per_stu > 0
    self.frame_per_stu = frame_per_stu

  def animation_running(self,b:bool):
    self.running = b = ((not self.running) if b is None else b)
    if b: self.anim.resume()
    else: self.anim.pause()
    return b

#==================================================================================================
class widget_animation_player (app,animation_player_base):
  r"""
Instances of this class are players for :mod:`matplotlib` animations controlled from `mod:ipywidgets` widgets.

:param fig_kw: configuration of the animation figure
  """
#==================================================================================================
  def __init__(self,display:Callable[[float],None],fig_kw={},children=(),toolbar=(),**ka):
    from ipywidgets import Text, FloatText, IntSlider
    w_play_toggler = SimpleButton(icon='')
    w_track_manager = IntSlider(0,min=0,readout=False)
    w_track_manager.active = True
    w_clockb = SimpleButton(icon='stopwatch',tooltip='manually reset clock')
    w_clock = Text('',layout=dict(width='1.6cm',padding='0cm'),disabled=True)
    w_clock2 = FloatText(0,min=0,layout=dict(width='1.6cm',padding='0cm',display='none'))
    w_clock2.active = False
    app.__init__(self,children,toolbar=[w_play_toggler,w_track_manager,w_clockb,w_clock,w_clock2,*toolbar])
    self.board = board = self.mpl_figure(**fig_kw)
    w_play_toggler.on_click(lambda b: setrunning())
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
    ctrack = -1,0,1
    def setval(v=None,d=None,submit=False):
      nonlocal ctrack
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
        ctrack = tuple(int(v_*self.frame_per_stu) for v_ in track_)
        w_track_manager.max = span = ctrack[1]-ctrack[0]
        ctrack += (span,)
        d = n-ctrack[0]
      w_track_manager.value = d
      w_track_manager.active = True
      self.frame = n
      if submit: display(v); board.canvas.draw_idle()
    self.setval = setval
    def setrunning(b=None):
      w_play_toggler.icon = 'pause' if self.animation_running(b) else 'play'
    self.setrunning = setrunning
    animation_player_base.__init__(self,display,**ka)

#==================================================================================================
class mpl_animation_player (animation_player_base):
  r"""
Instances of this class are players for :mod:`matplotlib` animations, controlled within matplotlib.

:param display: the function which takes a frame number and displays the corresponding frame
:param track: track generator decomposing the simulation interval into contiguous simulation time periods (tracks)
:param fig_kw: configuration of the animation figure (excluding the toolbar)
:param tbsize: size of the toolbar
  """
# ==================================================================================================
  def __init__(self,display:Callable[[float],None],fig_kw={},tbsize=((.15,1.,.8),.15),**ka):
    tbsize_ = sum(tbsize[0])
    figsize = fig_kw.pop('figsize',None)
    if figsize is None: figsize = rcParams['figure.figsize']
    figsize_ = list(figsize); figsize_[1] += tbsize[1]; figsize_[0] = max(figsize_[0],tbsize_)
    self.main = main = figure(figsize=figsize_,**fig_kw)
    r = tbsize[1],figsize[1]
    toolbar,self.board = main.subfigures(nrows=2,height_ratios=r)
    # widget definition
    r = list(tbsize[0])
    r[1] += figsize_[0]-tbsize_
    g = dict(width_ratios=r,wspace=0.,bottom=0.,top=1.,left=0.,right=1.)
    axes = toolbar.subplots(ncols=3,subplot_kw=dict(xticks=(),yticks=(),navigate=False),gridspec_kw=g)
    play_toogler = axes[0].text(.5,.5,'',ha='center',va='center',transform=axes[0].transAxes)
    axes[1].set(xlim=(0,1),ylim=(0,1))
    track_manager = axes[1].add_patch(Rectangle((0.,0.),0.,1.,fill=True))
    clock = axes[2].text(.5,.5,'',ha='center',va='center',transform=axes[2].transAxes)
    def on_button_press(ev):
      if ev.button == MouseButton.LEFT and ev.key is None:
        if ev.inaxes is play_toogler.axes: setrunning()
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
    ctrack = -1.,0.,1.
    def setval(v=None,d=+1,submit=False):
      nonlocal ctrack
      if v is None: n = self.frame+d; v = n/self.frame_per_stu
      else: n = int(v*self.frame_per_stu)
      x = (v-ctrack[0])/ctrack[2]
      if x<0 or x>=1:
        track_ = self.track(v)
        if track_ is None: return True
        ctrack = tuple(track_)+(track_[1]-track_[0],)
        x = (v-ctrack[0])/ctrack[2]
      if not edit_value: clock.set(text=f'{v:.02f}',color='k')
      track_manager.set(width=x)
      self.frame = n
      if submit: display(v); main.canvas.draw_idle()
    self.setval = setval
    def setrunning(b=None):
      play_toogler.set(text='II' if self.animation_running(b) else '|>')
      toolbar.canvas.draw_idle()
    self.setrunning = setrunning
    super().__init__(display,**ka)

  def _ipython_display_(self): return repr(self)
