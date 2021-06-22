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

import traitlets, traceback
from functools import wraps, partial
from matplotlib import rcParams
from matplotlib.pyplot import figure, Figure
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import MouseButton
try: from .ipywidgets import app, SimpleButton # so this works even if ipywidgets is not available
except: app = object

#==================================================================================================
def track_function(track:Union[float,Sequence[float],Callable[[float],Tuple[float,float]]],stype:type):
  r"""
A track function decomposes an interval of scalar (containing 0) into contiguous intervals. It can be specified either as a callable of one scalar input returning two scalar outputs (bounds of its track interval), or as

* a scalar, in which case the intervals are of constant length equal to that scalar, starting at 0
* or an increasing sequence of scalars, in which case the intervals are the consecutive pairs in the sequence
  """
# ==================================================================================================
  if not callable(track):
    if isinstance(track,stype):
      assert track>0
      def track(x,T=track): x -= x%T; return x,x+T
    else:
      L = tuple((0,*track))
      assert len(L)>1 and all(isinstance(x,stype) for x in L[1:]) and all(x<x_ for (x,x_) in zip(L[:-1],L[1:]))
      from bisect import bisect
      def track(x,L=L,imax=len(L)): i = bisect(L,x); return (L[i-1],L[i]) if i<imax else None
  return track

#==================================================================================================
class animation_player_base:
  r"""
Instances of this class are players for :mod:`matplotlib` animations.

:param display: the function which takes a simulation time and displays the corresponding (closest) frame
:param track: track generator decomposing the simulation interval into contiguous simulation time periods (tracks)
:param rate: frames per simulation time
:param ka: passed to the :func:`matplotlib.animation.FuncAnimation` constructor
  """
#==================================================================================================
  board: Figure
  r"""The figure on which to start the animation"""
  running: bool
  r"""The running state of the animation"""
  frame: int
  r"""The current frame displayed by the animation"""
  anim: FuncAnimation
  r"""The animation"""
  rate: float
  r"""The frame rate, in frames per simulation time"""

  def __init__(self,display:Callable[[float],None],track=None,rate:float=None,**ka):
    self.track = track_function(track,float)
    def frames():
      self.setrunning(False)
      yield self.setval(0.)
      while (n:=self.setval()) is not None: yield n
    self.anim = anim = FuncAnimation(self.board,(lambda n: display(n*rate)),frames,init_func=(lambda:None),repeat=False,**ka)
    if rate is None: rate = anim._interval/1000. # real time
    self.rate = rate

  def animation_running(self,b:bool):
    self.running = b = ((not self.running) if b is None else b)
    if b: self.anim.resume()
    else: self.anim.pause()
    return b

  def setrunning(self,b:bool):
    r"""Sets the running state of the animation. Must be refined in subclasses or in instances."""
    raise NotImplementedError()

  def setval(self,v:float):
    r"""Sets the current frame to be displayed by the animation. Must be refined in subclasses or in instances."""
    raise NotImplementedError()

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
    def set_clock():
      w_clock2.value = self.frame*self.rate
      w_clockb.layout.visibility,w_clock.layout.display,w_clock2.layout.display,w_clock2.active = 'hidden','none','',True
    w_clockb.on_click(lambda b: set_clock())
    def clock_set(v):
      w_clockb.layout.visibility,w_clock.layout.display,w_clock2.layout.display,w_clock2.active = 'visible','','none',False
      setval(v,submit=True)
    w_clock2.observe((lambda c: clock_set(c.new) if w_clock2.active else None),'value')
    w_track_manager.observe((lambda c: setval(d=c.new,submit=True) if w_track_manager.active else None),'value')
    ctrack = -1,0,1
    def setval(v=None,d=None,submit=False):
      nonlocal ctrack
      if v is None:
        if d is None: self.frame += 1
        else: self.frame = ctrack[0]+d
        v = self.frame*self.rate
      else: self.frame = int(v/self.rate)
      w_clock.value = f'{v:.02f}'
      x = self.frame-ctrack[0]
      w_track_manager.active = False
      if x<0 or x>=ctrack[2]:
        track_ = self.track(v)
        if track_ is None: return
        ctrack = tuple(int(v_/self.rate) for v_ in track_)
        w_track_manager.max = span = ctrack[1]-ctrack[0]
        ctrack += (span,)
        x = self.frame-ctrack[0]
      w_track_manager.value = x
      w_track_manager.active = True
      if submit:
        display(v)
        board.canvas.draw_idle()
      return self.frame
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
    track_manager = axes[1].add_patch(Rectangle((0.,0.),0.,1.,fill=True,transform=axes[1].transAxes))
    clock = axes[2].text(.5,.5,'',ha='center',va='center',transform=axes[2].transAxes)
    pix2pc = track_manager.axes.transAxes.inverted().transform
    def on_button_press(ev):
      if ev.button == MouseButton.LEFT and ev.key is None:
        if ev.inaxes is play_toogler.axes: setrunning()
        elif ev.inaxes is track_manager.axes: setval(ctrack[0]+pix2pc((ev.x,ev.y))[0]*ctrack[2],submit=True)
    edit_value = ''
    def on_key_press(ev):
      nonlocal edit_value
      key = ev.key
      if key == 'left' or key == 'right':
        if ev.inaxes is not None: setval(d=(+1 if key=='right' else -1),submit=True)
      elif ev.inaxes is clock.axes:
        v,c = f'{self.frame*self.rate:.02f}','k'
        if key=='enter':
          try: v_ = float(edit_value)
          except: return
          edit_value = ''
          if setval(v_,submit=True) is not None: return
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
    ctrack = -1.,0.,1.
    def setval(v=None,d=+1,submit=False):
      nonlocal ctrack
      if v is None: self.frame += d; v = self.frame*self.rate
      else: self.frame = int(v/self.rate)
      x = (v-ctrack[0])/ctrack[2]
      if x<0 or x>=1:
        track_ = self.track(v)
        if track_ is None: return
        ctrack = tuple(track_)+(track_[1]-track_[0],)
        x = (v-ctrack[0])/ctrack[2]
      if not edit_value: clock.set(text=f'{v:.02f}',color='k')
      track_manager.set(width=x)
      if submit:
        display(v)
        main.canvas.draw_idle()
      return self.frame
    self.setval = setval
    def setrunning(b=None):
      play_toogler.set(text='II' if self.animation_running(b) else '|>')
      toolbar.canvas.draw_idle()
    self.setrunning = setrunning
    toolbar.canvas.mpl_connect('button_press_event',on_button_press)
    toolbar.canvas.mpl_connect('key_press_event',on_key_press)
    super().__init__(display,**ka)

  def _ipython_display_(self): return repr(self)
