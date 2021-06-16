# File:                 simpy.py
# Creation date:        2018-05-18
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              utilities for simpy based simulation
#
r"""
:mod:`PYTOOLS.simpy` --- Utilities for simpy
============================================

This module provides some utilities for simpy based simulation.

Available types and functions
-----------------------------
"""

from __future__ import annotations
from typing import Mapping
from functools import cached_property

import simpy

__all__ = 'RobustEnvironment', 'SimpySimulation'

# ==================================================================================================
class RobustEnvironment (simpy.Environment):
  """
Instances of this class are :class:`simpy.Environment` instances with method :meth:`run_robust`, which extends method :class:`run` to allow jumping back in the past (by rerunning since the initial time until the specified point in the past).
  """
# ==================================================================================================

  def __init__(self,init_t=0):
    self.init_t = init_t
    self.n_reset = -1
    self.reset()

  def reset(self):
    super().__init__(self.init_t)
    self.n_reset += 1

  def run_robust(self,s=None):
    # same as run(s) but allows going back in time
    if s is None or s>self.now: self.run(s)
    elif s<self.now:
      self.reset()
      if s!=self.now: self.run(s)

class NaturalSpanMixin (RobustEnvironment):
  @cached_property # for environments which end naturally
  def span(self):
    self.run_robust()
    return self.now-self.init_t

# ==================================================================================================
class SimpySimulation:
  """
An instance of this class controls the rollout of a set *content* of pairs where the first component is a :class:`RobustEnvironment` instance and the second component is display specification. Attribute :attr:`player` holds a player object, created by the instance, which executes the rollout in a timely fashion.
A display specification is a function which takes as input an environment and a part as returned by generator method :math:`parts`, and returns a display function with no input which displays the environment (at time of invocation) on the part.

The speed of the simulation is controlled by two parameters whose product determines the number of real milli-seconds per simulation time unit (stu):

* parameter *frame_per_stu*: the number of frames per stu
* parameter *interval*: the number of real milli-seconds per frame (in the player)

Note that the *interval* is bounded below by the real time needed to construct and display the frames. Furthermore, below 40ms per frame, the human eye tends to loose vision persistency.

:param content: list of environments with displayers
:type content: :class:`Sequence[Tuple[simpy.Environment,Callable[[simpy.Environment,Any],Callable[[],None]],...]]`
:param play_kw: a dictionary of keyword arguments, passed to method :meth:`init_player`
:param frame_per_stu: the number of frames per stu
:param ka: passed to method :meth:`parts`
  """
# ==================================================================================================
  player: None
  r"""The player object to run the simulation"""
  play_default = {'interval':40}
  r"""The default arguments passed to the player factory"""

  def __init__(self,*content,play_kw:Mapping={},frame_per_stu:float=None,**ka):
    assert all((isinstance(env,RobustEnvironment) and all(callable(f) for f in L)) for env,*L in content)
    self.frame_per_stu = frame_per_stu = float(frame_per_stu)
    assert frame_per_stu > 0
    play_kw = dict(play_kw)
    track = play_kw.pop('track')
    assert track is not None
    if callable(track):
      track = lambda n,track_=track: None if (tr:=track_(n/frame_per_stu)) is None else tuple(int(t*frame_per_stu) for t in tr)
    elif isinstance(track,float):
      track = int(track*frame_per_stu)
    else:
      track = (int(t*frame_per_stu) for t in tuple(track))
    play_kw['track'] = track
    self.player = self.make_player(**dict(self.play_default,**play_kw))
    board = self.player.board
    self.content = [(env,tuple(f(env,part) for f in L)) for (env,*L),part in zip(content,self.parts(board,**ka))]
    self.player.bind(self.display_content)
    self.display_content(0)

  def display_content(self,n):
    dt = n/self.frame_per_stu
    for env,L in self.content:
      env.run_robust(env.init_t+dt)
      for disp in L: disp()

  def make_player(self,**ka):
    r"""
Instantiate attribute :attr:`player`. This implementation uses an instance of class :class:`animation_player`.
    """
    from matplotlib import get_backend
    backend = get_backend()
    if 'ipympl' in backend:
      from .animation import widget_animation_player
      fig_kw = ka.pop('fig_kw',{})
      player = widget_animation_player(**ka).add_clock(self.frame_per_stu)
      player.board = player.mpl_figure(**fig_kw)
    else:
      from .animation import mpl_animation_player
      player = mpl_animation_player(rate=self.frame_per_stu,**ka) # board attribute is already included
    return player

  def bind(self,*a): self.player.bind(*a)
  def unbind(self,*a): self.player.unbind(*a)

  ax_default = {'aspect':'equal','gridlines':True}
  r"""The default arguments passed to the ``ax_kw`` parameter in method :meth:`parts`"""
  def parts(self,fig,nrows=1,ncols=1,sharex=False,sharey=False,gridspec_kw={},**ka):
    r"""
Generator of parts. This implementation assumes the board is a :mod:`matplotlib` grid-figure, and yields its subplots (each subplot is a part). The number of parts must be at least equal to the number of environments to which they are assigned, otherwise some environments are not processed.

:param ax_kw: a dictionary of keyword arguments passed to the :meth:`set` method of each part (key ``gridlines`` is also allowed and denotes whether gridlines should be displayed)
    """
    from numpy import zeros
    share = dict(all=(lambda row,col: (0,0)),row=(lambda row,col: (row,0)),col=(lambda row,col: (0,col)),none=(lambda row,col: (-1,-1)))
    share.update({True:share['all'],False:share['none']}) # aliases
    share = dict((dim,share[s]) for dim,s in (('sharex',sharex),('sharey',sharey)))
    ka = dict(self.ax_default,**ka)
    gridlines = ka.pop('gridlines')
    gridspec = fig.add_gridspec(nrows=nrows,ncols=ncols,**gridspec_kw)
    self.axes = axes = zeros((nrows,ncols),dtype=object); axes[...] = None
    for row in range(nrows):
      for col in range(ncols):
        ax = fig.add_subplot(gridspec[row,col],**dict((dim,axes[s(row,col)]) for dim,s in share.items()),**ka)
        ax.grid(gridlines)
        axes[row,col] = ax
        yield ax
    raise Exception(f'Insufficient number of parts on this board: {nrows*ncols}')

  def _ipython_display_(self): return self.player._ipython_display_()
