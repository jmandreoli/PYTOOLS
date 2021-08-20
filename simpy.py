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

  def span(self): # caution: only for environments which end naturally
    self.run_robust()
    return self.now-self.init_t

# ==================================================================================================
class SimpySimulation:
  """
An instance of this class controls the rollout of a set *content* of pairs where the first component is a :class:`RobustEnvironment` instance and the second component is display specification. Attribute :attr:`player` holds a player object, created by method :math:`player_factory` with keyword arguments provided by *play_kw*. The player object executes the rollout in a timely fashion under user control on a display board (typically a :mod:`matplotlib` figure).

A display specification is a function which takes as input a :mod:`simpy.Environment` instance and a part of the display board as returned by generator method :meth:`parts`, and returns a display function with no input which displays the environment on the specified part as of the time of invocation.

:param content: list of environments with displayers
:type content: :class:`Sequence[Tuple[simpy.Environment,Callable[[simpy.Environment,Any],Callable[[],None]],...]]`
:param play_kw: a dictionary of keyword arguments, passed to the player constructor (attribute :attr:`factory`)
:param ka: passed to method :meth:`parts`
  """
# ==================================================================================================
  player: None
  r"""The player object to run the simulation"""
  play_default = {'interval':40}
  r"""The default arguments passed to the player factory"""

  def __init__(self,*content,play_kw:Mapping={},**ka):
    assert all((isinstance(env,RobustEnvironment) and all(callable(f) for f in L)) for env,*L in content)
    def displayer(board):
      content_ = [(env,tuple(f(env,part) for f in L)) for (env,*L),part in zip(content,self.parts(board,**ka))]
      def disp_(v):
        for env,L in content_:
          env.run_robust(env.init_t+v)
          for disp in L: disp()
      return disp_
    self.player = self.player_factory(displayer,**dict(self.play_default,**play_kw))

  @cached_property
  def player_factory(self):
    from matplotlib import get_backend
    from .animation import widget_animation_player, mpl_animation_player
    return widget_animation_player if 'ipympl' in get_backend() else mpl_animation_player

  ax_default = {'aspect':'equal','gridlines':True}
  r"""The default arguments passed to the ``ax_kw`` parameter in method :meth:`parts`"""
  def parts(self,fig,nrows=1,ncols=1,sharex=False,sharey=False,gridspec_kw={},**ka):
    r"""
Generator of parts. This implementation assumes the board is a :mod:`matplotlib` grid-figure, and yields its subplots (each subplot is a part). The number of parts must be at least equal to the number of environments to which they are assigned, otherwise some environments are not processed.

:param ka: a dictionary of keyword arguments passed to the :meth:`matplotlib.figure.add_subplot` method of each part (key ``gridlines`` is also allowed and denotes whether gridlines should be displayed)
    """
    from numpy import zeros
    share = dict(all=(lambda row,col: (0,0)),row=(lambda row,col: (row,0)),col=(lambda row,col: (0,col)),none=(lambda row,col: (-1,-1)))
    share.update({True:share['all'],False:share['none']}) # aliases
    share = tuple((dim,share[s]) for dim,s in (('sharex',sharex),('sharey',sharey)))
    ka = dict(self.ax_default,**ka)
    gridlines = ka.pop('gridlines')
    gridspec = fig.add_gridspec(nrows=nrows,ncols=ncols,**gridspec_kw)
    self.axes = axes = zeros((nrows,ncols),dtype=object); axes[...] = None
    for row in range(nrows):
      for col in range(ncols):
        ax = fig.add_subplot(gridspec[row,col],**dict((dim,axes[s(row,col)]) for dim,s in share),**ka)
        ax.grid(gridlines)
        axes[row,col] = ax
        yield ax
    raise Exception(f'Insufficient number of parts on this board: {nrows*ncols}')

  def _ipython_display_(self): return self.player._ipython_display_()
