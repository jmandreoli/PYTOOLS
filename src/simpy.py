# File:                 simpy.py
# Creation date:        2018-05-18
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              utilities for simpy based simulation
#

from __future__ import annotations
from typing import Mapping, Callable, Tuple, Any
from functools import cached_property

import simpy

__all__ = 'RobustEnvironment', 'SimpySimulation'

# ==================================================================================================
class RobustEnvironment (simpy.Environment):
  """
Instances of this class are :class:`simpy.Environment` instances with method :meth:`run_robust`, which extends method :class:`run` to allow jumping back in the past (by rerunning since the initial time until the specified point in the past).
  """
# ==================================================================================================
  displayers: dict[str,Callable[[Any,...],Callable[[],None]]]
  r"""A list of displayers (each in charge of a specific aspect) as used in method :meth:`displayer`"""

  def __init__(self,init_t=0):
    self.init_t = init_t
    self.n_reset = -1
    self.displayers = {}
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

  def displayer(self,pane:Any,**ka:dict[str,Any])->Callable[[],None]:
    r"""
Returns a function which displays all aspects of this environment, at the time of invocation, onto some abstract object *pane*, typically a :class:`matplotlib.axes.Axes` instance.
    """
    L = [f(pane,**kw) for aspect,kw in ka.items() if (f:=self.displayers.get(aspect)) is not None] if ka else [f(pane) for f in self.displayers.values()]
    def D():
      for disp in L: disp()
    return D

# ==================================================================================================
class SimpySimulation:
  """
An instance of this class controls the rollout of a set :class:`RobustEnvironment` instances. Attribute :attr:`player` holds a :class:`.simpy.BaseControlledAnimation` instance, created by invoking method :meth:`player_factory` with keyword arguments *play_kw*. The player object executes the rollout in a timely fashion under user control on a display board (typically a :class:`matplotlib.Figure`).

:param envs: list of environments
:param play_kw: a dictionary of keyword arguments, passed to the player constructor (attribute :attr:`player_factory`)
:param ka: passed to method :meth:`parts`
  """
# ==================================================================================================
  player = None
  r"""The player object to run the simulation"""
  player_defaults = {'interval':40}
  r"""The default arguments passed to the player factory"""
  @cached_property
  def player_factory(self)->Callable:
    r"""
  The factory to create the :attr:`player` object. This implementation assumes the board is a :mod:`matplotlib` figure, and the control panel factory is the default one as defined in :mod:`.simpy`.
    """
    from .animation import ControlledAnimation
    return ControlledAnimation

  def __init__(self,*envs:RobustEnvironment,displayer_kw:Mapping[str,Any]|None=None,panes_kw:Mapping[str,Any]|None=None,**ka):
    assert all(isinstance(env,RobustEnvironment) for env in envs)
    display_list:list[tuple[RobustEnvironment,Callable[[],None]]] = []
    def display(v):
      for env,disp in display_list: env.run_robust(v); disp()
    self.player = self.player_factory(display,**(self.player_defaults|ka))
    display_list[:] = ((env,env.displayer(pane,**(displayer_kw or {}))) for env,pane in zip(envs,self.player.panes(**(panes_kw or {}))))
    if (b:=getattr(self.player,'_repr_mimebundle_',None)) is not None: self._repr_mimebundle_ = b
