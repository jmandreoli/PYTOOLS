# File:                 simpy.py
# Creation date:        2018-05-18
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              utilities for simpy based simulation
#

from __future__ import annotations
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence
import logging; logger = logging.getLogger(__name__)

from collections import defaultdict
from functools import cached_property
from simpy import Environment

__all__ = 'SimpySimulation',

#==================================================================================================
class SimpySimulation:
  """
An instance of this class controls a simulation driven by an :class:`Environment` instance, initialised by a list of configurators launched at startup (typically starting :mod:`simpy` processes in the environment). Each pane on the board (typically a :class:`matplotlib.axes.Axes`) is painted by a list of displayers. A player, created by method :meth:`player_factory`, produces a sequence of frames (simulation time instant of type :class:`float`). While the sequence is non decreasing, the environment is advanced until that simulation time. If a frame is smaller than its previous occurrence, a new environment is created, re-initialised by all the configurations, and advanced until the requested simulation time.
  """
#==================================================================================================
  player = None
  r"""The player object to run the simulation"""
  player_defaults = {'interval':40}
  r"""The default arguments passed to the player factory"""
  @cached_property
  def player_factory(self)->Callable:
    r"""
  The factory to create the :attr:`player` object. This implementation assumes the board is a :mod:`matplotlib` figure, and the control panel factory is the default one as defined in :mod:`.animation`.
    """
    from .animation import ControlledAnimation
    return ControlledAnimation
  displayers:dict[tuple[int,int],list[tuple[str,Callable[[Any],Callable[[float],None]]]]]
  r"""A dictionary mapping each pane position to a list of named displayers"""
  configs:list[Callable[[Environment],Iterable[Any]]]
  r"""A list of environment configurations"""
  def __init__(self): self.displayers = defaultdict(list); self.configs = []
  def add_displayer(self,D:Callable[[Any],Callable[[float],None]],aspect:str,pos:tuple[int,int]=(0,0)): self.displayers[pos].append((aspect,D)); return self
  def add_config(self,C:Callable[[Environment],Iterable[Any]]): self.configs.append(C); return self
  def play(self,init_t:float=0.,displayer_kw:Mapping[str,Mapping[str,Any]]|None=None,panes_kw:Mapping[str,Any]|None=None,**ka):
    now,env = float('inf'),Environment()
    display_list:list[Callable[[float],None]] = []
    def display(v:float):
      nonlocal now,env
      if v<now:
        now,env = init_t,Environment(init_t)
        for c in self.configs: c(env)
      if v>now: env.run(v)
      now = v
      for f in display_list: f(v)
    player = self.player_factory(display,**(self.player_defaults|ka))
    panes = player.panes(**(panes_kw or {}))
    if displayer_kw is None: displayer_kw = {}
    display_list[:] = (D(panes[pos],**displayer_kw.get(aspect,{})) for pos,L in self.displayers.items() for aspect,D in L)
    return player
