from __future__ import annotations
from functools import cached_property
import simpy

__all__ = 'RobustEnvironment', 'SimpySimulation'

# ==================================================================================================
class RobustEnvironment (simpy.Environment):
  """
Instances of this class are :class:`simpy.Environment` instances with method :meth:`run_robust`, which extends method :class:`run` to allow jumping back in the past (by rerunning since the initial time until the specified point in the past).
  """
# ==================================================================================================

  def __init__(self,s_init=0):
    self.s_init = s_init
    self.n_reset = -1
    self.reset()

  def reset(self):
    super().__init__(self.s_init)
    self.n_reset += 1

  def run_robust(self,s=None):
    # same as run(s) but allows going back in time
    if s!=self.now:
      try: self.run(s)
      except ValueError as e:
        if not str(e).startswith('until'): raise # they should have raised a proper exception
        self.reset()
        if s!=self.now: self.run(s)

class NaturalSpanMixin (RobustEnvironment):
  @cached_property # for environments which end naturally
  def span(self): self.run_robust(); return self.now-self.s_init

# ==================================================================================================
class SimpySimulation:
  """
An instance of this class controls the rollout of a set *L* of pairs where the first component is a :class:`RobustEnvironment` instance and the second component is some display function taking as input a :class:`simpy.Environment` instance.

The speed of the simulation is controlled by two parameters whose product determines the number of real milli-seconds per simulation time unit (stu):
* *frame_per_stu*: the number of frames per stu
* *interval*: the number of real milli-seconds per frame (in the player)

Note that the *interval* is bounded below by the real time needed to construct and display the frames. Furthermore, below 40ms per frame, the human eye tends to loose vision persistency.
  """
# ==================================================================================================
  def __init__(self,*L,frame_per_stu:float=1.,length_stu=None,player):
    assert all((isinstance(env,RobustEnvironment) and callable(display)) for env,display in L)
    self.content = L
    length_stu = max(env.span for env,_ in self.content) if length_stu is None else length_stu
    self.frame_per_stu = frame_per_stu
    self.offset = 0
    self.player = player
    player.vmin = 0
    player.value = 0
    player.vmax = int(length_stu*frame_per_stu+1)
    player.bind(self.display_content)
    self.display_content(0)
  def display_content(self,n:int):
    for env,display in self.content:
      env.run_robust(env.s_init+(n+self.offset)/self.frame_per_stu)
      display(env)
  def extend(self):
    self.offset += self.player.value
    self.player.value = 0
  def _ipython_display_(self): return self.player._ipython_display_()
