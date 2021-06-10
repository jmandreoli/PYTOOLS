from __future__ import annotations
from functools import cached_property, partial
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
    if s!=self.now:
      try: self.run(s)
      except ValueError as e:
        if not str(e).startswith('until'): raise # they should have raised a proper exception
        self.reset()
        if s!=self.now: self.run(s)

class NaturalSpanMixin (RobustEnvironment):
  @cached_property # for environments which end naturally
  def span(self): self.run_robust(); return self.now-self.init_t

# ==================================================================================================
class SimpySimulation:
  """
An instance of this class controls the rollout of a set *content* of pairs where the first component is a :class:`RobustEnvironment` instance and the second component is display specification. Attribute :attr:`player` holds a player object, created by the instance, which executes the rollout in a timely fashion.
A display specification is a function which takes as input an environment and a part as returned by generator method :math:`parts`, and returns a display function with no input which displays the environment (at time of invocation) on the part.

The speed of the simulation is controlled by two parameters whose product determines the number of real milli-seconds per simulation time unit (stu):
* *frame_per_stu*: the number of frames per stu
* *interval*: the number of real milli-seconds per frame (in the player)

Note that the *interval* is bounded below by the real time needed to construct and display the frames. Furthermore, below 40ms per frame, the human eye tends to loose vision persistency.

:param content: list of environments with displayers
:param play_kw: a dictionary of keyword arguments, passed to method :meth:`init_player`
:param frame_per_stu: the number of frames per stu
:param ka: passed to method :meth:`parts`
  """
# ==================================================================================================
  player: None
  r"""The player object to run the simulation"""
  play_default = {'interval':40}
  r"""The default arguments passed to the player factory"""

  def __init__(self,*content,play_kw={},frame_per_stu:float=None,**ka):
    assert all((isinstance(env,RobustEnvironment) and all(callable(f) for f in L)) for env,*L in content)
    frame_per_stu = float(frame_per_stu)
    assert frame_per_stu > 0
    self.frame_per_stu = frame_per_stu
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
    self.init_player(**dict(self.play_default,**play_kw))
    self.content = [(env,tuple(f(env,part) for f in L)) for (env,*L),part in zip(content,self.parts(**ka))]
    self.player.bind(self.display_content)
    self.display_content(0)

  def display_content(self,n):
    dt = n/self.frame_per_stu
    for env,L in self.content:
      env.run_robust(env.init_t+dt)
      for disp in L: disp()

  def init_player(self,**ka):
    r"""
Instantiate attribute :attr:`player`. This implementation uses an instance of class :class:`animation_player`.
    """
    from matplotlib import get_backend
    backend = get_backend()
    if 'ipympl' in backend:
      from myutil.ipywidgets import animation_player, SimpleButton
      from ipywidgets import Output,Text,FloatText
      w_out = Output()
      def mpl_start(f):
        with w_out: return f()
      self.mpl_start = mpl_start
      player = animation_player((w_out,),**ka)
      w_clockb = SimpleButton(icon='history',tooltip='manually reset clock')
      w_clock = Text('',layout=dict(width='1.6cm',padding='0cm'),disabled=True)
      w_clock2 = FloatText(0,min=0,layout=dict(width='1.6cm',padding='0cm',display='none'))
      w_clock2.active = False
      def tick(n): w_clock.value = f'{n/self.frame_per_stu:.2f}'
      player.bind(tick)
      def set_clock():
        player.pause()
        w_clock2.value = player.value/self.frame_per_stu
        w_clockb.layout.visibility,w_clock.layout.display,w_clock2.layout.display,w_clock2.active = 'hidden','none','',True
      w_clockb.on_click(lambda b: set_clock())
      def clock_set():
        if not w_clock2.active: return
        w_clockb.layout.visibility,w_clock.layout.display,w_clock2.layout.display,w_clock2.active = 'visible','','none',False
        player.value = int(w_clock2.value*self.frame_per_stu)
      w_clock2.observe((lambda c: clock_set()),'value')
      player.toolbar.children += (w_clockb,w_clock,w_clock2)
    else:
      raise Exception('Unsupported backend',backend)
    self.player = player

  def bind(self,*a): self.player.bind(*a)
  def unbind(self,*a): self.player.unbind(*a)

  ax_default = {'aspect':'equal','grid':True}
  r"""The default arguments passed to the ``ax_kw`` parameter in method :meth:`parts`"""
  def parts(self,ax_kw={},**ka):
    r"""
Generator of parts. This implementation assumes the board is a :mod:`matplotlib` grid-figure, and yields its subplots (each subplot is a part). All the parts are initially invisible, and they are made visible only when returned by the generator for assignment to an environment. The number of parts must be at least equal to the number of environments to which they are assigned, otherwise some environments are not processed.

:param ax_kw: a dictionary of keyword arguments passed to the :meth:`set` method of each part (key ``grid`` is also allowed and denotes whether a grid should be displayed)
    """
    from matplotlib.pyplot import subplots,close
    ax_kw = dict(self.ax_default,**ax_kw)
    g = ax_kw.pop('grid')
    fig,axes = self.mpl_start(partial(subplots,squeeze=False,**ka))
    self.player.w_closeb.on_click(lambda b: close(fig)) # should be done but seems forgotten
    def proc(ax): ax.grid(g); ax.set(visible=False,**ax_kw); return ax
    axes = [proc(ax) for ax_row in axes for ax in ax_row]
    for ax in axes: ax.set(visible=True); yield ax

  def _ipython_display_(self): return self.player._ipython_display_()
