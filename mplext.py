# File:                 mplext.py
# Creation date:        2013-02-15
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              A few matplotlib utilities
#

import os,sys,logging
from contextlib import contextmanager
logger = logging.getLogger(__name__)

#==============================================================================
class Cell:
  r"""
Instances of this class identify a cell, ie. a matplotlib figure equipped with a subplot specification. Instances can be incarnated, either into matplolib axes or into a grid, ie. an array of sub-cells.

:param fig: matplotlib figure.
:type fig: :class:`matplotlib.figure.Figure`
:param sps: matplotlib subplot specification.
:type sps: :class:`matplotlib.gridspec.SubplotSpec`

Attributes:

.. attribute:: figure

   from the initialisation

.. attribute:: sps

   from the initialisation

.. attribute:: axes

   :const:`None` or an instance of :class:`matplotlib.artists.Axes`
   (see method :meth:`make_axes`)

.. attribute:: grid

   :const:`None` or an instance of :class:`matplotlib.gridspec.GridSpecFromSubplotSpec`
   (see method :meth:`make_grid`)

.. attribute:: callbacks

   :const:`None` if :attr:`axes` is :const:`None`,
   otherwise a list of matplotlib callbacks, attached to the axes.

.. attribute:: offspring

   :const:`None` if :attr:`grid` is :const:`None`
   otherwise an array of :class:`Cell` instances or :const:`None`.
   The elements of the array are all initially :const:`None`
   and are lazily created when accessed by method :meth:`__getitem__`.

Methods:
  """
#==============================================================================

  def __init__(self,fig,sps):
    self.figure = fig
    self.sps = sps
    self.grid = None
    self.offspring = None
    self.axes = None
    self.callbacks = []

#------------------------------------------------------------------------------
  def make_grid(self,rows,cols,**ka):
    r"""Incarnates *self* as grid and returns :attr:`grid`."""
#------------------------------------------------------------------------------
    assert self.axes is None and self.grid is None, 'Cell must be cleared before shaped'
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    import numpy
    self.grid = GridSpecFromSubplotSpec(rows,cols,self.sps,**ka)
    self.shape = shape = rows,cols
    self.offspring = numpy.zeros(shape,object)
    self.offspring[...] = None
    return self.grid

  def make_axes(self,**ka):
    r"""Incarnates *self* as axes and returns :attr:`axes`."""
    assert self.axes is None and self.grid is None, 'Cell must be cleared before shaped'
    self.axes = self.figure.add_subplot(self.sps,**ka)
    return self.axes


#------------------------------------------------------------------------------
  def __getitem__(self,ij):
    r"""
(only when :attr:`grid` is not :const:`None`). Lazily accesses the cell at row *i*, col *j*. The argument must be a tuple *i,j* where *i* and *j* can be numbers (type :class:`int`) or ranges (type :class:`slice`).
    """
#------------------------------------------------------------------------------
    assert self.grid is not None
    def start(i): return (0 if i.start is None else i.start) if isinstance(i,slice) else i
    i,j = ij
    i0 = start(i)
    j0 = start(j)
    c = self.offspring[i0,j0]
    if c is None: self.offspring[i,j] = c = Cell(self.figure,self.grid[i,j])
    return c

#------------------------------------------------------------------------------
  def connect(self,D=dict(button_press=0,button_release=1,draw=2,key_press=3,key_release=4,motion_notify=5,pick=6,resize=7,scroll=8,figure_enter=9,figure_leave=10,axes_enter=11,axes_leave=12,close=13),**ka):
    r"""
Connects slot *f* to signals of type specified by *D*.
    """
#------------------------------------------------------------------------------
    assert all(D.get(evt) is not None for evt in ka)
    for evt,f in ka.items():
      b = self.figure.canvas.mpl_connect(evt+'_event',(lambda ev: f(ev)))
      self.callbacks.append(b)

#------------------------------------------------------------------------------
  def clear(self):
    r"""
Clears *self*. All callbacks are disconnected.
If :attr:`axes` is not :const:`None`, the axes is detached from the figure.
If :attr:`grid` is not :const:`None`, all offspring cells are cleared.
After execution, both :attr:`axes` and :attr:`grid` are reset to :const:`None`.
    """
#------------------------------------------------------------------------------
    if self.grid is not None:
      self.grid = None
      for cr in self.offspring:
        for c in cr:
          if c is not None: c.clear()
      self.offspring = None
    elif self.axes is not None:
      self.figure.delaxes(self.axes)
      self.axes = None
    for b in self.callbacks: self.figure.canvas.mpl_disconnect(b)
    del self.callbacks[:]

#------------------------------------------------------------------------------
  @contextmanager
  def clearm(self):
    r"""
Returns a context manager which clears *self* on enter and redraws :attr:`figure` on exit.
    """
#------------------------------------------------------------------------------
    self.clear()
    yield
    self.figure.canvas.draw()

#------------------------------------------------------------------------------
  @staticmethod
  def create(fig=None,sps={},**ka):
    r"""
Factory of :class:`Cell` objects.

:param fig: description of a target figure.
:type fig: :class:`matplotlib.figure.Figure`
:param sps: description of a subplot specification.
:type sps: :const:`Dict`

If *fig* is :const:`None`, a matplotlib figure in a new window is created, using *ka* as keyword arguments. Otherwise, *fig* must be an existing matplotlib figure and *ka* empty. Argument *sps* is a dict passed to :class:`matplotlib.gridspec.GridSpec`.
    """
#------------------------------------------------------------------------------
    if fig is None:
      from matplotlib.pyplot import figure
      fig = figure(**ka)
    else:
      from matplotlib.figure import Figure
      assert isinstance(fig,Figure) and not ka
    from matplotlib.gridspec import GridSpec
    return Cell(fig,GridSpec(1,1,**sps)[0,0])

#==============================================================================
class Timer (object):
  r"""
Wraps a timer to make it pausable.

:param timer: base timer
:type timer: :class:`matplotlib.backend_bases.TimerBase`
:param paused: initial pause status
:type paused: :const:`bool`

Attributes and methods:
  """
#==============================================================================

  @property
  def interval(self): return self.interval_
  r"""The wrapped timer's interval"""

  @interval.setter
  def interval(self,v):
    self.interval_ = v
    if not self.paused: self.timer.interval = v

  def __init__(self,timer,paused=False):
    self.timer = timer
    self.paused = paused
    self.animation = None
    self.interval_ = 0
    self.running = False

  def start(self):
    assert self.animation is not None # should come from the current animation
    self.running = True
    if not self.paused: self.timer.start()

  def stop(self):
    assert self.animation is not None # should come from the current animation
    self.running = False
    self.timer.stop()

  def add_callback(self,f):
    self.timer.add_callback(f)

  def remove_callback(self,f):
    self.timer.remove_callback(f)

  def togglepause(self):
    r"""Toggles the pause status of the timer."""
    self.paused = not self.paused
    self.set_paused()
    if not self.paused and self.running: self.timer.start()

  def set_paused(self):
    if self.paused:
      self.timer.single_shot = True
      self.timer.interval = 0
      self.status(-1)
    else:
      self.timer.single_shot = False
      self.timer.interval = self.interval_
      self.status(1)

  def onestep(self):
    r"""Performs a single shot of the timer if paused."""
    assert self.paused
    self.timer.start()

  def launch(self,anim):
    r"""Launches *self* and attaches *anim* to it. *anim* must be a :class:`matplotlib.animation.TimedAnimation` instance."""
    from matplotlib.animation import TimedAnimation
    assert self.animation is None and isinstance(anim,TimedAnimation)
    anim.event_source = self
    self.animation = anim
    anim._interval = self.interval_
    self.set_paused()

  def shutdown(self):
    r"""Shutdowns *self* and stops and detachs its animation, if present."""
    if self.animation is None: return
    self.animation._stop()
    self.timer.stop()
    self.animation = None
    self.status(0)

  def status(self,s):
    r"""
Invoked when the pause status of the timer changes (or it ends). This implementation does nothing. It can be redefined in subclasses or at the instance level. *s* is :const:`0` when the timer is stopped, :const:`-1` when it is paused, :const:`1` when it is unpaused.
    """
    pass

#==============================================================================
class Toolbar:
  r"""
:param fig: a matplotlib figure
:type fig: :class:`matplotlib.figure.Figure`
:param size: the size of the toolbar in figure coordinates (hence between 0. and 1.)
:type size: :class:`float`
:param ka: passed to the button factory :class:`matplotlib.text.Text`

Objects of this class control a toolbar added to a figure. The toolbar is located on the top of the figure, initially invisible. Two buttons at the top corners remain always visible and allow toggling the visibility of the toolbar. Simulates a very restricted subset of the API of the QT toolbar (which should be preferred when the backend is QT).
  """
#==============================================================================

  bstyle = dict(backgroundcolor='k',color='w',fontsize='small')

  def __init__(self,fig,size=.02,**ka):
    from matplotlib.patches import Rectangle
    from collections import OrderedDict
    toggles = [Rectangle((x,1.),size*e,-size,transform=fig.transFigure,picker=True,figure=fig,zorder=2,fc='gray') for x,e in ((0.,1.),(1.,-1.))]
    fig.patches.extend(toggles)
    self.ax = fig.add_axes((size,1.-size,1.-2*size,size),xticks=(),yticks=(),frameon=False,zorder=1,visible=False)
    def onpick(ev,ax=self.ax,toggles=toggles):
      if ev.artist in toggles:
        ax.set_visible(not ax.get_visible())
        ax.figure.canvas.draw()
        return
      a = self.actions.get(ev.artist)
      if a is not None: a()
    fig.canvas.mpl_connect('pick_event',onpick)
    self.actions = OrderedDict()
    for k,v in self.bstyle.items(): ka.setdefault(k,v)
    self.bstyle = ka.items()

  def addAction(self,label,callback,**ka):
    r"""Adds (and returns) a button with given label and callback."""
    for k,v in self.bstyle: ka.setdefault(k,v)
    b = self.ax.text(0.,1.,label,transform=self.ax.transAxes,va='top',ha='center',picker=True,**ka)
    self.actions[b] = callback
    self.setpos()
    return b

  def delAction(self,b):
    r"""Deletes a button previously added by :meth:`addAction`."""
    del self.actions[b]
    self.setpos()

  def setpos(self):
    n = len(self.actions)+1
    for i,b in enumerate(self.actions,1): b.set_x(i/n)
    self.ax.figure.canvas.draw()

#==================================================================================================
# Utilities
#==================================================================================================

#------------------------------------------------------------------------------
def pager(L,shape,vmake,vpaint,offset=0,toolbar=None,save=None,savedefaults=dict(dest='pager-sav',format='svg'),bstyle={},**_ka):
  r"""
:param L: a list of arbitrary objects other than :const:`None`
:type L: :class:`List[object]`
:param shape: a pair (number of rows, number of columns) or a single number if they are equal
:type shape: :class:`Union[int,Tuple[int,int]]`
:param vmake: a function to initialise the display (see below)
:type vmake: :class:`Callable[[Cell],None]`
:param vpaint: a function to instantiate the display for a given page (see below)
:type vpaint: :class:`Callable[[Cell,object],None]`
:param savedefaults: used as default keyword arguments of method :meth:`savefig` when saving pages
:type savedefaults: :class:`Dict`

This function first create a :class:`Cell` instance with all the remaining arguments, then splits it into a grid of sub-cells according to *shape*, then displays *L* page per page on the grid. Each page displays a slice of *L* of length equal to the product of the components of *shape* (or less, for the final page). The toolbar is enriched with page navigation buttons. A save button also allows to save the whole collection of pages in a given directory (beware: may be long).

Function *vmake* takes as input a :class:`Cell` instance and instantiates it as needed. It can store information (e.g. about the specific role of each artist created in the cell), if needed, by simply setting attributes in the cell. This is called once for each cell at the begining of the display.

Function *vpaint* takes as input a cell and an element of *L* or None, and displays that element in the cell (or resets the cell to indicate a missing value), possibly using the artists created by *vmake* and stored in the cell. This is called once at each page display and for each cell.

Unfortunately, matplotlib toolbars are not standardised: the depend on the backend and may not support adding button.
  """
#------------------------------------------------------------------------------
  from numpy import ceil, rint, clip
  from matplotlib.text import Text
  from matplotlib.widgets import Slider
  from matplotlib.pyplot import close
  from pathlib import Path
  from shutil import rmtree
  def gen(L):
    yield from L
    while True: yield None
  def genc(cell):
    Nr,Nc = cell.shape
    yield from (cell[row,col] for row in range(Nr) for col in range(Nc))
  def paintp(cell,p,draw=True):
    for c,x in zip(genc(cell),gen(L[p*cellpp:])): vpaint(c,x)
    if draw: cell.figure.canvas.draw()
  def toggle_ctrl():
    ctrl.ax.set_visible(not ctrl.ax.get_visible())
    cell.figure.canvas.draw()
  def save_all():
    ka = _ka.copy()
    ka.update(fig=None,figsize=((cell.figure.get_figwidth(),cell.figure.get_figheight())))
    #import multiprocessing
    #multiprocessing.get_context('spawn').Process(target=pager,args=(L,shape,vmake,vpaint),kwargs=dict(save={},savedefaults=savedefaults,**ka)).start()
    pager(L,shape,vmake,vpaint,save={},savedefaults=savedefaults,**ka)
  cell = Cell.create(**_ka)
  Nr,Nc = (shape,shape) if isinstance(shape,int) else shape
  cell.make_grid(Nr,Nc)
  for c in genc(cell): vmake(c)
  cellpp = Nr*Nc
  npage = int(ceil(len(L)/cellpp))
  if save is None:
    actions = [
      ('<<',(lambda:ctrl.set_val(clip(ctrl.val-1,1,npage)))),
      ('>>',(lambda:ctrl.set_val(clip(ctrl.val+1,1,npage)))),
      ('toggle-ctrl',toggle_ctrl),
      ('save-all',save_all),
      ]
    if toolbar is None:
      try: toolbar = cell.figure.canvas.toolbar; toolbar.addAction
      except: toolbar = Toolbar(cell.figure,**bstyle)
    for a,f in actions: toolbar.addAction(a,f)
    ctrl = Slider(cell.figure.add_axes((0.1,0.,.8,.03),visible=False,zorder=1),'page',.5,npage+.5,valinit=0,valfmt='%.0f/{}'.format(npage),closedmin=False,closedmax=False)
    ctrl.on_changed(lambda p:paintp(cell,int(rint(p))-1))
    ctrl.set_val(1+offset/cellpp)
  else:
    s = savedefaults.copy()
    s.update(save)
    pth = Path(s.pop('dest'))
    try:
      assert pth.is_dir()
      for f in list(pth.iterdir()): rmtree(str(f))
    except Exception as e:
      logger.warn('Error on save directory %s: %s',path,e)
      raise
    try:
      for p in range(npage):
        paintp(cell,p,False)
        cell.figure.savefig(str((pth/'p{:02d}'.format(p)).with_suffix('.'+s['format'])),**s)
    except Exception as e: logger.warn('Error saving page %s: %s',p,e)
    close(cell.figure.number)
