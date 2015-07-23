from functools import partial

#--------------------------------------------------------------------------------------------------
def matplotlibcfg():
  """Configurations for matplotlib animations"""
#--------------------------------------------------------------------------------------------------
  from matplotlib.animation import FuncAnimation
  del FuncAnimation.new_saved_frame_seq # fixes a bug
  from matplotlib import rcParams
  rcParams['animation.mencoder_path'] = '/opt/Linux/bin/mencoder'
matplotlibcfg()
del matplotlibcfg

#--------------------------------------------------------------------------------------------------
def launch(syst=None,animate=None,axes=(lambda fig: fig.add_subplot(1,1,1)),**ka):
#--------------------------------------------------------------------------------------------------
    from matplotlib.pyplot import figure, show
    from matplotlib.animation import FuncAnimation
    fig = figure()
    syst.display(axes(fig),animate=partial(FuncAnimation,**animate),**ka)
    show()

#--------------------------------------------------------------------------------------------------
def launchui(cfg,axes=(lambda view: view.make_axes()),width=None):
#--------------------------------------------------------------------------------------------------
  from myutil import quickui, mplext

  class MyExperimentUI (quickui.ExperimentUI):

    def setup(self):
      Q = quickui.cbook
      view = Q.mplview(self,'main')
      timer = Q.mpltimer(view.figure)
      main = lambda view,syst=None,**ka: syst.display(axes(view),**ka)
      def exper(ka):
        timer.shutdown()
        with view.clearm(): main(view,**ka)
      super(MyExperimentUI,self).setup(cfg(timer),exper)

  with quickui.startup():
    w = MyExperimentUI()
    if width is not None: w.main.setFixedWidth(width)

#--------------------------------------------------------------------------------------------------
def cfg_anim(basetimer,modifier={}):
#--------------------------------------------------------------------------------------------------
  p = pdict(
    repeat_delay=0,
    timer=pdict(
      paused=True,
      interval=300,
      ),
    save=pdict(
      fps=5,
      writer='mencoder',
      extra_args='',
      metadata=pdict(
        title='',
        artist='Jean-Marc Andreoli <jean.marc.andreoli@xrce.xerox.com>',
        ),
      )
    )
  p.update(**modifier)
  from myutil import quickui
  Q = quickui.cbook
  return (
    Q.Cclosr(partial(animate,basetimer,**p),multi=False),
    ('timer',dict(sel=True),
     Q.Ctransf(partial(dict,**p._.timer)),
     ('interval',dict(accept=Q.i,tooltip='inter-frame interval in ms'),
      Q.Cbase.LinScalar(step=1,vmin=10,vmax=1000,vtype=int)
      ),
     ('paused',dict(accept=Q.i,tooltip='whether the animation starts as paused'),
      Q.Cbase.Boolean()
      ),
     ),
    ('save',dict(sel=False,anchor=True),
     Q.Ctransf(partial(dict,**p._.save),multi=True),
     ('fps',dict(accept=Q.i,sel=False,tooltip='frame per seconds'),
      Q.Cbase.LinScalar(step=1,vmin=1,vmax=50,vtype=int)
      ),
     ('filename',dict(accept=Q.i,sel=None),
      Q.Cbase.Filename(op='save',caption='Save animation',filter='mpeg4 (*.mp4)')
      ),
     ('extra_args',dict(accept=Q.i,sel=False,tooltip='tuple of strings passed to writer command line'),
      Q.Cbase.Object(initv=(),transf=(lambda s:s.strip().split()),itransf=(lambda x:' '.join(x)),fmt=(lambda x: ' '.join(x)),editor_location='bottom')
      ),
     ('metadata',dict(),
      Q.Ctransf(partial(dict,**p._.save._.metadata)),
      ('artist',dict(accept=Q.i),
       c_str()
       ),
      ('title',dict(accept=Q.i),
       c_str()
       ),
      ),
     ),
    ('repeat_delay',dict(accept=Q.i,sel=None,tooltip='delay before next iteration of animation loop in ms (no looping behaviour if null)'),
     Q.Cbase.LinScalar(step=1000,vmin=0,vmax=10000,vtype=int)
     ),
    )

#--------------------------------------------------------------------------------------------------
# Utilities
#--------------------------------------------------------------------------------------------------

def c_str():
  from myutil import quickui
  Q = quickui.cbook
  return Q.Cbase.Object(initv='',transf=(lambda s:s),itransf=(lambda s:s),editor_location='bottom')

def animate(basetimer,fig,timer=None,save=None,**ka):
  ka['repeat'] = ka['repeat_delay']>0
  from matplotlib.animation import FuncAnimation
  anim = FuncAnimation(fig,save_count=1,**ka)
  if timer is not None:
    config(basetimer,**timer)
    basetimer.launch(anim)
  if save is not None: anim.save(**save)

def config(x,**ka):
  for k,v in ka.items(): setattr(x,k,v)
  return x

class pdict (dict):
  class obj(object): pass
  def __init__(self,**kwd):
    super(pdict,self).__init__((k,v) for k,v in kwd.items() if not isinstance(v,pdict))
    self._ = config(self.obj(),**dict((k,v) for k,v in kwd.items() if isinstance(v,pdict)))
  def update(self,**d):
    if not d: return
    for k,v in self._.__dict__.items():
      try: dd = d.pop(k)
      except: pass
      else: v.update(**dd)
    if not d: return
    super(pdict,self).update(**d)
