# File:                 demo/quickui.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the quickui module

if __name__=='__main__':
  import sys
  from myutil.demo.quickui import demo
  demo()
  sys.exit(0)

#--------------------------------------------------------------------------------------------------

from pathlib import Path
from functools import partial
from ..quickui import ExperimentUI, cbook as Q, configuration, startup
from ..pyqt import QtCore, QtGui
automatic = False

def demo():
  with startup() as app:
    w = Demo()
    w.main.setFixedWidth(500)
    if automatic: autoplay(w)

class Demo (ExperimentUI):

  def setup(self):
    def SUM(a=100,b=30): return a+b
    def DIF(a=450,b=200): return a-b
    def CMP(a=(3,5),b='abcde'):
      try: return 1 if a<b else -1 if a>b else 0
      except TypeError: return None

    # the configuration
    c = configuration(
        Q.Ccomp(tuple,{},multi=True),
        ('sum',dict(sel=True,tooltip='Summation operator'),Q.Ctransf(SUM),
         ('a',dict(accept=Q.i,tooltip='First operand of sum'),Q.Cbase.LinScalar(vtype=int)),
         ('b',dict(accept=Q.i,tooltip='Second operand of sum'),Q.Cbase.LinScalar(vtype=int)),
         ),
        ('dif',{},Q.Ctransf(DIF),
         ('a',dict(accept=Q.i),Q.Cbase.LinScalar(vtype=int)),
         ('b',dict(accept=Q.i),Q.Cbase.LinScalar(vtype=int)),
         ),
        ('cmp',dict(sel=None),Q.Ctransf(CMP),
         ('a',dict(accept=Q.i),Q.Cbase.Object(initv=None,transf=eval,itransf=repr,tooltip='Enter any python expression')),
         ('b',dict(accept=Q.i),Q.Cbase.Object(initv=None,transf=eval,itransf=repr,tooltip='Enter any python expression')),
         ),
        )

    # the result tab ('main')
    wr = QtGui.QLabel('')
    QtGui.QVBoxLayout(self.addtab('main')).addWidget(wr)


    # the experiment
    def exper(r): wr.setText(str(r))
    super(Demo,self).setup(c,exper)

def autoplay(w):
  import time
  from threading import Thread
  def save(wid=w.main.winId(),DIR=Path(__file__).resolve().parent):
    QtGui.QPixmap.grabWindow(wid).save(str(DIR/'quickui{}.png'.format(w.forsave)))
  actionSave = QtGui.QAction(w.main)
  QtCore.QObject.connect(actionSave, QtCore.SIGNAL("triggered()"), save)
  def play(w):
    time.sleep(.5)
    w.forsave = 1; actionSave.trigger(); time.sleep(.1)
    w.actionRelaunch.trigger(); time.sleep(.1)
    w.forsave = 2; actionSave.trigger(); time.sleep(.1)
    w.actionQuit.trigger()
  Thread(target=play,args=(w,),daemon=True).start()

