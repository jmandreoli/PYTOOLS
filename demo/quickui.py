# File:                 demo/quickui.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the quickui module
from make import RUN; RUN(__name__,__file__,2)
#--------------------------------------------------------------------------------------------------

from pathlib import Path
from functools import partial
from ..quickui import ExperimentUI, cbook as Q, configuration, startup
from qtpy import QtCore, QtGui, QtWidgets

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

    # the result tab ('main'): a simple QLabel widget
    wr = QtWidgets.QLabel('')
    QtWidgets.QVBoxLayout(self.addtab('main')).addWidget(wr)

    # setting up the experiment: each time the "Relaunch" button is clicked
    # the configuration *c* is invoked and its result *r* displayed in the main widget *wr*.
    super().setup(c,lambda r: wr.setText(str(r)))

#--------------------------------------------------------------------------------------------------

def demo():
  with startup() as app:
    w = Demo()
    w.main.setFixedWidth(500)
    def actions():
      from itertools import count; cnt = count(1)
      def snap():
        QtGui.QGuiApplication.primaryScreen().grabWindow(w.main.winId()).save(str(RUN.dir/'quickui{}.png'.format(next(cnt))),'PNG')
      actionSnap = QtWidgets.QAction(w.main); actionSnap.triggered.connect(snap)
      #actionSnap.setText('Snap'); w.toolbar.addAction(actionSnap); return
      for t,a in (.5,actionSnap),(1.,w.actionRelaunch),(1.5,actionSnap),(2.,w.actionQuit): yield t,a.trigger
    RUN.play(*actions())
