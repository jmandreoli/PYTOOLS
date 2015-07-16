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

from functools import partial
from ..quickui import ExperimentUI, cbook as Q, configuration, startup
from ..pyqt import QtCore, QtGui

def demo():
  with startup() as app:
    w = Demo()
    w.main.setFixedWidth(500)
    return w

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

