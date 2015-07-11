# File:                 demo/quickui.py
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Illustration of the quickui module

from functools import partial
from myutil import set_qtbinding; set_qtbinding() # to be set according to installation
from myutil import quickui
from myutil.qtbinding import QtCore, QtGui

class MyExperimentUI (quickui.ExperimentUI):

    def setup(self):
      def SUM(a=100,b=30): return a+b
      def DIF(a=450,b=200): return a-b
      def CMP(a=(3,5),b='abcde'):
          try: return 1 if a<b else -1 if a>b else 0
          except TypeError: return None
      Q = quickui.cbook

      # the configuration
      c = quickui.configuration(
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
      super(MyExperimentUI,self).setup(c,exper)

#--------------------------------------------------------------------------------------------------

if __name__=='__main__':
    with quickui.startup(): MyExperimentUI()

