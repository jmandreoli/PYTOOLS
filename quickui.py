# File:                 quickui.py
# Creation date:        2012-08-12
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Quick Qt5 UI design
#
# *** Copyright (c) 2012 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import os,sys,logging,traceback,functools,collections
from contextlib import contextmanager
from qtpy import QtCore, QtGui, QtWidgets
logger = logging.getLogger(__name__)

#--------------------------------------------------------------------------------------------------
class BaseTabUI (object):
  r"""
An instance of this class builds a Qt5 application with a single main window having a set of tabs.

UI components are accessible through attributes:
:attr:`main`, :attr:`centralwidget`, :attr:`tabw`,
:attr:`menubar`, :attr:`statusbar`, :attr:`toolbar`,
:attr:`menuFile`, :attr:`actionQuit`.

Methods:
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self):
    self.main = QtWidgets.QMainWindow()
    self.main.setObjectName("MainWindow")
    self.centralwidget = QtWidgets.QWidget(self.main)
    self.centralwidget.setObjectName("centralwidget")
    layout = QtWidgets.QVBoxLayout(self.centralwidget)
    self.tabw = QtWidgets.QTabWidget(self.centralwidget)
    self.tabw.setObjectName("tabw")
    layout.addWidget(self.tabw)
    self.main.setCentralWidget(self.centralwidget)
    self.menubar = QtWidgets.QMenuBar(self.main)
    self.menubar.setObjectName("menubar")
    self.menuFile = QtWidgets.QMenu(self.menubar)
    self.menuFile.setObjectName("menuFile")
    self.main.setMenuBar(self.menubar)
    self.statusbar = QtWidgets.QStatusBar(self.main)
    self.statusbar.setObjectName("statusbar")
    self.main.setStatusBar(self.statusbar)
    self.toolbar = QtWidgets.QToolBar(self.main)
    self.toolbar.setObjectName("toolbar")
    self.main.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)
    self.actionQuit = QtWidgets.QAction(self.main)
    self.actionQuit.setObjectName("actionQuit")
    self.actionQuit.triggered.connect(QtWidgets.QApplication.quit)
    self.menuFile.addAction(self.actionQuit)
    self.menubar.addAction(self.menuFile.menuAction())
    self.toolbar.addAction(self.actionQuit)
    QtCore.QMetaObject.connectSlotsByName(self.main)

    self.main.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "MainWindow"))
    self.menuFile.setTitle(QtWidgets.QApplication.translate("MainWindow", "File"))
    self.toolbar.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "toolbar"))
    self.actionQuit.setText(QtWidgets.QApplication.translate("MainWindow", "Quit"))

    self.main.show()

  def addtab(self,label):
    r"""Adds a new tab with label *label* and returns it."""
    w = QtWidgets.QWidget()
    w.setObjectName('tab_'+label)
    self.tabw.addTab(w,'')
    self.tabw.setTabText(self.tabw.indexOf(w), QtWidgets.QApplication.translate("MainWindow", label))
    return w

#--------------------------------------------------------------------------------------------------
class ExperimentUI (BaseTabUI):

  r"""
The first tab is reserved to a configuration pane and the other tabs to the display of results.

The result panes are disabled if the configuration has been modified. To re-enable them, the experiment must be (re-)launched. It can be relaunched any number of times (useful only if the experiment is non deterministic). Each launch is assumed to refresh the content of the result panes based on the content of the configuration pane.

Additional UI components are accessible through attributes:
:attr:`actionRelaunch`, :attr:`actionLoad`, :attr:`actionSave`,
:attr:`configlayout`, :attr:`navigbar`, :attr:`navigback`, :attr:`navigpage`,
:attr:`console`.

Attribute:

.. attribute:: configstylesheet

   Stylesheet specification to apply to the configuration tab.

Methods:
  """
#--------------------------------------------------------------------------------------------------

  configstylesheet = 'QWidget[configlabel] {border-top: 2px solid blue; border-left: 1px dashed blue}\nQWidget[configclass] {border: 1px dashed black}'

  def __init__(self):
    super(ExperimentUI,self).__init__()
    self.actionRelaunch = self.action('relaunch')
    self.toolbar.addAction(self.actionRelaunch)
    def relaunch():
      self.run()
      self.main.show()
    self.actionRelaunch.triggered.connect(relaunch)
    self.actionLoad = self.action('load')
    self.actionSave = self.action('save')
    self.menuFile.insertAction(self.actionQuit,self.actionLoad)
    self.menuFile.insertAction(self.actionQuit,self.actionSave)
    def fileop(f,ka=dict(parent=self.main)):
      def F():
        s = 'failed'
        try:
          s = f(self.navig[0],**ka)
          if s: self.tabw.setCurrentIndex(0); s = 'successful'
        except: self.consolemessage()
        finally:
          if s: self.statusbar.showMessage('Operation: {0}'.format(s),2000)
      return F
    self.actionLoad.triggered.connect(fileop(lambda c,**ka: c.parload(**ka)))
    self.actionSave.triggered.connect(fileop(lambda c,**ka: c.parsave(**ka)))

    self.synched = None
    self.navig = None
    self.exper = None

    self.configlayout = QtWidgets.QVBoxLayout(self.addtab('config'))
    self.configlayout.parentWidget().setStyleSheet(self.configstylesheet)
    self.navigbar = QtWidgets.QWidget()
    self.configlayout.addWidget(self.navigbar)
    layout = QtWidgets.QHBoxLayout(self.navigbar)
    self.navigpage = QtWidgets.QLabel('')
    self.navigback = QtWidgets.QPushButton('<<')
    layout.addWidget(self.navigpage)
    layout.addWidget(self.navigback)
    layout.addStretch(0)
    self.navigbar.setVisible(False)
    self.navigback.clicked.connect(lambda: self.select(-1))
    self.console = InfoPage(self.main,('Console',))
    self.configlayout.addWidget(self.console.widget)
    self.console.widget.setVisible(False)
    self.setup()

  def setsynch(self,flag):
    if self.synched != flag:
      self.synched = flag
      for i in range(1,self.tabw.count()): self.tabw.setTabEnabled(i,flag)

  def run(self):
    QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
    self.statusbar.clearMessage()
    if self.navig[-1] is self.console: self.select(-1)
    s = 'failed'
    try: self.exper() ; s = 'successful'
    except: self.consolemessage(); raise
    finally:
      self.statusbar.showMessage('Launch: {0}'.format(s),2000)
      QtWidgets.QApplication.restoreOverrideCursor()
    self.setsynch(True)
    if self.tabw.currentIndex()==0: self.tabw.setCurrentIndex(1)

  def consolemessage(self,msg=None):
    r"""Displays the console with message *msg* (a plain text string)."""
    if msg is None: msg = traceback.format_exc()
    self.console.widget.clear()
    self.console.widget.setPlainText(msg)
    self.select(self.console)
    self.tabw.setCurrentIndex(0)

  def select(self,c):
    h = self.navig
    h[-1].widget.setVisible(False)
    if c == -1:
      h.pop(-1)
      if len(h)==1: self.navigbar.setVisible(False)
    else:
      h.append(c)
      self.navigbar.setVisible(True)
    self.navigpage.setText('.'.join(h[-1].title))
    h[-1].widget.setVisible(True)

  def action(self,label):
    label = label.capitalize()
    a = QtWidgets.QAction(self.main)
    a.setObjectName("action"+label)
    a.setText(QtWidgets.QApplication.translate("MainWindow", label))
    return a

  def setup(self,c0,exper):
    r"""
Completes the initialisation of *self*, using configurator *c0* and one-input, one-output experiment function *exper*. At the end of :meth:`__init__`, :meth:`setup` is invoked with no argument. Subclasses must override method :meth:`setup` with no argument to make it invoke this method :meth:`setup` with the appropriate arguments.
    """
    assert self.exper is None
    self.navig = [c0]
    self.configlayout.addWidget(c0.widget)
    c0.onchange(lambda: self.setsynch(False))
    for c in c0.domain():
      self.configlayout.addWidget(c.widget)
      c.widget.setVisible(False)
      c.select = lambda c=c: self.select(c)
    self.setsynch(False)
    self.exper = lambda: exper(c0())

class InfoPage (object):

  def __init__(self,parent=None,title=()):
    self.widget = QtWidgets.QTextEdit(parent)
    self.widget.setReadOnly(True)
    self.title = title

#--------------------------------------------------------------------------------------------------
class Configurator (object):
  r"""
Abstract. A configurator is basically a callable object which can be parameterised through a set of widgets. Used essentially to set up the configuration pane of :class:`ExperimentUI` objects.

Methods:
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self):
    self.widget = None
    self.title = ()

  def __call__(self):
    r"""Returns the result of the execution of *self*. Defined in subclasses."""
    raise NotImplementedError

  def parset(self,v):
    r"""Sets *v* as parameter of *self*. Defined in subclasses."""
    raise NotImplementedError

  def parget(self):
    r"""Returns the parameter of *self*. Defined in subclasses."""
    raise NotImplementedError

  def onchange(self,f):
    r"""Attaches callback *f* (a 0-input callable) to any change in the parameter. Defined in subclasses."""
    raise NotImplementedError

  def hide(self):
    raise NotImplementedError

  def parload(self,**ka):
    r"""Loads parameter of *self* from a file."""
    import pickle
    ka.setdefault('filter','Quickui config (*.qcfg)')
    filen = QtWidgets.QFileDialog.getOpenFileName(**ka)[0]
    if filen:
      with open(filen,'rb') as u: self.parset(pickle.load(u))
      return True

  def parsave(self,**ka):
    r"""Saves parameter of *self* into a file."""
    import pickle
    ka.setdefault('filter','Quickui config (*.qcfg)')
    filen = QtWidgets.QFileDialog.getSaveFileName(**ka)[0]
    if filen:
      if not os.path.splitext(filen)[1]: filen += '.qcfg'
      with open(filen,'wb') as v: pickle.dump(self.parget(),v)
      return True

  def domain(self): return ()

#--------------------------------------------------------------------------------------------------
class EmptyConfigurator (Configurator):
  r"""
An instance of this class, when called, always returns its initial value, which is immutable.

:param v: initial value of the configurator.
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,v):
    super(EmptyConfigurator,self).__init__()
    self.value = v

  def __call__(self):
    return self.value

  def parget(self): return None
  def parset(self,v): pass

  def onchange(self,f): pass

  def hide(self,flag): pass

#--------------------------------------------------------------------------------------------------
class NonEmptyConfigurator (Configurator):
#--------------------------------------------------------------------------------------------------

  def __init__(self):
    super(NonEmptyConfigurator,self).__init__()
    self.widget = QtWidgets.QWidget()

  def hide(self,flag):
    self.widget.setDisabled(flag)

  def select(self):
    raise NotImplementedError

#--------------------------------------------------------------------------------------------------
class BaseConfigurator (NonEmptyConfigurator):
  r"""
An instance of this class, when called, returns the value held by an editor widget, such as instances of :class:`EditWidget`.

:param w: editor widget.
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,w,getval=None,setval=None,onchange=None):
    super(BaseConfigurator,self).__init__()
    ws = QtWidgets.QStackedWidget()
    ws.setProperty('configclass',w.__class__.__name__)
    QtWidgets.QVBoxLayout(self.widget).addWidget(ws)
    ws.addWidget(w)
    if getval is None: getval = lambda: w.value
    if setval is None: setval = lambda v: w.setValue(v)
    if onchange is None: onchange = lambda f: w.valueChanged.connect(lambda *a: f())
    self.parget = getval
    self.parset = setval
    self.onchange = onchange

  def __call__(self):
    return self.parget()

#--------------------------------------------------------------------------------------------------
class CompoundConfigurator (NonEmptyConfigurator):
  r"""
A compound configurator merges several labeled child configurators (offspring).

:param multi: whether multiple child configurators can be active simultaneously.
:type multi: :const:`NoneType`\|\ :const:`bool`
:param proc: how to merge the results of the execution of the active children.
:type proc: 1-input callable

If *multi* is :const:`None`, all child configurators are active. If *multi* is not :const:`None`, some labels may be selectable to activate/deactivate the corresponding child.

The execution of a compound configurator consists of the execution of its active children, followed by invocation of *proc* on the list of pairs (*label*, *res*), where *label* is the label of each active child and *res* the result of its execution.

The widget of a compound configurator is a grid with 2 columns and one row for each child. The label of each child appears in the first column, and its widget in the second column, except for anchored children, where the widget is replaced by a push button which displays the widget when clicked. This allows to control the size of the compound configurator widgets.

Methods:
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,multi=None,proc=tuple):
    super(CompoundConfigurator,self).__init__()
    self.multi = multi
    self.proc = proc
    self.anchors = []
    self.offspring = []
    self.layout = layout = QtWidgets.QGridLayout(self.widget)
    layout.setColumnStretch(0,0)
    layout.setColumnStretch(1,1)
    layout.setContentsMargins(2,1,2,1)
    layout.setSpacing(5)
    if multi is None:
      self.onselchange = (lambda f: None)
    else:
      assert isinstance(multi,bool)
      self.bgroup = bgroup = QtWidgets.QButtonGroup(self.widget)
      bgroup.setExclusive(not multi)
      if multi:
        def onselchange(f):
          bgroup.buttonClicked[int].connect(lambda k: f())
        def sethide(k):
          self.offspring[k][1].hide(not bgroup.button(k).isChecked())
        bgroup.buttonClicked[int].connect(sethide)
      else:
        def onselchange(f):
          bgroup.buttonPressed[int].connect(lambda k: f() if k != bgroup.checkedId() else None)
        def sethide(k):
          h = bgroup.checkedId()
          if k != h:
            self.offspring[k][1].hide(False)
            self.offspring[h][1].hide(True)
        bgroup.buttonPressed[int].connect(sethide)
      self.onselchange = onselchange

  def parget(self):
    return tuple((label,None if button is None else button.isChecked(),c.parget()) for label,c,button in self.offspring)

  def parset(self,l):
    d = dict((label,(sel,v)) for label,sel,v in l)
    for label,c,button in self.offspring:
      x = d.get(label)
      if x is None: continue
      sel,v = x
      if sel is not None and button is not None:
        button.setChecked(sel)
        c.hide(not sel)
      c.parset(v)

  def onchange(self,f):
    for label,c,button in self.offspring: c.onchange(f)
    self.onselchange(f)

  def domain(self):
    for label,c,button in self.offspring:
      if c in self.anchors: yield c
      for cc in c.domain(): yield cc

  def __iter__(self):
    for label,c,button in self.offspring:
      if button is None or button.isChecked(): yield label, c

  def addconfig(self,label,c,anchor=False,sel=-1,accept=None,tooltip=None):
    r"""
Adds a child configurator *c* to *self*.

:param anchor: whether the child should be anchored.
:type anchor: :const:`bool`
:param sel: whether the label should be initially selected.
:type sel: :const:`NoneType`\|\ :const:`bool`
:param accept: immediately invoked with *self*, *label*, *c* as input
:type accept: 3-input function
:param tooltip: tooltip to appear on the child label
:type tooltip: :const:`str`

If *sel* is not given, it defaults to *self*'s child activation status. If the latter is :const:`None` (no child activation is permitted) then *sel* must be :const:`None`. The *accept* function typically encodes repetitive operations on joints (like assigning a default parameter to the child obtained from the parent, or checking conformance to a rule).
    """
    assert isinstance(c,Configurator)
    assert isinstance(label,str)
    if sel==-1: sel = self.multi
    row = len(self.offspring)
    if sel is None:
      qlabel = QtWidgets.QLabel(label)
      qlabel.setAlignment(QtCore.Qt.AlignVCenter|QtCore.Qt.AlignHCenter)
      button = None
    else:
      assert isinstance(sel,bool) and self.multi is not None
      button = qlabel = QtWidgets.QPushButton(label)
      qlabel.setCheckable(True)
      qlabel.setChecked(sel)
      self.bgroup.addButton(qlabel)
      self.bgroup.setId(qlabel,row)
      c.hide(not sel)
    if accept is not None: accept(self,label,c)
    w = QtWidgets.QStackedWidget()
    self.layout.addWidget(w,row,0)
    w.setProperty('configlabel',button is not None)
    w.addWidget(qlabel)
    if tooltip is not None: w.setToolTip(tooltip)
    self.offspring.append((label,c,button))
    c.title = self.title+(label,)
    if isinstance(c,NonEmptyConfigurator):
      if anchor:
        w = QtWidgets.QPushButton('>>')
        #w.clicked.connect(lambda c=c: c.select()) # bug in python3
        w.clicked.connect(activeselect(c))
        self.anchors.append(c)
      else:
        w = c.widget
      self.layout.addWidget(w,row,1)
    self.layout.setRowStretch(row,1)

  def hide(self,flag):
    super(CompoundConfigurator,self).hide(flag)
    for c in self.anchors: c.hide(flag)

  def __call__(self):
    return self.proc((label,c()) for label,c in self)

def activeselect(c):
  return lambda: c.select()

#--------------------------------------------------------------------------------------------------
def configuration(c,*offspring):
  r"""
Returns compound configurator *c*, modified by the *offspring* specification.

:param c: a configurator
:type c: :class:`CompoundConfigurator`
:param offspring: a specification
:type offspring: :const:`tuple` (see below)

The offspring specification is a tuple of tuples, each consisting of the following components:

* Item 0: a label *lbl*
* Item 1: a possibly empty dictionary *ka*
* Item 2: a configurator *cc*
* Further items, forming a possibly empty offspring specification *o*

Configurator *cc* is appended as offspring to *c* with label *lbl* and *ka* passed as keyword parameters to :meth:`addconfig`. Function :func:`configuration` is recursively called on configurator *cc* with offspring specification *o*.
  """
#--------------------------------------------------------------------------------------------------
  assert isinstance(c,Configurator)
  for spec in offspring:
    c.addconfig(spec[0],configuration(*spec[2:]),**dict(spec[1]))
  return c

#--------------------------------------------------------------------------------------------------
class EditWidget (QtWidgets.QWidget):
  r"""
Abstract. An instance of this class is a widget which allows editing of some value.

Attributes:

.. attribute:: valueChanged

   A signal triggered when the edited value is changed, either due to the editor activity or to external assignment.

.. attribute:: ready

   A signal triggered when the editor is in a coherent state. Available only where the editor supports transient incoherent states.

.. attribute:: value

   The edited value.

.. attribute:: position

   Should only be used in a slot associated with signal :attr:`ready`. Holds the current edited value.

Methods:

.. method:: setValue(v)

   Set *v* as the edited value.

.. method:: setPosition(v)

   Should only be used in a slot associated with signal :attr:`ready`. Modifies the current edited value.

.. method:: setOrientation(o)

   Set the orientation of the widget. *o* must be either :const:`QtCore.Qt.Horizontal` or :const:`QtCore.Qt.Vertical`.
  """
#--------------------------------------------------------------------------------------------------
  pass

#--------------------------------------------------------------------------------------------------
class ObjectEditWidget (EditWidget):
  r"""
An instance of this classes edits an arbitrary value which can be parsed from a single line string.

:param initv: initial value.
:param transf: mapping from string space to value space (parser).
:param itransf: mapping from value space to string space (generator).
:param parent: parent widget.
:param fmt: mapping from value space to informal strings (viewer).

The widget consists of a viewer widget of class :class:`QtWidgets.QLabel` and an editor widget of class :class:`QtWidgets.QLineEdit`. The editor widget is hidden by default, and becomes visible when the viewer widget is clicked. Signal :attr:`ready` is emitted when the return key is hit in the editor.

.. attribute:: editor

   The editor widget.

.. attribute:: viewer

   The viewer widget.
  """
#--------------------------------------------------------------------------------------------------

  valueChanged = QtCore.Signal(object)
  ready = QtCore.Signal()

  def __init__(self,initv,transf,itransf,parent=None,fmt=str,editor_location='right',tooltip=None):
    super(ObjectEditWidget,self).__init__(parent)
    self.position = initv
    self.value = None

    layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.Direction(dict(right=QtWidgets.QBoxLayout.LeftToRight,left=QtWidgets.QBoxLayout.RightToLeft,bottom=QtWidgets.QBoxLayout.TopToBottom,top=QtWidgets.QBoxLayout.BottomToTop)[editor_location]),self)
    self.viewer = viewer = QtWidgets.QLabel()
    self.viewer.setToolTip('Focus to edit')
    self.editor = editor = QtWidgets.QLineEdit()
    if tooltip is not None: self.editor.setToolTip(tooltip)
    layout.addWidget(viewer)
    layout.addWidget(editor)

    viewer.setTextFormat(QtCore.Qt.PlainText)
    def startedit(ev):
      editor.setText(itransf(self.value))
      editor.setFocus()
      editor.setVisible(True)
    viewer.setFocusPolicy(QtCore.Qt.ClickFocus)
    viewer.focusInEvent = startedit
    def keypressed(ev):
      if ev.key()==QtCore.Qt.Key_Escape:
        ev.ignore()
        editor.setVisible(False)
      editor.__class__.keyPressEvent(editor,ev)
    editor.keyPressEvent = keypressed
    editor.setVisible(False)

    def update():
      v = self.position
      viewer.setText(fmt(v))
      if v != self.value:
        self.value = v
        self.valueChanged.emit(v)
    def setPosition(v):
      self.position = v
    self.setPosition = setPosition
    def setValue(v):
      setPosition(v)
      update()
    self.setValue = setValue
    def editoraction():
      try: v = transf(str(editor.text()))
      except:
        QtWidgets.QMessageBox.warning(self.viewer,'Operation failed',traceback.format_exc())
      else:
        editor.setVisible(False)
        setPosition(v)
        self.ready.emit()
        update()
    editor.returnPressed.connect(editoraction)

    def setOrientation(o):
      d = QtWidgets.QBoxLayout.Direction(QtWidgets.QBoxLayout.LeftToRight if o == QtCore.Qt.Horizontal else QtWidgets.QBoxLayout.TopToBottom)
      editor.setOrientation(o)
      layout.setDirection(d)
    self.setOrientation = setOrientation

    update()

#--------------------------------------------------------------------------------------------------
class ScalarEditWidget (EditWidget):
  r"""
An instance of this class edits an object in a value space which is iso-morphic to an integer interval (index space).

:param vmin: lower bound, value space.
:param vmax: upper bound, value space.
:param transf: mapping from index space to value space.
:param itransf: mapping from value space to index space (inverse of transf).
:param itransfd: mapping from value-delta space to index-delta space.
:param parent: parent widget.
:param vtype: coercing mapping into value space.
:param fmt: mapping from value space to informal strings (viewer).

The ordering of values is assumed to follow the ordering of indices. The widget consists of a viewer widget of class :class:`QtWidgets.QLabel` and an editor widget of class :class:`QtWidgets.QSlider`. Signal :attr:`ready` is emitted  each time the slider is on one of its degrees.

.. attribute:: editor

   The editor widget.

.. attribute:: viewer

   The viewer widget.
  """
#--------------------------------------------------------------------------------------------------

  valueChanged = QtCore.Signal(object)
  ready = QtCore.Signal(int)

  def __init__(self,vmin,vmax,transf,itransf,itransfd,parent=None,vtype=None,fmt=str,tooltip=None):
    vmin = vtype(vmin)
    vmax = vtype(vmax)
    assert vmax > vmin
    super(ScalarEditWidget,self).__init__(parent)
    self.minimum = vmin
    self.maximum = vmax
    self.position = vmin
    self.value = None
    rng = itransf(vmin),itransf(vmax)

    layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.Direction(QtWidgets.QBoxLayout.LeftToRight),self)
    self.viewer = viewer = QtWidgets.QLabel(None)
    self.editor = editor = QtWidgets.QSlider(None)
    layout.addWidget(editor)
    layout.addWidget(viewer)

    editor.setOrientation(QtCore.Qt.Horizontal)
    editor.setRange(*rng)
    editor.setToolTip('{}[{},{},{}]'.format(vtype.__name__,fmt(vmin),fmt(vmax),tooltip))

    def update():
      v = self.position
      editor.setSliderPosition(itransf(v))
      viewer.setText(fmt(v))
      if v != self.value:
        self.value = v
        self.valueChanged.emit(v)
    def setPosition(v):
      self.position = max(min(vtype(v),self.maximum),self.minimum)
    self.setPosition = setPosition
    def setValue(v):
      setPosition(v)
      update()
    self.setValue = setValue
    def editoraction(a):
      setPosition(transf(editor.sliderPosition()))
      self.ready.emit(a)
      update()
    editor.actionTriggered.connect(editoraction)

    self.setTickPosition = editor.setTickPosition
    self.setPageStep = lambda v: editor.setPageStep(max(itransfd(v),1))
    self.setSingleStep = lambda v: editor.setSingleStep(max(itransfd(v),1))
    self.setTickInterval = lambda v:editor.setTickInterval(max(itransfd(v),1))
    def setOrientation(o):
      d = QtWidgets.QBoxLayout.Direction(QtWidgets.QBoxLayout.LeftToRight if o == QtCore.Qt.Horizontal else QtWidgets.QBoxLayout.TopToBottom)
      editor.setOrientation(o)
      layout.setDirection(d)
    self.setOrientation = setOrientation

    update()

#--------------------------------------------------------------------------------------------------
class LinScalarEditWidget (ScalarEditWidget):
  r"""
The value space is a linear scaled grid in an interval of whole or real numbers. 

:param step: additive size of grid step, starting at *vmin*.
:param nval: number of grid steps (exclusive with *step* ).
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,step=None,vmin=0,vmax=1000,nval=1000,vtype=None,**ka):
    from math import floor
    if step is None:
      assert isinstance(nval,int) and nval>1
      tt = 'n={}'.format(nval)
      step = (vmax-vmin)/nval
      if step == 0: step = vtype(1)
    else:
      tt = 's={}'.format(step)
      step = vtype(step)
    transf = lambda k: vmin+k*step
    itransf = lambda v,vref=vmin: int(floor(.5+(v-vref)/step))
    itransfd = lambda v: itransf(v,vtype(0))
    super(LinScalarEditWidget,self).__init__(vmin,vmax,transf,itransf,itransfd,vtype=vtype,tooltip=tt,**ka)

#--------------------------------------------------------------------------------------------------
class LogScalarEditWidget (ScalarEditWidget):
  r"""
The value space is a log scaled grid in an interval of real numbers. 

:param mstep: multiplicative size of grid step, starting at *vmin*.
:param nval: number of grid steps (exclusive with *mstep* ).
  """
#--------------------------------------------------------------------------------------------------

  def __init__(self,vmin=.01,vmax=100,mstep=2,nval=None,fmt='{0:.2e}'.format,**ka):
    from math import log,floor
    if nval is None:
      assert mstep > 1.
      tt = 's=*{}'.format(mstep)
      lstep=log(mstep)
    else:
      assert isinstance(nval,int) and nval>1
      tt = 'n=*{}'.format(nval)
      r = vmax/vmin
      lstep = log(r)/nval
      mstep = r**(1./nval)
    transf = lambda k: vmin*(mstep**k)
    itransf = lambda v,vref=vmin: int(floor(.5+log(v/vref)/lstep))
    itransfd = lambda v: itransf(v,1.)
    super(LogScalarEditWidget,self).__init__(vmin,vmax,transf,itransf,itransfd,fmt=fmt,vtype=float,tooltip=dict(mstep=mstep),**ka)

#--------------------------------------------------------------------------------------------------
class SetEditWidget (EditWidget):
  r"""
An instance of this class edits a subset from an explicit set of string options. 

:param options: list of options or option-tooltip pairs.
:type options: iterable(:const:`str`\|tuple(:const:`str`,\ :const:`str`))
:param multi: if :const:`False`, one option at most is selected.
:type multi: :const:`bool`
:param parent: parent widget.

The widget consists of one checkable widget of class :class:`QtWidgets.QPressButton` for each option. The edited value consists of the set of checked options. This editor does not support the :attr:`ready` signal.
  """
#--------------------------------------------------------------------------------------------------

  valueChanged = QtCore.Signal(object)

  @property
  def value(self):
    return self.getval()

  def __init__(self,options=(),multi=False,parent=None):
    assert isinstance(multi,bool)
    super(SetEditWidget,self).__init__(parent)

    layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.Direction(QtWidgets.QBoxLayout.TopToBottom),self)
    bgroup = QtWidgets.QButtonGroup(self)
    optval = {}
    for k,lbl in enumerate(options):
      if isinstance(lbl,tuple): lbl,tooltip = lbl
      else: tooltip = None
      w = QtWidgets.QPushButton(lbl)
      layout.addWidget(w)
      bgroup.addButton(w)
      bgroup.setId(w,k)
      optval[lbl] = k
      if tooltip is not None: w.setToolTip(tooltip)
      w.setCheckable(True)
    inds = set(optval.values())

    self.multi = multi
    self.options = options
    bgroup.setExclusive(not multi)
    layout.setContentsMargins(1,1,1,1)
    layout.setSpacing(1)

    if multi:
      def getval(): return tuple(bgroup.id(w) for w in bgroup.buttons() if w.isChecked())
      def addValue(k):
        w = bgroup.button(k)
        if not w.isChecked():
          w.setChecked(True)
          self.valueChanged.emit(getval())
      def delValue(k):
        w = bgroup.button(k)
        if w.isChecked():
          w.setChecked(False)
          self.valueChanged.emit(getval())
      def togValue(k):
        w = bgroup.button(k)
        w.setChecked(not w.isChecked())
        self.valueChanged.emit(getval())
      def setValue(s):
        change = False
        for w in bgroup.buttons():
          flag = bgroup.id(w) in s
          if flag != w.isChecked():
            change = True
            w.setChecked(flag)
        if change: self.valueChanged.emit(tuple(set(s)&inds))
      self.addValue = addValue
      self.delValue = delValue
      self.togValue = togValue
      self.addLabelled = lambda lbl: addValue(optval[lbl])
      self.delLabelled = lambda lbl: delValue(optval[lbl])
      self.togLabelled = lambda lbl: togValue(optval[lbl])
      self.setLabelled = lambda lbls: setValue(set(optval[lbl] for lbl in lbls))
      bgroup.buttonClicked[int].connect(lambda k: self.valueChanged.emit(getval()))
    else:
      def getval(): return bgroup.checkedId()
      def setValue(k):
        if k in inds and k != bgroup.checkedId():
          bgroup.button(k).setChecked(True)
          self.valueChanged.emit(k)
      self.setLabelled = lambda lbl: setValue(optval[lbl])
      bgroup.button(0).setChecked(True)
      def pressed(k):
        if k != bgroup.checkedId(): self.valueChanged.emit(k)
      bgroup.buttonPressed[int].connect(pressed)
    self.setValue = setValue
    self.getval = getval
    def setOrientation(o):
      d = QtWidgets.QBoxLayout.Direction(QtWidgets.QBoxLayout.LeftToRight if o == QtCore.Qt.Horizontal else QtWidgets.QBoxLayout.TopToBottom)
      layout.setDirection(d)
    self.setOrientation = setOrientation

#--------------------------------------------------------------------------------------------------
class BooleanEditWidget (EditWidget):
  r"""
An instance of this class edits a Boolean flag. 

:param parent: parent widget.

The widget consists of a single widget of class :class:`QtWidgets.QCheckBox`. This editor does not support the :attr:`ready` signal.
  """
#--------------------------------------------------------------------------------------------------

  valueChanged = QtCore.Signal(bool)

  @property
  def value(self):
    return self.getval()

  def __init__(self,parent=None):
    super(BooleanEditWidget,self).__init__(parent)
    layout = QtWidgets.QVBoxLayout(self)
    w = QtWidgets.QCheckBox()
    layout.addWidget(w)
    self.getval = w.isChecked
    def setval(v):
      if v!=w.isChecked():
        w.setChecked(v)
        self.valueChanged.emit(v)
    self.setValue = setval
    w.toggled.connect(self.valueChanged)

#--------------------------------------------------------------------------------------------------
class FilenameEditWidget (EditWidget):
  r"""
An instance of this class edits a file path.

:param parent: parent widget.

The widget consists of a viewer widget of class :class:`QtWidgets.QLabel`. This editor does not support the :attr:`ready` signal.
  """
#--------------------------------------------------------------------------------------------------

  valueChanged = QtCore.Signal(str)

  @property
  def value(self):
    return self.getval()

  def __init__(self,parent=None,op='open',**ka):
    assert op=='open' or op=='save'
    super(FilenameEditWidget,self).__init__(parent)
    layout = QtWidgets.QHBoxLayout(self)
    editbutton = QtWidgets.QPushButton('^')
    viewer = QtWidgets.QLabel()
    layout.addWidget(editbutton)
    layout.addWidget(viewer)
    viewer.setTextFormat(QtCore.Qt.PlainText)
    self.getval = lambda: str(viewer.text())
    def setval(v):
      if v != str(viewer.text()):
        viewer.setText(v)
        self.valueChanged.emit(v)
    self.setValue = setval
    def edit():
      v = str((QtWidgets.QFileDialog.getOpenFileName if op=='open' else QtWidgets.QFileDialog.getSaveFileName)(viewer,**ka)[0])
      if v: setval(v)
    editbutton.clicked.connect(edit)

#--------------------------------------------------------------------------------------------------
class Animator (QtCore.QObject):
  r"""
An instance of this class is used to create animations.

Attributes:

.. attribute:: state

   Can be any object. Each animation step can either update it inline or return a new object.

.. attribute:: paused

   A boolean holding whether the animator is paused.

.. attribute:: statusChanged

   A signal, emitted when the animator changes status, with argument either
   :const:`0` (not running),
   :const:`1` (running, not paused),
   :const:`-1` (running, paused),

Methods:
  """
#--------------------------------------------------------------------------------------------------

  statusChanged = QtCore.Signal(int)

  def __init__(self,parent=None):
    super(Animator,self).__init__(parent)
    self.timer = QtCore.QTimer()
    self.step = None
    self.state = None
    self.paused = None

  def shutdown(self):
    r"""Terminates the animation."""
    if self.step is None: return
    self.timer.timeout.disconnect()
    self.timer.stop()
    self.step = None
    self.state = None
    self.paused = None
    self.statusChanged.emit(0)

  def launch(self,t,f,s=None,paused=False):
    r"""
Launches the animation.

:param t: time between steps in ms.
:type t: :const:`int`
:param f: state transformer executed at each step (input=oldstate; output=newstate).
:type f: 1-input, 1-output callable
:param s: initial :attr:`state`.
:param paused: initial :attr:`paused`.
:type paused: :const:`bool`
    """
    assert self.step is None
    self.step = f
    self.state = s
    self.paused = paused
    self.timer.timeout.connect(lambda: self.tick())
    self.timer.start(t)
    self.statusChanged.emit(-1 if paused else 1)

  def tick(self):
    if self.paused: return
    else: self.onestep()

  def onestep(self):
    r"""Performs one step of the animation."""
    try: self.state = self.step(self.state)
    except StopIteration: self.shutdown()
    except: self.shutdown(); raise

  def pause(self,paused=None):
    r"""Pauses the animation."""
    if self.step is None: return
    if paused is None: self.paused = not self.paused
    elif self.paused == paused: return
    self.statusChanged.emit(-1 if self.paused else 1)

#--------------------------------------------------------------------------------------------------
# Utilities
#--------------------------------------------------------------------------------------------------

def func_keywords(f):
  def dec(f):
    if isinstance(f,functools.partial):
      F,kwd = dec(f.func)
      if f.keywords is not None: kwd.update(f.keywords)
      return F,kwd
    else:
      assert callable(f)
      try: d,c = f.func_defaults, f.func_code
      except:
        try: d,c = f.__defaults__, f.__code__
        except: return f,{}
      if d is None: return f,{}
      n = len(d)
      return f,dict(zip(c.co_varnames[:c.co_argcount][-n:],d))
  return dec(f)

def figure(x,withtoolbar=True,**ka):
  r"""
Creates and returns a matplotlib figure.

:parameter x: a layout or a widget.
:type x: :class:`QtWidgets.QBoxLayout`\|\ :class:`QtWidgets.QWidget`
:parameter withtoolbar: whether a toolbar should also be added to the layout.
:type withtoolbar: :const:`bool`
:parameter ka: passed as keyword arguments to the figure constructor.

The canvas widget of the created figure is added to *layout*, together with a matplotlib toolbar if *withtoolbar* is :const:`True`. *layout* is either *x* if that is a :class:`QtWidgets.QBoxLayout` instance or a :class:`QtWidgets.QVBoxLayout` instance associated to *x* if *x* is a :class:`QtWidgets.QWidget` instance.
  """
  if isinstance(x,QtWidgets.QWidget): layout = QtWidgets.QVBoxLayout(x)
  else: layout = x ; assert isinstance(x,QtWidgets.QLayout)
  from matplotlib.figure import Figure
  from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
  fig = Figure(**ka)
  w = FigureCanvasQTAgg(fig)
  layout.addWidget(w)
  if withtoolbar:
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
    t = NavigationToolbar2QT(w,None)
    layout.addWidget(t)
  return fig

class AppError (Exception): pass

@contextmanager
def startup(*a):
  app = QtWidgets.QApplication(list(a))
  yield app
  n = app.exec_()
  if n: raise AppError(n)

#--------------------------------------------------------------------------------------------------
class cbook:
  r"""
This class is not meant to be instantiated. It offers a list (cookbook) of utilities as static methods.
  """
#--------------------------------------------------------------------------------------------------

  @staticmethod
  def mpltimer(fig,toolbar=None):
    r"""
Creates and returns a pausable matplotlib timer attached to a figure.

:param fig: a figure with a canvas having a toolbar
:type fig: :class:`matplotlib.figure.Figure`
:param toolbar: a toolbar
:type toolbar: :const:`NoneType`\|\ :class:`QtWidgets.QToolbar`
:rtype: :class:`myutil.mplext.Timer`

A 'pause' and a `1-step` buttons are added to *toolbar*. The former allows to toggle the pause status of the timer, and the latter, only visible when the timer is paused, allows to simulate one tick of the timer. If *toolbar* is :const:`None`, the figure's own toolbar is used.
    """
    from .mplext import Timer
    timer = Timer(fig.canvas.new_timer())
    if toolbar is None: toolbar = fig.canvas.toolbar
    actionPause = a = toolbar.addAction('pause')
    a.triggered.connect(timer.togglepause)
    actionStep = a = toolbar.addAction('1-step')
    a.triggered.connect(timer.onestep)
    def status(s):
      actionPause.setDisabled(s==0)
      actionStep.setVisible(s==-1)
    status(0)
    timer.status = status
    return timer

  @staticmethod
  def mplview(e,label):
    r"""
Adds a new tab to a tabulated UI, holding a mplext cell.

:param e: a tabulated UI
:type e: :class:`BaseTabUI`
:param label: label of the tab
:type label: :const:`str`
:rtype: :class:`myutil.mplext.Cell`
    """
    assert isinstance(e,BaseTabUI) and isinstance(label,str)
    from .mplext import Cell
    return Cell.new(figure(e.addtab(label)))

  @staticmethod
  def i(c1,lbl,c2):
    r"""
Sets the parameter of configurator *c2* as the value associated to key *lbl* in the :attr:`keyword` dict of configurator *c1*.
    """
    c2.parset(c1.keywords[lbl])

  @staticmethod
  def selectv1(l):
    r"""
When passed the result of a compound configurator, returns the first value (drops the label).
    """
    (label,x), = l
    return x

  @staticmethod
  def selectv(l):
    r"""
When passed the result of a compound configurator, returns the list of values (drops the labels).
    """
    return tuple(x[1] for x in l)

  @staticmethod
  def Ccomp(f,D,**ka):
    r"""
Returns a compound configurator.

:param D: stored as attribute :attr:`keywords` (see :meth:`i`).
:type D: :const:`dict`
:param f: stored as attribute :attr:`proc` (see the :class:`CompoundConfigurator` constructor).
:type f: callable
    """
    assert callable(f)
    c = CompoundConfigurator(proc=f,**ka)
    c.keywords = D
    return c

  @staticmethod
  def Ctransf(f,**ka):
    r"""
Returns a compound configurator.

:param f: a transfer function.
:type f: callable

* Attribute :attr:`proc` is set to the function which converts its argument to a dictionary *D* and returns *f*\ (** *D*\ ) .
* Attribute :attr:`keyword` is set to the dictionary of default values of *f* (which can be a function or obtained by :func:`functools.partial`).
* Child configurators must be added separately.
    """
    assert callable(f)
    c = CompoundConfigurator(proc=(lambda a: f(**dict(a))),**ka)
    c.keywords = func_keywords(f)[1]
    return c

  @staticmethod
  def Cclosr(f,**ka):
    r"""
Returns a compound configurator.

:param f: a transfer function.
:type f: callable

* Attribute :attr:`proc` is set to the function which converts its argument to a dictionary *D* and returns the closure (obtained by :func:`functools.partial`) of *f* by *D*.
* Attribute :attr:`keyword` is set to the dictionary of default values of *f* (which can be a function or obtained by :func:`functools.partial`).
* Child configurators must be added separately.
    """
    assert callable(f)
    c = CompoundConfigurator(proc=(lambda a: functools.partial(f,**dict(a))),**ka)
    c.keywords = func_keywords(f)[1]
    return c

  class Cbase:
    r"""
The cookbook also exposes the following functions which returns instances of :class:`myutil.quickui.BaseConfigurator` encapsulating instances of :class:`myutil.quickui.EditWidget`
    """
    def make(factory):
      f = lambda **ka: BaseConfigurator(factory(**ka))
      f.__doc__ = 'Base configurator for a :class:`{0}`.'.format(factory.__name__)
      return staticmethod(f)
    Object = make(ObjectEditWidget)
    Scalar = make(ScalarEditWidget)
    LinScalar = make(LinScalarEditWidget)
    LogScalar = make(LogScalarEditWidget)
    Set = make(SetEditWidget)
    Boolean = make(BooleanEditWidget)
    Filename = make(FilenameEditWidget)
    del make

