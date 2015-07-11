# File:                 viewer.py
# Creation date:        2014-05-13
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              docstack viewer
#
# *** Copyright (c) 2014 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import subprocess, whoosh.qparser
from ..qtbinding import QtCore, QtGui
from lxml.etree import tounicode as html2str
from .editorui import Ui_MainWindow as Ui_EditorWindow
from .browserui import Ui_MainWindow as Ui_BrowserWindow

#==================================================================================================
class ListWindow:
  """
An object of this class dispays a viewer of indexed entries.
  """
#==================================================================================================

  def __init__(self,mgr):
    self.subprocmgr = SubprocManager()
    self.mgr = mgr

  def setup(self,win):
    self.rollback = False
    def selectitem(item,prev):
      # pretty ugly but no clean way to rollback a currentItemChanged signal
      if self.rollback: self.rollback = False
      elif self.cleanup(): self.rollback = True;QtCore.QTimer.singleShot(0,lambda w=self.listWidget,prev=prev: w.setCurrentItem(prev))
      else: self.displaydoc(item.document)
    self.win = win
    self.win.closeEvent = lambda ev: ev.ignore() if self.cleanup() else ev.accept()
    self.listWidget.currentItemChanged.connect(selectitem)
    self.viewsrcButton.clicked.connect(self.viewsrc)
    self.viewlogButton.clicked.connect(self.viewlog)
    self.refreshButton.clicked.connect(self.displaylist)
    self.actionQuit.triggered.connect(self.quit)
    self.displaylist()

  def cleanup(self):
    self.subprocmgr.stop()
    self.oidField.clear()
    self.viewsrcButton.setEnabled(False)
    self.viewlogButton.setEnabled(False)

  def displaylist(self):
    if self.cleanup(): return
    self.listWidget.clear()
    for doc in self.doclist():
      item = QtGui.QListWidgetItem('',self.listWidget)
      item.document = doc
      self.displayitem(item)
    self.listWidget.setFocus()
    self.listWidget.setCurrentItem(self.initialitem())

  def initialitem(self):
    return self.listWidget.item(0)

  def displayitem(self,item,scroll=None):
    # scroll: QtGui.QAbstractItemView.{EnsureVisible,PositionAtTop,PositionAtBottom,PositionAtCenter}
    doc = item.document
    item.setText(self.docname(doc))
    item.setForeground((QtCore.Qt.blue if doc['final'] else QtCore.Qt.darkBlue) if doc['active'] else QtCore.Qt.black)
    if scroll is not None: self.listWidget.scrollToItem(item,scroll)

  def displaydoc(self,doc):
    self.viewsrcButton.setEnabled(True)
    self.viewlogButton.setEnabled(True)
    self.oidField.setText(doc['oid'])
    self.oidField.setStyleSheet(('color: Blue; font-weight: bold;' if doc['final'] else 'color: DarkBlue; font-weight: normal;') if doc['active'] else 'color: black')

  def viewsrc(self):
    doc = self.listWidget.currentItem().document
    self.subprocmgr.launch('viewsrc',*self.mgr.srcviewer(doc['src']))
  def viewlog(self):
    doc = self.listWidget.currentItem().document
    self.subprocmgr.launch('viewlog',*self.mgr.logviewer(str((self.mgr.META/doc['oid']).with_suffix('.log'))))

  def quit(self):
    if self.cleanup(): return
    self.win.close()

#==================================================================================================
class EditorWindow (Ui_EditorWindow,ListWindow):
#==================================================================================================

  def __init__(self,*a,**ka):
    super(EditorWindow,self).__init__(*a,**ka)
    self.meditor = self.meditorfactory()

  def setup(self,win):
    self.setupUi(win)
    self.meditor.setup(self.editorBox)
    self.meditor.save.connect(self.updateitem)
    super(EditorWindow,self).setup(win)

  def initialitem(self):
    for i in range(self.listWidget.count()):
      item = self.listWidget.item(i)
      if not item.document['final']: return item
    else: return None

  def cleanup(self):
    if self.meditor.editing() and QtGui.QMessageBox.question(self.win,'Item editor','Discard edits ?',QtGui.QMessageBox.Discard|QtGui.QMessageBox.Cancel)==QtGui.QMessageBox.Cancel: return True
    elif super(EditorWindow,self).cleanup(): return True
    else: self.meditor.clear()

  def updateitem(self,s):
    item = self.listWidget.currentItem()
    doc = item.document
    doc.update(s)
    with self.mgr.ix.writer() as ixw: ixw.update_document(**doc)
    self.displayitem(item)
    self.displaydoc(doc)
    self.listWidget.setFocus()
    self.statusbar.showMessage('Document update successful',1000)

  def doclist(self):
    with self.mgr.ix.searcher() as ixs:
      L = tuple(sorted(ixs.documents(),key=(lambda doc: doc['version']),reverse=True))
    self.nbentriesField.setText('{}({} active)'.format(len(L),len(tuple(doc for doc in L if doc['active']))))
    return L

#==================================================================================================
class BrowserWindow (Ui_BrowserWindow,ListWindow):
#==================================================================================================

  LIMIT = 100

  def __init__(self,*a,**ka):
    super(BrowserWindow,self).__init__(*a,**ka)
    self.wqp = whoosh.qparser.QueryParser('content',schema=self.mgr.ix.schema)
    self.wqp.replace_plugin(whoosh.qparser.FieldsPlugin(remove_unknown=False))
    self.wqp.replace_plugin(whoosh.qparser.SingleQuotePlugin())
    self.filterq = None

  def setup(self,win):
    self.setupUi(win)
    super(BrowserWindow,self).setup(win)
    self.filterEditor.returnPressed.connect(self.filteredited)
    self.filterButton.clicked.connect(self.filteredit)
    self.filterEditor.setVisible(False)

  def filteredit(self):
    self.filterEditor.setVisible(True)
    self.filterEditor.setText(self.filterField.text())
    self.filterField.setVisible(False)
    self.filterEditor.setFocus()
    self.filterEditor.setSelection(0,1000)

  def filteredited(self):
    filtr = self.filterEditor.text()
    if filtr:
      try:
        q = self.wqp.parse(filtr)
        assert all(k in self.mgr.ix.schema.names() for k,v in q.iter_all_terms())
      except:
        QtGui.QMessageBox.warning(self.win,'Item browser','Syntax error in query')
        return
    else: q = None
    self.filterEditor.setVisible(False)
    self.filterField.setText(filtr)
    self.filterField.setVisible(True)
    self.filterq = q
    self.displaylist()

  def cleanup(self):
    if super(BrowserWindow,self).cleanup(): return True
    else: self.browserWidget.clear()

  def displaydoc(self,doc):
    super(BrowserWindow,self).displaydoc(doc)
    self.browserWidget.setHtml(html2str(self.dochtml(doc)))

  def doclist(self):
    if self.filterq is None: results = ()
    else:
      with self.mgr.ix.searcher() as ixs:
        results = tuple(r.fields() for r in ixs.search(self.filterq,sortedby='version',reverse=True,limit=self.LIMIT))
    self.nbentriesField.setText(str(len(results)))
    return results

#==================================================================================================
# Utilities
#==================================================================================================

def run(obj,args):
  app = QtGui.QApplication(args)
  win = QtGui.QMainWindow()
  obj.setup(win)
  win.show()
  return app.exec_()

class SubprocManager (object):

  def __init__(self):
    self.subs = {}

  def stop(self,name=None):
    if name is None:
      for s in self.subs.values():
        try: s.terminate(); s.wait()
        except: pass
      self.subs.clear()
    else:
      s = self.subs.pop(name)
      try: s.terminate(); s.wait()
      except: pass

  def launch(self,name,*args):
    s = self.subs.get(name)
    if s is None: a = True
    else: s.poll(); a = s.returncode is not None
    if a: self.subs[name] = subprocess.Popen(args)
