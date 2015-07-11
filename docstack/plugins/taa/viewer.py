# File:                 viewer.py
# Creation date:        2014-05-13
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              docstack viewer for "taa" plugin
#
# *** Copyright (c) 2014 Xerox Corporation  ***
# *** Xerox Research Centre Europe - Grenoble ***
#

import unicodedata
from datetime import datetime
from lxml.html.builder import E,CLASS
from myutil.docstack.viewer import EditorWindow as BaseEditorWindow, BrowserWindow as BaseBrowserWindow, QtGui, QtCore
from .meditorui import Ui_Form as Ui_MEditorForm

#==================================================================================================
class MEditorForm (Ui_MEditorForm,QtCore.QObject):
#==================================================================================================

  save = QtCore.pyqtSignal(dict)

  def __init__(self):
    super(MEditorForm,self).__init__()
    self.txtedit,self.txtget,self.txtmod = None,None,(lambda: False)

  def editing(self):
    return self.titleEditor.isModified() or self.authorsEditor.isModified() or self.abstractEditor.document().isModified() or self.txtmod()

  def setup(self,widget):
    self.widget = widget
    self.setupUi(widget)
    self.edittxtButton.clicked.connect(lambda: self.txtedit())
    self.saveButton.clicked.connect(self.savefinal)
    self.savedraftButton.clicked.connect(self.savedraft)
    def modstatus(): self.modifField.setVisible(self.editing())
    self.titleEditor.textEdited.connect(lambda t: modstatus())
    self.authorsEditor.textEdited.connect(lambda t: modstatus())
    self.abstractEditor.modificationChanged.connect(lambda b: modstatus())

  def savefinal(self): self.save.emit(self.currentstate(True))
  def savedraft(self): self.save.emit(self.currentstate(False))

  def display(self,doc,txt):
    self.modifField.setVisible(False)
    if doc['active']:
      self.txtget,self.txtedit,self.txtmod = txt
      self.widget.setEnabled(True)
      self.titleEditor.setText(doc['title'])
      self.authorsEditor.setText(doc['authors'])
      self.abstractEditor.document().setPlainText(doc['abstract'])
      self.abstractEditor.document().setModified(False)
      self.edittxtButton.setEnabled(True)
    else:
      self.clear()

  def clear(self):
    self.txtedit,self.txtget,self.txtmod = None,None,(lambda: False)
    self.widget.setEnabled(False)
    self.titleEditor.clear()
    self.authorsEditor.clear()
    self.abstractEditor.document().clear()
    self.edittxtButton.setEnabled(False)

  def currentstate(self,final):
    return dict(
      title=normalise(self.titleEditor.text()),
      authors=normalise(self.authorsEditor.text()),
      abstract=normalise(self.abstractEditor.toPlainText()),
      content=self.txtget(),
      final=final,
      )

#==================================================================================================
class EditorWindow (BaseEditorWindow):
#==================================================================================================

  meditorfactory = MEditorForm

  def displaydoc(self,doc):
    super(EditorWindow,self).displaydoc(doc)
    cnt = dict((k,doc[k]) for k in ('active','title','authors','abstract'))
    txt = (self.mgr.META/doc['oid']).with_suffix('.txt')
    def txtedit(s=self.subprocmgr,a=self.mgr.txteditor(str(txt))):
      return s.launch('edittxt',*a)
    def txtget():
      with txt.open('r') as u: return u.read()
    def txtmod(r=txt.stat().st_mtime):
      return txt.stat().st_mtime!=r
    self.meditor.display(cnt,(txtget,txtedit,txtmod))

  def docname(self,doc):
    t = datetime.fromtimestamp(doc['version']).strftime('%Y-%m-%d')
    if doc['active']: return '[{0}] {title} [[ {authors} ]]'.format(t,**doc)
    else: return '[{0}] {src}'.format(t,**doc)

#==================================================================================================
class BrowserWindow (BaseBrowserWindow):
#==================================================================================================

  STYLE = '''
table { width: 100%; border-width: 1px; border-spacing: 0px;  border-collapse: collapse; background-color: #e0e0e0; }
td.title { color: blue; font-weight: bold; }
td.authors { font-style: italic; }
td.source { font-size: small; }
'''

  def docname(self,doc):
    t = datetime.fromtimestamp(doc['version']).strftime('%Y-%m-%d')
    if doc['active']: return '[{0}] {title} [[ {authors} ]]'.format(t,**doc)
    else: return '[{0}] {src}'.format(t,**doc)

  def dochtml(self,doc):
    return E.html(
      E.head(
        E.title(doc['oid']),
        E.style(self.STYLE,type='text/css'),
        ),
      E.body(
        E.table(
          E.tr(E.th('title'),E.td(doc.get('title',''),**CLASS('title'))),
          E.tr(E.th('authors'),E.td(doc.get('authors',''),**CLASS('authors'))),
          E.tr(E.th('source'),E.td('{} [{}]'.format(doc.get('src',''),datetime.fromtimestamp(doc.get('version')).strftime('%Y-%m-%d %H:%M:%S')),**CLASS('source'))),
          ),
        E.p(
          doc.get('abstract',''),
          ),
        ),
      )

#==================================================================================================
# Utilities
#==================================================================================================

def normalise(x):
  # all unicode normalised NFKC
  # "-" at end of line interpreted as line continuation
  x = unicodedata.normalize('NFKC',x).replace('-\n','')
  return ' '.join(x.strip().split())
