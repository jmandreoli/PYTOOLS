# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'plugins/taa/meditor.ui'
#
# Created: Wed Jun 18 15:19:43 2014
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(716, 377)
        self.verticalLayout = QtGui.QVBoxLayout(Form)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label = QtGui.QLabel(Form)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_2.addWidget(self.label)
        self.titleEditor = QtGui.QLineEdit(Form)
        self.titleEditor.setObjectName(_fromUtf8("titleEditor"))
        self.horizontalLayout_2.addWidget(self.titleEditor)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout.addWidget(self.label_2)
        self.authorsEditor = QtGui.QLineEdit(Form)
        self.authorsEditor.setObjectName(_fromUtf8("authorsEditor"))
        self.horizontalLayout.addWidget(self.authorsEditor)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.label_3 = QtGui.QLabel(Form)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.verticalLayout.addWidget(self.label_3)
        self.abstractEditor = QtGui.QPlainTextEdit(Form)
        self.abstractEditor.setObjectName(_fromUtf8("abstractEditor"))
        self.verticalLayout.addWidget(self.abstractEditor)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.edittxtButton = QtGui.QPushButton(Form)
        self.edittxtButton.setObjectName(_fromUtf8("edittxtButton"))
        self.horizontalLayout_3.addWidget(self.edittxtButton)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.savedraftButton = QtGui.QPushButton(Form)
        self.savedraftButton.setObjectName(_fromUtf8("savedraftButton"))
        self.horizontalLayout_3.addWidget(self.savedraftButton)
        self.saveButton = QtGui.QPushButton(Form)
        self.saveButton.setObjectName(_fromUtf8("saveButton"))
        self.horizontalLayout_3.addWidget(self.saveButton)
        self.modifField = QtGui.QLabel(Form)
        self.modifField.setObjectName(_fromUtf8("modifField"))
        self.horizontalLayout_3.addWidget(self.modifField)
        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.label.setText(_translate("Form", "title", None))
        self.label_2.setText(_translate("Form", "authors", None))
        self.label_3.setText(_translate("Form", "abstract", None))
        self.edittxtButton.setText(_translate("Form", "edit txt", None))
        self.savedraftButton.setText(_translate("Form", "save draft", None))
        self.saveButton.setText(_translate("Form", "save", None))
        self.modifField.setText(_translate("Form", "*", None))

