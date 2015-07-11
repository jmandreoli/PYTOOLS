# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'browser.ui'
#
# Created: Fri May 16 19:25:53 2014
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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(800, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.nbentriesField = QtGui.QLabel(self.centralwidget)
        self.nbentriesField.setObjectName(_fromUtf8("nbentriesField"))
        self.horizontalLayout.addWidget(self.nbentriesField)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.refreshButton = QtGui.QPushButton(self.centralwidget)
        self.refreshButton.setObjectName(_fromUtf8("refreshButton"))
        self.horizontalLayout.addWidget(self.refreshButton)
        self.filterButton = QtGui.QPushButton(self.centralwidget)
        self.filterButton.setObjectName(_fromUtf8("filterButton"))
        self.horizontalLayout.addWidget(self.filterButton)
        self.filterField = QtGui.QLabel(self.centralwidget)
        self.filterField.setText(_fromUtf8(""))
        self.filterField.setObjectName(_fromUtf8("filterField"))
        self.horizontalLayout.addWidget(self.filterField)
        self.filterEditor = QtGui.QLineEdit(self.centralwidget)
        self.filterEditor.setObjectName(_fromUtf8("filterEditor"))
        self.horizontalLayout.addWidget(self.filterEditor)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.listWidget = QtGui.QListWidget(self.centralwidget)
        self.listWidget.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.listWidget.setObjectName(_fromUtf8("listWidget"))
        self.verticalLayout.addWidget(self.listWidget)
        self.line = QtGui.QFrame(self.centralwidget)
        self.line.setLineWidth(4)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.verticalLayout.addWidget(self.line)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.oidField = QtGui.QLabel(self.centralwidget)
        self.oidField.setObjectName(_fromUtf8("oidField"))
        self.horizontalLayout_2.addWidget(self.oidField)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.viewlogButton = QtGui.QPushButton(self.centralwidget)
        self.viewlogButton.setObjectName(_fromUtf8("viewlogButton"))
        self.horizontalLayout_2.addWidget(self.viewlogButton)
        self.viewsrcButton = QtGui.QPushButton(self.centralwidget)
        self.viewsrcButton.setObjectName(_fromUtf8("viewsrcButton"))
        self.horizontalLayout_2.addWidget(self.viewsrcButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.browserWidget = QtGui.QTextBrowser(self.centralwidget)
        self.browserWidget.setObjectName(_fromUtf8("browserWidget"))
        self.verticalLayout.addWidget(self.browserWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionQuit = QtGui.QAction(MainWindow)
        self.actionQuit.setObjectName(_fromUtf8("actionQuit"))
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label.setText(_translate("MainWindow", "entries:", None))
        self.nbentriesField.setText(_translate("MainWindow", "0", None))
        self.refreshButton.setText(_translate("MainWindow", "refresh", None))
        self.filterButton.setText(_translate("MainWindow", "filter", None))
        self.oidField.setText(_translate("MainWindow", "oid", None))
        self.viewlogButton.setText(_translate("MainWindow", "view log", None))
        self.viewsrcButton.setText(_translate("MainWindow", "view source", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.actionQuit.setText(_translate("MainWindow", "Quit", None))

