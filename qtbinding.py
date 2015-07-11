# Dummy module
# Use set_qtbinding to override
from . import set_qtbinding
try: mod = set_qtbinding('pyside',False)
except: mod = set_qtbinding('pyqt4',False)
QtCore, QtGui = mod.QtCore, mod.QtGui

