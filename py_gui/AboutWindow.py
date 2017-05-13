from py_gui.Ui_About import Ui_About
from PySide import QtGui


class AboutWindow(QtGui.QWidget, Ui_About):
    def __init__(self, parent=None):
        super(AboutWindow, self).__init__()
        self.setupUi(self)
    def show_test(self):
        self.show()