from PySide import QtGui
from Ui_DialogSettings import Ui_Dialog


class SettingsWindow(QtGui.QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(SettingsWindow, self).__init__()
        self.setupUi(self)
        self.caller = parent
        self.accepted.connect(self.accept)
        self.rejected.connect(self.closed)

        self.accepted = False
        #self.buttonBox.clicked.connect(self.accept)
       

    def accept(self):
        self.accepted = True
        print "accepted"
        self.caller.file_name = self.lineEdit.text()
        self.done(0)
        #self.close()


    def closed(self):
        #
        if not self.accepted:
            self.caller.file_name = ""
            print "canceled"


