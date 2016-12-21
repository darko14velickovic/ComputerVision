from PySide import QtCore, QtGui


class NullObject:

    def __init__(self):
        self.color = QtGui.QColor("Red")
        self.center = None
        self.radius = None
        self.centerDot = None
        self.circle = None

    def message_box(self):
        msg = QtGui.QMessageBox()
        msg.setIcon(QtGui.QMessageBox.Information)
        msg.setText("No selected tool")
        msg.setInformativeText("Please select a tool (Good or Bad)")
        msg.setWindowTitle("Warning")
        msg.setDetailedText("Good button for green circle\nBad button for green circle")
        msg.exec_()

    def left_click(self, position, scene):
        self.message_box()

    def right_click(self, position, scene):
        self.message_box()

    def graphics_view_mouse_drag(self, drag, scene):
        return