# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main2.ui'
#
# Created: Sun Dec 18 12:39:18 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!
import sys
from PySide import QtCore, QtGui

from scene.CirlceObject import CircleObject
from scene.NullObject import NullObject
import py_gui.Ui_MainWindow as ui
import cv2


class MainWindow(QtGui.QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.init_actions()

        self.graphics_scene = QtGui.QGraphicsScene()
        self.graphics_scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 255)))
        self.graphicsView.setScene(self.graphics_scene)

        self.brush = QtGui.QBrush(QtCore.Qt.black)
        self.pen = QtGui.QPen()
        self.pen.setWidth(2)

        self.element = NullObject()
        self.init_graphics_component()

        self.items = list()
        self.view_capture = None

    def init_actions(self):
        self.goodButton.clicked.connect(lambda: self.good_button_action())
        self.badButton.clicked.connect(lambda: self.bad_button_action())

        self.nextButton.clicked.connect(lambda: self.next_frame())
        self.backButton.clicked.connect(lambda: self.back_frame())

        self.actionOpen.triggered.connect(self.open)

    def init_graphics_component(self):
        self.graphics_scene.mousePressEvent = self.graphics_view_click
        self.graphics_scene.mouseMoveEvent = self.graphics_view_mouse_drag

    def graphics_view_click(self, event):
        press_event = event.buttons()
        if press_event == QtCore.Qt.LeftButton:
            self.left_click(event)
        if press_event == QtCore.Qt.RightButton:
            self.right_click(event)

    def graphics_view_mouse_drag(self, drag):
        self.element.graphics_view_mouse_drag(drag, self.graphics_scene)

    def left_click(self, position):
        obj = self.element.left_click(position, self.graphics_scene)
        if obj is not None:
            self.items.append(obj)
            print(obj.rect().center())
            print(self.items)

    def right_click(self, position):
        self.element.right_click(position, self.graphics_scene, self.items)

    def good_button_action(self):
        self.element = CircleObject("Good")

    def bad_button_action(self):
        self.element = CircleObject("Bad")

    def open(self):
        file_name, _ = QtGui.QFileDialog.getOpenFileName(self, "Open File", QtCore.QDir.currentPath())
        if file_name:
            self.view_capture = cv2.VideoCapture(file_name)
            if not self.view_capture.isOpened():
                return
            self.label.setText("Selected file: " + file_name)
            flag, frame = self.view_capture.read()
            self.show_frame(frame)

    def show_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)
        frame = cv2.flip(frame, 1)
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
        self.graphics_scene.clear()
        self.graphics_scene.update()
        #self.graphics_scene.setSceneRect(image.rect())
        img = QtGui.QPixmap.fromImage(image)
        pixItem = QtGui.QGraphicsPixmapItem(img)
        self.graphics_scene.addItem(pixItem)
        x = self.graphicsView.width() - image.rect().width()
        y = self.graphicsView.height() - image.rect().height()
        pixItem.setPos(x/2, y/2)

        self.graphicsView.autoFillBackground()
        self.graphicsView.fitInView(self.graphicsView.rect())

    def next_frame(self):
        if self.view_capture is not None:
            flag, frame = self.view_capture.read()
            self.show_frame(frame)

    def back_frame(self):
        print("back")

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    MainWindow = MainWindow()
    MainWindow.show()

    sys.exit(app.exec_())