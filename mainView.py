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
import copy
import numpy as np
from sklearn.neural_network import MLPClassifier

class MainWindow(QtGui.QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.keyPressEvent = self.key_press_handler

        self.setupUi(self)
        self.init_actions()
        #flag for autoplay
        self.autoplaying = False
        #pointer to classifier
        self.classifier = None
        self.graphics_scene = QtGui.QGraphicsScene()
        self.graphics_scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 255)))

        self.graphicsView.setScene(self.graphics_scene)
        self.checkBox.stateChanged.connect(self.check_box_click)

        self.brush = QtGui.QBrush(QtCore.Qt.black)
        self.pen = QtGui.QPen()
        self.pen.setWidth(2)

        self.element = NullObject()
        self.init_graphics_component()

        self.items = list()

        self.trainingData = None

        self.view_capture = None

        #timer for autoplay
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)

    def init_learning(self):
        self.classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes= (15,), random_state= 1)

    def key_press_handler(self, event):
        if event.key() is 83:
            print "Saving"
        #print event.key()
    def init_actions(self):
        self.goodButton.clicked.connect(lambda: self.good_button_action())
        self.badButton.clicked.connect(lambda: self.bad_button_action())

        self.nextButton.clicked.connect(lambda: self.next_frame())
        self.backButton.clicked.connect(lambda: self.back_frame())

        self.actionOpen.triggered.connect(self.open)

        # autoplay action
        self.actionAutoplay.triggered.connect(self.autoplay)

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
        self.showCoords(position.scenePos())
        obj = self.element.left_click(position, self.graphics_scene)
        if obj is not None:
            self.items.append(obj)
            print(obj.rect().center().x())
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
            if flag:
                self.show_frame(frame)

    def back_frame(self):
        print("back")
    def check_box_click(self, event):
        if event == 2:
            print "Enabled"
            self.graphicsView.wheelEvent = self.scroll_to_next_frame
        elif event == 0:
            print "Disabled"
            self.graphicsView.wheelEvent = self.scroll_dummy_func
        print event


    def autoplay(self):
        if self.autoplaying:
            self.timer.stop()
            self.actionAutoplay.setText("Autoplay")
        else:
            self.timer.start(1000)
            self.actionAutoplay.setText("Stop autoplay")
        self.autoplaying = not self.autoplaying

    def scroll_to_next_frame(self,event):
        self.next_frame()

    def scroll_dummy_func(self, event):
        #status text print...
        print "Scroll video disabled"

    def showCoords(self, point):
        xPoint = point.x()
        yPoint = point.y()
        self.status_label.setText("X: " + str(xPoint) + "  Y: " + str(yPoint))

    def quadriatic_mean_circle(xCentar, yCentar, radius, picture):  # mora da se prosledi gray_scale image
        # ove 2 promenjive su za rucno izracunavanje, svuda odkomentarisati ako treba to
        sum = 0
        numberOfPixels = 0
        radius = radius * radius
        image = copy.deepcopy(picture)
        array = []
        width = np.size(picture, 0)
        height = np.size(picture, 1)
        for x in range(xCentar - radius, xCentar + 1):
            for y in range(yCentar - radius, yCentar + 1):
                if ((x - xCentar) * (x - xCentar) + (y - yCentar) * (y - yCentar) <= radius):  # radius * radius
                    xSym = xCentar - (x - xCentar)
                    ySym = yCentar - (y - yCentar)
                    # (x, y), (x, ySym), (xSym , y), (xSym, ySym) are in the circle
                    # Ovo je za slucaj bez uslova, ako zatreba brzina izvrsavanja
                    # -----------------------------------------------------
                    sum += int(picture[x, y]) * int(picture[x, y])
                    sum += int(picture[x, ySym]) * int(picture[x, ySym])
                    sum += int(picture[xSym, y]) * int(picture[xSym, y])
                    sum += int(picture[xSym, ySym]) * int(picture[xSym, ySym])
                    numberOfPixels += 4
                    # -----------------------------------------------------
                    if ((x >= 0) & (width > x) & (y >= 0) & (height > y)):
                        image[x, y] = 255
                        array.append(int(picture[x, y]))
                        sum += int(picture[x, y]) * int(picture[x, y])
                        numberOfPixels += 1
                    if ((x >= 0) & (width > x) & (ySym >= 0) & (height > ySym) & (y != ySym)):
                        image[x, ySym] = 255
                        array.append(int(picture[x, ySym]))
                        sum += int(picture[x, ySym]) * int(picture[x, ySym])
                        numberOfPixels += 1
                    if ((xSym >= 0) & (width > xSym) & (y >= 0) & (height > y) & (xSym != x)):
                        image[xSym, y] = 255
                        array.append(int(picture[xSym, y]))
                        sum += int(picture[xSym, y]) * int(picture[xSym, y])
                        numberOfPixels += 1
                    if ((xSym >= 0) & (width > xSym) & (ySym >= 0) & (height > ySym) & (xSym != x) & (y != ySym)):
                        image[xSym, ySym] = 255
                        array.append(int(picture[xSym, ySym]))
                        sum += int(picture[xSym, ySym]) * int(picture[xSym, ySym])
                        numberOfPixels += 1

        #cv2.imwrite('krugTest.png', image)  # sacuva sliku sa iscrtanim belim krugom, za proveru radiusa

        # cuva u fajl vrednosti 0-255 za izmenjenu sliku i original, za proveru vrednosti
        # povecati for petlje za duzinu slike ako zatreba, duze traje upisivanje u txt !!!
        # for i in range(0, width/3):
        #    for j in range(0, height/3):
        #        f = open('image.txt', 'a')
        #        f.write(str(image[i,j]) + " ")
        #        f.close()
        #        f = open('picture.txt', 'a')
        #        f.write(str(picture[i,j]) + " ")
        #        f.close()
        #    f = open('image.txt', 'a')
        #    f.write('\n')
        #    f.close()
        #    f = open('picture.txt', 'a')
        #    f.write('\n')
        #    f.close()

        #pom2 = int(np.mean(array, dtype=np.float64))
        pom = int(np.sqrt(sum/numberOfPixels)) # ako treba da se vrati rucno izracunavanje
        return pom


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    MainWindow = MainWindow()
    MainWindow.show()

    sys.exit(app.exec_())