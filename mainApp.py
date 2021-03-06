# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main2.ui'
#
# Created: Sun Dec 18 12:39:18 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!
import copy
import math
import sys

import cv2
import numpy as np
import time
from PySide import QtCore, QtGui
from PySide.QtCore import QObject
from PySide.QtCore import Signal
from PySide.QtGui import *
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

import py_gui.NetworkSettings as network_settings
import py_gui.SettingsWindow as settings
import py_gui.Ui_MainWindow as ui
import py_gui.AboutWindow as about

from image_processor import hue
from scene.NullObject import NullObject
import fnmatch
import os
import threading
import glob, os

class LongPythonThread(QObject):

    thread_finished = Signal(str)

    def __init__(self):
        super(LongPythonThread, self).__init__()

    def train_network(self, filter, file_name):
        filter.tf_learn(file_name)
        self.thread_finished.emit("Done")

class MainWindow(QtGui.QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.long_thread = LongPythonThread()
        self.long_thread.thread_finished.connect(self.training_finished)

        self.fov_size = 32
        self.scan_poly = None
        self.image_or_video_filename = ""

        self.polySnipet = None
        self.setupUi(self)
        self.init_actions()
        self.pix_item = None
        self.trainer = None
        self.cnn_filter = None

        self.training_subfolder = "notset"

        self.drawing_pen = QtGui.QPen("White")
        self.drawing_pen.setWidth(2)


        # self.keyPressEvent = self.key_press_handler

        self.scaningRect = None

        # flag for autoplay
        self.autoplaying = False
        # pointer to classifier
        self.classifier = None
        self.networkSettings = None
        self.aboutWindow = None
        self.file_name = ""

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

        self.trainingData = list()

        self.view_capture = None
        self.currentFrame = None

        # timer for autoplay
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)

        start_point_X_coord = 3 * (600 - 200) / 4
        start_point_Y_coord = 2 * (480 - 200) / 3

        # self.scaningRect = QGraphicsRectItem(start_point_X_coord, start_point_Y_coord, 200, 200)
        # self.scaningRect.setPen(self.drawing_pen)

        # self.graphics_scene.addItem(self.scaningRect)


    def training_finished(self):
        self.descriptionLable.setText("Training finished!")

    def save_network_dialog(self):

        SettingsWindow = settings.SettingsWindow(self)

        SettingsWindow.show()
        test = SettingsWindow.exec_()
        if self.file_name != "":
            fullFileName = self.file_name + '.pkl'
            joblib.dump(self.classifier, fullFileName)
            self.status_label.setText("Saved in the file: " + fullFileName)
        else:
            fullFileName = "network.pkl"
            joblib.dump(self.classifier, fullFileName)
            self.status_label.setText("Saved in the file: " + fullFileName)

    def key(self, event):
        if event.key() is 83:
            self.status_label.setText("Started saving...")
            if self.classifier is not None:
                self.save_network_dialog()

        elif event.key() is QtCore.Qt.Key_Right:
            self.next_frame()
            # print event.key()

    def init_actions(self):

        self.wrongButton.clicked.connect(lambda: self.wrong_button_action())
        self.goodButton.clicked.connect(lambda: self.good_button_action())
        self.badButton.clicked.connect(lambda: self.bad_button_action())

        self.nextButton.clicked.connect(lambda: self.next_frame())
        self.backButton.clicked.connect(lambda: self.back_frame())

        self.Relearn.clicked.connect(self.cnn_relearn)
        # self.frameResizer.valueChanged.connect(self.FOV_resize)

        self.actionOpen.triggered.connect(self.open)

        self.graphicsView.keyPressEvent = self.key
        self.graphicsView.mouseMoveEvent = self.FOV

        self.actionAutoplay.triggered.connect(self.autoplay)

        self.actionLoad_ANN.setText("Reload image")

        self.actionLoad_ANN.triggered.connect(self.load_cnn)

        sequence = QKeySequence("Ctrl+R")

        # shortcut = QShortcut(sequence, self)

        self.actionLoad_ANN.setShortcut(sequence)

        self.actionClear_classifier.setText("Camera")

        sequence = QKeySequence("Shift+C")
        self.actionClear_classifier.setShortcut(sequence)

        self.actionClear_classifier.triggered.connect(self.clear_classifier)

        self.actionAbout.triggered.connect(self.about)

        self.actionCreate_classifier.triggered.connect(self.open_network_settings)

    def closeEvent(self, e):
        if self.view_capture is not None and self.view_capture.isOpened():
            self.view_capture.release()

    def load_cnn(self):
        if self.image_or_video_filename != "":
            self.view_capture = cv2.VideoCapture(self.image_or_video_filename)
            if not self.view_capture.isOpened():
                return
            self.label.setText("Selected file: " + self.image_or_video_filename)
            flag, self.currentFrame = self.view_capture.read()
            if flag:
                self.show_frame_general(self.currentFrame)

    def about(self):

        self.aboutWindow = about.AboutWindow()
        self.aboutWindow.show_test()

    def FOV(self,ev):
        position = ev.pos()
        xPos = position.x()
        yPos = position.y()
        self.status_label.setText("X: "+str(xPos)+" Y: "+str(yPos))

        if(self.polySnipet != None):
            try:
                self.graphics_scene.removeItem(self.polySnipet)
            except Exception:
                print Exception

        half = self.fov_size / 2
        self.polySnipet = QtGui.QGraphicsRectItem(xPos - half, yPos - half, self.fov_size, self.fov_size)
        #self.polySnipet.setBrush(QtGui.QBrush(QtCore.Qt.green))

        self.polySnipet.setPen(self.drawing_pen)
        self.graphics_scene.addItem(self.polySnipet)

    def FOV_resize(self, change):
        self.fov_size = change

    def evaluate(self):
        if self.cnn_filter is not None:
            self.cnn_filter.evaluate_img()

    def cnn_relearn(self):
        if self.cnn_filter is None:
            return
        others = (r'/home/darko/Documents/ComputerVision/test data/other/', r'*.png')
        points = (r'/home/darko/Documents/ComputerVision/test data/point/', r'*.png')
        palms = (r'/home/darko/Documents/ComputerVision/test data/palm', r'*.png')
        fists = (r'/home/darko/Documents/ComputerVision/test data/fist', r'*.png')

        correct = 0
        incorrect = 0

        point_image = np.empty((60 * 10, 60 * 10, 3), dtype=np.uint8)
        currentRow = 0
        currentColomn = 0

        confused = []
        logic_names_dict = {0: "fist", 1: "palm", 2: "point", 3: "other"}

        for pathAndFilename in glob.iglob(os.path.join(points[0], points[1])):
            image = cv2.imread(pathAndFilename)



            prediction = self.cnn_filter.evaluate_img(image)
            maxIndex = np.argmax(prediction)
            if maxIndex != 2:
                incorrect += 1
                confused.append(maxIndex)

            else:
                correct += 1
                point_image[currentColomn:currentColomn + 60, currentRow:currentRow + 60, :] = image

            currentColomn += 60

            if currentColomn >= 600:
                currentColomn = 0
                currentRow += 60

        cv2.imwrite("PointImage.png", point_image)
        counts = np.bincount(confused)
        index = -1
        if counts.__len__() != 0:
            index = np.argmax(counts)

        print("For point images results are: Correct: " + str(correct) + "; Incorrect: " + str(incorrect) + "|Most incorrect with class: " + logic_names_dict.get(index, "none"))

        correct = 0
        incorrect = 0
        confused = []


        palms_image = np.empty((60 * 10, 60 * 10, 3), dtype=np.uint8)
        currentRow = 0
        currentColomn = 0

        for pathAndFilename in glob.iglob(os.path.join(palms[0], palms[1])):
            image = cv2.imread(pathAndFilename)



            prediction = self.cnn_filter.evaluate_img(image)
            maxIndex = np.argmax(prediction)
            if maxIndex != 1:
                incorrect += 1

                confused.append(maxIndex)
            else:
                correct += 1
                palms_image[currentColomn:currentColomn + 60, currentRow:currentRow + 60, :] = image

            currentColomn += 60
            if currentColomn >= 600:
                currentColomn = 0
                currentRow += 60
        cv2.imwrite("PalmsImage.png", palms_image)

        counts = np.bincount(confused)
        index = -1
        if counts.__len__() != 0:
            index = np.argmax(counts)
        print("For palms images results are: Correct: " + str(correct) + "; Incorrect: " + str(incorrect) + "|Most incorrect with class: " + logic_names_dict.get(index, "none"))

        correct = 0
        incorrect = 0

        fists_image = np.empty((60 * 10, 60 * 10, 3), dtype=np.uint8)
        currentRow = 0
        currentColomn = 0
        confused = []

        for pathAndFilename in glob.iglob(os.path.join(fists[0], fists[1])):
            image = cv2.imread(pathAndFilename)
            prediction = self.cnn_filter.evaluate_img(image)
            maxIndex = np.argmax(prediction)
            if maxIndex != 0:
                incorrect += 1

                confused.append(maxIndex)
            else:
                correct += 1
                fists_image[currentColomn:currentColomn + 60, currentRow:currentRow + 60, :] = image


            currentColomn += 60
            if currentColomn >= 600:
                currentColomn = 0
                currentRow += 60
        cv2.imwrite("FistsImage.png", fists_image)

        counts = np.bincount(confused)
        index = -1
        if counts.__len__() != 0:
            index = np.argmax(counts)
        print("For fists images results are: Correct: " + str(correct) + "; Incorrect: " + str(incorrect) + "|Most incorrect with class: " + logic_names_dict.get(index, "none"))

        correct = 0
        incorrect = 0

        others_image = np.empty((60 * 10, 60 * 10, 3), dtype=np.uint8)
        currentRow = 0
        currentColomn = 0
        confused = []

        for pathAndFilename in glob.iglob(os.path.join(others[0], others[1])):
            image = cv2.imread(pathAndFilename)
            prediction = self.cnn_filter.evaluate_img(image)
            maxIndex = np.argmax(prediction)
            if maxIndex != 3:
                incorrect += 1

                confused.append(maxIndex)
            else:
                correct += 1
                others_image[currentColomn:currentColomn + 60, currentRow:currentRow + 60, :] = image


            currentColomn += 60
            if currentColomn >= 600:
                currentColomn = 0
                currentRow += 60
        cv2.imwrite("OthersImages.png", others_image)

        counts = np.bincount(confused)
        index = -1
        if counts.__len__() != 0:
            index = np.argmax(counts)
        print("For other images results are: Correct: " + str(correct) + "; Incorrect: " + str(incorrect) + "|Most incorrect with class: " + logic_names_dict.get(index, "none"))

        # if self.cnn_filter is not None:
        #     self.status_label.setText("CNN trainer started relearning based on training data.")
        #     self.descriptionLable.setText("Training in progress...")
        #     # QThread and work on it.
        #     # t = threading.Thread(target=self.long_thread.train_network, args=(self.cnn_filter, self.file_name))
        #     # t.daemon = True
        #     # t.start()
        #
        #     self.cnn_filter.tf_learn(self.file_name)
        # else:
        #     self.status_label.setText("Trainer not set or not trained!")

    def open_network_settings(self):
        if self.networkSettings is None:
            self.networkSettings = network_settings.NetworkSettings(self)
            self.networkSettings.show_test()
        else:
            self.networkSettings.show_test()


    def create_classifier(self, network_layers, activation_string, solver_string, alpha_float, batch_size_string,
                          learning_rate_string, learning_rate_float, power_float,
                          max_iter_int, shuffle_bool, random_state_oint, tol_float, verbose_bool, warm_state_bool,
                          momentum_float, nesterovs_momentum_bool, early_stopping_bool,
                          validation_fraction_float, beta_1_float, beta_2_float, epsilon_float):

        self.classifier = MLPClassifier(hidden_layer_sizes=network_layers, activation=activation_string,
                                        solver=solver_string, alpha=alpha_float, batch_size=batch_size_string,
                                        learning_rate=learning_rate_string, learning_rate_init=learning_rate_float,
                                        power_t=power_float, max_iter=max_iter_int, shuffle=shuffle_bool,
                                        random_state=random_state_oint, tol=tol_float, verbose=verbose_bool,
                                        warm_start=warm_state_bool, momentum=momentum_float,
                                        nesterovs_momentum=nesterovs_momentum_bool, early_stopping=early_stopping_bool,
                                        validation_fraction=validation_fraction_float,
                                        beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)

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
        if self.element is not None:
            self.element.graphics_view_mouse_drag(drag, self.graphics_scene)

    def QImageToCvMat(self, incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_RGB32)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.constBits()
        arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
        return arr

    def left_click(self, position):
        self.showCoords(position.scenePos())
        if self.training_subfolder == 'notset':
            msgBox = QMessageBox()
            msgBox.setText("You need to select the tool before clicking on image!")
            msgBox.exec_()
            return
        pos = position.scenePos()
        xCord = int(pos.x())
        yCord = int(pos.y())

        # check state variable for folder
        folder_name = "training data/"

        folder_name = folder_name + self.training_subfolder
        imageCount = len(fnmatch.filter(os.listdir(folder_name), '*.png'))

        if self.pix_item is not None:
            pixMap = self.pix_item.pixmap()
            qimg = pixMap.toImage()
            matrix = self.QImageToCvMat(qimg)
            halfFov =  self.fov_size / 2
            part = matrix[yCord - halfFov: yCord + halfFov, xCord - halfFov: xCord + halfFov]

            name = str(imageCount+1) + ".png"

            folder_name = folder_name + "/" + name
            cv2.imwrite(folder_name, part)

            # cv2.imshow("full", part)

        # if self.trainer is None:
        #     self.descriptionLable.setText("Set state for trainer!")
        #     return
        #
        # if self.element is None:
        #     self.descriptionLable.setText("Left click in wrong state")
        #     return
        #
        # obj = self.element.left_click(position, self.graphics_scene)
        # if obj is not None:
        #     self.items.append(obj)
        #     self.trainingData.append(copy.deepcopy(obj))

    def right_click(self, position):

        if self.cnn_filter is not None and self.cnn_filter.modelTrained:
            pos = position.scenePos()
            xCord = int(pos.x())
            yCord = int(pos.y())
            pixMap = self.pix_item.pixmap()
            qimg = pixMap.toImage()
            matrix = self.QImageToCvMat(qimg)

            halfFov = self.fov_size / 2
            part = matrix[yCord - halfFov: yCord + halfFov, xCord - halfFov: xCord + halfFov]
            part = part[:, :, :3]

            prediction = self.cnn_filter.evaluate_img(part)
            max_index = np.argmax(prediction)

            self.scan_poly = QtGui.QGraphicsRectItem(xCord - halfFov, yCord - halfFov, self.fov_size, self.fov_size)

            pen = None
            if max_index == 3:
                pen = QtGui.QPen("Gray")
            if max_index == 0:
                pen = QtGui.QPen("Blue")
            elif max_index == 1:
                pen = QtGui.QPen("Green")
            elif max_index == 2:
                pen = QtGui.QPen("Red")

            self.scan_poly.setPen(pen)
            self.graphics_scene.addItem(self.scan_poly)
            self.graphics_scene.update()

        # if self.element is None:
        #     self.descriptionLable.setText("Right click in wrong state")
        #     return
        #
        # self.element.right_click(position, self.graphics_scene, self.items)
        # for object in self.trainingData:
        #
        #     clickX = position.scenePos().x()
        #     clickY = position.scenePos().y()
        #
        #     x = object.x
        #     y = object.y
        #     r = object.circle_radius
        #
        #     dist = math.sqrt(pow(x - clickX, 2) + pow(y - clickY, 2))
        #     if dist <= r:
        #         self.trainingData.remove(object)
        #         self.status_label.setText("Removed circle from training data!")

    def wrong_button_action(self):

        self.element = None
        self.drawing_pen.setColor("Blue")
        self.training_subfolder = 's2'
        # self.label_6.setPixmap(QtGui.QPixmap("res/wrong.png"))
        # self.descriptionLable.setText("Wrong tool is used for\n selection of the circles \nthat are not caps at all.")

    def good_button_action(self):
        self.element = None
        self.drawing_pen.setColor("Green")
        self.training_subfolder = 's1'
        # self.element = CircleObject("Good")
        # self.label_6.setPixmap(QtGui.QPixmap("res/green.png"))
        # self.descriptionLable.setText("Good tool is used for marking\nvaluable part of image.")

    def bad_button_action(self):
        self.element = None
        self.drawing_pen.setColor("Red")
        self.training_subfolder = 's3'
        # self.element = CircleObject("Bad")
        # self.label_6.setPixmap(QtGui.QPixmap("res/red.png"))
        # self.descriptionLable.setText("Bad tool is used for marking\nnegative part of image.")

    def open(self):
        self.image_or_video_filename, _ = QtGui.QFileDialog.getOpenFileName(self, "Open File", QtCore.QDir.currentPath())
        if self.image_or_video_filename:
            # self.init_learning()
            self.view_capture = cv2.VideoCapture(self.image_or_video_filename)
            if not self.view_capture.isOpened():
                return
            self.label.setText("Selected file: " + self.image_or_video_filename)
            flag, self.currentFrame = self.view_capture.read()
            if flag:
                self.show_frame_general(self.currentFrame)

    def load_pickle_file(self):
        file_name, _ = QtGui.QFileDialog.getOpenFileName(self, "Open File", QtCore.QDir.currentPath())
        if file_name:
            try:
                self.classifier = joblib.load(file_name)
                self.status_label.setText("Successfully loaded classifier data.")
            except Exception as ex:

                self.status_label.setText("Error ocurred. Error message: " + str(ex))

    def sliding_window(self, image, stepSize, windowSize):
        # slide a window across the image
        for y in xrange(0, image.shape[0], stepSize):
            for x in xrange(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


    def window_stack(self, a, stepsize, width):
        n = a.shape[0]
        return np.hstack(a[i:1 + n + i - width:stepsize] for i in range(0, width))

    def show_frame_general(self, frame):

        frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)

        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
        self.graphics_scene.removeItem(self.pix_item)
        self.graphics_scene.update()
        img = QtGui.QPixmap.fromImage(image)
        self.pix_item = QtGui.QGraphicsPixmapItem(img)

        # QtGui.QGraphicsRectItem(window[0] - half, window[1] - half, self.fov_size, self.fov_size)



        self.graphics_scene.addItem(self.pix_item)
        self.graphics_scene.update()

        window_size = 180

        start_point_X_coord = (frame.shape[1] / 2) - (window_size / 2)
        start_point_Y_coord = (frame.shape[0] / 2) - (window_size / 2)



        smaller_window = frame[start_point_X_coord:start_point_X_coord + window_size, start_point_Y_coord: start_point_Y_coord + window_size, :]
        cv2.imshow("FOV", smaller_window)
        vsplit = np.split(smaller_window, [60, 120])

        sliding_windows = []

        for split in vsplit:
            more = np.split(split, [60, 120], 1)
            sliding_windows.append(more)


        # half = self.fov_size / 2

        for window in sliding_windows:
            if self.cnn_filter is not None and self.cnn_filter.modelTrained:
                if window[2].shape[0] != self.fov_size or window[2].shape[1] != self.fov_size:
                    continue
                roi = np.divide(window[2], 1000.)
                # print "Obradjen prozor"
                prediction = self.cnn_filter.evaluate_img(roi)
                max_index = np.argmax(prediction[0])
                if prediction[0][max_index] > 0.85:
                    # self.scan_poly = QtGui.QGraphicsRectItem(window[0] - half, window[1] - half, self.fov_size, self.fov_size)


                    pen = None
                    if max_index == 3:
                        continue
                    if max_index == 0:
                        pen = QtGui.QPen("Blue")
                        self.status_label.setText("FIST!")
                    elif max_index == 1:
                        pen = QtGui.QPen("Green")
                        self.status_label.setText("PALM!")
                    elif max_index == 2:
                        pen = QtGui.QPen("Red")
                        self.status_label.setText("POINT!")
                    # self.scan_poly.setPen(pen)
                    # self.graphics_scene.addItem(self.scan_poly)
                    # self.graphics_scene.update()

                # print(prediction)
                # print(max_index)

            # if self.scan_poly is not None:
            #     self.graphics_scene.removeItem(self.scan_poly)
            #     self.graphics_scene.update()
            # self.scan_poly = QtGui.QGraphicsRectItem(window[0] - half, window[1] - half, self.fov_size, self.fov_size)

            # self.graphics_scene.addItem(self.scan_poly)
            # self.graphics_scene.update()
            # time.sleep(0.005)

            # print (window)





    def show_frame(self, frame):

        self.trainingData, colorImage = hue.process_color_image(frame)

        # print ("Hue results: ")
        # print self.trainingData
        frame = cv2.cvtColor(colorImage, cv2.cv.CV_BGR2RGB)

        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
        self.graphics_scene.clear()
        self.graphics_scene.update()

        img = QtGui.QPixmap.fromImage(image)
        self.pix_item = QtGui.QGraphicsPixmapItem(img)

        self.graphics_scene.addItem(self.pix_item)
        self.graphics_scene.update()


        #self.graphicsView.autoFillBackground()
        #self.graphicsView.fitInView(self.graphicsView.rect())

        if (self.trainingData != None):

            for object in self.trainingData:

                if self.cnn_filter is not None and self.cnn_filter.modelTrained:

                    roi = hue.ROI(self.fov_size, self.fov_size, int(object.y), int(object.x), frame)
                    if roi.shape != (self.fov_size, self.fov_size, 3):
                        continue
                    # cv2.imshow("ROI found", roi)
                    prediction = self.cnn_filter.evaluate_img(roi)
                    maxIndex = np.argmax(prediction)

                    # print prediction
                    # print ("Max index of prediction: " + str(maxIndex))
                    if maxIndex == 1:
                        continue
                    if(object.mode == 'Good' and maxIndex == 0):
                        object.draw(self.graphics_scene)
                    elif(object.mode == 'Bad' and maxIndex == 2):
                        object.draw(self.graphics_scene)
                    else:
                        print("Mismatch in prediction and HUE")
                        print("Prediction class: "+ str(maxIndex))
                        print("Hue class: "+object.mode)
                        if prediction[0][maxIndex] > 0.7:
                            if maxIndex == 0:
                                object.change_mode("Good")
                                object.draw(self.graphics_scene)
                            elif maxIndex == 2:
                                object.change_mode("Bad")
                                object.draw(self.graphics_scene)
                        else:
                            object.draw(self.graphics_scene)

                else:
                    self.network_status.setText("Convolutional network not created!!!")
                    object.draw(self.graphics_scene)


        self.graphics_scene.update()

    def next_frame(self):
        if self.view_capture is not None:
            go_next = True
            if self.items is not None and len(self.items) > 0:
                go_next = self.save_work_pop_up()
            if go_next:
                flag, self.currentFrame = self.view_capture.read()
                if flag:
                    self.show_frame_general(self.currentFrame)
                else:
                    self.view_capture.release()

    def save_work_pop_up(self):
        msgBox = QtGui.QMessageBox()
        msgBox.setText("The document has been modified.")
        msgBox.setInformativeText("Do you want to save your changes?")
        msgBox.setStandardButtons(QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel)
        msgBox.setDefaultButton(QtGui.QMessageBox.Save)
        ret = msgBox.exec_()
        if ret == QtGui.QMessageBox.Save:
            self.save()
            self.remove_all_items()
            self.trainingData = []
            return True
        elif ret == QtGui.QMessageBox.Discard:
            self.remove_all_items()
            self.trainingData = []
            return True
        elif ret == QtGui.QMessageBox.Cancel:
            return False
        else:
            return False

    def remove_all_items(self):
        items_len = len(self.items)
        if self.items is not None and items_len > 0:
            for i in range(0, items_len):
                self.graphics_scene.removeItem(self.items[i].circle)
        self.items = list()

    def save(self):
        # mesto sta uradis kad liknes save

        if self.classifier is None:
            self.status_label.setText("No classifier set!")
            return
        items_len = len(self.trainingData)
        trainingData = list()
        labels = list()

        if self.trainingData is not None and items_len > 0:
            for i in range(0, items_len):
                figure = self.trainingData[i].circle
                if self.trainingData[i].mode == "Good":
                    labels.append(1)
                elif self.trainingData[i].mode == "Bad":
                    labels.append(0)

                # c = figure.rect().center()
                # x = int(c.x())
                x = int(self.trainingData[i].x)
                xf = x / 1000.0
                # y = int(c.y())
                y = int(self.trainingData[i].y)
                yf = y / 1000.0

                grayscalepicture = cv2.cvtColor(self.currentFrame, cv2.COLOR_BGR2GRAY)
                val = self.quadriatic_mean_circle(y, x, 5, grayscalepicture)
                val = val / 1000.0
                trainingData.append([xf, yf, val])

        if items_len > 0:
            self.classifier.fit(trainingData, labels)

        print(trainingData)

        print(labels)

    def back_frame(self):
        print("back")

    def clear_classifier(self):
        self.status_label.setText("Starting camera...")
        self.view_capture = cv2.VideoCapture(0)
        customFont = QtGui.QFont("Times", 72, QtGui.QFont.Bold)

        self.status_label.setFont(customFont)

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
            self.timer.start(10)
            self.actionAutoplay.setText("Stop autoplay")
        self.autoplaying = not self.autoplaying

    def scroll_to_next_frame(self, event):
        self.next_frame()

    def scroll_dummy_func(self, event):
        # status text print...
        print "Scroll video disabled"

    def showCoords(self, point):
        xPoint = point.x()
        yPoint = point.y()
        self.status_label.setText("X: " + str(xPoint) + "  Y: " + str(yPoint))

    def quadriatic_mean_circle(self, xCentar, yCentar, radius, picture):  # mora da se prosledi gray_scale image
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

        # pom2 = int(np.mean(array, dtype=np.float64))
        pom = int(np.sqrt(sum / numberOfPixels))  # ako treba da se vrati rucno izracunavanje
        return pom


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    MainWindow = MainWindow()
    MainWindow.show()

    sys.exit(app.exec_())
