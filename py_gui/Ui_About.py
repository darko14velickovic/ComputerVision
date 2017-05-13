# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'about.ui'
#
# Created: Fri May 12 23:48:38 2017
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_About(object):
    def setupUi(self, About):
        About.setObjectName("About")
        About.resize(626, 483)
        self.verticalLayout_2 = QtGui.QVBoxLayout(About)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.about_app = QtGui.QLabel(About)
        self.about_app.setTextFormat(QtCore.Qt.LogText)
        self.about_app.setWordWrap(True)
        self.about_app.setObjectName("about_app")
        self.verticalLayout_2.addWidget(self.about_app)
        self.label_2 = QtGui.QLabel(About)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.label_3 = QtGui.QLabel(About)
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)

        self.retranslateUi(About)
        QtCore.QMetaObject.connectSlotsByName(About)

    def retranslateUi(self, About):
        About.setWindowTitle(QtGui.QApplication.translate("About", "About", None, QtGui.QApplication.UnicodeUTF8))
        self.about_app.setText(QtGui.QApplication.translate("About", "This application is Python 2.7 Pyside desktop application for hybrid image recognition.It uses OpenCV and TFLearn libraries for learning and processing of image/video input. Preprocessing algorithms used are Smoothing and Sobel filter, on which then we applied Hue transform for circle recognition. Purpose of the application is finding and recognizing bottle caps in video feed. Hue is used for finding bottles in the images( in the training process it can make training and sorting of images easier and for testing speeds up the process of finding bottles). Then Quadratic mean error is used in Region of Interest (circle given by Hue) to determine if the circle is Bottle cap or not (bottle caps usually dont have enough black color in Gray Scale). After that the Deep Neural network takes ROI and tests the results. If DNN is more then 70% sure of its classification we say it is right, in all other cases we use Quadratic Mean Error to say if the bottle has a cap or not.", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("About", "Neural network takes images as input (size can be adjusted but we used 40x40 images) with image augmentation (fliping and rotation) and given in full color to the network. Besides input layer, first layer is convolutional_2d layer with Rectefied Linear Activation number of inputs same as the input layer. Next in pipeline is MaxPool layer with factor 2, followed by two convolutional layers both with double of input size inputs, after that there is a fully connected layer with 512 inputs,then dropout layer with factor 0.5. At the end there is one more fully connected layer with 3 inputs and softmax activation. Learning rate is 0.001 because that would represent one value change of RGB intensity, loss function is from Tensorflow ategorical_crossentropy and gradient optimizer is Adaptive Moment Estimation (Adam).  ", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("About", "This project is a part of combined assignment for courses of Pattern Recognition & Advanced Image Processing at Faculty of Electronic Engineering,  University of Nis. Project team: Darko Velickovic, Aleksandar Milosevic & Nemanja Mladenovic.", None, QtGui.QApplication.UnicodeUTF8))

