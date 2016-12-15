# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created: Wed Dec 14 23:23:05 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(994, 871)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.VideoView = QtGui.QGraphicsView(self.centralwidget)
        self.VideoView.setEnabled(True)
        self.VideoView.setGeometry(QtCore.QRect(40, 30, 761, 781))
        self.VideoView.setMouseTracking(True)
        self.VideoView.setInteractive(True)
        self.VideoView.setSceneRect(QtCore.QRectF(0.0, 0.0, 480.0, 640.0))
        self.VideoView.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.VideoView.setObjectName("VideoView")
        self.StartButton = QtGui.QPushButton(self.centralwidget)
        self.StartButton.setGeometry(QtCore.QRect(860, 30, 75, 23))
        self.StartButton.setObjectName("StartButton")
        self.SetStateGoodButton = QtGui.QPushButton(self.centralwidget)
        self.SetStateGoodButton.setGeometry(QtCore.QRect(860, 90, 75, 23))
        self.SetStateGoodButton.setObjectName("SetStateGoodButton")
        self.SetStateBadButton = QtGui.QPushButton(self.centralwidget)
        self.SetStateBadButton.setGeometry(QtCore.QRect(860, 140, 75, 23))
        self.SetStateBadButton.setObjectName("SetStateBadButton")
        self.NextFrameButton = QtGui.QPushButton(self.centralwidget)
        self.NextFrameButton.setGeometry(QtCore.QRect(860, 190, 75, 23))
        self.NextFrameButton.setObjectName("NextFrameButton")
        self.coordLabel = QtGui.QLabel(self.centralwidget)
        self.coordLabel.setGeometry(QtCore.QRect(40, 10, 141, 16))
        self.coordLabel.setObjectName("coordLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 994, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.StartButton.setText(QtGui.QApplication.translate("MainWindow", "Start", None, QtGui.QApplication.UnicodeUTF8))
        self.SetStateGoodButton.setText(QtGui.QApplication.translate("MainWindow", "Good", None, QtGui.QApplication.UnicodeUTF8))
        self.SetStateBadButton.setText(QtGui.QApplication.translate("MainWindow", "Bad", None, QtGui.QApplication.UnicodeUTF8))
        self.NextFrameButton.setText(QtGui.QApplication.translate("MainWindow", "NextFrame", None, QtGui.QApplication.UnicodeUTF8))
        self.coordLabel.setText(QtGui.QApplication.translate("MainWindow", "TextLabel", None, QtGui.QApplication.UnicodeUTF8))

