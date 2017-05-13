# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'create_network.ui'
#
# Created: Fri May 12 22:30:28 2017
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setWindowModality(QtCore.Qt.NonModal)
        Form.resize(382, 234)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.input_number = QtGui.QSpinBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.input_number.sizePolicy().hasHeightForWidth())
        self.input_number.setSizePolicy(sizePolicy)
        self.input_number.setMinimum(32)
        self.input_number.setMaximum(64)
        self.input_number.setSingleStep(2)
        self.input_number.setObjectName("input_number")
        self.horizontalLayout_2.addWidget(self.input_number)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtGui.QLabel(Form)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.network_name = QtGui.QLineEdit(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.network_name.sizePolicy().hasHeightForWidth())
        self.network_name.setSizePolicy(sizePolicy)
        self.network_name.setObjectName("network_name")
        self.horizontalLayout.addWidget(self.network_name)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.save_button = QtGui.QPushButton(Form)
        self.save_button.setObjectName("save_button")
        self.verticalLayout.addWidget(self.save_button)
        self.load_button = QtGui.QPushButton(Form)
        self.load_button.setObjectName("load_button")
        self.verticalLayout.addWidget(self.load_button)
        self.loading_label = QtGui.QLabel(Form)
        self.loading_label.setObjectName("loading_label")
        self.verticalLayout.addWidget(self.loading_label)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtGui.QApplication.translate("Form", "Create network", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Form", "Number of inputs", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Form", "Network name:", None, QtGui.QApplication.UnicodeUTF8))
        self.save_button.setText(QtGui.QApplication.translate("Form", "Create network", None, QtGui.QApplication.UnicodeUTF8))
        self.load_button.setText(QtGui.QApplication.translate("Form", "Load network", None, QtGui.QApplication.UnicodeUTF8))
        self.loading_label.setText(QtGui.QApplication.translate("Form", "Loading...", None, QtGui.QApplication.UnicodeUTF8))

