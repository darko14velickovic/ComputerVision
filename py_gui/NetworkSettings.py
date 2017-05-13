from PySide import QtGui
from PySide import QtCore

from PySide.QtGui import *

from PySide.QtCore import QObject, Slot, Signal

from Ui_Network_settings import Ui_Form
from image_processor import Trainer
import threading


class LongPythonThread(QObject):
    thread_finished = Signal(str)

    def __init__(self):
        super(LongPythonThread,self).__init__()

    def long_thread(self, form, fov, name, flag):
        form.cnn_filter = Trainer.CnnTrainer(fov, name, flag)
        self.thread_finished.emit("Done")



class NetworkSettings(QtGui.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(NetworkSettings, self).__init__()
        self.setupUi(self)
        self.caller = parent
        # self.test = Qbutton()

        self.long_thread = LongPythonThread()
        self.long_thread.thread_finished.connect(self.creation_finished)

        # Obsolete

        #self.combobox_setup()
        #self.current_settings()
        #self.setWindowTitle("")
        #self.defaultButton.clicked.connect(self.default_settings)
        #self.saveButton.clicked.connect(self.save_settings)

        self.save_button.clicked.connect(self.save_click)
        self.load_button.clicked.connect(self.load_click)

    def creation_finished(self):
        # print("reply2 finished")
        msgBox = QMessageBox()
        msgBox.setText("Created network!")
        msgBox.exec_()
        self.caller.network_status.setText(
            "Network name: " + self.caller.file_name + ", FOV: " + str(self.caller.fov_size) + "x" + str(
                self.caller.fov_size))
        self.close()

    def load_click(self):
        file_name, _ = QtGui.QFileDialog.getOpenFileName(self, "Open File", QtCore.QDir.currentPath())
        if file_name:
            try:
                path = QtCore.QDir.currentPath()
                splitName = file_name.split('.')
                found = False
                for split in splitName:
                    if split == 'tfl':
                        found = True
                if not found:
                    self.loading_label('File is not tfl format!')
                    return

                netName = splitName[0]
                netName = netName.replace(path, '')
                if netName.startswith('/'):
                    netName = netName[1:]

                self.loading_label.setText("File is tfl format, trying to load network: "+ netName+ " from data!");

                self.caller.file_name = netName

                self.caller.fov_size = int(self.input_number.text())

                t = threading.Thread(target=self.long_thread.long_thread,
                                     args=(self.caller, self.caller.fov_size, netName, True))
                t.daemon = True
                t.start()


            except Exception as ex:
                self.loading_label.setText("Error ocurred. Error message: " + str(ex))
    def save_click(self):

        self.caller.file_name = self.network_name.text()

        self.caller.fov_size = int(self.input_number.text())

        t = threading.Thread(target=self.long_thread.long_thread, args=(self.caller, self.caller.fov_size, self.network_name.text(), False))
        t.daemon = True
        t.start()

        # self.caller.cnn_filter = Trainer.CnnTrainer(self.caller.fov_size)



    # Obsolete
    def combobox_setup(self):
        self.activationComboBox.clear()
        self.activationComboBox.addItems(['identity', 'logistic', 'tanh', 'relu'])

        self.solverComboBox.clear()
        self.solverComboBox.addItems(['lbfgs', 'sgd', 'adam'])

        self.learning_rateComboBox.clear()
        self.learning_rateComboBox.addItems(['constant', 'invscaling', 'adaptive'])

    # Obsolete
    def default_settings(self):

        if self.caller.classifier is not None:
            parameters = self.caller.classifier.get_params(False)
            print parameters


        self.numberOfLayersspinBox.setValue(1)
        self.lineEdit.setText("100,")

        self.learning_rateComboBox.setCurrentIndex(3)
        self.solverComboBox.setCurrentIndex(2)
        self.alphaLineEdit.setText('0.0001')
        self.batch_sizeLineEdit_2.setText('auto')
        self.learning_rateComboBox.setCurrentIndex(0)
        self.max_itereLineEdit.setText("200")
        self.random_stateLineEdit.setText("21")
        self.shuffleCheckBox.setChecked(True)
        self.tolLineEdit.setText('0.00001')
        self.learningRateLineEdit.setText('0.001')
        self.powerTLineEdit.setText('0.5')
        self.verboseCheckBox.setChecked(False)
        self.warmStartCheckBox.setChecked(False)
        self.momentumLineEdit.setText('0.9')
        self.nesterovusLheckBox.setChecked(True)
        self.earlyStoppingCheckBox.setChecked(False)
        self.validationFractionLineEdit.setText('0.1')
        self.veta_1_LineEdit.setText('0.9')
        self.beta_2_LineEdit.setText('0.999')
        self.epsilonLineEdit.setText('1e-8')

    # Obsolete
    def save_settings(self):
        #save
        try:

            # network arhitecture
            tuple = ()
            stringSplit = self.lineEdit.text().split(',')
            for var in stringSplit:
                if var:
                    tuple += (int(var),)
            ####################
            activation = self.activationComboBox.currentText()
            solver = self.solverComboBox.currentText()
            alpha = float(self.alphaLineEdit.text())
            batch_size = self.batch_sizeLineEdit_2.text()
            learning_rate_algo = self.learning_rateComboBox.currentText()
            max_iterations = int(self.max_itereLineEdit.text())
            random_state = int(self.random_stateLineEdit.text())
            shuffle = self.shuffleCheckBox.isChecked()
            tol = float(self.tolLineEdit.text())
            learning_rate_init = float(self.learningRateLineEdit.text())
            power_t = float(self.powerTLineEdit.text())
            verbose = self.verboseCheckBox.isChecked()
            warm_start = self.warmStartCheckBox.isChecked()
            momentum = float(self.momentumLineEdit.text())
            nesterov = self.nesterovusLheckBox.isChecked()
            early_stop = self.earlyStoppingCheckBox.isChecked()
            validation = float(self.validationFractionLineEdit.text())
            beta_1 = float(self.veta_1_LineEdit.text())
            beta_2 = float(self.beta_2_LineEdit.text())
            epsilon = float(self.epsilonLineEdit.text())

            self.caller.create_classifier(tuple, activation, solver, alpha, batch_size, learning_rate_algo,
                                          learning_rate_init, power_t, max_iterations, shuffle, random_state, tol,
                                          verbose, warm_start, momentum, nesterov,
                                          early_stop, validation, beta_1, beta_2, epsilon)

            self.caller.status_label.setText('Saved settings')

            self.close()
        except Exception as ex:
            self.caller.status_label.setText('Error happened: '+ex.message)

    # Obsolete
    def current_settings(self):
        if self.caller.classifier is not None:
            parameters = self.caller.classifier.get_params(False)
            print parameters

            self.numberOfLayersspinBox.setValue(parameters['hidden_layer_sizes'].__len__())

            arhitecture = ""
            for element in parameters['hidden_layer_sizes']:
                arhitecture = arhitecture + str(element)+','
            self.lineEdit.setText(arhitecture)

            index = self.activationComboBox.findText(parameters['activation'])
            self.activationComboBox.setCurrentIndex(index)

            index = self.learning_rateComboBox.findText(parameters['learning_rate'])
            self.learning_rateComboBox.setCurrentIndex(index)

            index  = self.solverComboBox.findText(parameters['solver'])
            self.solverComboBox.setCurrentIndex(index)

            self.alphaLineEdit.setText(str(parameters['alpha']))
            self.batch_sizeLineEdit_2.setText(parameters['batch_size'])


            self.max_itereLineEdit.setText(str(parameters['max_iter']))
            self.random_stateLineEdit.setText(str(parameters['random_state']))

            self.shuffleCheckBox.setChecked(parameters['shuffle'])
            self.tolLineEdit.setText(str(parameters['tol']))

            self.learningRateLineEdit.setText(str(parameters['learning_rate_init']))
            self.powerTLineEdit.setText(str(parameters['power_t']))
            self.verboseCheckBox.setChecked(parameters['verbose'])
            self.warmStartCheckBox.setChecked(parameters['warm_start'])
            self.momentumLineEdit.setText(str(parameters['momentum']))
            self.nesterovusLheckBox.setChecked(parameters['nesterovs_momentum'])
            self.earlyStoppingCheckBox.setChecked(parameters['early_stopping'])
            self.validationFractionLineEdit.setText(str(parameters['validation_fraction']))
            self.veta_1_LineEdit.setText(str(parameters['beta_1']))
            self.beta_2_LineEdit.setText(str(parameters['beta_2']))
            self.epsilonLineEdit.setText(str(parameters['epsilon']))

        else:
            self.default_settings()


    def show_test(self):
        self.show()
        #self.combobox_setup()
        #self.current_settings()
