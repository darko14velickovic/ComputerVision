
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import cv2
import numpy as np
import tensorflow as tf
import glob, os
from tflearn.datasets import cifar10


class CnnTrainer:

    def createNetwork(self, input_size):

        # Real-time data preprocessing
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()

        # Real-time data augmentation
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=25.)

        # Convolutional network building
        # network = input_data(shape=[None, input_size, input_size, 3],
        # data_preprocessing=img_prep,
        # data_augmentation=img_aug)

        network = input_data(shape=[None, input_size, input_size, 3])

        network = conv_2d(network, input_size, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, input_size * 2, 3, activation='relu')
        network = conv_2d(network, input_size * 2, 3, activation='relu')
        # network = max_pool_2d(network, 2)
        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 3, activation='softmax')
        network = regression(network, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        return network

    def __init__(self, input_size, name, loadOnInit = False):

        self.version = '1.0'

        self.network = self.createNetwork(input_size)

        # Train using classifier
        self.model = tflearn.DNN(self.network, tensorboard_verbose=3, tensorboard_dir='model/tflearn_logs/')

        # If flag set try to load, if it doesn't load shit hits the fan... look into this.
        if(loadOnInit):
            try:
                self.model.load(name + '.tfl')

                self.modelLoaded = True
                self.modelTrained = True
            except:
                print("Failed to load model with name: " + name + ". Learning needed.")

                self.modelTrained = False
                self.modelLoaded = False
        else:
            self.modelTrained = False
            self.modelLoaded = False

    def evaluate_img(self, image):
        if self.modelTrained is False:
            print ("Model not trained!")
            return [0, 1, 0]
        return self.model.predict([image])

    def rename(self, dir, pattern):
        counter = 1

        # find max
        for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
            title, ext = os.path.splitext(os.path.basename(pathAndFilename))
            os.rename(pathAndFilename,
                      os.path.join(dir, "Kappa" + str(counter) + ext))
            counter += 1

        counter = 1
        for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
            os.rename(pathAndFilename,
                      os.path.join(dir, str(counter) + ext))
            counter += 1

    def loadBottlePics(self):
        #
        """
        Loads from folders test data and training data and returns images
        :return: First tuple of Training set and Training result set & tuple of Test set and Test results
        """

        # self.renaming(True)
        # self.renaming(False)

        # training = tuple()
        training_images = []
        training_labels = []

        # testing = tuple()
        testing_images = []
        testing_labels = []

        # add loading from the folders
        folders = ['s1', 's2', 's3']
        label_counter = 0

        for folder in folders:
            for i in range(1, 51):
                image = cv2.imread('training data/' + folder + '/' + str(i) + '.png', 1)
                new = np.divide(image, 1000.)
                training_images.append(new)
                training_labels.append(label_counter)

            for i in range(1, 6):
                image = cv2.imread('test data/' + folder + '/' + str(i) + '.png', 1)
                new = np.divide(image, 1000.)
                testing_images.append(new)
                testing_labels.append(label_counter)

            label_counter += 1

        # training.__add__(training_images)
        # training.__add__(training_labels)

        # testing.__add__(testing_images)
        # testing.__add__(testing_labels)

        return (np.array(training_images), np.array(training_labels)), (
        np.array(testing_images), np.array(testing_labels))

    # parse data from folder test


    def renaming(self, trainingFlag):
        if trainingFlag:
            self.rename(r'training data/s1/', r'*.png')
            self.rename(r'training data/s2/', r'*.png')
            self.rename(r'training data/s3/', r'*.png')
        else:
            self.rename(r'test data/s1/', r'*.png')
            self.rename(r'test data/s2/', r'*.png')
            self.rename(r'test data/s3/', r'*.png')

    def tf_learn(self, modelName):
        if(self.modelTrained):
            print("Model already trained!")
            return

        # (X, Y), (X_test, Y_test) = cifar10.load_data()
        (X, Y), (X_test, Y_test) = self.loadBottlePics()

        X, Y = shuffle(X, Y)
        Y = to_categorical(Y, 3)
        Y_test = to_categorical(Y_test, 3)



        self.model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
                  show_metric=True, batch_size=96, run_id=modelName)

        self.model.save("model/"+modelName+".tfl")

        self.modelTrained = True
        return None

    def testing(self):

        map = {0: "Bottle without cap", 1: "Random thing", 2: "Bottle with cap"}

        (X, Y), (X_test, Y_test) = self.loadBottlePics()
        # Real-time data preprocessing
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()

        # Real-time data augmentation
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=25.)

        # Convolutional network building
        network = input_data(shape=[None, 32, 32, 3],
                             data_preprocessing=img_prep,
                             data_augmentation=img_aug)
        network = conv_2d(network, 32, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 64, 3, activation='relu')
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 3, activation='softmax')
        network = regression(network, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)

        # Train using classifier
        model = tflearn.DNN(network, tensorboard_verbose=3)

        model.load('init_model.tflearn')

        correct_predictions = 0
        wrong_predictions = 0
        totalTests = X_test.__len__()

        # Run the model on all examples
        for i in range(0, totalTests):
            prediction = model.predict([X_test[i]])
            # print("Prediction: %s" % str(prediction[0]))

            maxIndex = np.argmax(prediction)

            if maxIndex == Y_test[i]:
                correct_predictions += 1
            else:
                wrong_predictions += 1
                print("Wrong prediction, true label is " + str(Y_test[i]) + " Predicted label is: " + str(maxIndex))
                forShow = np.multiply(X_test[i], 100)
                forShow = np.asarray(forShow, int)

                cv2.imwrite("WrongPrediction" + str(maxIndex) + ".png", forShow)

        correct_percent = correct_predictions / float(totalTests)
        wrong_percent = wrong_predictions / float(totalTests)

        print("Correct predictions: " + str(correct_predictions))
        print("Wrong predictions: " + str(wrong_predictions))
        print("Total: " + str(totalTests))

        print("Correct percent: " + str(correct_percent))
        print("Wrong percent: " + str(wrong_percent))

        # Evaluate model
        # score = model.evaluate([X_test], Y_test)
        # print('Test accuarcy: %0.4f%%' % (score[0] * 100))

    def load_cnn(self, file_name):
        self.model.load("model/" +file_name+".tfl")