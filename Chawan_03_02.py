# Chawan, Varsha Rani
# 1001-553-524
# 2018-10-08
# Assignment-03-02

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import Chawan_03_03

minWB = -0.001
maxWB = 0.001


class HebbModel(object):

    #########################################################################
    #  Logic to create random weights, train and test model and
    #  confusion matrix
    #########################################################################

    def __init__(self, file, activation="Symmetrical Hard limit", learning_method="Filtered Learning",learning_rate=0.1):
        self.epochs = []
        self.total_epochs = 0
        self.error = []
        self.iteration = 0
        self.X_train, self.X_test, self.Y_train, self.Y_test = Chawan_03_03.readImages(file)
        self.weights = np.random.uniform(minWB, maxWB,
                                         (np.unique(self.Y_train).size, self.X_train.shape[1]))  # 10X784
        self.bias = np.random.uniform(minWB, maxWB, np.unique(self.Y_train).size)
        self.activation = activation
        self.lb = preprocessing.LabelBinarizer()
        self.lb.fit(self.Y_train)
        self.learning_method = learning_method
        self.learning_rate = learning_rate

    def train(self, activation="Symmetrical Hard limit", learning_method="Filtered Learning", learning_rate=0.1):
        #########################################################################
        #  Trains the model for various Hebbian rules, plotting the graph for
        #   every 100 epochs and expands
        #########################################################################
        self.learning_rate = learning_rate
        self.activation = activation
        self.learning_method = learning_method
        start = self.iteration
        self.total_epochs += 1
        end = 0 + self.total_epochs * 100

        for epoc in range(start, end):
            for j in range(0, self.X_train.shape[0]):
                label_vector = self.lb.transform(np.asmatrix(self.Y_train[j])).flatten()
                label_vector = label_vector.astype(float)

                if self.learning_method == "Delta Rule" or self.learning_method == "Unsupervised Heb":
                    net_values = self.weights.dot(np.transpose(self.X_train[j, :]))
                    net_values += self.bias
                    maximum = np.max(net_values)
                    net_values = net_values / maximum
                    activation_vector = Chawan_03_03.calculate_ActivationFunction(self.activation, net_values)

                    if self.learning_method == "Delta Rule":
                        idx = np.argmax(activation_vector)
                        output_vector = np.zeros(activation_vector.shape)
                        output_vector[idx] = 1
                        error_vector = label_vector - output_vector

                    elif self.learning_method == "Unsupervised Heb":
                        output_vector = activation_vector

                input_vector = np.asmatrix(self.X_train[j, :])
                if self.learning_method == "Delta Rule":
                    self.bias += self.learning_rate * error_vector.reshape((10,))
                    error_vector = np.transpose(np.asmatrix(error_vector))
                    self.weights += self.learning_rate * (np.dot(error_vector, input_vector))

                elif self.learning_method == "Filtered Learning":
                    self.bias = (1 - self.learning_rate) * self.bias + self.learning_rate * label_vector
                    self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * np.dot(label_vector[0],
                                                                                                    self.X_train[j, :])
                elif self.learning_method == "Unsupervised Heb":
                    self.bias += self.learning_rate * output_vector
                    output_vector = np.transpose(np.asmatrix(output_vector))
                    self.weights += self.learning_rate * (np.dot(output_vector, input_vector))

            self.evaluateForTestData(epoc,self.weights)
        self.iteration = end

    def evaluateForTestData(self, epoch,weights):

        #########################################################################
        #  Tests the model for 200 test data and calculates the error rate
        #########################################################################

        trueValue = 0
        for j in range(0, self.X_test.shape[0]):
            net_values = np.dot(weights, np.transpose(self.X_test[j, :]))  # (10 x 784) * (784 x 1)
            net_values += self.bias
            maximum = np.max(net_values)
            net_values = net_values / maximum

            activation_vector = Chawan_03_03.calculate_ActivationFunction(self.activation, net_values)
            idx = np.argmax(activation_vector)
            output_vector = np.zeros(activation_vector.shape)
            output_vector[idx] = 1

            label_vector = self.lb.transform(np.asmatrix(self.Y_test[j])).flatten()

            if (label_vector.astype(int) == output_vector.astype(int)).all():
                trueValue += 1

        error = ((self.X_test.shape[0] - trueValue )/ self.X_test.shape[0])*100

        self.epochs.append(epoch)
        self.error.append(error)

    def conf_matrix(self):
        #########################################################################
        #  Calculates confusion matrix for test data
        #########################################################################
        targets = []
        actual = []
        self.error = []
        for j in range(self.X_test.shape[0]):  # self.X_test.shape[0]):
            net_values = np.dot(self.weights, np.transpose(self.X_test[j, :]))
            net_values += self.bias
            maximum = np.max(net_values)
            net_values = net_values / maximum
            activation_vector = Chawan_03_03.calculate_ActivationFunction(self.activation, net_values)  # ()
            idx = np.argmax(activation_vector)
            label_vector = self.Y_test[j]
            targets.append(label_vector)
            actual.append(idx)
        cfg = confusion_matrix(targets, actual)
        self.total_epochs = 0
        self.iteration = 0
        self.epochs = []
        self.error = []
        return cfg