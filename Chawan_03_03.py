# Chawan, Varsha Rani
# 1001-553-524
# 2018-10-08
# Assignment-03-03

import numpy as np
import glob
import scipy.misc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os.path

def readImages(file):

    #########################################################################
    #  Logic to read the data file , retrieve the label from filename,
    #  shuffle and split the data
    #########################################################################
    data_list = []
    outputs = []
    directory = file + "/*.png"

    for fname in glob.glob(directory):
        img = scipy.misc.imread(fname).astype(np.float32)
        img = img.flatten()
        img = img / 127.5
        img = img - 1.0
        data_list.append(img)

        # key, value = fname.split("//")
        key, value = os.path.split(fname)
        tmp = int(value[0])
        outputs.append(tmp)

    data_list = np.array(data_list)
    outputs = np.array(outputs)

    data_list, outputs = shuffle(data_list, outputs)
    X_train, X_test, Y_train, Y_test = train_test_split(data_list, outputs)
    return X_train, X_test, Y_train, Y_test


def calculate_ActivationFunction(activation_function, net_value):

    ############################################
    # Calculates the activation function
    ############################################

    if activation_function == "Symmetrical Hard limit":
        activation = net_value
        activation[activation >= 0] = 1.0
        activation[activation < 0] = -1.0
    elif activation_function == "Hyperbolic Tangent":
        activation = ((np.exp(net_value)) - (np.exp(-net_value))) / ((np.exp(net_value)) + (np.exp(-net_value)))
    elif activation_function == "Linear":
        activation = net_value
    return activation
