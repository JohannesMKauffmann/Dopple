import numpy as np
from td_utils import *

import tensorflow as tf
import keras
import os
import datetime

from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import SGD
from keras.optimizers import Adam

# Use 1101 for 2sec input audio
Tx = 5511  # The number of time steps input to the model from the spectrogram
n_freq = 101  # Number of frequencies input to the model at each time step of the spectrogram


# Use 272 for 2sec input audio
Ty = 1375# The number of time steps in the output of our model


def generateModel(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape=input_shape)

    ### START CODE HERE ###

    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)  # CONV1D
    X = BatchNormalization()(X)  # Batch normalization
    X = Activation('relu')(X)  # ReLu activation
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization

    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)

    ### END CODE HERE ###

    model = Model(inputs=X_input, outputs=X)

    return model


def loadDataSet(setDirectory):
    X = np.load('ExportDataSets/' + setDirectory + '/0/X.npy')
    Y = np.load('ExportDataSets/' + setDirectory + '/0/Y.npy')

    for i in range(len(os.listdir('ExportDataSets/' + setDirectory)) - 1):
        x = np.load('ExportDataSets/' + setDirectory + '/' + str(i + 1) + '/X.npy')
        y = np.load('ExportDataSets/' + setDirectory + '/' + str(i + 1) + '/Y.npy')

        X = np.concatenate((X, x), axis=0)
        Y = np.concatenate((Y, y), axis=0)

    print(X.shape)
    print(Y.shape)
    return X, Y

def trainModel(X, Y, bath_size, epochs, optimizer, modelName):
    print("Start: " + modelName)
    print(datetime.datetime.now())
    model = generateModel(input_shape=(Tx, n_freq))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    model.fit(X, Y, batch_size=bath_size, epochs=epochs)
    model.save(modelName)
    print('model: ' + modelName + ' saved')
    print(datetime.datetime.now())


X, Y = loadDataSet('4000SetDopple')
# X, Y = loadDataSet('01-09-2020-11-52-48')
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)

trainModel(X, Y, 75, 5, opt, '4000_75_5.h5')
trainModel(X, Y, 75, 10, opt, '4000_75_10.h5')
trainModel(X, Y, 75, 15, opt, '4000_75_15.h5')
trainModel(X, Y, 75, 20, opt, '4000_75_20.h5')
trainModel(X, Y, 50, 5, opt, '4000_50_5.h5')
trainModel(X, Y, 50, 10, opt, '4000_50_10.h5')
trainModel(X, Y, 50, 15, opt, '4000_50_15.h5')
trainModel(X, Y, 50, 20, opt, '4000_50_20.h5')
trainModel(X, Y, 25, 5, opt, '4000_25_5.h5')
trainModel(X, Y, 25, 10, opt, '4000_25_10.h5')
trainModel(X, Y, 25, 15, opt, '4000_25_15.h5')
trainModel(X, Y, 25, 20, opt, '4000_25_20.h5')
