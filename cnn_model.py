from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, losses
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.contrib.opt import AdamWOptimizer
from parse_arguments import *


def cnn_model(filters, units, rates, name=None):
    # Create model instance
    model = Sequential()
    filter1, filter2, filter3 = filters  # [64, 128, 256]
    # We add our first convolutional layer with 32 neurons and filter size of 3 x 3
    if filter1:
        # model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation='relu', padding='same', input_shape=input_shape))
        model.add(Conv2D(filters=filter1, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same', input_shape=input_shape, name='layer_conv1'))
        # We add our max pooling layer
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxPool1'))
    # We add a second convolutional layer
    if filter2:
        # model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
        model.add(Conv2D(filters=filter2, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same', name='layer_conv2'))
        # We add our max pooling layer
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxPool2'))
    # We add a third convolutional layer
    if filter3:
        # model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
        model.add(Conv2D(filters=filter3, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same', name='layer_conv3'))
        # We add our max pooling layer
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxPool3'))
    # We flatten the features
    model.add(Flatten())
    unit1, unit2, unit3 = units  # [512, 256, 128]
    rate1, rate2, rate3 = rates
    # A fully connected layer
    if unit1:
        # model.add(Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(l=0.001)))
        # model.add(Dense(units=512, activation='relu'))
        # model.add(Dense(units=256))
        model.add(Dense(units=unit1, activation=tf.nn.relu, name='fc1'))
    # Our dropout layer
    if rate1:
        model.add(Dropout(rate=rate1, name='dropout1'))
    if unit2:
        # model.add(Dense(units=16, activation='relu', kernel_regularizer=regularizers.l2(l=0.001)))
        # model.add(Dense(units=256, activation='relu'))
        # model.add(Dense(units=16))
        # model.add(Dense(units=8))
        model.add(Dense(units=unit2, activation=tf.nn.relu, name='fc2'))
    # Another dropout layer with more dropouts
    if rate2:
        model.add(Dropout(rate=rate2, name='dropout2'))
    if unit3:
        model.add(Dense(units=unit3, activation=tf.nn.relu, name='fc3'))
    if rate3:
        model.add(Dropout(rate=rate3, name='dropout3'))
    # We add an output layer that uses softmax activation for the 4 classes
    model.add(Dense(units=num_classes, activation='softmax', name='output'))

    return model
