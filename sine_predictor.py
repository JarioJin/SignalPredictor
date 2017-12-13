# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import functools
import os
import svmformat_reader
import random
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
import spirelib as lib
from spirelib.ops.conv1d import Conv1D
from spirelib.ops.linear import Linear
from spirelib.ops.batchnorm import Batchnorm
import matplotlib.pyplot as plt
from spirelib.ops import rnn_impl

model_dir = "C:/spirenn/tf_models/"


HIDDEN_SIZE = 100
NUM_LAYERS = 1

TIMESTEPS = 50
TRAINING_STEPS = 20000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 2000
SAMPLE_GAP = 0.01


def generate_data(seq, nsteps):
    X = []
    y = []

    for i in range(len(seq) - nsteps - nsteps - 1):
        X.append([seq[i: i+nsteps]]) # np.random.normal(scale=0.05, size=1)
        y.append([seq[i+nsteps: i+nsteps+nsteps]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def get_weight_variable_xavier(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.contrib.layers.xavier_initializer())
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class SineModel:
    def __init__(self, data, target, is_training):
        print('__init__')
        self.data = data
        self.target = target
        self.is_training = is_training
        self.error
        if is_training:
            self.optimize

    @lazy_property
    def prediction(self):
        print('__prediction__')
        # batch_siz = self.data.get_shape()[0]
        # state = tf.Variable(tf.random_normal([batch_siz.value, HIDDEN_SIZE],stddev=0), name="state")

        lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, activation=tf.nn.relu)
        # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
        # lstm_cell_fw = tf.contrib.rnn.ResidualWrapper(lstm_cell_fw)
        cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw] * NUM_LAYERS)

        # lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, activation=tf.nn.relu)
        # lstm_cell_bw = tf.contrib.rnn.ResidualWrapper(lstm_cell_bw)
        # cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw] * NUM_LAYERS)
        # networks = tf.contrib.layers.fully_connected(self.data, HIDDEN_SIZE, None)
        networks = tf.contrib.layers.fully_connected(self.data, HIDDEN_SIZE)
        output, _ = rnn_impl.dynamic_rnn(cell_fw, networks, dtype=tf.float32)
        # output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, networks, dtype=tf.float32)
        # output = output[:,-1,:]
        # output = tf.concat(output, 2)
        # print(output)
        # output = tf.reshape(output, [-1, TIMESTEPS * HIDDEN_SIZE * 2])
        # print(output)
        output = Linear("fc1", TIMESTEPS * HIDDEN_SIZE, TIMESTEPS, output, initialization='he', biases=False)
        print(output)
        # networks = tf.contrib.layers.fully_connected(output, TIMESTEPS, None)
        # networks = tf.squeeze(networks, axis=2)
        # print(output)
        return output

    @lazy_property
    def optimize(self):
        print('__optimize__')
        # optimize
        train_step = tf.train.AdamOptimizer().minimize(self.error)
        # train_step = tf.train.AdagradOptimizer(1e-3).minimize(self.error)
        # train_step = tf.train.RMSPropOptimizer(1e-3).minimize(self.error)
        return train_step

    @lazy_property
    def error(self):
        print('__error__')
        # calculate cross entropy and average value
        loss = tf.losses.absolute_difference(self.target, self.prediction)
        return loss

def data_batch(datX, datY, startId, bsize, index_array):
    x_ids = index_array[startId:startId + bsize]
    return datX[x_ids, :], datY[x_ids, :]

def train_model():
    # with tf.device('/cpu:0'):
    batchX_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, TIMESTEPS, 1])
    batchY_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, TIMESTEPS])
    X_placeholder = tf.placeholder(tf.float32, [1, TIMESTEPS, 1])
    Y_placeholder = tf.placeholder(tf.float32, [1, TIMESTEPS])
    with tf.name_scope("train"):
        with tf.variable_scope("sine", reuse=None):
            model = SineModel(batchX_placeholder, batchY_placeholder, True)
    with tf.name_scope("eval"):
        with tf.variable_scope("sine", reuse=True):
            val_model = SineModel(X_placeholder, Y_placeholder, False)

    for variables in tf.global_variables():
        print(variables.name)
    writer = tf.summary.FileWriter("C:/tmp/sine", tf.get_default_graph())

    test_start = TRAINING_EXAMPLES * SAMPLE_GAP
    test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
    ls = np.linspace(
             0.5, test_start, TRAINING_EXAMPLES, dtype=np.float32
         )
    train_X, train_y = generate_data(np.sin(ls)/ls, TIMESTEPS)
    ls = np.linspace(
             test_start, test_end, TESTING_EXAMPLES, dtype=np.float32
         )
    test_X, test_y = generate_data(np.sin(ls)/ls, TIMESTEPS)
    train_X = np.swapaxes(train_X, 1, 2)
    train_y = np.squeeze(train_y)
    test_X = np.swapaxes(test_X, 1, 2)
    test_y = np.squeeze(test_y)
    test_y = test_y[:,0]
    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)

    datL = train_X.shape[0]
    index_array = np.arange(datL)
    np.random.shuffle(index_array)

    saver = tf.train.Saver()
    # start training
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        tid = 0
        for i in range(TRAINING_STEPS):
            # i % datL
            batX, batY = data_batch(train_X, train_y, tid, BATCH_SIZE, index_array)

            tid = tid + BATCH_SIZE
            if (tid + BATCH_SIZE) > datL:
                tid = 0
            if i % 500 == 0:
                err = sess.run(model.error,
                               feed_dict={
                                   batchX_placeholder: batX,
                                   batchY_placeholder: batY
                               })
                print("After %d training steps, loss is %g" % (i, err))
            sess.run(model.optimize,
                     feed_dict={
                         batchX_placeholder: batX,
                         batchY_placeholder: batY
                     })
        saver.save(sess, os.path.join(model_dir, "sine_model.ckpt"))

        predicted = []
        testL = test_y.shape[0]//TIMESTEPS
        test_y = test_y[0: testL*TIMESTEPS]

        batX = test_X[0, :]
        prdY = np.expand_dims(batX, 0)


        for i in range(testL):
            prdY = sess.run(val_model.prediction,
                            feed_dict={
                                X_placeholder: prdY
                            })
            if i == 0:
                predicted = prdY
            else:
                predicted = np.concatenate((predicted, prdY), axis=1)
            prdY = np.expand_dims(prdY, 2)

        # print(predicted.shape)
        # print(test_y.shape)
        predicted = np.squeeze(predicted)
        rmse = np.sqrt(((predicted - test_y)**2).mean(axis=0))
        print("Mean square error is: {}".format(rmse))

        coord.request_stop()
        coord.join(threads)
        sess.close()


    plot_predicted, = plt.plot(predicted, label='predicted')
    plot_test, = plt.plot(test_y, label='real_sin')
    plt.legend([plot_predicted, plot_test],['predicted','real_sin'])
    plt.show()

    writer.close()


def train_once():
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        train_model()


if __name__ == '__main__':
    train_once()
    # train_once()
    # ls = np.linspace(1, 100, TRAINING_EXAMPLES, dtype=np.float32)
    # a = np.sin(ls) / ls
    # plt.plot(a, label='a')
    # plt.show()

