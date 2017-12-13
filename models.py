# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def dynamic_rnn_model(input_steps, true_steps, params):
    input_steps = tf.expand_dims(input_steps, -1)

    lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(params.rnn_hidden)
    cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw] * params.rnn_layers)

    net = tf.contrib.layers.fully_connected(input_steps, params.rnn_hidden)
    net, _ = tf.nn.dynamic_rnn(cell_fw, net, dtype=tf.float32)

    net = tf.reshape(net, [-1, params.rnn_predict_steps * params.rnn_hidden])
    net = tf.contrib.layers.fully_connected(net, params.rnn_predict_steps, None)
    return net

