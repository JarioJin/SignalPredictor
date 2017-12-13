# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import


class HyperParameter(object):
    def __init__(self):
        # rnn parameters
        self.rnn_batch_size = 32
        self.rnn_input_steps = 35
        self.rnn_predict_steps = 20
        self.rnn_hidden = 20
        self.rnn_layers = 2
        self.rnn_train_epoch = 20
        self.rnn_model_dir = 'model'

