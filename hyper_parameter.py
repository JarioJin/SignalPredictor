# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import


class HyperParameter(object):
    def __init__(self):
        # rnn parameters
        self.rnn_batch_size = 32
        self.rnn_input_steps = 30
        self.rnn_predict_steps = 5
        self.rnn_input_depth = 1
        self.rnn_predict_depth = 1
        self.rnn_hidden = 20
        self.rnn_train_epoch = 1
        self.rnn_model_dir = 'model'

        self.encoder_stability_loss = 0.0
        self.encoder_activation_loss = 1e-05
        self.decoder_stability_loss = 0.0
        self.decoder_activation_loss = 1e-05

        self.encoder_rnn_layers = 1
        self.decoder_rnn_layers = 1

        # ar parameters
        self.ar_batch_size = 128
        self.ar_input_steps = 30
        self.ar_predict_steps = 1
        self.ar_input_depth = 1
        self.ar_predict_depth = 1
        self.ar_train_epoch = 2
        self.ar_model_dir = 'model'

