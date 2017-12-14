# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.util import nest


def rnn_stability_loss(rnn_output, beta):
    """
    REGULARIZING RNNS BY STABILIZING ACTIVATIONS
    https://arxiv.org/pdf/1511.08400.pdf
    :param rnn_output: [time, batch, features]
    :return: loss value
    """
    if beta == 0.0:
        return 0.0
    # [time, batch, features] -> [time, batch]
    l2 = tf.sqrt(tf.reduce_sum(tf.square(rnn_output), axis=-1))
    #  [time, batch] -> []
    return beta * tf.reduce_mean(tf.square(l2[1:] - l2[:-1]))


def rnn_activation_loss(rnn_output, beta):
    """
    REGULARIZING RNNS BY STABILIZING ACTIVATIONS
    https://arxiv.org/pdf/1511.08400.pdf
    :param rnn_output: [time, batch, features]
    :return: loss value
    """
    if beta == 0.0:
        return 0.0
    return tf.nn.l2_loss(rnn_output) * beta


def dynamic_rnn_model(input_steps, is_train, params):
    input_steps = tf.expand_dims(input_steps, -1)

    layer_size = [params.rnn_hidden for i in range(params.encoder_rnn_layers)]
    rnn_layers = [tf.contrib.rnn.LSTMBlockCell(size) for size in layer_size]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    rnn_out, rnn_state = tf.nn.dynamic_rnn(multi_rnn_cell, input_steps, dtype=tf.float32)

    rnn_out = rnn_out[:,-1,:]
    # net = tf.reshape(net, [-1, params.rnn_input_steps * params.rnn_hidden])
    net = tf.contrib.layers.fully_connected(rnn_out, params.rnn_predict_steps, None)
    return net


def convert_state_v1(h_state, params, seed, c_state=None, dropout=1.0):
    """
    Converts RNN state tensor from cuDNN representation to TF RNNCell compatible representation.
    :param h_state: tensor [num_layers, batch_size, depth]
    :param c_state: LSTM additional state, should be same shape as h_state
    :return: TF cell representation matching RNNCell.state_size structure for compatible cell
    """

    def squeeze(seq):
        return tuple(seq) if len(seq) > 1 else seq[0]

    def wrap_dropout(structure):
        if dropout < 1.0:
            return nest.map_structure(lambda x: tf.nn.dropout(x, keep_prob=dropout, seed=seed), structure)
        else:
            return structure

    # Cases:
    # decoder_layer = encoder_layers, straight mapping
    # encoder_layers > decoder_layers: get outputs of upper encoder layers
    # encoder_layers < decoder_layers: feed encoder outputs to lower decoder layers, feed zeros to top layers
    h_layers = tf.unstack(h_state)
    if params.encoder_rnn_layers >= params.decoder_rnn_layers:
        return squeeze(wrap_dropout(h_layers[params.encoder_rnn_layers - params.decoder_rnn_layers:]))
    else:
        lower_inputs = wrap_dropout(h_layers)
        upper_inputs = [tf.zeros_like(h_layers[0]) for _ in
                        range(params.decoder_rnn_layers - params.encoder_rnn_layers)]
        return squeeze(lower_inputs + upper_inputs)


def rnn_decoder(encoder_state, previous_y, params):
    """
    :param encoder_state: shape [batch_size, encoder_rnn_depth]
    :param previous_y: Last step value, shape [batch_size]
    :return: decoder rnn output
    """
    def build_cell(idx):
        with tf.variable_scope('decoder_cell'):
            cell = tf.contrib.rnn.GRUBlockCell(params.rnn_hidden)
            return cell

    if params.decoder_rnn_layers > 1:
        cells = [build_cell(idx) for idx in range(params.decoder_rnn_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cells)
    else:
        cell = build_cell(0)

    predict_steps = params.rnn_predict_steps

    # Return raw outputs for RNN losses calculation
    return_raw_outputs = params.decoder_stability_loss > 0.0 or params.decoder_activation_loss > 0.0

    # Stop condition for decoding loop
    def cond_fn(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
        return time < predict_steps

    # FC projecting layer to get single predicted value from RNN output
    def project_output(tensor):
        return tf.layers.dense(tensor, 1, name='decoder_output_proj')

    def loop_fn(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
        """
        Main decoder loop
        :param time: Step number
        :param prev_output: Output(prediction) from previous step
        :param prev_state: RNN state tensor from previous step
        :param array_targets: Predictions, each step will append new value to this array
        :param array_outputs: Raw RNN outputs (for regularization losses)
        :return:
        """

        # Append previous predicted value to input features
        next_input = prev_output

        # Run RNN cell
        output, state = cell(next_input, prev_state)
        # Make prediction from RNN outputs
        projected_output = project_output(output)
        # Append step results to the buffer arrays
        if return_raw_outputs:
            array_outputs = array_outputs.write(time, output)
        array_targets = array_targets.write(time, projected_output)
        # Increment time and return
        return time + 1, projected_output, state, array_targets, array_outputs

    # Initial values for loop
    loop_init = [tf.constant(0, dtype=tf.int32),
                 tf.expand_dims(previous_y, -1),
                 encoder_state,
                 tf.TensorArray(dtype=tf.float32, size=predict_steps),
                 tf.TensorArray(dtype=tf.float32, size=predict_steps) if return_raw_outputs else tf.constant(0)]
    # Run the loop
    _, _, _, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn, loop_init)

    # Get final tensors from buffer arrays
    targets = targets_ta.stack()
    # [time, batch_size, 1] -> [time, batch_size]
    targets = tf.squeeze(targets, axis=-1)
    raw_outputs = outputs_ta.stack() if return_raw_outputs else None
    return targets, raw_outputs


def seq2seq_model(input_steps, is_train, params):
    input_steps = tf.expand_dims(input_steps, -1)

    layer_size = [params.rnn_hidden for i in range(params.encoder_rnn_layers)]
    rnn_layers = [tf.contrib.rnn.GRUBlockCell(size) for size in layer_size]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    encoder_output, rnn_state = tf.nn.dynamic_rnn(multi_rnn_cell, input_steps, dtype=tf.float32)
    # Encoder activation losses
    enc_stab_loss = rnn_stability_loss(encoder_output, params.encoder_stability_loss / params.rnn_input_steps)
    enc_activation_loss = rnn_activation_loss(encoder_output, params.encoder_activation_loss / params.rnn_input_steps)

    # h_state = tf.stack([state.h for state in rnn_state])
    h_state = tf.stack([state for state in rnn_state])
    encoder_state = convert_state_v1(h_state, params, None)

    # Run decoder
    decoder_targets, decoder_outputs = rnn_decoder(encoder_state, tf.squeeze(input_steps[:, -1], axis=1), params)

    # Decoder activation losses
    dec_stab_loss = rnn_stability_loss(decoder_outputs, params.decoder_stability_loss / params.rnn_predict_steps)
    dec_activation_loss = rnn_activation_loss(decoder_outputs, params.decoder_activation_loss / params.rnn_predict_steps)

    decoder_targets = tf.transpose(decoder_targets)

    reg_loss = enc_stab_loss + enc_activation_loss + dec_stab_loss + dec_activation_loss
    return decoder_targets, reg_loss





