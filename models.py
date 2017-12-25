# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from functools import partial
from tensorflow.python.util import nest


CudaRNN = tf.contrib.cudnn_rnn.CudnnLSTM


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


def cuda_params_size(cuda_model_builder):
    """
    Calculates static parameter size for CUDA RNN
    :param cuda_model_builder:
    :return:
    """
    with tf.Graph().as_default():
        cuda_model = cuda_model_builder()
        params_size_t = cuda_model.params_size()
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=config) as sess:
            result = sess.run(params_size_t)
            return result


def make_encoder(time_inputs, encoder_features_depth, is_train, params, transpose_output=True):
    """
    Builds encoder, using CUDA RNN
    :param time_inputs: Input tensor, shape [batch, time, features]
    :param encoder_features_depth: Static size for features dimension
    :param is_train:
    :param params:
    :param transpose_output: Transform RNN output to batch-first shape
    :return:
    """

    def build_rnn():
        return CudaRNN(num_layers=params.encoder_rnn_layers, num_units=params.rnn_hidden,
                       input_size=encoder_features_depth,
                       direction='unidirectional',
                       dropout=params.encoder_dropout if is_train else 0)

    static_p_size = cuda_params_size(build_rnn)
    cuda_model = build_rnn()
    params_size_t = cuda_model.params_size()
    with tf.control_dependencies([tf.assert_equal(params_size_t, [static_p_size])]):
        cuda_params = tf.get_variable("cuda_rnn_params",
                                      initializer=tf.random_uniform([static_p_size], minval=-0.05, maxval=0.05,
                                                                    dtype=tf.float32)
                                      )

    def build_init_state():
        batch_len = tf.shape(time_inputs)[0]
        return tf.zeros([params.encoder_rnn_layers, batch_len, params.rnn_hidden], dtype=tf.float32)

    input_h = build_init_state()

    # [batch, time, features] -> [time, batch, features]
    time_first = tf.transpose(time_inputs, [1, 0, 2])
    rnn_time_input = time_first
    model = partial(cuda_model, input_data=rnn_time_input, input_h=input_h, params=cuda_params)
    if CudaRNN == tf.contrib.cudnn_rnn.CudnnLSTM:
        rnn_out, rnn_state, c_state = model(input_c=build_init_state())
    else:
        rnn_out, rnn_state = model()
        c_state = None
    if transpose_output:
        rnn_out = tf.transpose(rnn_out, [1, 0, 2])
    return rnn_out, rnn_state, c_state


def dynamic_rnn_model(input_steps, is_train, params):
    # input_steps = tf.expand_dims(input_steps, -1)

    layer_size = [params.rnn_hidden for i in range(params.encoder_rnn_layers)]
    rnn_layers = [tf.contrib.rnn.LSTMBlockCell(size) for size in layer_size]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    rnn_out, rnn_state = tf.nn.dynamic_rnn(multi_rnn_cell, input_steps, dtype=tf.float32)

    rnn_out = rnn_out[:,-1:,:]
    # net = tf.reshape(net, [-1, params.rnn_input_steps * params.rnn_hidden])
    net = tf.contrib.layers.fully_connected(rnn_out, params.rnn_predict_steps, None)
    return net


def position_rnn_model(input_steps, is_train, params):
    input_vel = input_steps[:,1:] - input_steps[:,:-1]
    paddings = tf.constant([[0,0], [1,0], [0,0]])
    input_vel = tf.pad(input_vel, paddings, "CONSTANT")

    inputs = tf.concat((input_steps, input_vel), axis=2)

    layer_size = [params.rnn_hidden for i in range(params.encoder_rnn_layers)]
    rnn_layers = [tf.contrib.rnn.LSTMBlockCell(size) for size in layer_size]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    rnn_out, rnn_state = tf.nn.dynamic_rnn(multi_rnn_cell, inputs, dtype=tf.float32)
    # Encoder activation losses
    enc_stab_loss = rnn_stability_loss(rnn_out, params.encoder_stability_loss / params.rnn_input_steps)
    enc_activation_loss = rnn_activation_loss(rnn_out, params.encoder_activation_loss / params.rnn_input_steps)

    rnn_out = rnn_out[:,-1,:]
    # net = tf.reshape(net, [-1, params.rnn_input_steps * params.rnn_hidden])
    net = tf.contrib.layers.fully_connected(rnn_out, params.rnn_predict_steps, None)
    net = tf.reshape(net, [-1, params.rnn_predict_steps, params.rnn_predict_depth])
    reg_loss = enc_stab_loss + enc_activation_loss
    return net, reg_loss


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


def rnn_decoder(encoder_state, previous_y, decoder_features_depth, params):
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
        return tf.layers.dense(tensor, decoder_features_depth, name='decoder_output_proj')

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
                 previous_y,
                 encoder_state,
                 tf.TensorArray(dtype=tf.float32, size=predict_steps),
                 tf.TensorArray(dtype=tf.float32, size=predict_steps) if return_raw_outputs else tf.constant(0)]
    # Run the loop
    _, _, _, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn, loop_init)

    # Get final tensors from buffer arrays
    targets = targets_ta.stack()
    # [time, batch_size, 1] -> [time, batch_size]
    # targets = tf.squeeze(targets, axis=-1)
    raw_outputs = outputs_ta.stack() if return_raw_outputs else None
    return targets, raw_outputs


def seq2seq_model(input_steps, is_train, params):
    # input_steps = tf.expand_dims(input_steps, -1)

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
    decoder_targets, decoder_outputs = rnn_decoder(encoder_state, input_steps[:, -1],
                                                   params.rnn_input_depth, params)

    # Decoder activation losses
    dec_stab_loss = rnn_stability_loss(decoder_outputs, params.decoder_stability_loss / params.rnn_predict_steps)
    dec_activation_loss = rnn_activation_loss(decoder_outputs, params.decoder_activation_loss / params.rnn_predict_steps)

    decoder_targets = tf.transpose(decoder_targets, [1, 0, 2])

    reg_loss = enc_stab_loss + enc_activation_loss + dec_stab_loss + dec_activation_loss
    return decoder_targets, reg_loss


def position_seq2seq_model(input_steps, is_train, params):
    # input_steps = tf.expand_dims(input_steps, -1)
    # inputs = tf.layers.conv1d(input_steps, params.rnn_hidden, 3, padding='same')

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
    decoder_targets, decoder_outputs = rnn_decoder(encoder_state, input_steps[:, -1], params.rnn_input_depth,  # position and velocity
                                                   params)

    # Decoder activation losses
    dec_stab_loss = rnn_stability_loss(decoder_outputs, params.decoder_stability_loss / params.rnn_predict_steps)
    dec_activation_loss = rnn_activation_loss(decoder_outputs, params.decoder_activation_loss / params.rnn_predict_steps)

    decoder_targets = tf.transpose(decoder_targets, [1, 0, 2])

    reg_loss = enc_stab_loss + enc_activation_loss + dec_stab_loss + dec_activation_loss
    decoder_targets = tf.layers.dense(decoder_targets, params.rnn_predict_depth, name='output_proj')
    return decoder_targets, reg_loss


