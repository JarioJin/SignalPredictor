# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
from signal_provider import SignalProvider
from hyper_parameter import HyperParameter
from models import dynamic_rnn_model, seq2seq_model, position_rnn_model, position_seq2seq_model


class SignalPredictor(object):
    def __init__(self, params):
        self._batch_size = params.rnn_batch_size
        self._input_steps = params.rnn_input_steps
        self._predict_steps = params.rnn_predict_steps
        self._input_depth = params.rnn_input_depth
        self._predict_depth = params.rnn_predict_depth
        self._hidden = params.rnn_hidden
        self._train_epoch = params.rnn_train_epoch
        self._model_dir = params.rnn_model_dir
        if not os.path.exists(self._model_dir):
            os.mkdir(self._model_dir)

        self._data_provider = SignalProvider(self._batch_size,
                                             input_steps=self._input_steps, predict_steps=self._predict_steps)

        self._input_X = tf.placeholder(tf.float32, [self._batch_size, self._input_steps, self._input_depth])
        self._truth_Y = tf.placeholder(tf.float32, [self._batch_size, self._predict_steps, self._predict_depth])
        self._X = tf.placeholder(tf.float32, [None, self._input_steps, self._input_depth])

        global_step = tf.train.get_or_create_global_step()
        inc_step = tf.assign_add(global_step, 1)

        with tf.name_scope("train"):
            with tf.variable_scope("spnn", reuse=None):
                self._trained_Y = dynamic_rnn_model(self._input_X, True, params)
                # self._trained_Y, self._reg_loss = position_seq2seq_model(self._input_X, True, params)
                # self._trained_Y, self._reg_loss = seq2seq_model(self._input_X, True, params)
        with tf.name_scope("eval"):
            with tf.variable_scope("spnn", reuse=True):
                self._predict = dynamic_rnn_model(self._X, False, params)
                # self._predict, _ = position_seq2seq_model(self._X, False, params)
                # self._predict, _ = seq2seq_model(self._X, False, params)

        self._loss = self._get_loss()
        self._train_step = self._get_train_step()
        self._sess = tf.InteractiveSession()

    def _get_loss(self):
        # loss = tf.losses.absolute_difference(self._trained_Y, self._truth_Y)
        loss = tf.losses.mean_squared_error(self._trained_Y, self._truth_Y)
        return loss

    def _get_train_step(self):
        # train_step = tf.train.AdamOptimizer().minimize(self._loss + self._reg_loss)
        train_step = tf.train.AdamOptimizer().minimize(self._loss)
        return train_step

    def train(self):
        saver = tf.train.Saver()
        # start training

        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self._sess, coord=coord)

        time_s = time.clock()
        epoch = 0
        i = 0
        loss_avg = 0
        while epoch < self._train_epoch:
            x, y, nid = self._data_provider.get_next_batch()
            y = y[:,:,:self._predict_depth]
            p, loss, _ = self._sess.run([self._trained_Y, self._loss, self._train_step],
                                        feed_dict={self._input_X: x, self._truth_Y: y})
            loss_avg += loss
            i += 1

            if nid == 0:
                epoch += 1
                print("[INFO] after {} training steps, loss is {}, "
                      "time elapse {}".format(epoch, loss_avg / i, time.clock() - time_s))
                i = 0
                loss_avg = 0
                time_s = time.clock()

        saver.save(self._sess, os.path.join(self._model_dir, "spnn_si{}so{}dp{}.ckpt".format(
            self._input_steps, self._predict_steps, self._hidden
        )))
        coord.request_stop()
        coord.join(threads)

    def load_model(self):
        model_fn = os.path.join(self._model_dir, "spnn_si{}so{}dp{}.ckpt".format(
            self._input_steps, self._predict_steps, self._hidden
        ))
        saver = tf.train.Saver()
        saver.restore(self._sess, model_fn)

    def evaluate_pn(self):
        init_data = self._data_provider.evaluate_dat()
        init_len = init_data.shape[1]
        assert init_len >= self._input_steps

        predicted = init_data
        predicted_len = self._data_provider.evaluate_length()
        while len(predicted[0]) < init_len + predicted_len:
            x = predicted[:, len(predicted[0]) - self._input_steps:]
            x = (x - self._data_provider.norm_mean) / self._data_provider.norm_std
            y = self._sess.run(self._predict,
                               feed_dict={self._X: x})
            y = y * self._data_provider.norm_std[:self._predict_depth] + \
                self._data_provider.norm_mean[:self._predict_depth]
            predicted = np.concatenate((predicted, y), axis=1)

        predicted = predicted[:, init_len: init_len + predicted_len]
        return self._data_provider.evaluate(predicted)

    def evaluate_p1(self):
        init_data = self._data_provider.evaluate_dat_v2()
        init_len = init_data.shape[2]
        assert init_len >= self._input_steps

        predicted = []
        for i in range(init_data.shape[1]):
            x = init_data[:, i, init_len - self._input_steps:]
            x = (x - self._data_provider.norm_mean) / self._data_provider.norm_std
            y = self._sess.run(self._predict,
                               feed_dict={self._X: x})
            y = y[:, 0] * self._data_provider.norm_std[:self._predict_depth] + \
                self._data_provider.norm_mean[:self._predict_depth]
            predicted.append(y)

        predicted = np.stack(predicted, axis=1)
        return self._data_provider.evaluate(predicted)


def evaluate_once(param1):
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        params = HyperParameter()
        params.rnn_predict_steps = param1
        sp = SignalPredictor(params)
        sp.load_model()
        rmse = sp.evaluate_pn()
    return rmse


def train_once(param1, param2):
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        params = HyperParameter()
        params.rnn_hidden = param2
        params.rnn_input_steps = param1
        sp = SignalPredictor(params)
        sp.train()
        rmse = sp.evaluate_p1()
    return rmse


if __name__ == '__main__':
    params1 = [5]
    params2 = [20]
    res = np.zeros((len(params1), len(params2)), dtype=np.float32)
    p = 0
    for i in range(len(params1)):
        for j in range(len(params2)):
            rmse = train_once(params1[i], params2[j])
            # rmse = evaluate_once(params[i])
            res[i, j] = rmse
            print(res)
    print(res)
