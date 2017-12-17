# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import signal_provider
from signal_provider import SignalProvider
from hyper_parameter import HyperParameter


def get_weight_variable(input_dim, output_dim, regularizer):
    shape = (input_dim, output_dim)
    # weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    weights = tf.get_variable("weights", shape, initializer=tf.contrib.layers.xavier_initializer())
    if regularizer is not None:
        tf.add_to_collection('reg_loss', regularizer(weights))
    return weights


class ARSignalPredictor(object):
    def __init__(self, params):
        self._batch_size = params.ar_batch_size
        self._input_steps = params.ar_input_steps
        self._predict_steps = params.ar_predict_steps
        self._input_depth = params.ar_input_depth
        self._predict_depth = params.ar_predict_depth

        self._train_epoch = params.ar_train_epoch
        self._model_dir = params.ar_model_dir
        if not os.path.exists(self._model_dir):
            os.mkdir(self._model_dir)

        self._data_provider = SignalProvider(self._batch_size,
                                             input_steps=self._input_steps, predict_steps=self._predict_steps)

        self._input_X = tf.placeholder(tf.float32, [self._batch_size, self._input_steps, self._input_depth])
        self._truth_Y = tf.placeholder(tf.float32, [self._batch_size, self._predict_steps, self._predict_depth])
        self._X = tf.placeholder(tf.float32, [None, self._input_steps, self._input_depth])

        with tf.name_scope("train"):
            with tf.variable_scope("ar", reuse=None):
                self._trained_Y, self._loss, self._train_step = self._graph(self._input_X, self._truth_Y, True)
        with tf.name_scope("eval"):
            with tf.variable_scope("ar", reuse=True):
                self._predict = self._graph(self._X, None, False)

        self._sess = tf.InteractiveSession()

    def _graph(self, inputX, truthY, is_train):
        regularizer = tf.contrib.layers.l2_regularizer(0.0001)
        input_dim = self._input_steps * self._input_depth
        output_dim = self._predict_steps * self._predict_depth
        weights = get_weight_variable(input_dim, output_dim, regularizer)

        targets = tf.reshape(inputX, [-1, input_dim])
        biases = tf.get_variable("biases", [self._predict_steps], initializer=tf.constant_initializer(0.0))

        targets = tf.matmul(targets, weights) + biases
        targets = tf.reshape(targets, [-1, self._predict_steps, self._predict_depth])

        if is_train:
            loss = tf.losses.mean_squared_error(targets, truthY)
            # loss = tf.losses.absolute_difference(targets, truthY)
            loss = loss + tf.add_n(tf.get_collection('reg_loss'))
            train_step = tf.train.AdamOptimizer().minimize(loss)
            # train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)
            return targets, loss, train_step
        else:
            return targets

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

        saver.save(self._sess, os.path.join(self._model_dir, "ar_input{}.ckpt".format(
            self._input_steps
        )))
        coord.request_stop()
        coord.join(threads)

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
            y = y*self._data_provider.norm_std + self._data_provider.norm_mean
            predicted = np.concatenate((predicted, y), axis=1)

        predicted = predicted[:, init_len: init_len + predicted_len]
        return self._data_provider.evaluate(predicted)

    def evaluate_p1(self):
        init_data = self._data_provider.evaluate_dat_v2()
        init_len = init_data.shape[2]
        assert init_len >= self._input_steps

        predicted = []
        for i in range(init_data.shape[1]):
            x = init_data[:, i, init_len-self._input_steps:]
            x = (x - self._data_provider.norm_mean) / self._data_provider.norm_std
            y = self._sess.run(self._predict,
                               feed_dict={self._X: x})
            y = y[:, 0]*self._data_provider.norm_std + self._data_provider.norm_mean
            predicted.append(y)

        predicted = np.stack(predicted, axis=1)  # np.array(predicted, dtype=np.float32)
        return self._data_provider.evaluate(predicted)


class ARSignalPredictorTF(object):
    def __init__(self, params):
        self._batch_size = params.ar_batch_size
        self._input_steps = params.ar_input_steps
        self._predict_steps = params.ar_predict_steps
        self._s1, self._s2 = signal_provider.generate_signal()

        self.prep_s = self._s2[:100]
        self.eval_s = self._s2[100:]

        x = np.array(range(len(self._s1)), dtype=np.float32)
        data = {
            tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: self._s1,
        }

        reader = tf.contrib.timeseries.NumpyReader(data)

        self.train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
            reader, batch_size=self._batch_size, window_size=self._input_steps+self._predict_steps)

        self.ar = tf.contrib.timeseries.ARRegressor(
            periodicities=628, input_window_size=self._input_steps, output_window_size=self._predict_steps,
            num_features=1,
            loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)

    def train(self):
        self.ar.train(input_fn=self.train_input_fn, steps=6000)

    def evaluate(self):
        # predicted = self.prep_s
        # while len(predicted) < len(self._s2):
        x = self._s1 # predicted[len(predicted) - self._input_steps:]
        t = np.array(range(len(x)), dtype=np.float32)
        data = {
            tf.contrib.timeseries.TrainEvalFeatures.TIMES: t,
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: x,
        }
        reader = tf.contrib.timeseries.NumpyReader(data)
        input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
        y = self.ar.evaluate(input_fn=input_fn, steps=1)

        (predictions,) = tuple(self.ar.predict(
            input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
                y, steps=1000)))

        plt.plot(data['times'].reshape(-1), data['values'].reshape(-1), label='origin')
        plt.plot(y['times'].reshape(-1), y['mean'].reshape(-1), label='evaluation')
        plt.plot(predictions['times'].reshape(-1), predictions['mean'].reshape(-1), label='prediction')
        plt.xlabel('time_step')
        plt.ylabel('values')
        plt.legend(loc=4)
        plt.show()

        return np.sqrt(((np.squeeze(predictions['mean']) - self._s2) ** 2).mean(axis=0))


def train_once(param1):
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        params = HyperParameter()
        params.ar_input_steps = param1
        sp = ARSignalPredictor(params)
        sp.train()
        rmse = sp.evaluate_pn()
    return rmse


if __name__ == '__main__':
    params = [10, 20]
    res = np.zeros((len(params), 1), dtype=np.float32)
    p = 0
    for i in range(len(params)):
        rmse = train_once(params[i])
        # rmse = evaluate_once(params[i])
        res[i] = rmse
        print(res)
    print(res)


