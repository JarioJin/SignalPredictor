# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import random
import os

_signal_type = 1
_data_dir = 'signal-rec'

_sample_rate = 100
_training_times = 50
_testing_times = 10


def sweep(sp):
    k = 0.0187 * (np.exp(4 * sp / (_training_times + _testing_times)) - 1)
    w = 0.1 + k * (2 - 0.1)
    return np.sin(w * sp)


def signal_fun(sp):
    switcher = {
        0: np.sin,
        1: sweep,
    }
    f = switcher.get(_signal_type)
    return f(sp)


def generate_signal():
    ls = np.linspace(0, _training_times, _training_times * _sample_rate, dtype=np.float32)
    f1 = signal_fun(ls)
    ls = np.linspace(_training_times, _training_times + _testing_times, _testing_times * _sample_rate,
                     dtype=np.float32)
    f2 = signal_fun(ls)
    f1 = f1 + wgn(f1, 50)
    return f1, f2


def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    x_power = np.sum(x ** 2) / len(x)
    n_power = x_power / snr
    return np.random.randn(len(x)) * np.sqrt(n_power)


def generate_data(seq, in_steps, out_steps):
    xs = []
    ys = []

    for i in range(len(seq) - in_steps - out_steps + 1):
        xs.append([seq[i: i+in_steps]])
        ys.append([seq[i+in_steps: i+in_steps+out_steps]])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class SignalProvider(object):
    def __init__(self, batch_size, sample_gap=1, shuffle=True, refresh_data=False, input_steps=20, predict_steps=20):
        if not os.path.exists(_data_dir):
            os.mkdir(_data_dir)
        self._signal_fn = os.path.join(_data_dir, 'signal-t{}.npz'.format(_signal_type))

        if os.path.isfile(self._signal_fn):
            data = np.load(self._signal_fn)
            self._x = data['x']
            self._y = data['y']
            if self._x.shape[1] != input_steps or self._y.shape[1] != predict_steps:
                refresh_data = True
            self._evaluate_dat = data['eval']
        else:
            refresh_data = True

        # re-sampling the data if needed
        if refresh_data:
            f1, f2 = generate_signal()
            self._x, self._y = generate_data(f1, input_steps, predict_steps)
            self._x = np.squeeze(self._x)
            self._y = np.squeeze(self._y)
            self._evaluate_dat = f2
            np.savez(self._signal_fn, x=self._x, y=self._y, eval=self._evaluate_dat)

        self._batch_size = batch_size
        self._sample_gap = sample_gap
        self._shuffle = shuffle
        self._fetch_id = 0
        self._sample_len = self._x.shape[0]
        self._index_arr = np.arange(self._sample_len)

        if self._shuffle:
            np.random.shuffle(self._index_arr)

    def get_next_batch(self):
        x_ids = self._index_arr[self._fetch_id:self._fetch_id + self._batch_size]
        x_ = self._x[x_ids, :]
        y_ = self._y[x_ids, :]

        self._fetch_id += self._sample_gap
        if self._fetch_id + self._batch_size > self._sample_len:
            if self._shuffle:
                np.random.shuffle(self._index_arr)
            self._fetch_id = 0

        return x_, y_

    def evaluate_length(self):
        return len(self._evaluate_dat) - 100

    def evaluate_dat(self):
        return self._evaluate_dat[0:100]

    def evaluate(self, predicted):
        # calculate root mean square error
        return np.sqrt(((predicted - self._evaluate_dat[100:]) ** 2).mean(axis=0))


if __name__ == '__main__':
    f1, f2 = generate_signal()
    plt.plot(f1, label='signal_gt')
    plt.show()