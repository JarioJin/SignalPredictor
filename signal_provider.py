# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import random
import os

_signal_type = 9
_data_dir = 'signal-rec'
_custom_data_fn = os.path.join(_data_dir, 'tracking_data_p.npz')

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

    if seq.ndim == 3:
        for d in range(seq.shape[0]):
            seq1 = seq[d]
            for i in range(len(seq1) - in_steps - out_steps + 1):
                xs.append([seq1[i: i+in_steps]])
                ys.append([seq1[i+in_steps: i+in_steps+out_steps]])
    else:
        for i in range(len(seq) - in_steps - out_steps + 1):
            xs.append([seq[i: i+in_steps]])
            ys.append([seq[i+in_steps: i+in_steps+out_steps]])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class SignalProvider(object):
    def __init__(self, batch_size=32, custom_data=True, sample_gap=1, shuffle=True, refresh_data=False,
                 input_steps=20, predict_steps=1):
        if not os.path.exists(_data_dir):
            os.mkdir(_data_dir)
        if custom_data and _signal_type != 9:
            custom_data = False
            print('[WARNING]: signal_type != 9, USE SELF GENERATE DATA!')

        self._signal_fn = os.path.join(_data_dir, 'signal-t{}.npz'.format(_signal_type))

        if os.path.isfile(self._signal_fn):
            data = np.load(self._signal_fn)
            self._x = data['x']
            self._y = data['y']
            if self._x.shape[1] != input_steps or self._y.shape[1] != predict_steps:
                refresh_data = True
            self._evaluate_dat = data['eval']
            self.norm_mean = data['norm_mean']
            self.norm_std = data['norm_std']
        else:
            refresh_data = True

        # re-sampling the data if needed
        if refresh_data:
            if custom_data:
                data = np.load(_custom_data_fn)
                f1 = data['x']
                f2 = data['y']
            else:
                f1, f2 = generate_signal()
                f1 = np.expand_dims(np.expand_dims(f1, axis=0), axis=-1)
                f2 = np.expand_dims(np.expand_dims(f2, axis=0), axis=-1)

            print('[DATA-INFO]: Train data shape: {} (nSeqs, nSteps, nFeats)'.format(f1.shape))
            print('[DATA-INFO]: Test  data shape: {} (nSeqs, nSteps, nFeats)'.format(f2.shape))

            depth = f1.shape[2]
            self.norm_mean = np.mean(np.reshape(f1, [-1, depth]), axis=0)
            self.norm_std = np.std(np.reshape(f1, [-1, depth]), axis=0)
            f1 = (f1 - self.norm_mean) / self.norm_std
            print('[DATA-INFO]: Train data mean: {}, std: {}'.format(self.norm_mean, self.norm_std))

            self._x, self._y = generate_data(f1, input_steps, predict_steps)
            self._x = np.squeeze(self._x, axis=1)
            self._y = np.squeeze(self._y, axis=1)
            self._evaluate_dat = f2

            np.savez(self._signal_fn, x=self._x, y=self._y, eval=self._evaluate_dat,
                     norm_mean=self.norm_mean, norm_std=self.norm_std)

        self._batch_size = batch_size
        self._sample_gap = sample_gap
        self._shuffle = shuffle
        self._fetch_id = 0
        self._sample_len = self._x.shape[0]
        self._index_arr = np.arange(self._sample_len)

        self._prep_steps = 200

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

        return x_, y_, self._fetch_id

    def evaluate_length(self):
        return len(self._evaluate_dat[0]) - self._prep_steps

    def evaluate_dat(self):
        return self._evaluate_dat[:, 0:self._prep_steps]

    def evaluate(self, predicted):
        '''
        e, = plt.plot(predicted[0, :], label='signal_e')
        g, = plt.plot(self._evaluate_dat[0, self._prep_steps:], label='signal_gt')
        plt.legend([e, g], ['predicted', 'gt'])
        plt.show()
        '''
        # calculate root mean square error
        return np.sqrt(((predicted - self._evaluate_dat[:, self._prep_steps:]) ** 2).mean())

    def evaluate_dat_v2(self):
        l = self.evaluate_length()
        tx = []
        for d in range(self._evaluate_dat.shape[0]):
            tx1 = []
            for i in range(l):
                d1 = self._evaluate_dat[d, i:i + self._prep_steps]
                tx1.append(d1)
            tx.append(np.array(tx1, dtype=np.float32))
        return np.array(tx, dtype=np.float32)


if __name__ == '__main__':
    sp = SignalProvider()
    dat = sp.evaluate_dat_v2()
    predicted = dat[:,:,-1]

    pv = predicted[:,:-1]-predicted[:,1:]
    p0 = np.zeros([pv.shape[0], 1, 1])
    pv = np.concatenate((p0, pv), axis=1)
    # predicted += pv*0.05
    print(sp.evaluate(predicted))

    # f1, f2 = generate_signal()
    # plt.plot(f1, label='signal_gt')
    # plt.show()