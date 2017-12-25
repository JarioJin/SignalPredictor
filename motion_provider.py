# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import random
import os

def generate_data(seq, seq_gt, in_steps, out_steps):
    assert seq.shape[:2] == seq_gt.shape[:2]
    xs = []
    ys = []

    for d in range(seq.shape[0]):
        seq1 = seq[d]
        seq2 = seq_gt[d]
        for i in range(len(seq1) - in_steps - out_steps + 1):
            xs.append([seq1[i: i+in_steps]])
            ys.append([seq2[i+in_steps: i+in_steps+out_steps]])
    xs = np.concatenate(xs, axis=0)
    if xs.ndim == 2:
        xs = np.expand_dims(xs, axis=-1)
    ys = np.concatenate(ys, axis=0)
    if ys.ndim == 2:
        ys = np.expand_dims(ys, axis=-1)
    return xs, ys


class MotionProvider(object):
    def __init__(self, batch_size=32, sample_gap=1, shuffle=True, refresh_data=True,
                 input_steps=20, predict_steps=1):
        self._sim_time = 30
        self._sim_dt = 0.005
        self._obs_dt = 0.05
        self._sim_step = round(self._sim_time/self._sim_dt)
        self._obs_step = round(self._sim_time/self._obs_dt)
        self._acceleration_noise_std = 1
        self._observe_noise_std = 0.2
        self._n_sim_train = 100
        self._n_sim_eval = 100

        self._signal_fn = os.path.join('signal-rec', 'motion.npz')
        if os.path.isfile(self._signal_fn):
            data = np.load(self._signal_fn)
            train_x = data['tx']
            train_y = data['ty']
            eval_x = data['ex']
            eval_y = data['ey']
        else:
            refresh_data = True

        if refresh_data:
            # position, diff of p, diff of dp, sum(above)
            train_x = np.zeros([self._n_sim_train, self._obs_step, 4], np.float32)
            train_y = np.zeros([self._n_sim_train, self._obs_step], np.float32)

            for i in range(self._n_sim_train):
                states, truth, times = self._sim_once()
                sv = np.diff(states) / self._obs_dt
                sv = np.concatenate((sv[:1], sv), axis=0)
                sa = np.diff(sv) / self._obs_dt
                sa = np.concatenate((sa[:1], sa), axis=0)
                es = states + sv*self._obs_dt + sa*self._obs_dt**2
                train_x[i] = np.stack((states, sv, sa, es), axis=1)
                train_y[i] = truth

            eval_x = np.zeros([self._n_sim_eval, self._obs_step, 4], np.float32)
            eval_y = np.zeros([self._n_sim_eval, self._obs_step], np.float32)

            for i in range(self._n_sim_eval):
                states, truth, times = self._sim_once()
                sv = np.diff(states) / self._obs_dt
                sv = np.concatenate((sv[:1], sv), axis=0)
                sa = np.diff(sv) / self._obs_dt
                sa = np.concatenate((sa[:1], sa), axis=0)
                es = states + sv*self._obs_dt + sa*self._obs_dt**2
                eval_x[i] = np.stack((states, sv, sa, es), axis=1)
                eval_y[i] = truth

            np.savez(self._signal_fn, tx=train_x, ty=train_y, ex=eval_x, ey=eval_y)

        self._tx, self._ty = generate_data(train_x, train_y, input_steps, predict_steps)
        self._ex, self._ey = generate_data(eval_x, eval_y, input_steps, predict_steps)

        self._batch_size = batch_size
        self._sample_gap = sample_gap
        self._shuffle = shuffle
        self._fetch_id = 0
        self._sample_len = self._tx.shape[0]
        self._index_arr = np.arange(self._sample_len)

        if self._shuffle:
            np.random.shuffle(self._index_arr)

    def get_next_batch(self):
        x_ids = self._index_arr[self._fetch_id:self._fetch_id + self._batch_size]
        x_ = self._tx[x_ids, :]
        y_ = self._ty[x_ids, :]

        self._fetch_id += self._sample_gap
        if self._fetch_id + self._batch_size > self._sample_len:
            if self._shuffle:
                np.random.shuffle(self._index_arr)
            self._fetch_id = 0

        return x_, y_, self._fetch_id

    def evaluate_dat(self):
        return self._ex

    def evaluate(self, predicted):
        '''
        e, = plt.plot(predicted[:500, 0, 0], label='signal_e')
        g, = plt.plot(self._ey[:500, 0, 0], label='signal_gt')
        plt.legend([e, g], ['predicted', 'gt'])
        plt.show()
        '''
        # calculate root mean square error
        return np.sqrt(((predicted - self._ey) ** 2).mean())

    def _sim_once(self):
        motion_type = 2  # random.randint(1, 4)
        change_step = random.uniform(0, 5)

        sine_mag = random.uniform(0, 10)
        sine_phase = random.uniform(0, 2 * np.pi)
        step_scope = [-10, 10]
        line_scope = [-15, 15]
        step_mag = random.uniform(step_scope[0], step_scope[1])
        line_mag = random.uniform(line_scope[0], line_scope[1])

        state = np.array([np.random.normal(0, 200), np.random.normal(0, 20), 0],
                         dtype=np.float32)
        dt = self._sim_dt
        F = np.array([[1,        dt, 0.5*dt**2],
                      [0,         1,        dt],
                      [0,         0,         1]], dtype=np.float32)
        times = np.zeros([self._sim_step], dtype=np.float32)
        sim_states = np.zeros([self._sim_step, 3], dtype=np.float32)

        sim_t10 = 0
        # different types of motion
        for i in range(self._sim_step):
            times[i] = i*dt
            acceleration_noise = np.random.normal(0, self._acceleration_noise_std)

            if motion_type == 1:
                state[2] = acceleration_noise
            elif motion_type == 2:
                state[2] = sine_mag*np.sin(times[i]/2 + sine_phase) + acceleration_noise
            elif motion_type == 3:
                if change_step < sim_t10 < change_step + 5:
                    state[2] = step_mag + acceleration_noise
                else:
                    state[2] = acceleration_noise
            else:
                if change_step < sim_t10 < change_step + 5:
                    t1 = sim_t10 - change_step
                    state[2] = line_mag/5*t1 + acceleration_noise
                else:
                    state[2] = acceleration_noise
            sim_states[i] = state
            state = np.dot(F, state)
            sim_t10 += dt
            if sim_t10 >= 10:
                step_mag = random.uniform(step_scope[0], step_scope[1])
                line_mag = random.uniform(line_scope[0], line_scope[1])
                sim_t10 = 0

        dt = self._obs_dt
        obs_times = np.zeros([self._obs_step], dtype=np.float32)
        obs_states = np.zeros([self._obs_step], dtype=np.float32)
        tru_states = np.zeros([self._obs_step, 3], dtype=np.float32)
        for i in range(self._obs_step):
            obs_times[i] = i*dt
            s = round(i*(self._obs_dt/self._sim_dt))
            state = sim_states[s,:]
            tru_states[i] = np.copy(state)
            # measurement noise
            observe_noise = np.random.normal(0, self._observe_noise_std)
            state_tmp = state[0] + observe_noise
            obs_states[i] = state_tmp

        # plt.plot(obs_states[:,2], label='signal_gt')
        # plt.show()

        return obs_states, tru_states[:,0], obs_times


if __name__ == '__main__':
    mp = MotionProvider()
    dat = mp.evaluate_dat()
    # dat = dat[:,-1:,:1] + dat[:,-1:,1:2]*0.05 + dat[:,-1:,2:3]*0.05*0.05
    print(mp.evaluate(dat[:,-1:,-1:]))

    print(dat.shape)