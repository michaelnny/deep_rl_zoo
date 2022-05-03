# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import time
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# algorithm
gamma = 0.99
v_min, v_max = -10, 10
n_atoms = 51
support = np.linspace(v_min, v_max, n_atoms)  # support, a vector

# spacing between atoms in the support, use it to find the closest support element index by equation: bj = (r - vmin)/dz
delta_z = (v_max - v_min) / (n_atoms - 1.0)


def generate_random_probs():
    '''
    generate a dummy probability distribution for size=n_atoms
    '''
    t = np.random.randint(low=10, high=20, size=n_atoms).astype(np.float32)
    t /= t.sum()
    return t


def categorical_l2_project(z_p, probs, z_q):
    """
    Code copied from deepmind RLax
    https://github.com/deepmind/rlax/blob/master/rlax/_src/value_learning.py


    Projects a categorical distribution (z_p, p) onto a different support z_q.
    The projection step minimizes an L2-metric over the cumulative distribution
    functions (CDFs) of the source and target distributions.
    Let kq be len(z_q) and kp be len(z_p). This projection works for any
    support z_q, in particular kq need not be equal to kp.
    See "A Distributional Perspective on RL" by Bellemare et al.
    (https://arxiv.org/abs/1707.06887).
    Args:
      z_p: support of distribution p.
      probs: probability values.
      z_q: support to project distribution (z_p, probs) onto.
    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """

    kp = z_p.shape[0]
    kq = z_q.shape[0]

    # Construct helper arrays from z_q.
    d_pos = np.roll(z_q, shift=-1)  # move first element in support to the last in new array
    d_neg = np.roll(z_q, shift=1)  # move last element in support to the first in new array

    # Clip z_p to be in new support range (vmin, vmax).
    z_p = np.clip(z_p, z_q[0], z_q[-1])[None, :]
    assert z_p.shape == (1, kp)

    # Get the distance between atom values in support.
    d_pos = (d_pos - z_q)[:, None]  # z_q[i+1] - z_q[i]
    d_neg = (z_q - d_neg)[:, None]  # z_q[i] - z_q[i-1]
    z_q = z_q[:, None]
    assert z_q.shape == (kq, 1)

    # Ensure that we do not divide by zero, in case of atoms of identical value.
    d_neg = np.where(d_neg > 0, 1.0 / d_neg, np.zeros_like(d_neg))
    d_pos = np.where(d_pos > 0, 1.0 / d_pos, np.zeros_like(d_pos))

    delta_qp = z_p - z_q  # clip(z_p)[j] - z_q[i]
    d_sign = (delta_qp >= 0.0).astype(probs.dtype)
    assert delta_qp.shape == (kq, kp)
    assert d_sign.shape == (kq, kp)

    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
    delta_hat = (d_sign * delta_qp * d_pos) - ((1.0 - d_sign) * delta_qp * d_neg)
    probs = probs[None, :]
    assert delta_hat.shape == (kq, kp)
    assert probs.shape == (1, kp)

    return np.sum(np.clip(1.0 - delta_hat, 0.0, 1.0) * probs, axis=-1)


def project_distribution(r, probs, support):
    '''
    the categorical projection implementation according to paper 4.2
    note not the algorithm 1 block, but the equation 7

    Args:
        r       (float):    the reward
        probs   (array):    Z for next state, tipically came from DQN model
        support (array):    the support we wish to project the probs on
    '''
    assert len(probs.shape) == 1 and probs.shape[0] == n_atoms

    m = np.zeros_like(probs).astype(np.float64)
    n = m.shape[0]
    for i in range(0, n):
        for j in range(0, n):
            Tzj = r + gamma * support[j]
            Tzj = np.clip(Tzj, v_min, v_max)
            delta_p = 1.0 - np.abs(Tzj - support[i]) / delta_z
            delta_p = np.clip(delta_p, 0.0, 1.0)
            m[i] += delta_p * probs[j]

    return m


def project_distribution_vec(r, probs, support):
    '''
    vectorized implementation of the equation 7 in the distributional RL paper
    Args:
        r       (float):    the reward
        probs   (array):    Z for next state, tipically came from DQN model
        support (array):    the support we wish to project the probs on
    '''

    support = support[:, None]  # [n_atoms, 1]
    target_z = r + gamma * support  # [n_atoms, 1]
    target_z = np.clip(target_z, v_min, v_max)
    target_z = np.reshape(target_z, (1, n_atoms))  # [1, n_atoms]

    numerator = np.abs(target_z - support)
    quotient = 1.0 - (numerator / delta_z)
    quotient = np.clip(quotient, 0.0, 1.0)

    probs = probs[None, :]  # [1, n_atoms]
    probs = np.clip(probs, v_min, v_max)

    m = quotient * probs  # [n_atoms, n_atoms]
    m = np.sum(m, axis=-1)

    return m


def visualize_projection_1(probs, num_steps, reward, support, anim_interval=100):
    probs_list = []  # store the projected distributions for plotting
    probs_list.append(probs)

    # calculate the projections for n steps
    for i in range(num_steps):
        Zt = project_distribution(reward, probs_list[-1], support)
        probs_list.append(Zt)
        Z = Zt

    # visualize the projections with some animation
    fig = plt.figure(figsize=(20, 10))
    bar_w = 0.35

    def init():
        pass

    def animate(i):
        fig.clear()
        plt.bar(support, probs_list[0], color='gray', edgecolor='lightskyblue', width=bar_w, label='t=0', alpha=0.4)

        if i > 0:
            t = i
            if i > 5:
                tm5 = t - 5
                plt.bar(
                    support, probs_list[tm5], color='r', edgecolor='lightskyblue', width=bar_w, label=f't={tm5+1}', alpha=0.4
                )
            # we want the last step Z to be on top of all existing charts
            plt.bar(support, probs_list[t], color='g', edgecolor='lightskyblue', width=bar_w, label=f't={t+1}', alpha=0.3)

        plt.xticks(support, rotation=45)
        plt.draw()
        plt.legend(loc='upper left')
        plt.ylabel('Probabilities')
        plt.xlabel('Value ranges')
        plt.title('Probability projection animations')

    ani = FuncAnimation(
        fig, func=animate, init_func=init, frames=num_steps, interval=anim_interval, repeat_delay=0, repeat=False
    )
    plt.show()


def visualize_projection_3(probs, num_steps, reward, support, anim_interval=100):
    probs_list = []  # store the projected distributions for plotting
    probs_list.append(probs)

    # calculate the projections for n steps
    for i in range(1, num_steps + 1):
        print(f'Calculating {i}/{num_steps} step projection')
        target_z = reward + gamma * support
        Zt = categorical_l2_project(target_z, probs_list[-1], support)

        # Zt = project_distribution(reward, probs_list[-1], support)
        Zt2 = project_distribution_vec(reward, probs_list[-1], support)

        set_diff = np.where(Zt != Zt2)

        a1 = Zt[set_diff]
        a2 = Zt2[set_diff]

        assert np.array_equal(np.round(Zt, 6), np.round(Zt2, 6))  # may have small precision difference
        probs_list.append(Zt)

    print('Plotting projections')
    # visualize the projections with some animation
    fig = plt.figure(figsize=(20, 8))
    bar_w = 0.35
    orig_bars = plt.bar(support, probs_list[0], color='gray', edgecolor='lightskyblue', width=bar_w, label='t=0', alpha=0.4)
    tm5_bars = plt.bar(support, probs_list[0], color='r', edgecolor='lightskyblue', width=bar_w, alpha=0.4)
    t_bars = plt.bar(support, probs_list[0], color='g', edgecolor='lightskyblue', width=bar_w, alpha=0.3)

    # plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1)
    plt.ylabel('Probabilities')
    plt.xlabel('Value ranges')
    plt.xticks(support, rotation=45)
    plt.title(f'Probability projection animations with reward={reward}')

    def init():
        pass

    def animate(i):
        if i > 5:
            y_tm5 = probs_list[i - 5]
            for rect, y in zip(tm5_bars, y_tm5):
                rect.set_visible(True)
                rect.set_height(y)
            t_bars.set_label(f't={i+1 -5}')
        else:
            for rect in tm5_bars:
                rect.set_visible(False)

        y_t = probs_list[i]
        for rect, y in zip(t_bars, y_t):
            rect.set_height(y)
        plt.ylim(0, max(y_t) + 0.001)  # dynamic y axis, add small space to top
        t_bars.set_label(f't={i+1}')
        # tm5_bars.set_label(f"t={i+1 - 5}")
        tm5_bars.set_label(f't-5')

        plt.legend(loc='best')

        fig.gca().autoscale_view()
        fig.gca().relim()
        return tm5_bars, probs_list

    ani = FuncAnimation(
        fig, func=animate, init_func=init, frames=num_steps, interval=anim_interval, repeat_delay=0, repeat=False
    )
    plt.show()


probs = generate_random_probs()
# demo and animation related
num_steps = 200
reward = 0
anim_interval = 20

if __name__ == '__main__':
    visualize_projection_3(probs, num_steps, reward, support, anim_interval)
