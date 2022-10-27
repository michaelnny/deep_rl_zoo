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
"""Components for bandit algorithm."""

import numpy as np


class SimplifiedSlidingWindowUCB:
    """A simplified Sliding window UCB algorithm for non-starionary MABP.

    Used in Agent57.


    From paper "Agent57: Outperforming the Atari Human Benchmark"
    https://arxiv.org/pdf/2003.13350

    From original paper "On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems":
    https://arxiv.org/abs/0805.3415

    """

    def __init__(
        self,
        num_arms: int,
        window_size: int,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        beta: float = 1.0,
        epsilon: float = 0.5,
    ) -> None:

        self.num_arms = num_arms
        self.window_size = window_size
        self._random_state = random_state
        self._beta = beta
        self._epsilon = epsilon

        self._rewards = np.zeros((window_size, num_arms), dtype=np.float32)
        self._count = np.zeros((window_size, num_arms), dtype=np.int32)

        self.t = 0

    def update(self, current_arm: int, reward: float) -> None:
        """Update statistics."""
        index = self.t % self.window_size
        self._count[index, current_arm] = 1
        self._rewards[index, current_arm] = reward
        self.t += 1

    def sample(self) -> int:
        """Sample an arm to play."""
        # Make sure all arms are played at least once.
        if self.t < self.num_arms:
            a_t = self.t
        elif self._random_state.rand() <= self._epsilon:
            a_t = self._random_state.randint(0, self.num_arms)
        else:
            # Use whatever data we've got to calculate mean rewards for each arm.
            i = min(self.t, self.window_size)
            rewards_sum = np.sum(self._rewards[:i], axis=0)
            count = np.sum(self._count[:i], axis=0)

            # Expected reward. Add some constant to avoid divide by zero.
            mean = rewards_sum / (count + 1e-8)
            c = self._beta * np.sqrt(1 / count)
            ucb_result = mean + c

            a_t = np.argmax(ucb_result)
        return int(a_t)
