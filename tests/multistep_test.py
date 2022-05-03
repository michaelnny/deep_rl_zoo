# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The functions has been modified by The Deep RL Zoo Authors
# to support PyTorch opeartion.
#
# ============================================================================
"""Tests for multistep.py."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import numpy as np
from deep_rl_zoo import multistep


class NStepBellmanTargetTest(parameterized.TestCase):
    def test_n_step_bellman_target_one_step(self):
        targets = multistep.n_step_bellman_target(
            r_t=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).t(),
            done=torch.tensor([[False] * 3]).t(),
            q_t=torch.tensor([[100, 200, 300]], dtype=torch.float32).t(),
            gamma=0.9,
            n_steps=1,
        )
        np.testing.assert_allclose(targets.numpy(), np.array([[1 + 0.9 * 100, 2 + 0.9 * 200, 3 + 0.9 * 300]]).T)

    def test_n_step_bellman_target_one_step_with_done(self):
        targets = multistep.n_step_bellman_target(
            r_t=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).t(),
            done=torch.tensor([[False, True, False]]).t(),
            q_t=torch.tensor([[100, 200, 300]], dtype=torch.float32).t(),
            gamma=0.9,
            n_steps=1,
        )
        np.testing.assert_allclose(targets.numpy(), np.array([[1 + 0.9 * 100, 2, 3 + 0.9 * 300]]).T)

    def test_n_step_bellman_target_two_steps(self):
        targets = multistep.n_step_bellman_target(
            r_t=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).t(),
            done=torch.tensor([[False, False, False]]).t(),
            q_t=torch.tensor([[100, 200, 300]], dtype=torch.float32).t(),
            gamma=0.9,
            n_steps=2,
        )
        np.testing.assert_allclose(
            targets.numpy(),
            np.array(
                [
                    [
                        1 + 0.9 * 2 + 0.9**2 * 200,
                        2 + 0.9 * 3 + 0.9**2 * 300,
                        # Last target is actually 1-step.
                        3 + 0.9 * 300,
                    ]
                ]
            ).T,
        )

    def test_n_step_bellman_target_three_steps_done(self):
        targets = multistep.n_step_bellman_target(
            r_t=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], dtype=torch.float32).t(),
            done=torch.tensor([[False, False, False, True, False, False, False]]).t(),
            q_t=torch.tensor([[100, 200, 300, 400, 500, 600, 700]], dtype=torch.float32).t(),
            gamma=0.9,
            n_steps=3,
        )
        np.testing.assert_allclose(
            targets.numpy(),
            np.array(
                [
                    [
                        1 + 0.9 * 2 + 0.9**2 * 3 + 0.9**3 * 300,
                        2 + 0.9 * 3 + 0.9**2 * 4,
                        3 + 0.9 * 4,
                        4,
                        5 + 0.9 * 6 + 0.9**2 * 7 + 0.9**3 * 700,
                        # Actually 2-step.
                        6 + 0.9 * 7 + 0.9**2 * 700,
                        # Actually 1-step.
                        7 + 0.9 * 700,
                    ]
                ]
            ).T,
        )


if __name__ == '__main__':
    absltest.main()
