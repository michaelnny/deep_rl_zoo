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
# to support PyTorch operation.
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


class TruncatedGeneralizedAdvantageEstimationTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()

        self.r_t = torch.tensor([0.0, 0.0, 1.0, 0.0, -0.5], dtype=torch.float32)
        self.v_t = torch.tensor([1.0, 4.0, -3.0, -2.0, -1.0], dtype=torch.float32)
        self.v_tp1 = torch.tensor([4.0, -3.0, -2.0, -1.0, -1.0], dtype=torch.float32)
        self.discount_tp1 = torch.tensor([0.99, 0.99, 0.99, 0.99, 0.99], dtype=torch.float32)

        # Different expected results for different values of lambda.
        self.expected = {}
        self.expected[1.0] = np.array([-1.45118, -4.4557, 2.5396, 0.5249, -0.49], dtype=np.float32)

        self.expected[0.7] = np.array([-0.676979, -5.248167, 2.4846, 0.6704, -0.49], dtype=np.float32)

        self.expected[0.4] = np.array([0.56731, -6.042, 2.3431, 0.815, -0.49], dtype=np.float32)

    @parameterized.named_parameters(('lambda1', 1.0), ('lambda0.7', 0.7), ('lambda0.4', 0.4))
    def test_truncated_gae(self, lambda_):
        """Tests truncated GAE for a full batch."""
        actual = multistep.truncated_generalized_advantage_estimation(
            self.r_t, self.v_t, self.v_tp1, self.discount_tp1, lambda_
        )
        np.testing.assert_allclose(self.expected[lambda_], actual, atol=1e-3)

    def test_truncated_gae_cross_episode_case1(self):
        r_t = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        v_t = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
        v_tp1 = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
        discount_tp1 = torch.tensor([0.9, 0.0, 0.9, 0.9], dtype=torch.float32)

        expected = np.array([1.3775, 0.5, 1.76225, 0.95], dtype=np.float32)
        actual = multistep.truncated_generalized_advantage_estimation(r_t, v_t, v_tp1, discount_tp1, 0.95)

        np.testing.assert_allclose(expected, actual, atol=1e-3)

    @parameterized.named_parameters(('lambda1', 1.0), ('lambda0.7', 0.7), ('lambda0.4', 0.4))
    def test_truncated_gae_cross_episode_case2(self, lambda_):
        r_t = torch.cat((self.r_t, torch.zeros(1), self.r_t), dim=0)
        v_t = torch.cat((self.v_t, torch.zeros(1), self.v_t), dim=0)
        v_tp1 = torch.cat((self.v_tp1, torch.zeros(1), self.v_tp1), dim=0)
        discount_tp1 = torch.cat((self.discount_tp1, torch.zeros(1), self.discount_tp1), dim=0)

        expected = torch.cat(
            (torch.tensor(self.expected[lambda_]), torch.zeros(1), torch.tensor(self.expected[lambda_])), dim=0
        )
        actual = multistep.truncated_generalized_advantage_estimation(r_t, v_t, v_tp1, discount_tp1, lambda_)

        np.testing.assert_allclose(expected, actual, atol=1e-3)


if __name__ == '__main__':
    absltest.main()
