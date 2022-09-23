# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
# Copyright 2018 The trfl Authors. All Rights Reserved.
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
"""Unit tests for `policy_gradient.py`."""

# Dependency imports
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

from deep_rl_zoo import policy_gradient as policy_gradients
from deep_rl_zoo import distributions


class EntropyLossTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.logits = torch.tensor(
            [[1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [-1.0, -2.0, -3.0]], dtype=torch.float32
        )  # torch.tensor([[0, 1], [1, 2], [0, 2], [1, 1], [0, -1000], [0, 1000]], dtype=torch.float32)
        self.expected_entropy = np.array(
            [1.0986123, 0.66557276, 0.83239555], dtype=np.float32
        )  # np.array([0.58220309, 0.58220309, 0.36533386, 0.69314718, 0, 0])

    def test_entropy_loss_2d(self):
        # Large values check numerical stability through the logs
        # B=3

        entropy_op = policy_gradients.entropy_loss(self.logits)
        np.testing.assert_allclose(entropy_op.loss.numpy(), self.expected_entropy, atol=1e-4)

    def test_entropy_loss_3d(self):
        # Large values check numerical stability through the logs
        # T=5, B=3
        logits = torch.stack([self.logits] * 5)

        entropy_op = policy_gradients.entropy_loss(logits)

        np.testing.assert_allclose(entropy_op.loss.numpy(), self.expected_entropy, atol=1e-4)


class PolicyGradientLossTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        # B=3, A=3
        self.logits = torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [-1.0, -2.0, -3.0]], dtype=torch.float32)
        self.advantages = torch.tensor([0.3, 0.2, 0.1], dtype=torch.float32)
        self.actions = torch.tensor([0, 1, 2], dtype=torch.int64)
        self.expected = np.array([-0.3296, -0.4479, -0.2408], dtype=np.float32)

    def test_policy_gradient_loss_2d_batch(self):
        """Tests for a full batch."""
        # B=3
        # Test outputs.
        actual = policy_gradients.policy_gradient_loss(self.logits, self.actions, self.advantages)
        self.assertEqual(actual.loss.shape, (3,))
        np.testing.assert_allclose(self.expected, actual.loss.numpy(), atol=1e-4)

    def test_policy_gradient_loss_3d_batch(self):
        """Tests for a full batch."""
        # T=5, B=3
        logits = torch.stack([self.logits, self.logits + 1, self.logits, self.logits - 1, self.logits])
        actions = torch.stack([self.actions, self.actions, self.actions, self.actions, self.actions])
        advantages = torch.stack([self.advantages, self.advantages + 1, self.advantages, self.advantages - 1, self.advantages])
        expected = np.mean(
            np.stack([self.expected, self.expected + 1, self.expected, self.expected - 1, self.expected], axis=0), axis=0
        )
        # Test outputs.
        actual = policy_gradients.policy_gradient_loss(logits, actions, advantages)
        self.assertEqual(actual.loss.shape, (3,))
        np.testing.assert_allclose(expected, actual.loss.numpy(), atol=1e-4)


class ClippedSurrogatePGLossTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()

        # B=3
        self.logits = torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [-1.0, -2.0, -3.0]], dtype=torch.float32)
        self.old_logits = torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [-3.0, -2.0, -1.0]], dtype=torch.float32)

        self.advantages = torch.tensor([0.3, 0.2, 0.1], dtype=torch.float32)
        self.actions = torch.tensor([0, 1, 2], dtype=torch.int64)
        self.epsilon = 0.2
        self.expected = np.array([0.3000, 0.2000, 0.0135])

    def test_clipped_surrogate_pg_loss_batch(self):
        """Tests for a full batch."""
        # B=3
        prob_ratios = distributions.categorical_importance_sampling_ratios(self.logits, self.old_logits, self.actions)

        actual = policy_gradients.clipped_surrogate_gradient_loss(prob_ratios, self.advantages, self.epsilon)
        self.assertEqual(actual.loss.shape, (3,))
        np.testing.assert_allclose(actual.loss.numpy(), self.expected, atol=1e-4)


if __name__ == '__main__':
    absltest.main()
