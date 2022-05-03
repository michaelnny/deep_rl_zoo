# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
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
# ============================================================================
"""Tests for distributions.py."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
import numpy as np
from deep_rl_zoo import distributions


def _shaped_arange(*shape):
    """Runs np.arange, converts to float and reshapes."""
    return torch.tensor(np.arange(np.prod(shape), dtype=np.float32).reshape(*shape))


def _softmax(logits):
    """Applies softmax non-linearity on inputs."""
    return torch.exp(logits) / torch.sum(torch.exp(logits), dim=-1, keepdims=True)


class CategoricalDistributionLogProbsFromLogitsAndActionsTest(parameterized.TestCase):
    def test_log_probs_from_logits_and_actions(self):
        """Tests categorical_distribution."""
        batch_size = 2
        seq_len = 7
        num_actions = 3

        policy_logits = _shaped_arange(seq_len, batch_size, num_actions) + 10
        actions = torch.from_numpy(np.random.randint(0, num_actions - 1, size=(seq_len, batch_size))).long()

        categorical_distribution = distributions.categorical_distribution(policy_logits)
        action_log_probs_tensor = categorical_distribution.log_prob(actions)

        # Ground Truth
        # Using broadcasting to create a mask that indexes action logits
        action_index_mask = actions[..., None] == torch.arange(num_actions)

        def index_with_mask(array, mask):
            return array[mask].reshape(*array.shape[:-1])

        # Note: Normally log(softmax) is not a good idea because it's not
        # numerically stable. However, in this test we have well-behaved values.
        ground_truth_v = index_with_mask(np.log(_softmax(policy_logits)), action_index_mask)

        torch.testing.assert_allclose(ground_truth_v, action_log_probs_tensor)


class ImportanceSamplingTest(absltest.TestCase):
    def setUp(self):
        super().setUp()

        self.pi_logits = torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float32)
        self.mu_logits = torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float32)
        self.actions = torch.tensor([1, 0], dtype=torch.int64)

        pi = F.softmax(self.pi_logits, dim=1)
        mu = F.softmax(self.mu_logits, dim=1)
        self.expected_rhos = torch.tensor([pi[0][1] / mu[0][1], pi[1][0] / mu[1][0]], dtype=torch.float32)

    def test_importance_sampling_ratios_batch(self):
        """Tests for a full batch."""

        # Test softmax output in batch.
        actual = distributions.categorical_importance_sampling_ratios(self.pi_logits, self.mu_logits, self.actions)
        np.testing.assert_allclose(self.expected_rhos.numpy(), actual.numpy(), atol=1e-4)


if __name__ == '__main__':
    absltest.main()
