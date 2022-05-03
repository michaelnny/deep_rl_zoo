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
"""Tests for base.py."""
from absl.testing import absltest
from absl.testing import parameterized
import torch
import numpy as np

from deep_rl_zoo import base


class BatchIndexingTest(parameterized.TestCase):
    @parameterized.parameters([True, False])
    def testOrdinaryValues(self, keepdims):
        """Indexing value functions by action for a minibatch of values."""
        values = torch.tensor(
            [
                [1.1, 1.2, 1.3],
                [1.4, 1.5, 1.6],
                [2.1, 2.2, 2.3],
                [2.4, 2.5, 2.6],
                [3.1, 3.2, 3.3],
                [3.4, 3.5, 3.6],
                [4.1, 4.2, 4.3],
                [4.4, 4.5, 4.6],
            ],
            dtype=torch.float32,
        )
        action_indices = torch.tensor([0, 2, 1, 0, 2, 1, 0, 2], dtype=torch.long)
        result = base.batched_index(values, action_indices, keepdims=keepdims)
        expected_result = np.array([1.1, 1.6, 2.2, 2.4, 3.3, 3.5, 4.1, 4.6])
        if keepdims:
            expected_result = np.expand_dims(expected_result, axis=-1)

        np.testing.assert_allclose(result.numpy(), expected_result)

    def testValueSequence(self):
        """Indexing value functions by action with a minibatch of sequences."""
        values = torch.tensor(
            [
                [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
                [[2.1, 2.2, 2.3], [2.4, 2.5, 2.6]],
                [[3.1, 3.2, 3.3], [3.4, 3.5, 3.6]],
                [[4.1, 4.2, 4.3], [4.4, 4.5, 4.6]],
            ],
            dtype=torch.float32,
        )
        action_indices = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 2]], dtype=torch.long)
        result = base.batched_index(values, action_indices)
        expected_result = np.array([[1.1, 1.6], [2.2, 2.4], [3.3, 3.5], [4.1, 4.6]])

        np.testing.assert_allclose(result.numpy(), expected_result)

    def testInputShapeChecks(self):
        """Input shape checks can catch some, but not all, shape problems."""
        # 1. Inputs have incorrect or incompatible ranks:
        for args in [
            dict(values=torch.tensor([[5, 5], [5, 5]]), indices=torch.tensor(1, dtype=torch.long)),
            dict(values=torch.tensor([[5, 5], [5, 5]]), indices=torch.tensor([1], dtype=torch.long)),
            dict(values=torch.tensor([[[5, 5], [5, 5]], [[5, 5], [5, 5]]]), indices=torch.tensor([1], dtype=torch.long)),
        ]:
            with self.assertRaisesRegex(ValueError, 'Error in rank and/or compatibility check'):
                base.batched_index(**args)


if __name__ == '__main__':
    absltest.main()
