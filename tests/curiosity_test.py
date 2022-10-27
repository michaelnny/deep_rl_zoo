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
"""Tests for curiosity.py."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import numpy as np
from deep_rl_zoo import curiosity


class KNNQueryTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.data = np.array([7.5, 1.0])
        self.memory = np.array([[2.0, 1.3], [7.5, 0.0], [40.0, 40.0]])

    def test_small_k_query(self):
        num_neighbors = 2
        expected_neighbors = np.array([[7.5, 0.0], [2.0, 1.3]])
        expected_distances = np.array([1.0, 30.34])
        expected_neighbor_indices = np.array([1, 0])

        def query_variant(data, points):
            return curiosity.knn_query(data, points, num_neighbors)

        actual = query_variant(torch.tensor(self.data, dtype=torch.float32), torch.tensor(self.memory, dtype=torch.float32))

        np.testing.assert_allclose(actual.neighbors.numpy(), expected_neighbors, atol=1e-6)
        np.testing.assert_allclose(actual.neighbor_indices.numpy(), expected_neighbor_indices, atol=1e-6)
        np.testing.assert_allclose(actual.neighbor_distances.numpy(), expected_distances, atol=1e-6)


if __name__ == '__main__':
    absltest.main()
