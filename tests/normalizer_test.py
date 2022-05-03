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
"""Tests for normalizer.py."""
from absl.testing import absltest
import torch
import numpy as np
from deep_rl_zoo import normalizer


class NormalizerTest(absltest.TestCase):
    def test_not_in_place_normalization(self):
        norm = normalizer.Normalizer(eps=0.0, clip_range=(-np.inf, np.inf))
        data = np.random.uniform(size=(100, 32))

        for _ in range(5):
            norm.update(torch.tensor(data))
        normalized = norm(torch.tensor(data))

        # a, b not equal
        np.testing.assert_equal(np.any(np.not_equal(data, normalized.numpy())), True)

    def test_normalization_1d(self):
        norm = normalizer.Normalizer(eps=0.0, clip_range=(-np.inf, np.inf))
        data = np.random.uniform(size=(32))

        for _ in range(5):
            norm.update(torch.tensor(data))
        normalized = norm(torch.tensor(data))

        np.testing.assert_allclose(np.mean(normalized.numpy(), axis=0), np.zeros(32), atol=1e-4)
        np.testing.assert_allclose(np.std(normalized.numpy(), axis=0), np.ones(32), atol=1e-4)

        np.testing.assert_allclose(np.mean(data, axis=0), norm.mean.numpy(), atol=1e-4)
        np.testing.assert_allclose(np.std(data, axis=0), norm.std.numpy(), atol=1e-4)

    def test_normalization_2d(self):
        norm = normalizer.Normalizer(eps=0.0, clip_range=(-np.inf, np.inf))
        data = np.random.uniform(size=(100, 32))

        for _ in range(5):
            norm.update(torch.tensor(data))
        normalized = norm(torch.tensor(data))

        np.testing.assert_allclose(np.mean(normalized.numpy(), axis=0), np.zeros(32), atol=1e-4)
        np.testing.assert_allclose(np.std(normalized.numpy(), axis=0), np.ones(32), atol=1e-4)

        np.testing.assert_allclose(np.mean(data, axis=0), norm.mean.numpy(), atol=1e-4)
        np.testing.assert_allclose(np.std(data, axis=0), norm.std.numpy(), atol=1e-4)

    def test_normalization_3d(self):
        norm = normalizer.Normalizer(eps=0.0, clip_range=(-np.inf, np.inf))
        data = np.random.uniform(size=(2, 100, 32))

        for _ in range(5):
            norm.update(torch.tensor(data))
        normalized = norm(torch.tensor(data))

        np.testing.assert_allclose(np.mean(normalized.numpy().reshape(-1, 32), axis=0), np.zeros(32), atol=1e-4)
        np.testing.assert_allclose(np.std(normalized.numpy().reshape(-1, 32), axis=0), np.ones(32), atol=1e-4)

        np.testing.assert_allclose(np.mean(data.reshape(-1, 32), axis=0), norm.mean.numpy(), atol=1e-4)
        np.testing.assert_allclose(np.std(data.reshape(-1, 32), axis=0), norm.std.numpy(), atol=1e-4)


if __name__ == '__main__':
    absltest.main()
