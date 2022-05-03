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
"""Tests for replay.py."""
from absl.testing import absltest
from absl.testing import parameterized
from typing import NamedTuple
import torch

from deep_rl_zoo import replay as replay_lib


class Transition(NamedTuple):
    s_t: torch.Tensor
    a_t: torch.Tensor
    r_t: torch.Tensor


Structure = Transition(s_t=None, a_t=None, r_t=None)


class SplitStructureTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.s_t = torch.tensor(
            [
                [[1, 1, 1], [9, 0, 9], [1, 0, 0], [0, 0, 9]],
                [[9, 9, 0], [9, 0, 0], [1, 1, 1], [9, -9, 0]],
                [[1, 1, 1], [9, 0, 9], [0, 0, 9], [1, 0, 0]],
                [[9, 9, 0], [9, 0, 0], [1, 1, 1], [9, -9, 0]],
                [[9, 9, 0], [9, 0, 0], [0, 9, 9], [9, -9, 0]],
            ],
            dtype=torch.float32,
        )
        self.a_t = torch.tensor([2, 1, 3, 0, 1], dtype=torch.int64)
        self.r_t = torch.tensor([0.5, 0.0, 0.5, 0.8, -0.1], dtype=torch.float32)

        self.transition = Transition(self.s_t, self.a_t, self.r_t)

    def test_split_structure_with_size_0(self):
        """Checks split structure."""

        prefix, suffix = replay_lib.split_structure(self.transition, Structure, 0)

        self.assertEqual(prefix, None)

        self.assertTrue(torch.equal(self.transition.s_t, suffix.s_t))
        self.assertTrue(torch.equal(self.transition.a_t, suffix.a_t))
        self.assertTrue(torch.equal(self.transition.r_t, suffix.r_t))

    @parameterized.named_parameters(('size_1', 1), ('size_3', 3), ('size_5', 5))
    def test_split_structure_with_size_n(self, split_size):
        """Checks split structure."""

        prefix, suffix = replay_lib.split_structure(self.transition, Structure, split_size)

        self.assertTrue(torch.equal(self.transition.s_t[:split_size], prefix.s_t))
        self.assertTrue(torch.equal(self.transition.a_t[:split_size], prefix.a_t))
        self.assertTrue(torch.equal(self.transition.r_t[:split_size], prefix.r_t))
        self.assertTrue(torch.equal(self.transition.s_t[split_size:], suffix.s_t))
        self.assertTrue(torch.equal(self.transition.a_t[split_size:], suffix.a_t))
        self.assertTrue(torch.equal(self.transition.r_t[split_size:], suffix.r_t))

    def test_split_structure_with_size_error(self):
        """Checks split structure."""

        with self.assertRaisesRegex(ValueError, 'Expect prefix_length to be less or equal to'):
            prefix, suffix = replay_lib.split_structure(self.transition, Structure, self.transition.s_t.shape[0] + 1)


if __name__ == '__main__':
    absltest.main()
