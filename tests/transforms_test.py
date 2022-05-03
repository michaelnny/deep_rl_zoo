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
"""Unit tests for `transforms.py`."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import torch
import numpy as np
from deep_rl_zoo import transforms as transforms


TWO_HOT_BINS = 5
TWO_HOT_SCALARS = [-5.0, -3.0, -1.0, -0.4, 0.0, 0.3, 1.0, 4.5, 10.0]
TWO_HOT_PROBABILITIES = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.8, 0.2, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.4, 0.6, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]


class TransformsTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.x = 0.5
        self.xs = torch.tensor([-2, -1, -0.5, 0, 0.5, 1, 2], dtype=torch.float32)

    def test_identity_scalar(self):
        identity = transforms.identity
        x = torch.tensor(self.x)
        # Test output.
        np.testing.assert_allclose(identity(x).numpy(), self.x)

    def test_identity_vector(self):
        identity = transforms.identity
        # Test output.
        np.testing.assert_allclose(identity(self.xs).numpy(), self.xs)

    def test_sigmoid_scalar(self):
        sigmoid = transforms.sigmoid
        logit = transforms.logit
        x = torch.tensor(self.x)
        # Test output.
        np.testing.assert_allclose(logit(sigmoid(x)).numpy(), self.x, atol=1e-3)

    def test_sigmoid_vector(self):
        sigmoid = transforms.sigmoid
        logit = transforms.logit
        # Test output.
        np.testing.assert_allclose(logit(sigmoid(self.xs)).numpy(), self.xs, atol=1e-3)

    def test_signed_log_exp_transform_scalar(self):
        signed_logp1 = transforms.signed_logp1
        signed_expm1 = transforms.signed_expm1
        x = torch.tensor(self.x)
        # Test inverse.
        np.testing.assert_allclose(signed_expm1(signed_logp1(x)).numpy(), self.x, atol=1e-3)

    def test_signed_log_exp_transform_vector(self):
        signed_logp1 = transforms.signed_logp1
        signed_expm1 = transforms.signed_expm1
        # Test inverse.
        np.testing.assert_allclose(signed_expm1(signed_logp1(self.xs)).numpy(), self.xs, atol=1e-3)

    def test_signed_hyper_parabolic_transform_scalar(self):
        signed_hyperbolic = transforms.signed_hyperbolic
        signed_parabolic = transforms.signed_parabolic
        x = torch.tensor(self.x)
        # Test inverse.
        np.testing.assert_allclose(signed_parabolic(signed_hyperbolic(x)).numpy(), self.x, atol=1e-3)

    def test_signed_hyper_parabolic_transform_vector(self):
        signed_hyperbolic = transforms.signed_hyperbolic
        signed_parabolic = transforms.signed_parabolic
        # Test inverse.
        np.testing.assert_allclose(signed_parabolic(signed_hyperbolic(self.xs)).numpy(), self.xs, atol=1e-3)

    def test_signed_power_transform_scalar(self):
        square = functools.partial(transforms.power, p=2.0)
        sqrt = functools.partial(transforms.power, p=1 / 2.0)
        x = torch.tensor(self.x)
        # Test inverse.
        np.testing.assert_allclose(square(sqrt(x)).numpy(), self.x, atol=1e-3)

    def test_signed_power_transform_vector(self):
        square = functools.partial(transforms.power, p=2.0)
        sqrt = functools.partial(transforms.power, p=1 / 2.0)
        # Test inverse.
        np.testing.assert_allclose(square(sqrt(self.xs)).numpy(), self.xs, atol=1e-3)

    def test_hyperbolic_sin_transform_scalar(self):
        sinh = transforms.hyperbolic_sin
        arcsinh = transforms.hyperbolic_arcsin
        x = torch.tensor(self.x)
        # Test inverse.
        np.testing.assert_allclose(sinh(arcsinh(x)).numpy(), self.x, atol=1e-3)
        np.testing.assert_allclose(arcsinh(sinh(x)).numpy(), self.x, atol=1e-3)

    def test_hyperbolic_sin_transform_vector(self):
        sinh = transforms.hyperbolic_sin
        arcsinh = transforms.hyperbolic_arcsin
        # Test inverse.
        np.testing.assert_allclose(sinh(arcsinh(self.xs)).numpy(), self.xs, atol=1e-3)
        np.testing.assert_allclose(arcsinh(sinh(self.xs)).numpy(), self.xs, atol=1e-3)

    def test_transform_to_2hot(self):
        y = transforms.transform_to_2hot(
            scalar=torch.tensor(TWO_HOT_SCALARS), min_value=-1.0, max_value=1.0, num_bins=TWO_HOT_BINS
        )

        np.testing.assert_allclose(y.numpy(), np.array(TWO_HOT_PROBABILITIES), atol=1e-4)

    def test_transform_from_2hot(self):
        y = transforms.transform_from_2hot(
            probs=torch.tensor(TWO_HOT_PROBABILITIES), min_value=-1.0, max_value=1.0, num_bins=TWO_HOT_BINS
        )

        np.testing.assert_allclose(y.numpy(), np.clip(np.array(TWO_HOT_SCALARS), -1, 1), atol=1e-4)

    def test_2hot_roundtrip(self):
        min_value = -1.0
        max_value = 1.0
        num_bins = 11

        value = torch.arange(min_value, max_value, 0.01)

        transformed = transforms.transform_to_2hot(value, min_value, max_value, num_bins)
        restored = transforms.transform_from_2hot(transformed, min_value, max_value, num_bins)

        np.testing.assert_almost_equal(value.numpy(), restored.numpy(), decimal=5)


if __name__ == '__main__':
    absltest.main()
