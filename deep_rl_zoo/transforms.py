# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
#
# The functions has been modified by The Deep RL Zoo Authors
# to support PyTorch operation.
#
# ==============================================================================
"""Common functions implementing custom non-linear transformations.

This is a collection of element-wise non-linear transformations that may be
used to transform losses, value estimates, or other multidimensional data.

Code adapted from DeepMind's RLax to support PyTorch.
"""

import torch
import torch.nn.functional as F

from deep_rl_zoo import base


def identity(x: torch.Tensor) -> torch.Tensor:
    """Identity transform."""
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return x


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid transform."""
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return torch.sigmoid(x)


def logit(x: torch.Tensor) -> torch.Tensor:
    """Logit transform, inverse of sigmoid."""
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return -torch.log(1.0 / x - 1.0)


def signed_logp1(x: torch.Tensor) -> torch.Tensor:
    """Signed logarithm of x + 1."""
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return torch.sign(x) * torch.log1p(torch.abs(x))


def signed_expm1(x: torch.Tensor) -> torch.Tensor:
    """Signed exponential of x - 1, inverse of signed_logp1."""
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return torch.sign(x) * torch.expm1(torch.abs(x))


def signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed hyperbolic transform, inverse of signed_parabolic."""
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def hyperbolic_sin(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic sinus transform."""
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return torch.sinh(x)


def hyperbolic_arcsin(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic arcsinus transform."""
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    return torch.arcsinh(x)


def signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed parabolic transform, inverse of signed_hyperbolic."""
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    z = torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(x))) / 2 / eps - 1 / 2 / eps
    return torch.sign(x) * (torch.square(z) - 1)


def power(x: torch.Tensor, p: float) -> torch.Tensor:
    """Power transform; `power_tx(_, 1/p)` is the inverse of `power_tx(_, p)`."""
    base.assert_dtype(x, (torch.float16, torch.float32, torch.float64))
    q = torch.sqrt(torch.tensor(p))
    return torch.sign(x) * (torch.pow(torch.abs(x) / q + 1.0, p) - 1) / q


def transform_to_2hot(scalar: torch.Tensor, min_value: float, max_value: float, num_bins: int) -> torch.Tensor:
    """Transforms a scalar tensor to a 2 hot representation."""
    scalar = torch.clamp(scalar, min_value, max_value)
    scalar_bin = (scalar - min_value) / (max_value - min_value) * (num_bins - 1)
    lower, upper = torch.floor(scalar_bin), torch.ceil(scalar_bin)
    lower_value = (lower / (num_bins - 1.0)) * (max_value - min_value) + min_value
    upper_value = (upper / (num_bins - 1.0)) * (max_value - min_value) + min_value
    p_lower = (upper_value - scalar) / (upper_value - lower_value + 1e-5)
    p_upper = 1 - p_lower
    lower_one_hot = F.one_hot(lower.long(), num_bins) * torch.unsqueeze(p_lower, -1)
    upper_one_hot = F.one_hot(upper.long(), num_bins) * torch.unsqueeze(p_upper, -1)
    return lower_one_hot + upper_one_hot


def transform_from_2hot(probs: torch.Tensor, min_value: float, max_value: float, num_bins: int) -> torch.Tensor:
    """Transforms from a categorical distribution to a scalar."""
    support_space = torch.linspace(min_value, max_value, num_bins)
    scalar = torch.sum(probs * torch.unsqueeze(support_space, 0), -1)
    return scalar
