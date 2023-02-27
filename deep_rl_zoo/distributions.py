# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
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
# ==============================================================================
"""Functions for working with probability distributions."""

import torch
from torch.distributions import Categorical, Normal

from deep_rl_zoo import base


def categorical_distribution(logits: torch.Tensor) -> torch.distributions.Categorical:
    """Returns categorical distribution that support sample(), entropy(), and log_prob()."""
    return Categorical(logits=logits)


def normal_distribution(mu: torch.Tensor, sigma: torch.Tensor) -> torch.distributions.Normal:
    """Returns normal distribution that support sample(), entropy(), and log_prob()."""
    return Normal(mu, sigma)


def categorical_importance_sampling_ratios(
    pi_logits_t: torch.Tensor, mu_logits_t: torch.Tensor, a_t: torch.Tensor
) -> torch.Tensor:
    """Compute importance sampling ratios from logits.

    Args:
      pi_logits_t: raw logits at time t for the target policy,
        shape [B, action_dim] or [T, B, action_dim].
      mu_logits_t: raw logits at time t for the behavior policy,
        shape [B, action_dim] or [T, B, action_dim].
      a_t: actions at time t, shape [B] or [T, B].

    Returns:
      importance sampling ratios, shape [B] or [T, B].
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(pi_logits_t, (2, 3), torch.float32)
    base.assert_rank_and_dtype(mu_logits_t, (2, 3), torch.float32)
    base.assert_rank_and_dtype(a_t, (1, 2), torch.long)

    pi_m = Categorical(logits=pi_logits_t)
    mu_m = Categorical(logits=mu_logits_t)

    pi_logprob_a_t = pi_m.log_prob(a_t)
    mu_logprob_a_t = mu_m.log_prob(a_t)

    return torch.exp(pi_logprob_a_t - mu_logprob_a_t)
