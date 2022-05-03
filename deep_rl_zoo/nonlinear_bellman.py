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
# The file has been modified by The Deep RL Zoo Authors
# to support PyTorch opeartion.
#
# ==============================================================================
"""Transformed value functions.

Canonical value functions map states onto the expected discounted sum of rewards
that may be collected by an agent from any starting state. Value functions may
also be defined as the fixed points of certain linear recursive relations known
as Bellman equations. It is sometimes useful to consider transformed values that
are the solution to non-linear generalization of traditional Bellman equations.
In this subpackage we provide a general utility for wrapping bootstrapped return
calculations to construct regression targets for these transformed values.
We also use this to implement different learning algorithms from the literature.
"""

from typing import NamedTuple, Callable, Any
import functools

import torch

from deep_rl_zoo import base
from deep_rl_zoo import value_learning
from deep_rl_zoo import multistep
from deep_rl_zoo import transforms


class TxPair(NamedTuple):
    apply: Callable[[Any], Any]
    apply_inv: Callable[[Any], Any]


# Example transform pairs; these typically consist of a monotonically increasing
# squashing fn `apply` and its inverse `apply_inv`. Other choices are possible.

IDENTITY_PAIR = TxPair(transforms.identity, transforms.identity)
SIGNED_LOGP1_PAIR = TxPair(transforms.signed_logp1, transforms.signed_expm1)
SIGNED_HYPERBOLIC_PAIR = TxPair(transforms.signed_hyperbolic, transforms.signed_parabolic)
HYPERBOLIC_SIN_PAIR = TxPair(transforms.hyperbolic_arcsin, transforms.hyperbolic_sin)


def transform_values(build_targets, *value_argnums):
    """Decorator to convert targets to use transformed value function."""

    @functools.wraps(build_targets)
    def wrapped_build_targets(tx_pair, *args, **kwargs):
        tx_args = list(args)
        for index in value_argnums:
            tx_args[index] = tx_pair.apply_inv(tx_args[index])

        targets = build_targets(*tx_args, **kwargs)
        return tx_pair.apply(targets)

    return wrapped_build_targets


transformed_general_off_policy_returns_from_action_values = transform_values(
    multistep.general_off_policy_returns_from_action_values, 0
)


def transformed_retrace(
    q_tm1: torch.Tensor,
    q_t: torch.Tensor,
    a_tm1: torch.Tensor,
    a_t: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    pi_t: torch.Tensor,
    mu_t: torch.Tensor,
    lambda_: float,
    eps: float = 1e-8,
    tx_pair: TxPair = IDENTITY_PAIR,
) -> base.LossOutput:
    """Calculates transformed Retrace errors.

    See "Recurrent Experience Replay in Distributed Reinforcement Learning" by
    Kapturowski et al. (https://openreview.net/pdf?id=r1lyTjAqYX).

    Args:
      q_tm1: Q-values at time t-1, this is from the online Q network, shape [T, B, num_actions].
      q_t: Q-values at time t, this is often from the target Q network, shape [T, B, num_actions].
      a_tm1: action index at time t-1, the action the agent took in state s_tm1, shape [T, B].
      a_t: action index at time t, the action the agent took in state s_t, shape [T, B].
      r_t: reward at time t, for state-action pair (s_tm1, a_tm1), shape [T, B].
      discount_t: discount at time t, shape [T, B].
      pi_t: target policy probs at time t, shape [T, B, num_actions].
      mu_t: behavior policy probs at time t, shape [T, B, num_actions].
      lambda_: scalar mixing parameter lambda.
      eps: small value to add to mu_t for numerical stability.

    Returns:
      Transformed retrace td error, shape [T, B].
    """
    base.assert_rank_and_dtype(q_tm1, 3, torch.float32)
    base.assert_rank_and_dtype(q_t, 3, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 2, torch.long)
    base.assert_rank_and_dtype(a_t, 2, torch.long)
    base.assert_rank_and_dtype(r_t, 2, torch.float32)
    base.assert_rank_and_dtype(discount_t, 2, torch.float32)
    base.assert_rank_and_dtype(pi_t, 3, torch.float32)
    base.assert_rank_and_dtype(mu_t, 2, torch.float32)

    pi_a_t = base.batched_index(pi_t, a_t)
    c_t = torch.minimum(torch.tensor(1.0), pi_a_t / (mu_t + eps)) * lambda_

    with torch.no_grad():
        target_tm1 = transformed_general_off_policy_returns_from_action_values(tx_pair, q_t, a_t, r_t, discount_t, c_t, pi_t)
    q_a_tm1 = base.batched_index(q_tm1, a_tm1)
    td_error = target_tm1 - q_a_tm1
    loss = 0.5 * torch.square(td_error)

    return base.LossOutput(loss, value_learning.QExtra(target=target_tm1, td_error=td_error))
