# Copyright 2018 The trfl Authors. All Rights Reserved.
#
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
# The file has been modified by The Deep RL Zoo Authors
# to support PyTorch opeartion.
#
# ============================================================================
"""Common ops for discrete-action Policy Gradient functions."""

# Dependency imports
from typing import NamedTuple, Optional
import torch
from torch.distributions import Categorical
from deep_rl_zoo import base


class EntropyExtra(NamedTuple):
    entropy: Optional[torch.Tensor]


def baseline_loss(delta: torch.Tensor) -> base.LossOutput:
    """Calculates the baseline loss.

    Args:
      delta: the difference between predicted baseline value and estimated target value, shape [B,] or [T, B].

    Returns:

      Returns:
        A namedtuple with fields:
        * `loss`: Baseline 'loss', shape `[B]`.
    """
    loss = torch.square(delta)  # 0.5 * torch.square(delta)

    if len(loss.shape) == 2:
        # Average over time dimension.
        loss = torch.mean(loss, dim=0)

    return base.LossOutput(loss, extra=None)


def entropy_loss(logits_t: torch.Tensor) -> base.LossOutput:
    """Calculates the entropy regularization loss.

    See "Function Optimization using Connectionist RL Algorithms" by Williams.
    (https://www.tandfonline.com/doi/abs/10.1080/09540099108946587)

    Args:
      logits_t: a sequence of unnormalized action preferences, shape [B, num_actions] or [T, B, num_actions].

    Returns:

      Returns:
        A namedtuple with fields:
        * `loss`: Entropy 'loss', shape `[B]`.
        * `extra`: a namedtuple with fields:
            * `entropy`: Entropy of the policy, shape `[B]`.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(logits_t, (2, 3), torch.float32)

    m = Categorical(logits=logits_t)
    entropy = m.entropy()
    loss = -entropy

    if len(loss.shape) == 2:
        # Average over time dimension.
        loss = torch.mean(loss, dim=0)
        entropy = torch.mean(entropy, dim=0)

    return base.LossOutput(loss, EntropyExtra(entropy))


def policy_gradient_loss(
    logits_t: torch.Tensor,
    a_t: torch.Tensor,
    adv_t: torch.Tensor,
) -> base.LossOutput:
    """Calculates the policy gradient loss.

    See "Simple Gradient-Following Algorithms for Connectionist RL" by Williams.
    (http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)

    Args:
      logits_t: a sequence of unnormalized action preferences, shape [B, num_actions] or [T, B, num_actions].
      a_t: a sequence of actions sampled from the preferences `logits_t`, shape [B] or [T, B].
      adv_t: the observed or estimated advantages from executing actions `a_t`, shape [B] or [T, B].

    Returns:
        A namedtuple with fields:
        * `loss`: policy gradient 'loss', shape `[B]`.

    """

    # Rank and compatibility checks.
    base.assert_rank_and_dtype(logits_t, (2, 3), torch.float32)
    base.assert_rank_and_dtype(a_t, (1, 2), torch.long)
    base.assert_rank_and_dtype(adv_t, (1, 2), torch.float32)

    base.assert_batch_dimension(a_t, logits_t.shape[0])
    base.assert_batch_dimension(adv_t, logits_t.shape[0])
    # For rank 3, check [T, B].
    if len(logits_t.shape) == 3:
        base.assert_batch_dimension(a_t, logits_t.shape[1], 1)
        base.assert_batch_dimension(adv_t, logits_t.shape[1], 1)

    m = Categorical(logits=logits_t)
    logprob_a_t = m.log_prob(a_t).view_as(adv_t)
    loss = -logprob_a_t * adv_t.detach()

    if len(loss.shape) == 2:
        # Average over time dimention.
        loss = torch.mean(loss, dim=0)

    return base.LossOutput(loss, extra=None)


def clipped_surrogate_gradient_loss(
    prob_ratios_t: torch.Tensor,
    adv_t: torch.Tensor,
    epsilon: float,
) -> base.LossOutput:
    """Computes the clipped surrogate policy gradient loss for PPO algorithms.

    L_clipₜ(θ) = - min(rₜ(θ)Âₜ, clip(rₜ(θ), 1-ε, 1+ε)Âₜ)

    Where rₜ(θ) = π_θ(aₜ| sₜ) / π_θ_old(aₜ| sₜ) and Âₜ are the advantages.

    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347

    Args:
      prob_ratios_t: Ratio of action probabilities for actions a_t:
          rₜ(θ) = π_θ(aₜ| sₜ) / π_θ_old(aₜ| sₜ), shape [B].
      adv_t: the observed or estimated advantages from executing actions a_t, shape [B].
      epsilon: Scalar value corresponding to how much to clip the objecctive.

    Returns:
      Loss whose gradient corresponds to a clipped surrogate policy gradient
          update, shape [B,].
    """
    base.assert_rank_and_dtype(prob_ratios_t, 1, torch.float32)
    base.assert_rank_and_dtype(adv_t, 1, torch.float32)

    clipped_ratios_t = torch.clamp(prob_ratios_t, 1.0 - epsilon, 1.0 + epsilon)
    clipped_objective = -torch.min(prob_ratios_t * adv_t.detach(), clipped_ratios_t * adv_t.detach())

    return base.LossOutput(clipped_objective, extra=None)
