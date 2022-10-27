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
# to support PyTorch operation.
#
# ==============================================================================
"""Common ops for multistep return evaluation."""

import torch
import numpy as np

from deep_rl_zoo import base


def n_step_bellman_target(
    r_t: torch.Tensor,
    done: torch.Tensor,
    q_t: torch.Tensor,
    gamma: float,
    n_steps: int,
) -> torch.Tensor:
    r"""Computes n-step Bellman targets.

    See section 2.3 of R2D2 paper (which does not mention the logic around end of
    episode).

    Args:
      rewards: This is r_t in the equations below. Should be non-discounted, non-summed,
        shape [T, B] tensor.
      done: This is done_t in the equations below. done_t should be true
        if the episode is done just after
        experimenting reward r_t, shape [T, B] tensor.
      q_t: This is Q_target(s_{t+1}, a*) (where a* is an action chosen by the caller),
        shape [T, B] tensor.
      gamma: Exponential RL discounting.
      n_steps: The number of steps to look ahead for computing the Bellman targets.

    Returns:
      y_t targets as <float32>[time, batch_size] tensor.
      When n_steps=1, this is just:

      $$r_t + gamma * (1 - done_t) * Q_{target}(s_{t+1}, a^*)$$

      In the general case, this is:

      $$(\sum_{i=0}^{n-1} \gamma ^ {i} * notdone_{t, i-1} * r_{t + i}) +
        \gamma ^ n * notdone_{t, n-1} * Q_{target}(s_{t + n}, a^*) $$

      where notdone_{t,i} is defined as:

      $$notdone_{t,i} = \prod_{k=0}^{k=i}(1 - done_{t+k})$$

      The last n_step-1 targets cannot be computed with n_step returns, since we
      run out of Q_{target}(s_{t+n}). Instead, they will use n_steps-1, .., 1 step
      returns. For those last targets, the last Q_{target}(s_{t}, a^*) is re-used
      multiple times.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(r_t, 2, torch.float32)
    base.assert_rank_and_dtype(done, 2, torch.bool)
    base.assert_rank_and_dtype(q_t, 2, torch.float32)

    base.assert_batch_dimension(done, q_t.shape[0])
    base.assert_batch_dimension(r_t, q_t.shape[0])
    base.assert_batch_dimension(done, q_t.shape[1], 1)
    base.assert_batch_dimension(r_t, q_t.shape[1], 1)

    # We append n_steps - 1 times the last q_target. They are divided by gamma **
    # k to correct for the fact that they are at a 'fake' indices, and will
    # therefore end up being multiplied back by gamma ** k in the loop below.
    # We prepend 0s that will be discarded at the first iteration below.
    bellman_target = torch.concat(
        [torch.zeros_like(q_t[0:1]), q_t] + [q_t[-1:] / gamma**k for k in range(1, n_steps)], dim=0
    )
    # Pad with n_steps 0s. They will be used to compute the last n_steps-1
    # targets (having 0 values is important).
    done = torch.concat([done] + [torch.zeros_like(done[0:1])] * n_steps, dim=0)
    rewards = torch.concat([r_t] + [torch.zeros_like(r_t[0:1])] * n_steps, dim=0)
    # Iteratively build the n_steps targets. After the i-th iteration (1-based),
    # bellman_target is effectively the i-step returns.
    for _ in range(n_steps):
        rewards = rewards[:-1]
        done = done[:-1]
        bellman_target = rewards + gamma * (1.0 - done.float()) * bellman_target[1:]

    return bellman_target


def truncated_generalized_advantage_estimation(
    r_t: torch.Tensor,
    value_t: torch.Tensor,
    value_tp1: torch.Tensor,
    discount_tp1: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Computes truncated generalized advantage estimates for a sequence length k.

    The advantages are computed in a backwards fashion according to the equation:
    Âₜ = δₜ + (γλ) * δₜ₊₁ + ... + ... + (γλ)ᵏ⁻ᵗ⁺¹ * δₖ₋₁
    where δₜ = rₜ + γₜ * v(sₜ₊₁) - v(sₜ).

    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347

    Args:
      r_t: Sequence of rewards at times [0, k]
      value_t: Sequence of values under π at times [0, k]
      value_tp1: Sequence of values under π at times [1, k+1]
      discount_tp1: Sequence of discounts at times [1, k+1]
      lambda_: a scalar

    Returns:
      Multistep truncated generalized advantage estimation at times [0, k-1].
    """

    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(value_t, 1, torch.float32)
    base.assert_rank_and_dtype(value_tp1, 1, torch.float32)
    base.assert_rank_and_dtype(discount_tp1, 1, torch.float32)

    lambda_ = torch.ones_like(discount_tp1) * lambda_  # If scalar, make into vector.

    delta_t = r_t + discount_tp1 * value_tp1 - value_t

    advantage_t = torch.zeros_like(delta_t, dtype=torch.float32)

    gae_t = 0
    for i in reversed(range(len(delta_t))):
        gae_t = delta_t[i] + discount_tp1[i] * lambda_[i] * gae_t
        advantage_t[i] = gae_t

    return advantage_t


def general_off_policy_returns_from_action_values(
    q_t: torch.Tensor,
    a_t: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    c_t: torch.Tensor,
    pi_t: torch.Tensor,
) -> torch.Tensor:
    """Calculates targets for various off-policy correction algorithms.

    Given a window of experience of length `K`, generated by a behavior policy μ,
    for each time-step `t` we can estimate the return `G_t` from that step
    onwards, under some target policy π, using the rewards in the trajectory, the
    actions selected by μ and the action-values under π, according to equation:

      Gₜ = rₜ₊₁ + γₜ₊₁ * (E[q(aₜ₊₁)] - cₜ * q(aₜ₊₁) + cₜ * Gₜ₊₁),

    where, depending on the choice of `c_t`, the algorithm implements:
      Importance Sampling             c_t = π(x_t, a_t) / μ(x_t, a_t),
      Harutyunyan's et al. Q(lambda)  c_t = λ,
      Precup's et al. Tree-Backup     c_t = π(x_t, a_t),
      Munos' et al. Retrace           c_t = λ min(1, π(x_t, a_t) / μ(x_t, a_t)).

    See "Safe and Efficient Off-Policy Reinforcement Learning" by Munos et al.
    (https://arxiv.org/abs/1606.02647).

    Args:
      q_t: Q-values at times [1, ..., K - 1], shape [T, B, num_actions].
      a_t: action index at times [1, ..., K - 1], shape [T, B].
      r_t: reward at times [1, ..., K - 1], shape [T, B].
      discount_t: discount at times [1, ..., K - 1], shape [T, B].
      c_t: importance weights at times [1, ..., K - 1], shape [T, B].
      pi_t: target policy probs at times [1, ..., K - 1], shape [T, B, num_actions].

    Returns:
      Off-policy estimates of the generalized returns from states visited at times
      [0, ..., K - 1], shape [T, B].
    """

    base.assert_rank_and_dtype(q_t, 3, torch.float32)
    base.assert_rank_and_dtype(a_t, 2, torch.long)
    base.assert_rank_and_dtype(r_t, 2, torch.float32)
    base.assert_rank_and_dtype(discount_t, 2, torch.float32)
    base.assert_rank_and_dtype(c_t, 2, torch.float32)
    base.assert_rank_and_dtype(pi_t, 3, torch.float32)

    for i in (0, 1):
        base.assert_batch_dimension(a_t, q_t.shape[i], i)
        base.assert_batch_dimension(r_t, q_t.shape[i], i)
        base.assert_batch_dimension(discount_t, q_t.shape[i], i)
        base.assert_batch_dimension(c_t, q_t.shape[i], i)
        base.assert_batch_dimension(pi_t, q_t.shape[i], i)

    # Get the expected values and the values of actually selected actions.
    exp_q_t = (pi_t * q_t).sum(axis=-1)
    # The generalized returns are independent of Q-values and cs at the final
    # state.
    q_a_t = base.batched_index(q_t, a_t)[:-1, ...]
    c_t = c_t[:-1, ...]

    return general_off_policy_returns_from_q_and_v(q_a_t, exp_q_t, r_t, discount_t, c_t)


def general_off_policy_returns_from_q_and_v(
    q_t: torch.Tensor,
    v_t: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    c_t: torch.Tensor,
) -> torch.Tensor:
    """Calculates targets for various off-policy evaluation algorithms.
    Given a window of experience of length `K+1`, generated by a behavior policy
    μ, for each time-step `t` we can estimate the return `G_t` from that step
    onwards, under some target policy π, using the rewards in the trajectory, the
    values under π of states and actions selected by μ, according to equation:
      Gₜ = rₜ₊₁ + γₜ₊₁ * (vₜ₊₁ - cₜ₊₁ * q(aₜ₊₁) + cₜ₊₁* Gₜ₊₁),
    where, depending on the choice of `c_t`, the algorithm implements:
      Importance Sampling             c_t = π(x_t, a_t) / μ(x_t, a_t),
      Harutyunyan's et al. Q(lambda)  c_t = λ,
      Precup's et al. Tree-Backup     c_t = π(x_t, a_t),
      Munos' et al. Retrace           c_t = λ min(1, π(x_t, a_t) / μ(x_t, a_t)).
    See "Safe and Efficient Off-Policy Reinforcement Learning" by Munos et al.
    (https://arxiv.org/abs/1606.02647).
    Args:
      q_t: Q-values under π of actions executed by μ at times [1, ..., K - 1].
      v_t: Values under π at times [1, ..., K].
      r_t: rewards at times [1, ..., K].
      discount_t: discounts at times [1, ..., K].
      c_t: weights at times [1, ..., K - 1].
      stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.
    Returns:
      Off-policy estimates of the generalized returns from states visited at times
      [0, ..., K - 1].
    """

    base.assert_rank_and_dtype(q_t, 2, torch.float32)
    base.assert_rank_and_dtype(v_t, 2, torch.float32)
    base.assert_rank_and_dtype(r_t, 2, torch.float32)
    base.assert_rank_and_dtype(discount_t, 2, torch.float32)
    base.assert_rank_and_dtype(c_t, 2, torch.float32)

    for i in (0, 1):
        base.assert_batch_dimension(v_t, r_t.shape[i], i)
        base.assert_batch_dimension(discount_t, r_t.shape[i], i)

    # Work backwards to compute `G_K-1`, ..., `G_1`, `G_0`.
    g = r_t[-1] + discount_t[-1] * v_t[-1]  # G_K-1.
    returns = [g]
    for i in reversed(range(q_t.shape[0])):  # [K - 2, ..., 0]
        g = r_t[i] + discount_t[i] * (v_t[i] - c_t[i] * q_t[i] + c_t[i] * g)
        returns.insert(0, g)

    return torch.stack(returns, dim=0).detach()
