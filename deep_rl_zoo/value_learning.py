# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
# Copyright 2018 The trfl Authors. All Rights Reserved.
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

"""Functions for state value and action-value learning.

Value functions estimate the expected return (discounted sum of rewards) that
can be collected by an agent under a given policy of behavior. This subpackage
implements a number of functions for value learning in discrete scalar action
spaces. Actions are assumed to be represented as indices in the range `[0, A)`
where `A` is the number of distinct actions.
"""

from typing import NamedTuple, Optional
import torch
import torch.nn.functional as F

from deep_rl_zoo import base
from deep_rl_zoo import multistep


class QExtra(NamedTuple):
    target: Optional[torch.Tensor]
    td_error: Optional[torch.Tensor]


class DoubleQExtra(NamedTuple):
    target: torch.Tensor
    td_error: torch.Tensor
    best_action: torch.Tensor


class Extra(NamedTuple):
    target: Optional[torch.Tensor]


def qlearning(
    q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    q_t: torch.Tensor,
) -> base.LossOutput:
    r"""Implements the Q-learning loss.

    The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
    the target `r_t + discount_t * max q_t`.

    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/book/ebook/node65.html).

    Args:
      q_tm1: Tensor holding Q-values for first timestep in a batch of
        transitions, shape `[B x action_dim]`.
      a_tm1: Tensor holding action indices, shape `[B]`.
      r_t: Tensor holding rewards, shape `[B]`.
      discount_t: Tensor holding discount values, shape `[B]`.
      q_t: Tensor holding Q-values for second timestep in a batch of
        transitions, shape `[B x action_dim]`.

    Returns:
      A namedtuple with fields:

      * `loss`: a tensor containing the batch of losses, shape `[B]`.
      * `extra`: a namedtuple with fields:
          * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`.
          * `td_error`: batch of temporal difference errors, shape `[B]`.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(q_tm1, 2, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 1, torch.long)
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(discount_t, 1, torch.float32)
    base.assert_rank_and_dtype(q_t, 2, torch.float32)

    base.assert_batch_dimension(a_tm1, q_tm1.shape[0])
    base.assert_batch_dimension(r_t, q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, q_tm1.shape[0])
    base.assert_batch_dimension(q_t, q_tm1.shape[0])

    # Q-learning op.
    # Build target and select head to update.
    with torch.no_grad():
        target_tm1 = r_t + discount_t * torch.max(q_t, dim=1)[0]
    qa_tm1 = base.batched_index(q_tm1, a_tm1)
    # B = q_tm1.shape[0]
    # qa_tm1 = q_tm1[torch.arange(0, B), a_tm1]

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target_tm1 - qa_tm1
    loss = 0.5 * td_error**2

    return base.LossOutput(loss, QExtra(target_tm1, td_error))


def double_qlearning(
    q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    q_t_value: torch.Tensor,
    q_t_selector: torch.Tensor,
) -> base.LossOutput:
    r"""Implements the double Q-learning loss.

    The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
    the target `r_t + discount_t * q_t_value[argmax q_t_selector]`.

    See "Double Q-learning" by van Hasselt.
    (https://papers.nips.cc/paper/3964-double-q-learning.pdf).

    Args:
      q_tm1: Tensor holding Q-values for first timestep in a batch of
        transitions, shape `[B x action_dim]`.
      a_tm1: Tensor holding action indices, shape `[B]`.
      r_t: Tensor holding rewards, shape `[B]`.
      discount_t: Tensor holding discount values, shape `[B]`.
      q_t_value: Tensor of Q-values for second timestep in a batch of transitions,
        used to estimate the value of the best action, shape `[B x action_dim]`.
      q_t_selector: Tensor of Q-values for second timestep in a batch of
        transitions used to estimate the best action, shape `[B x action_dim]`.

    Returns:
      A namedtuple with fields:

      * `loss`: a tensor containing the batch of losses, shape `[B]`.
      * `extra`: a namedtuple with fields:
          * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`
          * `td_error`: batch of temporal difference errors, shape `[B]`
          * `best_action`: batch of greedy actions wrt `q_t_selector`, shape `[B]`
    """

    # Rank and compatibility checks.
    base.assert_rank_and_dtype(q_tm1, 2, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 1, torch.long)
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(discount_t, 1, torch.float32)
    base.assert_rank_and_dtype(q_t_value, 2, torch.float32)
    base.assert_rank_and_dtype(q_t_selector, 2, torch.float32)

    base.assert_batch_dimension(a_tm1, q_tm1.shape[0])
    base.assert_batch_dimension(r_t, q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, q_tm1.shape[0])
    base.assert_batch_dimension(q_t_value, q_tm1.shape[0])
    base.assert_batch_dimension(q_t_selector, q_tm1.shape[0])

    # double Q-learning op.
    # Build target and select head to update.

    best_action = torch.argmax(q_t_selector, dim=1)
    # B = q_tm1.shape[0]
    # double_q_bootstrapped = q_t_value[torch.arange(0, B), best_action]
    double_q_bootstrapped = base.batched_index(q_t_value, best_action)

    with torch.no_grad():
        target_tm1 = r_t + discount_t * double_q_bootstrapped

    # qa_tm1 = q_tm1[torch.arange(0, B), a_tm1]
    qa_tm1 = base.batched_index(q_tm1, a_tm1)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target_tm1 - qa_tm1
    loss = 0.5 * td_error**2

    return base.LossOutput(loss, DoubleQExtra(target_tm1, td_error, best_action))


def _slice_with_actions(embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Slice a Tensor.

    Take embeddings of the form [batch_size, action_dim, embed_dim]
    and actions of the form [batch_size, 1], and return the sliced embeddings
    like embeddings[:, actions, :].

    Args:
      embeddings: Tensor of embeddings to index.
      actions: int Tensor to use as index into embeddings

    Returns:
      Tensor of embeddings indexed by actions
    """

    batch_size, action_dim = embeddings.shape[:2]

    # Values are the 'values' in a sparse tensor we will be setting
    act_idx = actions[:, None]

    values = torch.reshape(torch.ones(actions.shape, dtype=torch.int8, device=actions.device), [-1])

    # Create a range for each index into the batch
    act_range = torch.arange(0, batch_size, dtype=torch.int64)[:, None].to(device=actions.device)
    # Combine this into coordinates with the action indices
    indices = torch.concat([act_range, act_idx], 1)

    # Needs transpose indices before adding to torch.sparse_coo_tensor.
    actions_mask = torch.sparse_coo_tensor(indices.t(), values, [batch_size, action_dim])
    with torch.no_grad():
        actions_mask = actions_mask.to_dense().bool()

    sliced_emb = torch.masked_select(embeddings, actions_mask[..., None])
    # Make sure shape is the same as embeddings
    sliced_emb = sliced_emb.reshape(embeddings.shape[0], -1)
    return sliced_emb


def l2_project(z_p: torch.Tensor, p: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    r"""Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.

    The supports z_p and z_q are specified as tensors of distinct atoms (given
    in ascending order).

    Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
    support z_q, in particular Kq need not be equal to Kp.

    Args:
      z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
      p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
      z_q: Tensor holding support to project onto, shape `[Kq]`.

    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """
    # Broadcasting of tensors is used extensively in the code below. To avoid
    # accidental broadcasting along unintended dimensions, tensors are defensively
    # reshaped to have equal number of dimensions (3) throughout and intended
    # shapes are indicated alongside tensor definitions. To reduce verbosity,
    # extra dimensions of size 1 are inserted by indexing with `None` instead of
    # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
    # `[k, l]' to one of shape `[k, 1, l]`).

    # Extract vmin and vmax and construct helper tensors from z_q
    vmin, vmax = z_q[0], z_q[-1]
    d_pos = torch.concat([z_q, vmin[None]], 0)[1:]  # 1 x Kq x 1
    d_neg = torch.concat([vmax[None], z_q], 0)[:-1]  # 1 x Kq x 1
    # Clip z_p to be in new support range (vmin, vmax).
    z_p = torch.clamp(z_p, min=vmin, max=vmax)[:, None, :]  # B x 1 x Kp

    # Get the distance between atom values in support.
    d_pos = (d_pos - z_q)[None, :, None]  # z_q[i+1] - z_q[i]. 1 x B x 1
    d_neg = (z_q - d_neg)[None, :, None]  # z_q[i] - z_q[i-1]. 1 x B x 1
    z_q = z_q[None, :, None]  # 1 x Kq x 1

    # Ensure that we do not divide by zero, in case of atoms of identical value.
    d_neg = torch.where(d_neg > 0, 1.0 / d_neg, torch.zeros_like(d_neg))  # 1 x Kq x 1
    d_pos = torch.where(d_pos > 0, 1.0 / d_pos, torch.zeros_like(d_pos))  # 1 x Kq x 1

    delta_qp = z_p - z_q  # clip(z_p)[j] - z_q[i]. B x Kq x Kp
    d_sign = (delta_qp >= 0.0).to(dtype=p.dtype)  # B x Kq x Kp

    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
    # Shape  B x Kq x Kp.
    delta_hat = (d_sign * delta_qp * d_pos) - ((1.0 - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]  # B x 1 x Kp.
    return torch.sum(torch.clamp(1.0 - delta_hat, min=0.0, max=1.0) * p, 2)


def categorical_dist_qlearning(
    atoms_tm1: torch.Tensor,
    logits_q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    atoms_t: torch.Tensor,
    logits_q_t: torch.Tensor,
) -> base.LossOutput:
    """Implements Distributional Q-learning as TensorFlow ops.

    The function assumes categorical value distributions parameterized by logits.

    See "A Distributional Perspective on Reinforcement Learning" by Bellemare,
    Dabney and Munos. (https://arxiv.org/abs/1707.06887).

    Args:
      atoms_tm1: 1-D tensor containing atom values for first timestep,
        shape `[num_atoms]`.
      logits_q_tm1: Tensor holding logits for first timestep in a batch of
        transitions, shape `[B, action_dim, num_atoms]`.
      a_tm1: Tensor holding action indices, shape `[B]`.
      r_t: Tensor holding rewards, shape `[B]`.
      discount_t: Tensor holding discount values, shape `[B]`.
      atoms_t: 1-D tensor containing atom values for second timestep,
        shape `[num_atoms]`.
      logits_q_t: Tensor holding logits for second timestep in a batch of
        transitions, shape `[B, action_dim, num_atoms]`.

    Returns:
      A namedtuple with fields:

      * `loss`: a tensor containing the batch of losses, shape `[B]`.
      * `extra`: a namedtuple with fields:
          * `target`: a tensor containing the values that `q_tm1` at actions
          `a_tm1` are regressed towards, shape `[B, num_atoms]`.

    Raises:
      ValueError: If the tensors do not have the correct rank or compatibility.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(logits_q_tm1, 3, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 1, torch.long)
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(discount_t, 1, torch.float32)
    base.assert_rank_and_dtype(logits_q_t, 3, torch.float32)
    base.assert_rank_and_dtype(atoms_tm1, 1, torch.float32)
    base.assert_rank_and_dtype(atoms_t, 1, torch.float32)

    base.assert_batch_dimension(a_tm1, logits_q_tm1.shape[0])
    base.assert_batch_dimension(r_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(logits_q_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(atoms_tm1, logits_q_tm1.shape[-1])
    base.assert_batch_dimension(atoms_t, logits_q_tm1.shape[-1])

    # Categorical distributional Q-learning op.
    # Scale and shift time-t distribution atoms by discount and reward.
    target_z = r_t[:, None] + discount_t[:, None] * atoms_t[None, :]

    # Convert logits to distribution, then find greedy action in state s_t.
    q_t_probs = F.softmax(logits_q_t, dim=-1)
    q_t_mean = torch.sum(q_t_probs * atoms_t, 2)
    pi_t = torch.argmax(q_t_mean, 1)

    # Compute distribution for greedy action.
    p_target_z = _slice_with_actions(q_t_probs, pi_t)

    # Project using the Cramer distance
    with torch.no_grad():
        target_tm1 = l2_project(target_z, p_target_z, atoms_tm1)

    logit_qa_tm1 = _slice_with_actions(logits_q_tm1, a_tm1)

    loss = F.cross_entropy(input=logit_qa_tm1, target=target_tm1, reduction='none')

    return base.LossOutput(loss, Extra(target_tm1))


def categorical_dist_double_qlearning(
    atoms_tm1: torch.Tensor,
    logits_q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    atoms_t: torch.Tensor,
    logits_q_t: torch.Tensor,
    q_t_selector: torch.Tensor,
) -> base.LossOutput:
    """Implements Distributional Double Q-learning as TensorFlow ops.

    The function assumes categorical value distributions parameterized by logits,
    and combines distributional RL with double Q-learning.

    See "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
    Hessel, Modayil, van Hasselt, Schaul et al.
    (https://arxiv.org/abs/1710.02298).

    Args:
      atoms_tm1: 1-D tensor containing atom values for first timestep,
        shape `[num_atoms]`.
      logits_q_tm1: Tensor holding logits for first timestep in a batch of
        transitions, shape `[B, action_dim, num_atoms]`.
      a_tm1: Tensor holding action indices, shape `[B]`.
      r_t: Tensor holding rewards, shape `[B]`.
      discount_t: Tensor holding discount values, shape `[B]`.
      atoms_t: 1-D tensor containing atom values for second timestep,
        shape `[num_atoms]`.
      logits_q_t: Tensor holding logits for second timestep in a batch of
        transitions, shape `[B, action_dim, num_atoms]`.
      q_t_selector: Tensor holding another set of Q-values for second timestep
        in a batch of transitions, shape `[B, action_dim]`.
        These values are used for estimating the best action. In Double DQN they
        come from the online network.

    Returns:
      A namedtuple with fields:

      * `loss`: Tensor containing the batch of losses, shape `[B]`.
      * `extra`: a namedtuple with fields:
          * `target`:  Tensor containing the values that `q_tm1` at actions
          `a_tm1` are regressed towards, shape `[B, num_atoms]` .

    Raises:
      ValueError: If the tensors do not have the correct rank or compatibility.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(logits_q_tm1, 3, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 1, torch.long)
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(discount_t, 1, torch.float32)
    base.assert_rank_and_dtype(logits_q_t, 3, torch.float32)
    base.assert_rank_and_dtype(q_t_selector, 2, torch.float32)
    base.assert_rank_and_dtype(atoms_tm1, 1, torch.float32)
    base.assert_rank_and_dtype(atoms_t, 1, torch.float32)

    base.assert_batch_dimension(a_tm1, logits_q_tm1.shape[0])
    base.assert_batch_dimension(r_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(logits_q_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(q_t_selector, logits_q_tm1.shape[0])
    base.assert_batch_dimension(atoms_tm1, logits_q_tm1.shape[-1])
    base.assert_batch_dimension(atoms_t, logits_q_tm1.shape[-1])

    # Categorical distributional double Q-learning op.
    # Scale and shift time-t distribution atoms by discount and reward.
    target_z = r_t[:, None] + discount_t[:, None] * atoms_t[None, :]

    # Convert logits to distribution, then find greedy policy action in
    # state s_t.
    q_t_probs = F.softmax(logits_q_t, dim=-1)
    pi_t = torch.argmax(q_t_selector, dim=1)
    # Compute distribution for greedy action.
    p_target_z = _slice_with_actions(q_t_probs, pi_t)

    # Project using the Cramer distance
    with torch.no_grad():
        target_tm1 = l2_project(target_z, p_target_z, atoms_tm1)

    logit_qa_tm1 = _slice_with_actions(logits_q_tm1, a_tm1)

    loss = F.cross_entropy(input=logit_qa_tm1, target=target_tm1, reduction='none')

    return base.LossOutput(loss, Extra(target_tm1))


def huber_loss(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Returns huber-loss."""
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def _quantile_regression_loss(
    dist_src: torch.Tensor,
    tau_src: torch.Tensor,
    dist_target: torch.Tensor,
    huber_param: float = 0.0,
) -> torch.Tensor:
    """Compute (Huber) QR loss between two discrete quantile-valued distributions.

    See "Distributional Reinforcement Learning with Quantile Regression" by
    Dabney et al. (https://arxiv.org/abs/1710.10044).

    Args:
      dist_src: source probability distribution, shape `[B, num_taus]`.
      tau_src: source distribution probability thresholds, shape `[B, num_taus]`.
      dist_target: target probability distribution, shape `[B, num_taus]`.
      huber_param: Huber loss parameter, defaults to 0 (no Huber loss).

    Returns:
      Quantile regression loss.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(dist_src, 2, torch.float32)
    base.assert_rank_and_dtype(tau_src, 2, torch.float32)
    base.assert_rank_and_dtype(dist_target, 2, torch.float32)

    base.assert_batch_dimension(tau_src, dist_src.shape[0])
    base.assert_batch_dimension(dist_target, dist_src.shape[0])

    # Calculate quantile error.
    delta = dist_target.unsqueeze(1) - dist_src.unsqueeze(-1)

    delta_neg = (delta < 0.0).float().detach()
    weight = torch.abs(tau_src.unsqueeze(-1) - delta_neg)

    # Calculate Huber loss.
    if huber_param > 0.0:
        loss = huber_loss(delta, huber_param)
    else:
        loss = torch.abs(delta)
    loss *= weight

    # Averaging over target-samples dimension, sum over src-samples dimension.
    return torch.sum(torch.mean(loss, dim=-1), dim=1)


def quantile_q_learning(
    dist_q_tm1: torch.Tensor,
    tau_q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    dist_q_t: torch.Tensor,
    huber_param: float = 0.0,
) -> base.LossOutput:
    """Implements Q-learning for quantile-valued Q distributions.

    See "Distributional Reinforcement Learning with Quantile Regression" by
    Dabney et al. (https://arxiv.org/abs/1710.10044).

    Args:
      dist_q_tm1: Tensor holding Q distribution at time t-1, shape `[B, num_taus, action_dim]`.
      tau_q_tm1: Q distribution probability thresholds, , shape `[B, num_taus]`.
      a_tm1: Tensor holding action indices, shape `[B]`.
      r_t: Tensor holding rewards, shape `[B]`.
      discount_t: Tensor holding discount values, shape `[B]`.
      dist_q_t: Tensor holding target Q distribution at time t, shape `[B, num_taus, action_dim]`.
      huber_param: Huber loss parameter, defaults to 0 (no Huber loss).

    Returns:
      A namedtuple with fields:

      * `loss`: Tensor containing the batch of losses, shape `[B]`.
      * `extra`: a namedtuple with fields:
          * `target`:  Tensor containing the values that `q_tm1` at actions
          `a_tm1` are regressed towards, shape `[B, num_taus]` .

    """

    # Rank and compatibility checks.
    base.assert_rank_and_dtype(dist_q_tm1, 3, torch.float32)
    base.assert_rank_and_dtype(tau_q_tm1, 2, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 1, torch.long)
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(discount_t, 1, torch.float32)
    base.assert_rank_and_dtype(dist_q_t, 3, torch.float32)

    base.assert_batch_dimension(a_tm1, dist_q_tm1.shape[0])
    base.assert_batch_dimension(r_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(dist_q_t, dist_q_tm1.shape[0])

    # Quantile Regression q learning op.
    # Only update the taken actions.
    dist_qa_tm1 = base.batched_index(dist_q_tm1, a_tm1, 2)  # [batch_size, num_taus]
    # dist_qa_tm1 = dist_q_tm1[torch.arange(0, B), :, a_tm1] # [batch_size, num_taus]

    # Select target action according to greedy policy w.r.t. q_t_selector.
    q_t_selector = torch.mean(dist_q_t, dim=1)  # q_t_values
    a_t = torch.argmax(q_t_selector, dim=1)
    # dist_qa_t = dist_q_t[torch.arange(0, B), :, a_t]  # [batch_size, num_taus]
    dist_qa_t = base.batched_index(dist_q_t, a_t, 2)  # [batch_size, num_taus]

    # Compute target, do not backpropagate into it.
    with torch.no_grad():
        dist_target_tm1 = r_t[:, None] + discount_t[:, None] * dist_qa_t  # [batch_size, num_taus]

    loss = _quantile_regression_loss(dist_qa_tm1, tau_q_tm1, dist_target_tm1, huber_param)
    return base.LossOutput(loss, Extra(dist_target_tm1))


def quantile_double_q_learning(
    dist_q_tm1: torch.Tensor,
    tau_q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    dist_q_t: torch.Tensor,
    q_t_selector: torch.Tensor,
    huber_param: float = 0.0,
) -> base.LossOutput:
    """Implements Q-learning for quantile-valued Q distributions.

    See "Distributional Reinforcement Learning with Quantile Regression" by
    Dabney et al. (https://arxiv.org/abs/1710.10044).

    Args:
      dist_q_tm1: Tensor holding Q distribution at time t-1, shape `[B, num_taus, action_dim]`.
      tau_q_tm1: Q distribution probability thresholds, , shape `[B, num_taus]`.
      a_tm1: Tensor holding action indices, shape `[B]`.
      r_t: Tensor holding rewards, shape `[B]`.
      discount_t: Tensor holding discount values, shape `[B]`.
      dist_q_t: Tensor holding target Q distribution at time t, shape `[B, num_taus, action_dim]`.
      q_t_selector: Tensor holding Q distribution at time t for selecting greedy action in
        target policy. This is separate from dist_q_t as in Double Q-Learning, but
        can be computed with the target network and a separate set of samples,
        shape `[B, num_taus, action_dim]`.
      huber_param: Huber loss parameter, defaults to 0 (no Huber loss).

    Returns:
      A namedtuple with fields:

      * `loss`: Tensor containing the batch of losses, shape `[B]`.
      * `extra`: a namedtuple with fields:
          * `target`:  Tensor containing the values that `q_tm1` at actions
          `a_tm1` are regressed towards, shape `[B, num_taus]` .

    """

    # Rank and compatibility checks.
    base.assert_rank_and_dtype(dist_q_tm1, 3, torch.float32)
    base.assert_rank_and_dtype(tau_q_tm1, 2, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 1, torch.long)
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(discount_t, 1, torch.float32)
    base.assert_rank_and_dtype(dist_q_t, 3, torch.float32)
    base.assert_rank_and_dtype(q_t_selector, 3, torch.float32)

    base.assert_batch_dimension(a_tm1, dist_q_tm1.shape[0])
    base.assert_batch_dimension(r_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(dist_q_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(q_t_selector, dist_q_tm1.shape[0])

    # Quantile Regression double q learning op.
    # Only update the taken actions.
    dist_qa_tm1 = base.batched_index(dist_q_tm1, a_tm1, 2)  # [batch_size, num_taus]
    # dist_qa_tm1 = dist_q_tm1[torch.arange(0, B), :, a_tm1] # [batch_size, num_taus]

    # Select target action according to greedy policy w.r.t. q_t_selector.
    q_t_selector = torch.mean(q_t_selector, dim=1)
    a_t = torch.argmax(q_t_selector, dim=1)
    # dist_qa_t = dist_q_t[torch.arange(0, B), :, a_t]  # [batch_size, num_taus]
    dist_qa_t = base.batched_index(dist_q_t, a_t, 2)  # [batch_size, num_taus]

    # Compute target, do not backpropagate into it.
    with torch.no_grad():
        dist_target_tm1 = r_t[:, None] + discount_t[:, None] * dist_qa_t  # [batch_size, num_taus]

    loss = _quantile_regression_loss(dist_qa_tm1, tau_q_tm1, dist_target_tm1, huber_param)
    return base.LossOutput(loss, Extra(dist_target_tm1))


def retrace(
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
) -> base.LossOutput:
    """Calculates Retrace errors.

    See "Safe and Efficient Off-Policy Reinforcement Learning" by Munos et al.
    (https://arxiv.org/abs/1606.02647).

    Args:
      q_tm1: Q-values at time t-1, this is from the online Q network, shape [T, B, action_dim].
      q_t: Q-values at time t, this is often from the target Q network, shape [T, B, action_dim].
      a_tm1: action index at time t-1, the action the agent took in state s_tm1, shape [T, B].
      a_t: action index at time t, the action the agent took in state s_t, shape [T, B].
      r_t: reward at time t, for state-action pair (s_tm1, a_tm1), shape [T, B].
      discount_t: discount at time t, shape [T, B].
      pi_t: target policy probs at time t, shape [T, B, action_dim].
      mu_t: behavior policy probs at time t, shape [T, B, action_dim].
      lambda_: scalar mixing parameter lambda.
      eps: small value to add to mu_t for numerical stability.

    Returns:
      * `loss`: Tensor containing the batch of losses, shape `[T, B]`.
      * `extra`: a namedtuple with fields:
          * `target`:  Tensor containing the values that `q_tm1` at actions
          `a_tm1` are regressed towards, shape `[T, B]` .
          * `td_error`: batch of temporal difference errors, shape `[T, B]`
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
        target_tm1 = multistep.general_off_policy_returns_from_action_values(q_t, a_t, r_t, discount_t, c_t, pi_t)

    qa_tm1 = base.batched_index(q_tm1, a_tm1)

    td_error = target_tm1 - qa_tm1
    loss = 0.5 * td_error**2

    return base.LossOutput(loss, QExtra(target=target_tm1, td_error=td_error))
