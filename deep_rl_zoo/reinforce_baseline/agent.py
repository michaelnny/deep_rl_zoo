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
"""REINFORCE with baseline agent class.

From the paper "Policy Gradient Methods for Reinforcement Learning with Function Approximation"
https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf.
"""
from typing import Tuple
import collections
import numpy as np
import torch
from torch import nn

# pylint: disable=import-error
import deep_rl_zoo.replay as replay_lib
import deep_rl_zoo.types as types_lib
import deep_rl_zoo.policy_gradient as rl
from deep_rl_zoo import distributions
from deep_rl_zoo import base

# torch.autograd.set_detect_anomaly(True)


class ReinforceBaseline(types_lib.Agent):
    """Reinforce agent with baseline"""

    def __init__(
        self,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        discount: float,
        baseline_network: nn.Module,
        baseline_optimizer: torch.optim.Optimizer,
        transition_accumulator: replay_lib.TransitionAccumulator,
        normalize_returns: bool,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
    ) -> None:
        """
        Args:
            policy_network: the policy network we want to train.
            policy_optimizer: the optimizer for policy network.
            discount: the gamma discount for future rewards.
            baseline_network: the critic state-value network.
            baseline_optimizer: the optimizer for state-value network.
            transition_accumulator: external helper class to build n-step transition.
            normalize_returns: if True, normalize episode returns.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to in the range [0.0, 1.0], got {discount}')

        self.agent_name = 'REINFORCE-BASELINE'
        self._device = device
        self._policy_network = policy_network.to(device=self._device)
        self._policy_optimizer = policy_optimizer
        self._discount = discount

        self._baseline_network = baseline_network.to(device=self._device)
        self._baseline_optimizer = baseline_optimizer

        self._transition_accumulator = transition_accumulator
        self._trajectory = collections.deque(maxlen=108000)

        self._normalize_returns = normalize_returns
        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        # Counters and stats
        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._baseline_loss_t = np.nan

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Agent take a step at timestep, return the action a_t,
        and record episode tranjectory, learn after the enpisode terminated"""
        self._step_t += 1

        a_t = self.act(timestep)

        # Try to build transition and add into episodic replay
        for transition in self._transition_accumulator.step(timestep, a_t):
            self._trajectory.append(transition)

        # Start to learn
        if timestep.done:
            self._learn()

        return a_t

    def reset(self):
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()
        self._trajectory.clear()

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        'Given timestep, return an action.'
        a_t = self._choose_action(timestep)
        return a_t

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, choose action a_t"""
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        logits_t = self._policy_network(s_t).pi_logits
        # Sample an action
        a_t = distributions.categorical_distribution(logits_t).sample()
        return a_t.cpu().item()

    def _learn(self) -> None:
        # Turn entire episode tranjectory into one Transition object
        transitions = replay_lib.np_stack_list_of_transitions(list(self._trajectory), replay_lib.TransitionStructure)
        self._update(transitions)

    def _update(self, transitions: replay_lib.Transition) -> None:
        self._baseline_optimizer.zero_grad()
        self._policy_optimizer.zero_grad()

        policy_loss, baseline_loss = self._calc_loss(transitions)

        # Backpropagate value network
        baseline_loss.backward()
        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._baseline_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )
        self._baseline_optimizer.step()

        # Backpropagate policy network
        policy_loss.backward()
        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )
        self._policy_optimizer.step()
        self._update_t += 1

    def _calc_loss(self, transitions: replay_lib.Transition) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate loss sumed over the trajectories of a single episode"""
        s_tm1 = torch.from_numpy(transitions.s_tm1).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        a_tm1 = torch.from_numpy(transitions.a_tm1).to(device=self._device, dtype=torch.int64)  # [batch_size]
        r_t = torch.from_numpy(transitions.r_t).to(device=self._device, dtype=torch.float32)  # [batch_size]
        done = torch.from_numpy(transitions.done).to(device=self._device, dtype=torch.bool)  # [batch_size]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_tm1, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_tm1, 1, torch.long)
        base.assert_rank_and_dtype(r_t, 1, torch.float32)
        base.assert_rank_and_dtype(done, 1, torch.bool)

        # Compute episode returns.
        discount_t = (~done) * self._discount
        seq_len = len(r_t)
        returns = torch.empty(seq_len, device=self._device)
        g = 0.0
        # Calculate returns from t=T-1, T-2, ..., 1, 0
        for t in reversed(range(0, seq_len)):
            g = r_t[t] + discount_t[t] * g
            returns[t] = g

        # Get policy action logits for s_tm1.
        logits_tm1 = self._policy_network(s_tm1).pi_logits
        # Get baseline state-value
        baseline_tm1 = self._baseline_network(s_tm1).baseline.squeeze(1)  # [batch_size]

        delta = returns - baseline_tm1

        # Compute policy gradient a.k.a. log-likelihood loss.
        policy_loss = rl.policy_gradient_loss(logits_tm1, a_tm1, delta).loss

        # Compute baseline state-value loss.
        baseline_loss = rl.baseline_loss(returns, baseline_tm1).loss

        # Averaging over batch dimension.
        baseline_loss = torch.mean(baseline_loss, dim=0)

        # Negative sign to indicate we want to maximize the policy gradient objective function
        policy_loss = -torch.mean(policy_loss, dim=0)

        # For logging only.
        self._policy_loss_t = policy_loss.detach().cpu().item()
        self._baseline_loss_t = baseline_loss.detach().cpu().item()

        return policy_loss, baseline_loss

    @property
    def statistics(self):
        """Returns current agent statistics as a dictionary."""
        return {
            # 'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            # 'baseline_learning_rate': self._baseline_optimizer.param_groups[0]['lr'],
            'baseline_loss': self._baseline_loss_t,
            'policy_loss': self._policy_loss_t,
            # 'discount': self._discount,
            'updates': self._update_t,
        }
