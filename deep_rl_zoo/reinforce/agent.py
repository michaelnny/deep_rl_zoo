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
"""REINFORCE agent class.

From the paper "Policy Gradient Methods for Reinforcement Learning with Function Approximation"
https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf.
"""
import collections
import numpy as np
import torch
from torch import nn

# pylint: disable=import-error
import deep_rl_zoo.replay as replay_lib
import deep_rl_zoo.types as types_lib
import deep_rl_zoo.policy_gradient as rl
from deep_rl_zoo import base
from deep_rl_zoo import distributions

# torch.autograd.set_detect_anomaly(True)


class Reinforce(types_lib.Agent):
    """Reinforce agent"""

    def __init__(
        self,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        discount: float,
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
            transition_accumulator: external helper class to build n-step transition.
            normalize_returns: if True, normalize episode returns.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to in the range [0.0, 1.0], got {discount}')

        self.agent_name = 'REINFORCE'
        self._device = device
        self._policy_network = policy_network.to(device=self._device)
        self._policy_optimizer = policy_optimizer
        self._discount = discount

        self._transition_accumulator = transition_accumulator
        self._trajectory = collections.deque(maxlen=108000)

        self._normalize_returns = normalize_returns
        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        # Counters and stats
        self._step_t = -1
        self._update_t = 0
        self._loss_t = np.nan

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
        self._policy_optimizer.zero_grad()
        loss = self._calc_loss(transitions)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )
        self._policy_optimizer.step()
        self._update_t += 1

        # For logging only.
        self._loss_t = loss.detach().cpu().item()

    def _calc_loss(self, transitions: replay_lib.Transition) -> torch.Tensor:
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
        # returns = multistep.lambda_returns(r_t=r_t[..., None], discount_t=discount_t[..., None], )

        seq_len = len(r_t)
        returns = torch.empty(seq_len, device=self._device)
        g = 0.0
        # Calculate returns from t=T-1, T-2, ..., 1, 0
        for t in reversed(range(0, seq_len)):
            g = r_t[t] + discount_t[t] * g
            returns[t] = g

        # Get policy action logits for s_tm1.
        logits_tm1 = self._policy_network(s_tm1).pi_logits

        # Compute policy gradient a.k.a. log-likelihood loss.
        loss = rl.policy_gradient_loss(logits_tm1, a_tm1, returns).loss

        # Averaging over batch dimension.
        # Negative sign to indicate we want to maximize the policy gradient objective function
        loss = -torch.mean(loss, dim=0)

        return loss

    @property
    def statistics(self):
        """Returns current agent statistics as a dictionary."""
        return {
            # 'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            'policy_loss': self._loss_t,
            # 'discount': self._discount,
            'updates': self._update_t,
        }
