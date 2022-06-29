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
"""Actor-Critic agent class.

From the paper "Actor-Critic Algorithms"
https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf.
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


class ActorCritic(types_lib.Agent):
    """Actor-Critic agent"""

    def __init__(
        self,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        transition_accumulator: replay_lib.NStepTransitionAccumulator,
        discount: float,
        n_step: int,
        batch_size: int,
        entropy_coef: float,
        baseline_coef: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
    ) -> None:
        """
        Args:
            policy_network: the policy network we want to train.
            policy_optimizer: the optimizer for policy network.
            transition_accumulator: external helper class to build n-step transition.
            discount: the gamma discount for future rewards.
            n_step: TD n-step returns.
            batch_size: sample batch_size of transitions.
            entropy_coef: the coefficient of entryopy loss.
            baseline_coef: the coefficient of state-value loss.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to be [0.0, 1.0], got {discount}')
        if not 1 <= n_step:
            raise ValueError(f'Expect n_step to be integer geater than 1, got {n_step}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to be [1, 512], got {batch_size}')
        if not 0.0 < entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to be (0.0, 1.0], got {entropy_coef}')
        if not 0.0 < baseline_coef <= 1.0:
            raise ValueError(f'Expect baseline_coef to be (0.0, 1.0], got {baseline_coef}')

        self.agent_name = 'Actor-Critic'
        self._device = device
        self._policy_network = policy_network.to(device=self._device)
        self._policy_optimizer = policy_optimizer
        self._discount = discount

        self._transition_accumulator = transition_accumulator
        self._n_step = n_step
        self._batch_size = batch_size

        self._storage = collections.deque(maxlen=1000)

        self._entropy_coef = entropy_coef
        self._baseline_coef = baseline_coef

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        # Counters and stats
        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._baseline_loss_t = np.nan
        self._entropy_loss_t = np.nan

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Agent take a step at timestep, return the action a_t,
        and record episode tranjectory, start to learn when the replay is ready"""
        self._step_t += 1

        a_t = self.act(timestep)

        # Try to build transition
        for transition in self._transition_accumulator.step(timestep, a_t):
            self._storage.append(transition)

        # Start learning when replay reach batch_size limit
        if len(self._storage) >= self._batch_size:
            self._learn()

        return a_t

    def reset(self):
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()

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
        transitions = list(self._storage)
        transitions = replay_lib.np_stack_list_of_transitions(transitions, replay_lib.TransitionStructure, 0)
        self._update(transitions)
        self._storage.clear()  # discard old samples after using it

    def _update(self, transitions: replay_lib.Transition) -> None:
        self._policy_optimizer.zero_grad()
        loss = self._calc_loss(transitions)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(), max_norm=self._max_grad_norm, error_if_nonfinite=True
            )
        self._policy_optimizer.step()
        self._update_t += 1

    def _calc_loss(self, transitions: replay_lib.Transition) -> torch.Tensor:
        """Calculate loss sumed over the trajectories of a single episode"""
        s_tm1 = torch.from_numpy(transitions.s_tm1).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        a_tm1 = torch.from_numpy(transitions.a_tm1).to(device=self._device, dtype=torch.int64)  # [batch_size]
        r_t = torch.from_numpy(transitions.r_t).to(device=self._device, dtype=torch.float32)  # [batch_size]
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        done = torch.from_numpy(transitions.done).to(device=self._device, dtype=torch.bool)  # [batch_size]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_tm1, (2, 4), torch.float32)
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_tm1, 1, torch.long)
        base.assert_rank_and_dtype(r_t, 1, torch.float32)
        base.assert_rank_and_dtype(done, 1, torch.bool)

        discount_t = (~done).float() * self._discount**self._n_step

        # Get policy action logits and baseline for s_tm1.
        policy_output = self._policy_network(s_tm1)
        logits_tm1 = policy_output.pi_logits
        baseline_s_tm1 = policy_output.baseline.squeeze(1)  # [batch_size]

        # Calculates TD n-step target and advantages.
        with torch.no_grad():
            baseline_s_t = self._policy_network(s_t).baseline.squeeze(1)  # [batch_size]
            target_baseline = r_t + discount_t * baseline_s_t
            advantages = target_baseline - baseline_s_tm1

        # Compute policy gradient loss.
        policy_loss = rl.policy_gradient_loss(logits_tm1, a_tm1, advantages).loss

        # Compute entropy loss.
        entropy_loss = rl.entropy_loss(logits_tm1).loss

        # Compute baseline state-value loss.
        baseline_loss = rl.baseline_loss(baseline_s_tm1 - target_baseline).loss

        # Average over batch dimension.
        policy_loss = torch.mean(policy_loss, dim=0)
        entropy_loss = self._entropy_coef * torch.mean(entropy_loss, dim=0)
        baseline_loss = self._baseline_coef * torch.mean(baseline_loss, dim=0)

        # Combine policy loss, baseline loss, entropy loss.
        loss = policy_loss + baseline_loss + entropy_loss

        # For logging only.
        self._policy_loss_t = policy_loss.detach().cpu().item()
        self._baseline_loss_t = baseline_loss.detach().cpu().item()
        self._entropy_loss_t = entropy_loss.detach().cpu().item()

        return loss

    @property
    def statistics(self):
        """Returns current agent statistics as a dictionary."""
        return {
            'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            'policy_loss': self._policy_loss_t,
            'baseline_loss': self._baseline_loss_t,
            'entropy_loss': self._entropy_loss_t,
            'discount': self._discount,
            'updates': self._update_t,
        }
