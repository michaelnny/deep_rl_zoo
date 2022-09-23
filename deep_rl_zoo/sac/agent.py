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
"""SAC (for Discrete Action) agent class.

From the paper "Soft Actor-Critic for Discrete Action Settings"
https://arxiv.org/abs/1910.07207.

From the paper "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
https://arxiv.org/abs/1801.01290.
"""

from typing import Iterable, Mapping, Tuple, Text
import copy
import multiprocessing
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

# pylint: disable=import-error
import deep_rl_zoo.replay as replay_lib
import deep_rl_zoo.types as types_lib
from deep_rl_zoo import value_learning
from deep_rl_zoo import base

# torch.autograd.set_detect_anomaly(True)


class Actor(types_lib.Agent):
    """SAC actor"""

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        policy_network: torch.nn.Module,
        transition_accumulator: replay_lib.NStepTransitionAccumulator,
        min_replay_size: int,
        num_actions: int,
        device: torch.device,
    ) -> None:
        """
        Args:
            rank: the rank for the actor.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            policy_network: the policy network for worker to make action choice.
            transition_accumulator: external helper class to build n-step transition.
            min_replay_size: minimum replay size before do learning.
            num_actions: number of actions for the environment.
            device: PyTorch runtime device.
        """

        if not 1 <= min_replay_size:
            raise ValueError(f'Expect min_replay_size to be integer greater than or equal to 1, got {min_replay_size}')
        if not 1 <= num_actions:
            raise ValueError(f'Expect num_actions to be integer greater than or equal to 1, got {num_actions}')

        self.rank = rank
        self.agent_name = f'SAC-actor{rank}'
        self._policy_network = policy_network.to(device=device)
        self._device = device

        self._queue = data_queue
        self._transition_accumulator = transition_accumulator
        self._num_actions = num_actions
        self._min_replay_size = min_replay_size

        self._step_t = -1

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1

        a_t = self.act(timestep)

        # Try build transition and put into global queue
        for transition in self._transition_accumulator.step(timestep, a_t):
            self._queue.put(transition)

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        'Given timestep, return an action.'
        a_t = self._choose_action(timestep)
        return a_t

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, choose action a_t"""
        if self._step_t < self._min_replay_size:  # Act randomly when staring out.
            a_t = np.random.randint(0, self._num_actions)
            return a_t
        else:
            s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
            logits_t = self._policy_network(s_t).pi_logits
            m = Categorical(logits=logits_t)
            a_t = m.sample()
            return a_t.cpu().item()

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {}


class Learner(types_lib.Learner):
    """SAC learner"""

    def __init__(
        self,
        replay: replay_lib.UniformReplay,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        q1_network: nn.Module,
        q1_optimizer: torch.optim.Optimizer,
        q2_network: nn.Module,
        q2_optimizer: torch.optim.Optimizer,
        discount: float,
        n_step: int,
        batch_size: int,
        num_actions: int,
        min_replay_size: int,
        learn_frequency: int,
        q_target_tau: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
    ) -> None:
        """
        Args:
            replay: simple experience replay to store transitions.
            policy_network: the policy network we want to train.
            policy_optimizer: the optimizer for policy network.
            q1_network: the first Q network.
            q1_optimizer: the optimizer for the first Q network.
            q2_network: the second Q network.
            q2_optimizer: the optimizer for the second Q network.
            discount: the gamma discount for future rewards.
            n_step: TD n-step returns.
            batch_size: sample batch_size of transitions.
            num_actions: number of actions for the environment.
            min_replay_size: minimum replay size before do learning.
            learn_frequency: how often should the agent learn.
            q_target_tau: the coefficient of target Q network weights.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to in the range [0.0, 1.0], got {discount}')
        if not 1 <= n_step:
            raise ValueError(f'Expect n_step to be integer geater than 1, got {n_step}')

        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to in the range [1, 512], got {batch_size}')
        if not 1 <= num_actions:
            raise ValueError(f'Expect num_actions to be integer geater than or equal to 1, got {num_actions}')
        if not 1 <= learn_frequency:
            raise ValueError(f'Expect learn_frequency to be positive integer, got {learn_frequency}')
        if not 1 <= min_replay_size:
            raise ValueError(f'Expect min_replay_size to be positive integer, got {min_replay_size}')
        if not batch_size <= min_replay_size <= replay.capacity:
            raise ValueError(f'Expect min_replay_size >= {batch_size} and <= {replay.capacity} and, got {min_replay_size}')
        if not 0.0 < q_target_tau <= 1.0:
            raise ValueError(f'Expect q_target_tau to (0.0, 1.0], got {q_target_tau}')

        self.agent_name = 'SAC-learner'
        self._device = device
        self._policy_network = policy_network.to(device=device)
        self._policy_optimizer = policy_optimizer
        self._q1_network = q1_network.to(device=device)
        self._q1_optimizer = q1_optimizer
        self._q2_network = q2_network.to(device=device)
        self._q2_optimizer = q2_optimizer

        # Lazy way to create target Q networks
        self._q1_target_network = copy.deepcopy(q1_network).to(device=device)
        self._q2_target_network = copy.deepcopy(q2_network).to(device=device)
        # Disable require gradients for target Q networks to improve performance
        for p1, p2 in zip(self._q1_target_network.parameters(), self._q2_target_network.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False

        # Entropy temperature parameters is learned
        # Automating Entropy Adjustment for Maximum Entropy RL section of https://arxiv.org/abs/1812.05905
        self._target_entropy = -np.log(1.0 / num_actions) * 0.98
        # Use log is more numerical stable as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        self._log_ent_coef = torch.log(torch.ones(1, device=device)).requires_grad_(True)
        lr = self._policy_optimizer.param_groups[0]['lr']  # Copy learning rate from policy network optimizer
        self._ent_coef_optimizer = torch.optim.Adam([self._log_ent_coef], lr=lr)

        self._q_target_tau = q_target_tau

        self._replay = replay
        self._discount = discount
        self._n_step = n_step
        self._batch_size = batch_size
        self._min_replay_size = min_replay_size
        self._learn_frequency = learn_frequency
        self._num_actions = num_actions

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        # Counters
        self._step_t = -1
        self._update_t = 0
        self._q1_loss_t = np.nan
        self._q2_loss_t = np.nan
        self._policy_loss_t = np.nan

    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.

        Yields:
            learner statistics if network parameters update occurred, otherwise returns None.
        """
        self._step_t += 1

        if self._replay.size < self._batch_size or self._step_t % self._batch_size != 0:
            return

        self._learn()
        yield self.statistics

    def reset(self) -> None:
        """Should be called at the begining of every iteration."""
        self._replay.reset()

    def received_item_from_queue(self, item) -> None:
        """Received item send by actors through multiprocessing queue."""
        self._replay.add(item)

    def _learn(self) -> None:
        # Note we don't clear old samples since this off-policy learning
        transitions = self._replay.sample(self._batch_size)
        self._update(transitions)

    def _update(self, transitions: replay_lib.Transition) -> None:
        self._update_q(transitions)  # Policy evaluation
        self._update_pi(transitions)  # Policy improvement
        self._update_target_q_networks()
        self._update_t += 1

    def _update_q(self, transitions: replay_lib.Transition) -> None:
        self._q1_optimizer.zero_grad()
        self._q2_optimizer.zero_grad()

        q1_loss, q2_loss = self._calc_q_loss(transitions)

        q1_loss.backward()
        q2_loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._q1_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )
            torch.nn.utils.clip_grad_norm_(
                self._q2_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._q1_optimizer.step()
        self._q2_optimizer.step()

        # For logging only.
        self._q1_loss_t = q1_loss.detach().cpu().item()
        self._q2_loss_t = q2_loss.detach().cpu().item()

    def _update_pi(self, transitions: replay_lib.Transition) -> None:
        self._policy_optimizer.zero_grad()
        self._ent_coef_optimizer.zero_grad()

        # Calculate policy loss
        loss, ent_coef_loss = self._calc_policy_loss(transitions=transitions)

        # Backpropagate policy network
        loss.backward()

        # Entropy temperature parameters
        ent_coef_loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._policy_optimizer.step()
        self._ent_coef_optimizer.step()

        # For logging only.
        self._policy_loss_t = loss.detach().cpu().item()

    def _calc_q_loss(self, transitions: replay_lib.Transition) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Calculate estimated q values for state-action pair (s_tm1, a_tm1) using two Q networks.
        q1_tm1 = self._q1_network(s_tm1).q_values
        q2_tm1 = self._q2_network(s_tm1).q_values

        # Computes target q value
        with torch.no_grad():
            # Get action a_t probabilities for s_t from current policy
            logits_t = self._policy_network(s_t).pi_logits  # [batch_size, num_actions]

            # Calculate log probabilities for all actions
            logprob_t = F.log_softmax(logits_t, dim=1)
            prob_t = F.softmax(logits_t, dim=1)

            # Get estimated q values from target Q networks
            q1_s_t = self._q1_target_network(s_t).q_values  # [batch_size, num_actions]
            q2_s_t = self._q2_target_network(s_t).q_values  # [batch_size, num_actions]
            q_s_t = torch.min(q1_s_t, q2_s_t)  # [batch_size, num_actions]

            # Calculate soft state-value for s_t with respect to current policy
            target_q_t = prob_t * (q_s_t - self.ent_coef * logprob_t)  # eq 10, (batch_size, num_actions)

        # Compute q loss is 0.5 * square(td_errors)
        q1_loss = value_learning.qlearning(q1_tm1, a_tm1, r_t, discount_t, target_q_t).loss
        q2_loss = value_learning.qlearning(q2_tm1, a_tm1, r_t, discount_t, target_q_t).loss

        # Average over batch dimension.
        q1_loss = q1_loss.mean()
        q2_loss = q2_loss.mean()

        return q1_loss, q2_loss

    def _calc_policy_loss(self, transitions: replay_lib.Transition) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate policy network loss and entrypo temperature loss"""
        s_tm1 = torch.from_numpy(transitions.s_tm1).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_tm1, (2, 4), torch.float32)

        # Compute action logits for s_tm1.
        logits_tm1 = self._policy_network(s_tm1).pi_logits  # [batch_size, num_actions]
        logprob_tm1 = F.log_softmax(logits_tm1, dim=1)
        prob_tm1 = F.softmax(logits_tm1, dim=1)

        # Compute the minimun q values for s_tm1 from the two Q networks.
        with torch.no_grad():
            q1_tm1 = self._q1_network(s_tm1).q_values
            q2_tm1 = self._q2_network(s_tm1).q_values
            min_q_tm1 = torch.min(q1_tm1, q2_tm1)

        # Compute expected q values with action probabilities from current policy.
        q_tm1 = torch.sum(min_q_tm1 * prob_tm1, dim=1, keepdim=True)

        # Compute entropy temperature parameter loss.
        ent_coef_losses = (
            self._log_ent_coef * (logprob_tm1 + self._target_entropy).detach()
        )  # eq 11, (batch_size, num_actions)

        # Compute SAC policy gradient loss.
        policy_losses = prob_tm1 * (q_tm1 - self.ent_coef * logprob_tm1)  # [batch_size, num_actions]
        # alternative, we can calculate it according to original paper eq 12
        # policy_losses = prob_tm1 * (self.ent_coef * logprob_tm1 - q_tm1)  # eq 12, (batch_size, num_actions)

        # Sum over all actions, average over batch dimension.
        # Negative sign to indicate we want to maximize the policy gradient objective function
        ent_coef_loss = -torch.mean(torch.sum(ent_coef_losses, dim=-1), dim=0)
        policy_loss = -torch.mean(torch.sum(policy_losses, dim=-1), dim=0)

        return policy_loss, ent_coef_loss

    def _update_target_q_networks(self) -> None:
        self._polyak_update_target_q(self._q1_network, self._q1_target_network, self._q_target_tau)
        self._polyak_update_target_q(self._q2_network, self._q2_target_network, self._q_target_tau)

    def _polyak_update_target_q(self, q: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
        with torch.no_grad():
            for param, target_param in zip(q.parameters(), target.parameters()):
                target_param.data.mul_(tau)
                target_param.data.add_((1 - tau) * param.data)

    @property
    def ent_coef(self) -> torch.Tensor:
        """Detached entropy temperature parameter, avoid passing into policy or Q networks"""
        return torch.exp(self._log_ent_coef.detach())

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            'discount': self._discount,
            'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            'q1_learning_rate': self._q1_optimizer.param_groups[0]['lr'],
            'q2_learning_rate': self._q2_optimizer.param_groups[0]['lr'],
            'policy_loss': self._policy_loss_t,
            'q1_loss': self._q1_loss_t,
            'q2_loss': self._q2_loss_t,
        }
