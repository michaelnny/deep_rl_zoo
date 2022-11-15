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
"""A2C agent class.

Specifically:
    * Actors sample batch of transitions to calculate loss, but not optimization step.
    * Actors collects local gradients, and send to master through multiprocessing.Queue.
    * Learner will aggregates batch of gradients then do the optimization step.
    * Learner update policy network parameters for workers (shared_memory).

Note only supports training on single machine.

From the paper "Asynchronous Methods for Deep Reinforcement Learning"
https://arxiv.org/abs/1602.01783

Synchronous, Deterministic variant of A3C
https://openai.com/blog/baselines-acktr-a2c/.
"""
import collections
from typing import Iterable, List, Mapping, Text
import multiprocessing
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


def extract_gradients(network: torch.nn.Module, compress: bool):
    """Extract network gradients into compressed numpy.ndarrays,
    for better performance when passing multiprocessing.Queue"""
    compress_fn = replay_lib.compress_array if compress else lambda g: g
    return [compress_fn(param.grad.data.cpu().numpy()) for param in network.parameters()]


class Actor(types_lib.Agent):
    """A2C GRAD actor"""

    def __init__(
        self,
        rank: int,
        lock: multiprocessing.Lock,
        gradient_queue: multiprocessing.Queue,
        policy_network: nn.Module,
        transition_accumulator: replay_lib.TransitionAccumulator,
        discount: float,
        batch_size: int,
        entropy_coef: float,
        baseline_coef: float,
        compress_gradient: bool,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
    ) -> None:
        """
        Args:
            rank: the rank for the actor.
            lock: multiprocessing.Lock to synchronize with learner process.
            gradient_queue: a multiprocessing.Queue to get collected local gradients to worker processes.
            policy_network: the policy network we want to train.
            transition_accumulator: external helper class to build n-step transition.
            discount: the gamma discount for future rewards.
            batch_size: sample batch_size of transitions.
            entropy_coef: the coefficient of entropy loss.
            baseline_coef: the coefficient of state-value loss.
            compress_gradient, if True, compress numpy arrays before put on to multiprocessing.Queue.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """

        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to be [0.0, 1.0], got {discount}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to be [1, 512], got {batch_size}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to be (0.0, 1.0], got {entropy_coef}')
        if not 0.0 <= baseline_coef <= 1.0:
            raise ValueError(f'Expect baseline_coef to be (0.0, 1.0], got {baseline_coef}')

        self.rank = rank
        self.agent_name = f'A2C-GRAD-actor{rank}'
        self._queue = gradient_queue
        self._lock = lock

        self._device = device
        self._policy_network = policy_network.to(device=device)
        self._compress_gradient = compress_gradient
        self._transition_accumulator = transition_accumulator
        self._discount = discount
        self._batch_size = batch_size
        self._trajectory = collections.deque(maxlen=batch_size)

        self._entropy_coef = entropy_coef
        self._baseline_coef = baseline_coef

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        self._step_t = -1
        self._policy_loss_t = np.nan
        self._baseline_loss_t = np.nan
        self._entropy_loss_t = np.nan

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Agent take a step at timestep, return the action a_t,
        and record episode trajectory, start to learn
        when the replay is ready and put local gradients into global queue"""

        # Wait for learner process update to finished before make any move
        with self._lock:
            self._step_t += 1

        a_t = self.act(timestep)

        # Try to build transition
        for transition in self._transition_accumulator.step(timestep, a_t):
            self._trajectory.append(transition)

        # start learning
        if len(self._trajectory) == self._trajectory.maxlen:
            gradients = self._learn()
            self._queue.put(gradients)  # add local gradients to global queue

        return a_t

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        'Given timestep, return an action.'
        a_t = self._choose_action(timestep)
        return a_t

    def reset(self):
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, choose action a_t"""
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        logits_t = self._policy_network(s_t).pi_logits
        # Sample an action
        a_t = distributions.categorical_distribution(logits_t).sample()
        return a_t.cpu().item()

    def _learn(self) -> None:
        # Turn list of transitions into one Transition object
        transitions = replay_lib.np_stack_list_of_transitions(list(self._trajectory), replay_lib.TransitionStructure)
        gradients = self._update(transitions)

        self._trajectory.clear()  # discard old samples after using it
        return gradients

    def _update(self, transitions: replay_lib.Transition) -> None:
        # Clear network gradients before calculate loss
        self._policy_network.zero_grad()
        loss = self._calc_loss(transitions)
        loss.backward()
        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        # Make a copy of local gradients as numpy.ndarrays and compress them.
        gradients = extract_gradients(self._policy_network, self._compress_gradient)
        return gradients

    def _calc_loss(self, transitions: replay_lib.Transition) -> torch.Tensor:
        """Calculate loss for a batch transitions"""
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

        discount_t = (~done).float() * self._discount

        # Get policy action logits and baseline for s_tm1.
        policy_output = self._policy_network(s_tm1)
        logits_tm1 = policy_output.pi_logits
        baseline_tm1 = policy_output.baseline.squeeze(1)  # [batch_size]

        # Calculates TD n-step target and advantages.
        with torch.no_grad():
            baseline_s_t = self._policy_network(s_t).baseline.squeeze(1)  # [batch_size]
            target_baseline = r_t + discount_t * baseline_s_t
            advantages = target_baseline - baseline_tm1

        # Compute policy gradient a.k.a. log-likelihood loss.
        policy_loss = rl.policy_gradient_loss(logits_tm1, a_tm1, advantages).loss

        # Compute entropy loss.
        entropy_loss = rl.entropy_loss(logits_tm1).loss

        # Compute baseline state-value loss.
        baseline_loss = rl.baseline_loss(target_baseline, baseline_tm1).loss

        # Averaging over batch dimension.
        policy_loss = torch.mean(policy_loss, dim=0)
        entropy_loss = self._entropy_coef * torch.mean(entropy_loss, dim=0)
        baseline_loss = self._baseline_coef * torch.mean(baseline_loss, dim=0)

        # Combine policy loss, baseline loss, entropy loss.
        # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
        loss = -(policy_loss + entropy_loss) + baseline_loss

        # For logging only.
        self._policy_loss_t = policy_loss.detach().cpu().item()
        self._baseline_loss_t = baseline_loss.detach().cpu().item()
        self._entropy_loss_t = entropy_loss.detach().cpu().item()

        return loss

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            'policy_loss': self._policy_loss_t,
            'baseline_loss': self._baseline_loss_t,
            'entropy_loss': self._entropy_loss_t,
        }


class Learner(types_lib.Learner):
    """A2C GRAD learner"""

    def __init__(
        self,
        lock: multiprocessing.Lock,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        gradient_replay: replay_lib.GradientReplay,
        batch_size: int,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
    ) -> None:
        """
        Args:
            lock: multiprocessing.Lock to synchronize with worker processes.
            policy_network: the policy network we want to train.
            policy_optimizer: the optimizer for policy network.
            gradient_replay: simple storage to store gradients.
            batch_size: sample batch_size of number of accumulated gradients.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """

        self.agent_name = 'A2C-GRAD-learner'
        self._device = device
        self._policy_network = policy_network.to(device=device)
        self._policy_network.train()
        self._policy_optimizer = policy_optimizer

        self._lock = lock
        self._replay = gradient_replay
        self._batch_size = batch_size

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        self._step_t = -1
        self._update_t = 0

    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.

        Yields:
            learner statistics if network parameters update occurred, otherwise returns None.
        """
        self._step_t += 1

        if self._replay.size < self._batch_size:
            return

        # Blocking while master is updating network parameters
        with self._lock:
            self._learn()
        yield self.statistics

    def reset(self) -> None:
        """Should be called at the beginning of every iteration."""
        self._replay.reset()

    def received_item_from_queue(self, item) -> None:
        """Received item send by actors through multiprocessing queue."""
        self._replay.add(item)

    def _learn(self) -> None:
        # Get aggregate batch gradients
        gradients = self._replay.sample()
        self._update(gradients)
        self._update_t += 1
        assert self._replay.size == 0

    def _update(self, gradients: List[np.ndarray]) -> None:
        # Clear out old gradients
        self._policy_optimizer.zero_grad()

        # Manually set gradients
        for param, grad in zip(self._policy_network.parameters(), gradients):
            # Averaging over batch dimension
            param.grad = torch.tensor(grad).to(device=self._device, dtype=torch.float32).mean(dim=0)

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._policy_optimizer.step()

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            # 'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            'updates': self._update_t,
        }
