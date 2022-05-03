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
"""PPO-RND agent class.

From the paper "Exploration by Random Network Distillation"
https://arxiv.org/abs/1810.12894
"""

from typing import Mapping, Tuple
import itertools
import queue
import multiprocessing
import numpy as np
import torch
from torch import nn

# pylint: disable=import-error
import deep_rl_zoo.replay as replay_lib
import deep_rl_zoo.types as types_lib
from deep_rl_zoo.schedule import LinearSchedule
import deep_rl_zoo.policy_gradient as rl
from deep_rl_zoo import distributions
from deep_rl_zoo import base
from deep_rl_zoo import normalizer

# torch.autograd.set_detect_anomaly(True)


class Actor(types_lib.Agent):
    """PPO-RND actor"""

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        policy_network: torch.nn.Module,
        transition_accumulator: replay_lib.PgNStepTransitionAccumulator,
        unroll_length: int,
        device: torch.device,
    ) -> None:
        """
        Args:
            rank: the rank for the actor.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            policy_network: the policy network for worker to make action choice.
            transition_accumulator: external helper class to build n-step transition.
            unroll_length: rollout length.
            device: PyTorch runtime device.
        """
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer geater than or equal to 1, got {unroll_length}')

        self.rank = rank
        self.agent_name = f'PPO-RND-actor{rank}'
        self._queue = data_queue
        self._transition_accumulator = transition_accumulator
        self._policy_network = policy_network.to(device=device)
        self._device = device
        self._unroll_length = unroll_length
        self._storage = []
        self._step_t = -1

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given current timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1

        a_t, pi_logits_t = self.act(timestep)

        # Try build transition and put into global queue
        for transition in self._transition_accumulator.step(timestep, a_t, pi_logits_t):
            self._storage.append(transition)
            if len(self._storage) == self._unroll_length:
                self._queue.put(self._storage)
                self._storage = []

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()

    def act(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action, np.ndarray]:
        'Given timestep, return an action.'
        a_t, pi_logits_t = self._choose_action(timestep)
        return a_t, pi_logits_t

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action, np.ndarray]:
        """Given timestep, choose action a_t"""
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        logits_t = self._policy_network(s_t).pi_logits
        # Sample an action
        a_t = distributions.categorical_distribution(logits_t).sample()
        return a_t.cpu().item(), logits_t.squeeze(0).cpu().numpy()

    @property
    def statistics(self) -> Mapping[str, float]:
        """Returns current agent statistics as a dictionary."""
        return {}


class Learner:
    """PPO-RND learner"""

    def __init__(
        self,
        data_queue: multiprocessing.Queue,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        old_policy_network: nn.Module,
        rnd_target_network: nn.Module,
        rnd_predictor_network: nn.Module,
        observation_normalizer: normalizer.Normalizer,
        clip_epsilon: LinearSchedule,
        discount: float,
        rnd_discount: float,
        n_step: int,
        batch_size: int,
        update_k: int,
        unroll_length: int,
        extrinsic_reward_coef: float,
        intrinsic_reward_coef: float,
        rnd_experience_proportion: float,
        num_actors: int,
        entropy_coef: float,
        baseline_coef: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
    ) -> None:
        """
        Args:
            data_queue: a multiprocessing.Queue to get collected transitions from worker processes.
            policy_network: the policy network we want to train.
            policy_optimizer: the optimizer for policy network.
            old_policy_network: the old policy network used for workers.
            rnd_target_network: RND target fixed random network.
            rnd_predictor_network: RND predictor network.
            observation_normalizer: observation normalizer, only for RND networks.
            clip_epsilon: external scheduler to decay clip epsilon.
            replay: simple experience replay to store transitions.
            discount: the gamma discount for future rewards.
            rnd_discount: the gamma discount for future rewards for RND networks.
            n_step: TD n-step returns.
            batch_size: sample batch_size of transitions.
            update_k: update k times when it's time to do learning.
            unroll_length: worker rollout horizon.
            extrinsic_reward_coef: weights extrinsic reward from environment.
            intrinsic_reward_coef: weights intrinsic reward from RND bonus.
            rnd_experience_proportion: proportion of experience used for traning RND predictor.
            num_actors: number of worker processes.
            entropy_coef: the coefficient of entryopy loss.
            baseline_coef: the coefficient of state-value loss.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """

        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to in the range [0.0, 1.0], got {discount}')
        if not 0.0 <= rnd_discount <= 1.0:
            raise ValueError(f'Expect rnd_discount to in the range [0.0, 1.0], got {rnd_discount}')
        if not 1 <= n_step:
            raise ValueError(f'Expect n_step to be integer geater than 1, got {n_step}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to in the range [1, 512], got {batch_size}')
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k to be integer geater than or equal to 1, got {update_k}')
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer geater than or equal to 1, got {unroll_length}')
        if not 0.0 <= extrinsic_reward_coef <= 10.0:
            raise ValueError(f'Expect extrinsic_reward_coef to in the range [0.0, 10.0], got {extrinsic_reward_coef}')
        if not 0.0 <= intrinsic_reward_coef <= 10.0:
            raise ValueError(f'Expect intrinsic_reward_coef to in the range [0.0, 10.0], got {intrinsic_reward_coef}')
        if not 0.0 <= rnd_experience_proportion <= 10.0:
            raise ValueError(f'Expect rnd_experience_proportion to in the range [0.0, 10.0], got {rnd_experience_proportion}')
        if not 1 <= num_actors:
            raise ValueError(f'Expect num_actors to be integer geater than or equal to 1, got {num_actors}')
        if not 0.0 < entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to (0.0, 1.0], got {entropy_coef}')
        if not 0.0 < baseline_coef <= 1.0:
            raise ValueError(f'Expect baseline_coef to (0.0, 1.0], got {baseline_coef}')

        self.agent_name = 'PPO-RND-learner'
        self._policy_network = policy_network.to(device=device)
        self._old_policy_network = old_policy_network.to(device=device)
        self._policy_optimizer = policy_optimizer
        self._rnd_predictor_network = rnd_predictor_network.to(device=device)
        self._rnd_target_network = rnd_target_network.to(device=device)
        # Disable autograd for RND target networks.
        for p in self._rnd_target_network.parameters():
            p.requires_grad = False
        self._device = device

        self._observation_normalizer = observation_normalizer

        # Original paper uses 25% of experience for traning RND predictor
        self._rnd_loss_mask = torch.rand(batch_size).to(device=self._device)
        self._rnd_loss_mask = (self._rnd_loss_mask < rnd_experience_proportion).to(device=self._device, dtype=torch.float32)

        # Acummulate running statistics to calcualte mean and std online,
        # this will also clip intrinsic reward values in the range [-10, 10]
        self._intrinsic_reward_normalizer = normalizer.Normalizer(eps=0.0001, clip_range=(-10, 10), device=self._device)

        self._ext_reward_coef = extrinsic_reward_coef
        self._int_reward_coef = intrinsic_reward_coef

        self._storage = []
        self._discount = discount
        self._rnd_discount = rnd_discount
        self._n_step = n_step
        self._batch_size = batch_size
        self._unroll_length = unroll_length
        self._update_k = update_k

        self._entropy_coef = entropy_coef
        self._baseline_coef = baseline_coef
        self._clip_epsilon = clip_epsilon

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        self._queue = data_queue
        self._num_actors = num_actors

        # Counters
        self._step_t = -1
        self._update_t = -1
        self._done_workers = 0

    def run_train_loop(self) -> None:
        """Start the train loop, only break if all worker processes are done."""
        self.reset()
        while True:
            self._step_t += 1
            # Pull one item off queue
            try:
                item = self._queue.get()
                if item == 'PROCESS_DONE':  # worker process is done
                    self._done_workers += 1
                else:
                    self._storage.append(item)  # This will store a list (length unroll_length) of  transitions.
            except queue.Empty:
                pass
            except EOFError:
                pass

            # Only break if all worker processes are done
            if self._done_workers == self._num_actors:
                break

            # This is to check num_actors * unroll_length
            if len(self._storage) < self._num_actors:
                continue

            self._learn()

    def reset(self) -> None:
        """Should be called at the begining of every iteration."""
        self._done_workers = 0
        self._storage = []

    def _learn(self) -> None:
        # Merge list of lists into a single big list, this contains transitions from all workers.
        all_transitions = list(itertools.chain.from_iterable(self._storage))

        # Split indices into 'bins' with batch_size.
        bined_indices = replay_lib.split_indices_into_bins(self._batch_size, len(all_transitions))

        # Run update for K times
        for _ in range(self._update_k):
            # Update on a batch of transitions.
            for indices in bined_indices:
                transitions = [all_transitions[i] for i in indices]
                # Stack list of transitions, follow our code convention.
                stacked_transition = replay_lib.np_stack_list_of_transitions(
                    transitions, replay_lib.OffPolicyTransitionStructure
                )
                self._update(stacked_transition)
        self._storage = []  # discard old samples after using it
        self._update_old_policy()

    def _update(self, transitions: replay_lib.OffPolicyTransition) -> None:
        self._policy_optimizer.zero_grad()
        loss = self._calc_loss(transitions=transitions)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )
            torch.nn.utils.clip_grad_norm_(
                self._rnd_predictor_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._policy_optimizer.step()
        self._update_t += 1

    def _calc_rnd_loss(self, s_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update observation normalization statistics and normalize state
        s_t = self._normalize_observation(s_t)  # [batch_size, state_shape]

        pred_s_t = self._rnd_predictor_network(s_t)
        with torch.no_grad():
            target_s_t = self._rnd_target_network(s_t)

        abs_error = torch.sum(torch.abs(pred_s_t - target_s_t), dim=-1)  # Sums over latent features (batch_size, )
        assert abs_error.shape == self._rnd_loss_mask.shape

        # Apply porpotion mask, average over batch dimension
        rnd_loss = torch.mean(0.5 * torch.square(abs_error * self._rnd_loss_mask.detach()), dim=0)

        # Update intrinsic reward normalization statistics
        self._intrinsic_reward_normalizer.update(abs_error.detach())
        # Normalize intrinsic reward
        intrinsic_reward = self._intrinsic_reward_normalizer(abs_error.detach())

        if len(rnd_loss.shape) != 0:
            raise RuntimeError(f'Expect rnd_loss to be a scalar tensor, got {rnd_loss.shape}')

        if len(intrinsic_reward.shape) != 1:
            raise RuntimeError(f'Expect intrinsic_reward to be a 1D tensor, got {intrinsic_reward.shape}')

        return rnd_loss, intrinsic_reward

    def _calc_loss(self, transitions: replay_lib.OffPolicyTransition) -> torch.Tensor:
        """Calculate loss for a batch transitions"""
        s_tm1 = torch.from_numpy(transitions.s_tm1).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        a_tm1 = torch.from_numpy(transitions.a_tm1).to(device=self._device, dtype=torch.int64)  # [batch_size]
        r_t = torch.from_numpy(transitions.r_t).to(device=self._device, dtype=torch.float32)  # [batch_size]
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        done = torch.from_numpy(transitions.done).to(device=self._device, dtype=torch.bool)  # [batch_size]
        behavior_logits_tm1 = torch.from_numpy(transitions.logits_tm1).to(
            device=self._device, dtype=torch.float32
        )  # [batch_size, num_actions]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_tm1, (2, 4), torch.float32)
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_tm1, 1, torch.long)
        base.assert_rank_and_dtype(r_t, 1, torch.float32)
        base.assert_rank_and_dtype(done, 1, torch.bool)
        base.assert_rank_and_dtype(behavior_logits_tm1, 2, torch.float32)

        # Compute intrinsic reward using RND module.
        rnd_loss, intrinsic_reward = self._calc_rnd_loss(s_t)

        discount_t = (~done).float() * self._discount**self._n_step

        # Get policy action logits and baseline for s_tm1.
        policy_output = self._policy_network(s_tm1)
        logits_tm1 = policy_output.pi_logits
        # Two baseline heads
        ext_baseline_tm1 = policy_output.ext_baseline.squeeze(1)  # [batch_size]
        int_baseline_tm1 = policy_output.int_baseline.squeeze(1)  # [batch_size]

        # Calculates TD n-step target and advantages
        with torch.no_grad():
            target_output = self._old_policy_network(s_t)
            target_ext_baseline_s_t = target_output.ext_baseline.squeeze(1)  # [batch_size]
            target_int_baseline_s_t = target_output.int_baseline.squeeze(1)  # [batch_size]

            # Calculate advantages for target based on extrinsic rewards
            target_ext = r_t + discount_t * target_ext_baseline_s_t  # [batch_size]
            ext_advantages = target_ext - ext_baseline_tm1  # [batch_size]

            # Calculate advantages for target based on intrinsic rewards
            target_int = intrinsic_reward + discount_t * target_int_baseline_s_t  # [batch_size]
            int_advantages = target_int - int_baseline_tm1  # [batch_size]

            # Combined advantages
            advantages = self._ext_reward_coef * ext_advantages + self._int_reward_coef * int_advantages

        # Calculate importance sampling ratio
        ratio = distributions.categorical_importance_sampling_ratios(logits_tm1, behavior_logits_tm1, a_tm1)

        if ratio.shape != advantages.shape:
            raise RuntimeError(f'Expect ratio and advantages have same shape, got {ratio.shape} and {advantages.shape}')

        # Compute PPO policy gradient loss.
        policy_loss = rl.clipped_surrogate_gradient_loss(ratio, advantages, self.clip_epsilon).loss

        # Compute entropy loss.
        entropy_loss = rl.entropy_loss(logits_tm1).loss

        # Compute baseline state-value loss, combine extrinsic and intrinsic losses.
        ext_baseline_loss = rl.baseline_loss(ext_baseline_tm1 - target_ext).loss
        int_baseline_loss = rl.baseline_loss(int_baseline_tm1 - target_int).loss
        baseline_loss = ext_baseline_loss + int_baseline_loss

        # Average over batch dimension.
        policy_loss = torch.mean(policy_loss, dim=0)
        entropy_loss = torch.mean(entropy_loss, dim=0)
        baseline_loss = torch.mean(baseline_loss, dim=0)

        # Combine policy loss, baseline loss, entropy loss.
        loss = policy_loss + self._baseline_coef * baseline_loss + self._entropy_coef * entropy_loss

        # Add RND predictor loss.
        loss += rnd_loss

        return loss

    def _normalize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        # Normalize observation using online incremental algorithm
        if len(observation.shape) > 2:
            # Unstack frames, RND normalize one frame.
            _unstacked = torch.unbind(observation, dim=1)
            _results = []
            for obs in _unstacked:
                # Add last dimension as channel, we normalize images by channel.
                obs = obs[..., None]
                self._observation_normalizer.update(obs)
                norm_states = self._observation_normalizer(obs).squeeze(-1)  # Remove last channel dimension
                _results.append(norm_states)

            # Restack frames
            normalized_obs = torch.stack(_results, dim=1)
        else:
            self._observation_normalizer.update(observation)
            normalized_obs = self._observation_normalizer(observation)
        return normalized_obs

    def _update_old_policy(self):
        self._old_policy_network.load_state_dict(self._policy_network.state_dict())

    @property
    def clip_epsilon(self):
        """Call external clip epsilon scheduler"""
        return self._clip_epsilon(self._step_t)

    @property
    def statistics(self) -> Mapping[str, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            'discount': self._discount,
            'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
        }
