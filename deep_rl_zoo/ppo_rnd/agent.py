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

Notice in this implementation we follow the following naming convention when referring to unroll sequence:
sₜ, aₜ, rₜ, sₜ₊₁, aₜ₊₁, rₜ₊₁, ...

From the paper "Exploration by Random Network Distillation"
https://arxiv.org/abs/1810.12894
"""

from typing import Mapping, Tuple, Iterable, Text, Optional, NamedTuple
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
from deep_rl_zoo import multistep
from deep_rl_zoo import utils
from deep_rl_zoo import base
from deep_rl_zoo import normalizer

# torch.autograd.set_detect_anomaly(True)


class Transition(NamedTuple):
    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    logprob_a_t: Optional[float]
    ext_returns_t: Optional[float]
    int_returns_t: Optional[float]
    advantage_t: Optional[float]


TransitionStructure = Transition(
    s_t=None, a_t=None, logprob_a_t=None, ext_returns_t=None, int_returns_t=None, advantage_t=None
)


class Actor(types_lib.Agent):
    """PPO-RND actor"""

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        policy_network: torch.nn.Module,
        unroll_length: int,
        device: torch.device,
    ) -> None:
        """
        Args:
            rank: the rank for the actor.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            policy_network: the policy network for worker to make action choice.
            unroll_length: rollout length.
            device: PyTorch runtime device.
        """
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer greater than or equal to 1, got {unroll_length}')

        self.rank = rank
        self.agent_name = f'PPO-RND-actor{rank}'
        self._queue = data_queue
        self._policy_network = policy_network.to(device=device)
        self._device = device
        self._unroll_length = unroll_length
        self._unroll_sequence = []

        self._step_t = -1

        self._s_tm1 = None
        self._a_tm1 = None
        self._logprob_a_tm1 = None

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given current timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1

        a_t, logprob_a_t = self.act(timestep)

        if self._a_tm1 is not None:
            self._unroll_sequence.append(
                (
                    self._s_tm1,  # s_t
                    self._a_tm1,  # a_t
                    self._logprob_a_tm1,  # logprob_a_t
                    timestep.reward,  # r_t
                    timestep.observation,  # s_tp1
                    timestep.done,
                )
            )

            if len(self._unroll_sequence) == self._unroll_length:
                self._queue.put(self._unroll_sequence)
                self._unroll_sequence = []

        self._s_tm1 = timestep.observation
        self._a_tm1 = a_t
        self._logprob_a_tm1 = logprob_a_t

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode."""
        self._s_tm1 = None
        self._a_tm1 = None
        self._logprob_a_tm1 = None

    def act(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action]:
        'Given timestep, return an action.'
        return self._choose_action(timestep)

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action]:
        """Given timestep, choose action a_t"""
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        pi_logits_t = self._policy_network(s_t).pi_logits
        # Sample an action
        pi_dist_t = distributions.categorical_distribution(pi_logits_t)

        a_t = pi_dist_t.sample()
        logprob_a_t = pi_dist_t.log_prob(a_t)
        return a_t.cpu().item(), logprob_a_t.cpu().item()

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {}


class Learner(types_lib.Learner):
    """PPO-RND learner"""

    def __init__(
        self,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        rnd_target_network: nn.Module,
        rnd_predictor_network: nn.Module,
        observation_normalizer: normalizer.Normalizer,
        clip_epsilon: LinearSchedule,
        discount: float,
        rnd_discount: float,
        gae_lambda: float,
        total_unroll_length: int,
        batch_size: int,
        update_k: int,
        rnd_experience_proportion: float,
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
            rnd_target_network: RND target fixed random network.
            rnd_predictor_network: RND predictor network.
            observation_normalizer: observation normalizer, only for RND networks.
            clip_epsilon: external scheduler to decay clip epsilon.
            discount: the gamma discount for future rewards.
            rnd_discount: the gamma discount for future rewards for RND networks.
            gae_lambda: lambda for the GAE general advantage estimator.
            total_unroll_length: wait until collected this many transitions before update parameters.
            batch_size: sample batch_size of transitions.
            update_k: update k times when it's time to do learning.
            unroll_length: worker rollout horizon.
            rnd_experience_proportion: proportion of experience used for training RND predictor.
            entropy_coef: the coefficient of entropy loss.
            baseline_coef: the coefficient of state-value loss.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """

        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to in the range [1, 512], got {batch_size}')
        if not batch_size <= total_unroll_length:
            raise ValueError(
                f'Expect total_unroll_length to be integer greater than or equal to {batch_size}, got {total_unroll_length}'
            )
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k to be integer greater than or equal to 1, got {update_k}')
        if not 0.0 <= rnd_experience_proportion <= 10.0:
            raise ValueError(f'Expect rnd_experience_proportion to in the range [0.0, 10.0], got {rnd_experience_proportion}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to [0.0, 1.0], got {entropy_coef}')
        if not 0.0 <= baseline_coef <= 1.0:
            raise ValueError(f'Expect baseline_coef to [0.0, 1.0], got {baseline_coef}')

        self.agent_name = 'PPO-RND-learner'
        self._policy_network = policy_network.to(device=device)
        self._policy_optimizer = policy_optimizer
        self._rnd_predictor_network = rnd_predictor_network.to(device=device)
        self._rnd_target_network = rnd_target_network.to(device=device)
        # Disable autograd for RND target networks.
        for p in self._rnd_target_network.parameters():
            p.requires_grad = False
        self._device = device

        self._observation_normalizer = observation_normalizer

        # Original paper uses 25% of experience for training RND predictor
        rnd_loss_mask = torch.rand(batch_size).to(device=self._device)
        self._rnd_loss_mask = (rnd_loss_mask < rnd_experience_proportion).to(device=self._device, dtype=torch.float32)

        # Accumulate running statistics to calculate mean and std online,
        # this will also clip intrinsic reward values in the range [-10, 10]
        self._intrinsic_reward_normalizer = normalizer.Normalizer(eps=0.0001, clip_range=(-10, 10), device=self._device)

        self._storage = []
        self._batch_size = batch_size
        self._total_unroll_length = total_unroll_length
        self._update_k = update_k

        self._entropy_coef = entropy_coef
        self._baseline_coef = baseline_coef
        self._clip_epsilon = clip_epsilon

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm
        self._discount = discount
        self._rnd_discount = rnd_discount
        self._lambda = gae_lambda

        # Counters
        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._baseline_loss_t = np.nan
        self._entropy_loss_t = np.nan
        self._rnd_loss_t = np.nan

    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.

        Yields:
            learner statistics if network parameters update occurred, otherwise returns None.
        """
        self._step_t += 1

        if len(self._storage) < self._batch_size:
            return

        return self._learn()

    def reset(self) -> None:
        """Should be called at the beginning of every iteration."""
        self._storage = []

    def received_item_from_queue(self, unroll_sequences: Iterable[Tuple]) -> None:
        """Received item send by actors through multiprocessing queue."""

        # Unpack list of tuples into separate lists.
        s_t, a_t, logprob_a_t, r_t, s_tp1, done_tp1 = map(list, zip(*unroll_sequences))

        ext_returns_t, int_returns_t, advantage_t = self._compute_returns_and_advantages(s_t, r_t, s_tp1, done_tp1)

        # Zip multiple lists into list of tuples, only keep relevant data
        zipped_sequence = zip(s_t, a_t, logprob_a_t, ext_returns_t, int_returns_t, advantage_t)

        self._storage += zipped_sequence

    @torch.no_grad()
    def _compute_returns_and_advantages(
        self,
        s_t: Iterable[np.ndarray],
        r_t: Iterable[float],
        s_tp1: Iterable[np.ndarray],
        done_tp1: Iterable[bool],
    ):
        """Compute returns, GAE estimated advantages, and log probabilities for the given action a_t under s_t."""
        stacked_s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self._device, dtype=torch.float32)
        stacked_r_t = torch.from_numpy(np.stack(r_t, axis=0)).to(device=self._device, dtype=torch.float32)
        stacked_s_tp1 = torch.from_numpy(np.stack(s_tp1, axis=0)).to(device=self._device, dtype=torch.float32)
        stacked_done_tp1 = torch.from_numpy(np.stack(done_tp1, axis=0)).to(device=self._device, dtype=torch.bool)

        discount_tp1 = (~stacked_done_tp1).float() * self._discount
        rnd_discount_tp1 = (~stacked_done_tp1).float() * self._rnd_discount

        output_t = self._policy_network(stacked_s_t)

        ext_v_t = output_t.ext_baseline.squeeze(1)  # [batch_size]
        int_v_t = output_t.int_baseline.squeeze(1)  # [batch_size]

        output_tp1 = self._policy_network(stacked_s_tp1)

        ext_v_tp1 = output_tp1.ext_baseline.squeeze(-1)
        int_v_tp1 = output_tp1.int_baseline.squeeze(-1)

        # Compute intrinsic reward
        normalized_s_t = self._normalize_observation(stacked_s_t)  # [B, C, H, W] or [B, N]

        rnd_pred_s_t = self._rnd_predictor_network(normalized_s_t)
        rnd_target_s_t = self._rnd_target_network(normalized_s_t)

        abs_error = torch.sum(torch.abs(rnd_pred_s_t - rnd_target_s_t), dim=-1)  # Sums over latent features [batch_size,]

        # Normalize intrinsic reward
        intrinsic_reward_t = self._intrinsic_reward_normalizer(abs_error)

        # Calculate advantages using extrinsic rewards
        ext_advantage_t = multistep.truncated_generalized_advantage_estimation(
            stacked_r_t, ext_v_t, ext_v_tp1, discount_tp1, self._lambda
        )
        ext_returns_t = ext_advantage_t + ext_v_t

        # Calculate advantages using intrinsic rewards
        int_advantage_t = multistep.truncated_generalized_advantage_estimation(
            intrinsic_reward_t, int_v_t, int_v_tp1, rnd_discount_tp1, self._lambda
        )
        int_returns_t = int_advantage_t + int_v_t

        advantage_t = 2.0 * ext_advantage_t + 1.0 * int_advantage_t

        # Normalize advantages
        advantage_t = (advantage_t - advantage_t.mean()) / advantage_t.std()

        ext_returns_t = ext_returns_t.cpu().numpy()
        int_returns_t = int_returns_t.cpu().numpy()
        advantage_t = advantage_t.cpu().numpy()

        return (ext_returns_t, int_returns_t, advantage_t)

    def _learn(self) -> Iterable[Mapping[Text, float]]:
        # Run update for K times
        for _ in range(self._update_k):
            # For each update epoch, split indices into 'bins' with batch_size.
            binned_indices = utils.split_indices_into_bins(self._batch_size, len(self._storage), shuffle=True)
            for indices in binned_indices:
                transitions = [self._storage[i] for i in indices]

                # Stack list of transitions, follow our code convention.
                stacked_transition = replay_lib.np_stack_list_of_transitions(transitions, TransitionStructure)
                self._update(stacked_transition)
                yield self.statistics

        del self._storage[:]  # discard old samples after using it

    def _update(self, transitions: Transition) -> None:
        self._policy_optimizer.zero_grad()
        loss = self._calc_policy_loss(transitions=transitions)
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
        s_t = self._normalize_observation(s_t)  # [B, C, H, W] or [B, N]

        pred_s_t = self._rnd_predictor_network(s_t)
        with torch.no_grad():
            target_s_t = self._rnd_target_network(s_t)

        abs_error = torch.sum(torch.abs(pred_s_t - target_s_t), dim=-1)  # Sums over latent features [batch_size,]
        assert abs_error.shape == self._rnd_loss_mask.shape

        # Apply proportion mask, averaging over batch dimension
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

    def _calc_policy_loss(self, transitions: Transition) -> torch.Tensor:
        """Calculate loss for a batch transitions"""
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        a_t = torch.from_numpy(transitions.a_t).to(device=self._device, dtype=torch.int64)  # [batch_size]
        behavior_logprob_a_t = torch.from_numpy(transitions.logprob_a_t).to(
            device=self._device, dtype=torch.float32
        )  # [batch_size]
        ext_returns_t = torch.from_numpy(transitions.ext_returns_t).to(
            device=self._device, dtype=torch.float32
        )  # [batch_size]
        int_returns_t = torch.from_numpy(transitions.int_returns_t).to(
            device=self._device, dtype=torch.float32
        )  # [batch_size]
        advantage_t = torch.from_numpy(transitions.advantage_t).to(device=self._device, dtype=torch.float32)  # [batch_size]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_t, 1, torch.long)
        base.assert_rank_and_dtype(ext_returns_t, 1, torch.float32)
        base.assert_rank_and_dtype(int_returns_t, 1, torch.float32)
        base.assert_rank_and_dtype(advantage_t, 1, torch.float32)
        base.assert_rank_and_dtype(behavior_logprob_a_t, 1, torch.float32)

        # Compute intrinsic reward using RND module.
        rnd_loss, _ = self._calc_rnd_loss(s_t)

        # Get policy action logits and baseline for s_t.
        policy_output = self._policy_network(s_t)
        pi_logits_t = policy_output.pi_logits
        # Two baseline heads
        ext_v_t = policy_output.ext_baseline.squeeze(1)  # [batch_size]
        int_v_t = policy_output.int_baseline.squeeze(1)  # [batch_size]

        pi_dist_t = distributions.categorical_distribution(pi_logits_t)

        # Compute entropy loss.
        entropy_loss = pi_dist_t.entropy()

        # Compute clipped surrogate policy gradient loss.
        pi_logprob_a_t = pi_dist_t.log_prob(a_t)
        ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t)

        if ratio.shape != advantage_t.shape:
            raise RuntimeError(f'Expect ratio and advantage_t have same shape, got {ratio.shape} and {advantage_t.shape}')

        # Compute clipped surrogate policy gradient loss.
        policy_loss = rl.clipped_surrogate_gradient_loss(ratio, advantage_t, self.clip_epsilon).loss

        # Compute baseline state-value loss, combine extrinsic and intrinsic losses.
        ext_baseline_loss = rl.baseline_loss(ext_returns_t, ext_v_t).loss
        int_baseline_loss = rl.baseline_loss(int_returns_t, int_v_t).loss
        baseline_loss = ext_baseline_loss + int_baseline_loss

        # Averaging over batch dimension.
        policy_loss = torch.mean(policy_loss, dim=0)
        entropy_loss = self._entropy_coef * torch.mean(entropy_loss, dim=0)
        baseline_loss = self._baseline_coef * torch.mean(baseline_loss, dim=0)

        # Combine policy loss, baseline loss, entropy loss.
        # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
        loss = -(policy_loss + entropy_loss) + baseline_loss

        # Add RND predictor loss.
        loss += rnd_loss

        # For logging only.
        self._policy_loss_t = policy_loss.detach().cpu().item()
        self._baseline_loss_t = baseline_loss.detach().cpu().item()
        self._entropy_loss_t = entropy_loss.detach().cpu().item()
        self._rnd_loss_t = rnd_loss.detach().cpu().item()

        return loss

    def _normalize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        # Normalize observation using online incremental algorithm
        if len(observation.shape) > 2:  # shape of observation is [B, C, H, W]
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

    @property
    def clip_epsilon(self):
        """Call external clip epsilon scheduler"""
        return self._clip_epsilon(self._step_t)

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            # 'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            'policy_loss': self._policy_loss_t,
            'baseline_loss': self._baseline_loss_t,
            'entropy_loss': self._entropy_loss_t,
            'rnd_loss': self._rnd_loss_t,
            # 'discount': self._discount,
            # 'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
        }
