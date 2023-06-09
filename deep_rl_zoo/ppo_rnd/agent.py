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

from typing import Mapping, Tuple, Iterable, Text
import multiprocessing
import numpy as np
import torch
from torch import nn

# pylint: disable=import-error
import deep_rl_zoo.types as types_lib
from deep_rl_zoo.schedule import LinearSchedule
import deep_rl_zoo.policy_gradient as rl
from deep_rl_zoo import distributions
from deep_rl_zoo import multistep
from deep_rl_zoo import utils
from deep_rl_zoo import base
from deep_rl_zoo import normalizer

torch.autograd.set_detect_anomaly(True)


class Actor(types_lib.Agent):
    """PPO-RND actor"""

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        policy_network: torch.nn.Module,
        unroll_length: int,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        """
        Args:
            rank: the rank for the actor.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            policy_network: the policy network for worker to make action choice.
            unroll_length: rollout length.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer greater than or equal to 1, got {unroll_length}')

        self.rank = rank
        self.agent_name = f'PPO-RND-actor{rank}'
        self._queue = data_queue
        self._policy_network = policy_network.to(device=device)
        # Disable autograd for actor networks.
        for p in self._policy_network.parameters():
            p.requires_grad = False

        self._device = device

        self._shared_params = shared_params

        self._unroll_length = unroll_length
        self._unroll_sequence = []

        self._step_t = -1

        self._s_tm1 = None
        self._a_tm1 = None
        self._logprob_a_tm1 = None

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given current timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1

        a_t, logprob_a_t, ext_value, int_value = self.act(timestep)

        if self._a_tm1 is not None:
            self._unroll_sequence.append(
                (
                    self._s_tm1,  # s_t
                    self._a_tm1,  # a_t
                    self._logprob_a_tm1,  # logprob_a_t
                    ext_value,
                    int_value,
                    timestep.reward,  # r_t
                    timestep.done,
                )
            )

            if len(self._unroll_sequence) == self._unroll_length:
                self._queue.put(self._unroll_sequence)
                self._unroll_sequence = []
                self._update_actor_network()

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

    def _update_actor_network(self):
        state_dict = self._shared_params['policy_network']
        if state_dict is not None:
            if self._device != 'cpu':
                state_dict = {k: v.to(device=self._device) for k, v in state_dict.items()}
            self._policy_network.load_state_dict(state_dict)

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action, float, float, float]:
        """Given timestep, choose action a_t"""
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        output = self._policy_network(s_t)
        pi_logits_t = output.pi_logits
        # Sample an action
        pi_dist_t = distributions.categorical_distribution(pi_logits_t)

        a_t = pi_dist_t.sample()
        logprob_a_t = pi_dist_t.log_prob(a_t)

        ext_value, int_value = output.ext_baseline, output.int_baseline
        return (
            a_t.cpu().item(),
            logprob_a_t.cpu().item(),
            ext_value.squeeze(0).cpu().item(),
            int_value.squeeze(0).cpu().item(),
        )

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
        rnd_optimizer: torch.optim.Optimizer,
        rnd_obs_clip: float,
        clip_epsilon: LinearSchedule,
        ext_discount: float,
        int_discount: float,
        gae_lambda: float,
        total_unroll_length: int,
        update_k: int,
        rnd_experience_proportion: float,
        entropy_coef: float,
        value_coef: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        """
        Args:
            policy_network: the policy network we want to train.
            policy_optimizer: the optimizer for policy network.
            rnd_target_network: RND target fixed random network.
            rnd_predictor_network: RND predictor network.
            rnd_optimizer: the optimizer for RND predictor network.
            rnd_obs_clip: clip norm of the RND observation.
            clip_epsilon: external scheduler to decay clip epsilon.
            discount: the gamma discount for future rewards.
            int_discount: the gamma discount for future rewards for RND networks.
            gae_lambda: lambda for the GAE general advantage estimator.
            total_unroll_length: wait until collects this samples before update networks, computed as num_actors x rollout_length.
            update_k: update k times when it's time to do learning.
            rnd_experience_proportion: proportion of experience used for training RND predictor.
            entropy_coef: the coefficient of entropy loss.
            value_coef: the coefficient of state-value loss.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
           shared_params: a shared dict, so we can later update the parameters for actors.
        """

        if not 1 <= total_unroll_length:
            raise ValueError(f'Expect total_unroll_length to be greater than 1, got {total_unroll_length}')
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k to be integer greater than or equal to 1, got {update_k}')
        if not 0.0 <= rnd_experience_proportion <= 1.0:
            raise ValueError(f'Expect rnd_experience_proportion to in the range [0.0, 1.0], got {rnd_experience_proportion}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to [0.0, 1.0], got {entropy_coef}')
        if not 0.0 <= value_coef <= 1.0:
            raise ValueError(f'Expect value_coef to [0.0, 1.0], got {value_coef}')

        self.agent_name = 'PPO-RND-learner'
        self._policy_network = policy_network.to(device=device)
        self._policy_optimizer = policy_optimizer
        self._rnd_predictor_network = rnd_predictor_network.to(device=device)
        self._rnd_target_network = rnd_target_network.to(device=device)

        self._rnd_optimizer = rnd_optimizer
        # Disable autograd for RND target networks.
        for p in self._rnd_target_network.parameters():
            p.requires_grad = False

        self._device = device

        self._shared_params = shared_params

        self._rnd_experience_proportion = rnd_experience_proportion
        self._rnd_obs_clip = rnd_obs_clip

        # Accumulate running statistics to calculate mean and std
        self._int_reward_normalizer = normalizer.RunningMeanStd(shape=(1,))
        self._rnd_obs_normalizer = normalizer.TorchRunningMeanStd(shape=(1, 84, 84), device=self._device)

        self._storage = []
        self._total_unroll_length = total_unroll_length

        # For each update epoch, try best to process all samples in 4 batches
        self._batch_size = min(512, int(np.ceil(total_unroll_length / 4).item()))

        self._update_k = update_k

        self._entropy_coef = entropy_coef
        self._value_coef = value_coef
        self._clip_epsilon = clip_epsilon

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm
        self._ext_discount = ext_discount
        self._int_discount = int_discount
        self._gae_lambda = gae_lambda

        # Counters
        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._value_loss_t = np.nan
        self._entropy_loss_t = np.nan
        self._rnd_loss_t = np.nan

    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.

        Yields:
            learner statistics if network parameters update occurred, otherwise returns None.
        """
        self._step_t += 1

        if len(self._storage) < self._total_unroll_length:
            return

        return self._learn()

    def reset(self) -> None:
        """Should be called at the beginning of every iteration."""
        self._storage = []

    def received_item_from_queue(self, unroll_sequences: Iterable[Tuple]) -> None:
        """Received item send by actors through multiprocessing queue."""

        # Unpack list of tuples into separate lists.
        observations, actions, logprob_actions, ext_values, int_values, rewards, dones = map(list, zip(*unroll_sequences))

        s_t = observations[:-1]
        a_t = actions[:-1]
        logprob_a_t = logprob_actions[:-1]
        ext_v_t = ext_values[:-1]
        ext_r_t = rewards[1:]

        ext_v_tp1 = ext_values[1:]
        done_tp1 = dones[1:]

        int_v_t = int_values[:-1]
        int_v_tp1 = int_values[1:]

        # Compute extrinsic returns and advantages
        (ext_return_t, ext_advantage_t) = self._compute_returns_and_advantages(
            ext_v_t, ext_r_t, ext_v_tp1, done_tp1, self._ext_discount
        )

        # Get observation for RND, note we only need last frame
        rnd_s_t = [s[-1:, ...] for s in s_t]

        # Compute intrinsic rewards
        int_r_t = self._compute_int_reward(rnd_s_t)

        # Compute intrinsic returns and advantages
        (int_return_t, int_advantage_t) = self._compute_returns_and_advantages(
            int_v_t,
            int_r_t,
            int_v_tp1,
            np.zeros_like(done_tp1),  # No dones for intrinsic reward.
            self._int_discount,
        )

        # Zip multiple lists into list of tuples
        zipped_sequence = list(
            zip(s_t, a_t, logprob_a_t, ext_return_t, ext_advantage_t, rnd_s_t, int_return_t, int_advantage_t)
        )

        self._storage += zipped_sequence

    def get_policy_state_dict(self):
        # To keep things consistent, we move the parameters to CPU
        return {k: v.cpu() for k, v in self._policy_network.state_dict().items()}

    def init_rnd_obs_stats(self, rnd_obs_list):
        self._normalize_rnd_obs(rnd_obs_list, True)

    def _learn(self) -> Iterable[Mapping[Text, float]]:
        num_samples = len(self._storage)

        # Go over the samples for K epochs
        for i in range(self._update_k):
            # For each update epoch, split indices into 'bins' with batch_size.
            binned_indices = utils.split_indices_into_bins(self._batch_size, num_samples, shuffle=True)
            for indices in binned_indices:
                mini_batch = [self._storage[i] for i in indices]

                self._update_policy_network(mini_batch)
                self._update_rnd_predictor_network(mini_batch)

                self._update_t += 1
                yield self.statistics

        self._shared_params['policy_network'] = self.get_policy_state_dict()

        del self._storage[:]  # discard old samples after using it

    def _update_rnd_predictor_network(self, samples):
        self._rnd_optimizer.zero_grad()

        # Unpack list of tuples into separate lists
        _, _, _, _, _, rnd_s_t, _, _ = map(list, zip(*samples))

        normed_s_t = self._normalize_rnd_obs(rnd_s_t, True)
        normed_s_t = normed_s_t.to(device=self._device, dtype=torch.float32)

        pred_t = self._rnd_predictor_network(normed_s_t)
        with torch.no_grad():
            target_t = self._rnd_target_network(normed_s_t)

        rnd_loss = torch.square(pred_t - target_t).mean(dim=1)

        # Proportion of experience used for train RND predictor
        if self._rnd_experience_proportion < 1:
            mask = torch.rand(rnd_loss.size())
            mask = torch.where(mask < self._rnd_experience_proportion, 1.0, 0.0).to(device=self._device, dtype=torch.float32)
            rnd_loss = rnd_loss * mask

        # Averaging over batch dimension
        rnd_loss = torch.mean(rnd_loss)

        # Compute gradients
        rnd_loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._rnd_predictor_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        # Update parameters
        self._rnd_optimizer.step()

        # Logging
        self._rnd_loss_t = rnd_loss.detach().cpu().item()

    def _update_policy_network(self, mini_batch):
        self._policy_optimizer.zero_grad()

        # Unpack list of tuples into separate lists
        (s_t, a_t, logprob_a_t, ext_return_t, ext_advantage_t, _, int_return_t, int_advantage_t) = map(list, zip(*mini_batch))

        s_t = torch.from_numpy(np.stack(s_t, axis=0)).to(device=self._device, dtype=torch.float32)
        a_t = torch.from_numpy(np.stack(a_t, axis=0)).to(device=self._device, dtype=torch.int64)
        behavior_logprob_a_t = torch.from_numpy(np.stack(logprob_a_t, axis=0)).to(device=self._device, dtype=torch.float32)
        ext_return_t = torch.from_numpy(np.stack(ext_return_t, axis=0)).to(device=self._device, dtype=torch.float32)
        ext_advantage_t = torch.from_numpy(np.stack(ext_advantage_t, axis=0)).to(device=self._device, dtype=torch.float32)
        int_return_t = torch.from_numpy(np.stack(int_return_t, axis=0)).to(device=self._device, dtype=torch.float32)
        int_advantage_t = torch.from_numpy(np.stack(int_advantage_t, axis=0)).to(device=self._device, dtype=torch.float32)

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_t, 1, torch.long)
        base.assert_rank_and_dtype(ext_return_t, 1, torch.float32)
        base.assert_rank_and_dtype(ext_advantage_t, 1, torch.float32)
        base.assert_rank_and_dtype(int_return_t, 1, torch.float32)
        base.assert_rank_and_dtype(int_advantage_t, 1, torch.float32)
        base.assert_rank_and_dtype(behavior_logprob_a_t, 1, torch.float32)

        pi_logits_t, ext_v_t, int_v_t = self._policy_network(s_t)

        pi_dist_t = distributions.categorical_distribution(pi_logits_t)
        pi_logprob_a_t = pi_dist_t.log_prob(a_t)
        entropy_loss = pi_dist_t.entropy()

        # Combine extrinsic and intrinsic advantages together
        advantage_t = 2.0 * ext_advantage_t + 1.0 * int_advantage_t

        ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t)

        if ratio.shape != advantage_t.shape:
            raise RuntimeError(f'Expect ratio and advantages have same shape, got {ratio.shape} and {advantage_t.shape}')
        policy_loss = rl.clipped_surrogate_gradient_loss(ratio, advantage_t, self.clip_epsilon).loss

        ext_v_loss = rl.value_loss(ext_return_t, ext_v_t.squeeze(-1)).loss
        int_v_loss = rl.value_loss(int_return_t, int_v_t.squeeze(-1)).loss

        value_loss = ext_v_loss + int_v_loss

        # Averaging over batch dimension
        policy_loss = torch.mean(policy_loss)
        entropy_loss = torch.mean(entropy_loss)
        value_loss = torch.mean(value_loss)

        # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
        loss = -(policy_loss + self._entropy_coef * entropy_loss) + self._value_coef * value_loss

        # Compute gradients
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        # Update parameters
        self._policy_optimizer.step()

        # Logging
        self._policy_loss_t = policy_loss.detach().cpu().item()
        self._value_loss_t = value_loss.detach().cpu().item()
        self._entropy_loss_t = entropy_loss.detach().cpu().item()

    @torch.no_grad()
    def _compute_int_reward(self, rnd_s_t):
        normed_s_t = self._normalize_rnd_obs(rnd_s_t)

        normed_s_t = normed_s_t.to(device=self._device, dtype=torch.float32)

        pred = self._rnd_predictor_network(normed_s_t)
        target = self._rnd_target_network(normed_s_t)

        int_r_t = torch.square(pred - target).mean(dim=1).detach().cpu().numpy()

        # Normalize intrinsic reward
        normed_int_r_t = self._normalize_int_rewards(int_r_t)

        return normed_int_r_t

    @torch.no_grad()
    def _normalize_rnd_obs(self, rnd_obs_list, update_stats=False):
        # GPU could be much faster
        tacked_obs = torch.from_numpy(np.stack(rnd_obs_list, axis=0)).to(device=self._device, dtype=torch.float32)

        normed_obs = self._rnd_obs_normalizer.normalize(tacked_obs)
        normed_obs = normed_obs.clamp(-self._rnd_obs_clip, self._rnd_obs_clip)

        if update_stats:
            self._rnd_obs_normalizer.update(tacked_obs)

        return normed_obs

    @torch.no_grad()
    def _compute_returns_and_advantages(
        self,
        v_t: Iterable[np.ndarray],
        r_t: Iterable[float],
        v_tp1: Iterable[np.ndarray],
        done_tp1: Iterable[bool],
        discount: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute returns, GAE estimated advantages"""

        v_t = torch.from_numpy(np.stack(v_t, axis=0)).to(device=self._device, dtype=torch.float32)
        r_t = torch.from_numpy(np.stack(r_t, axis=0)).to(device=self._device, dtype=torch.float32)
        v_tp1 = torch.from_numpy(np.stack(v_tp1, axis=0)).to(device=self._device, dtype=torch.float32)
        done_tp1 = torch.from_numpy(np.stack(done_tp1, axis=0)).to(device=self._device, dtype=torch.bool)

        discount_tp1 = (~done_tp1).float() * discount

        advantage_t = multistep.truncated_generalized_advantage_estimation(r_t, v_t, v_tp1, discount_tp1, self._gae_lambda)

        return_t = advantage_t + v_t

        # Normalize seems to hurt performance on MontezumaRevenge
        # advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-8)

        advantage_t = advantage_t.cpu().numpy()
        return_t = return_t.cpu().numpy()

        return (return_t, advantage_t)

    def _normalize_int_rewards(self, int_rewards):
        """Compute returns then normalize the intrinsic reward based on these returns"""

        # From https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/ppo_agent.py#L257
        intrinsic_returns = []
        rewems = 0
        for t in reversed(range(len(int_rewards))):
            rewems = rewems * self._int_discount + int_rewards[t]
            intrinsic_returns.insert(0, rewems)
        self._int_reward_normalizer.update(np.ravel(intrinsic_returns).reshape(-1, 1))

        normed_int_rewards = int_rewards / np.sqrt(self._int_reward_normalizer.var + 1e-8)

        return normed_int_rewards.tolist()

    @property
    def clip_epsilon(self):
        """Call external clip epsilon scheduler"""
        return self._clip_epsilon(self._step_t)

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            # 'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            # 'discount': self._discount,
            'policy_loss': self._policy_loss_t,
            'value_loss': self._value_loss_t,
            'entropy_loss': self._entropy_loss_t,
            'rnd_loss': self._rnd_loss_t,
            'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
        }
