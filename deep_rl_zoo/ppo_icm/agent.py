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
"""PPO-ICM agent class.

Notice in this implementation we follow the following naming convention when referring to unroll sequence:
sₜ, aₜ, rₜ, sₜ₊₁, aₜ₊₁, rₜ₊₁, ...

From the paper "Curiosity-driven Exploration by Self-supervised Prediction"
https://arxiv.org/abs/1705.05363

From the paper "Proximal Policy Optimization Algorithms"
https://arxiv.org/abs/1707.06347.
"""
from typing import NamedTuple, Mapping, Tuple, Optional, Iterable, Text
import multiprocessing
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# pylint: disable=import-error
import deep_rl_zoo.replay as replay_lib
import deep_rl_zoo.types as types_lib
import deep_rl_zoo.policy_gradient as rl
from deep_rl_zoo.schedule import LinearSchedule
from deep_rl_zoo import utils
from deep_rl_zoo import distributions
from deep_rl_zoo import multistep
from deep_rl_zoo import base
from deep_rl_zoo import normalizer

# torch.autograd.set_detect_anomaly(True)


class IcmModuleOutput(NamedTuple):
    """ICM module output"""

    inverse_loss: Optional[torch.Tensor]
    forward_loss: Optional[torch.Tensor]
    intrinsic_reward: Optional[torch.Tensor]


class Transition(NamedTuple):
    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    logprob_a_t: Optional[float]
    returns_t: Optional[float]
    advantage_t: Optional[float]
    s_tp1: Optional[np.ndarray]


class Actor(types_lib.Agent):
    """PPO-ICM actor"""

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
            raise ValueError(f'Expect unroll_length to be integer geater than or equal to 1, got {unroll_length}')

        self.rank = rank
        self.agent_name = f'PPO-ICM-actor{rank}'
        self._queue = data_queue
        self._policy_network = policy_network.to(device=device)
        self._policy_network.eval()
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
    """PPO-ICM learner"""

    def __init__(
        self,
        policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        old_policy_network: nn.Module,
        icm_network: nn.Module,
        icm_optimizer: torch.optim.Optimizer,
        clip_epsilon: LinearSchedule,
        discount: float,
        gae_lambda: float,
        total_unroll_length: int,
        batch_size: int,
        update_k: int,
        intrinsic_lambda: float,
        icm_beta: float,
        policy_loss_coef: float,
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
            old_policy_network: the old policy network used for workers.
            icm_network: the ICM module network.
            icm_optimizer: the optimizer for ICM module network.
            clip_epsilon: external scheduler to decay clip epsilon.
            discount: the gamma discount for future rewards.
            gae_lambda: lambda for the GAE general advantage estimator.
            total_unroll_length: wait until collected this many transitions before update parameters.
            batch_size: sample batch_size of transitions.
            update_k: update k times when it's time to do learning.
            unroll_length: worker rollout horizon.
            intrinsic_lambda: scaling facotr for intrinsic reward when calculate using equaltion 6.
            icm_beta: weights inverse model loss against the forward model loss.
            policy_loss_coef: weights policy loss against the importance of learning the intrinsic reward.
            entropy_coef: the coefficient of entryopy loss.
            baseline_coef: the coefficient of state-value loss.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to in the range [0.0, 1.0], got {discount}')
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k to be integer geater than or equal to 1, got {update_k}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to in the range [1, 512], got {batch_size}')
        if not batch_size <= total_unroll_length:
            raise ValueError(
                f'Expect total_unroll_length to be integer geater than or equal to {batch_size}, got {total_unroll_length}'
            )
        if not 0.0 <= intrinsic_lambda:
            raise ValueError(f'Expect intrinsic_lambda to be greater than or equal to 0.0, got {intrinsic_lambda}')
        if not 0.0 <= icm_beta <= 1.0:
            raise ValueError(f'Expect icm_beta to in the range [0.0, 1.0], got {icm_beta}')
        if not 0.0 <= policy_loss_coef <= 1.0:
            raise ValueError(f'Expect policy_loss_coef to in the range [0.0, 1.0], got {policy_loss_coef}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to [0.0, 1.0], got {entropy_coef}')
        if not 0.0 <= baseline_coef <= 1.0:
            raise ValueError(f'Expect baseline_coef to [0.0, 1.0], got {baseline_coef}')

        self.agent_name = 'PPO-ICM-learner'
        self._policy_network = policy_network.to(device=device)
        self._old_policy_network = old_policy_network.to(device=device)
        self._policy_optimizer = policy_optimizer
        self._icm_network = icm_network.to(device=device)
        self._icm_optimizer = icm_optimizer
        self._device = device

        self._update_old_policy()

        # Acummulate running statistics to calcualte mean and std online,
        # this will also clip intrinsic reward values in the range [-10, 10]
        self._intrinsic_reward_normalizer = normalizer.Normalizer(eps=0.0001, clip_range=(-10, 10), device=self._device)

        self._intrinsic_lambda = intrinsic_lambda
        self._icm_beta = icm_beta
        self._policy_loss_coef = policy_loss_coef

        self._storage = []
        self._total_unroll_length = total_unroll_length
        self._batch_size = batch_size
        self._update_k = update_k

        self._entropy_coef = entropy_coef
        self._baseline_coef = baseline_coef
        self._clip_epsilon = clip_epsilon

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm
        self._discount = discount
        self._lambda = gae_lambda

        # Counters
        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._baseline_loss_t = np.nan
        self._entropy_loss_t = np.nan
        self._icm_inverse_loss_t = np.nan
        self._icm_forward_loss_t = np.nan

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
        """Should be called at the begining of every iteration."""
        self._storage = []

    def received_item_from_queue(self, unroll_sequences: Iterable[Tuple]) -> None:
        """Received item send by actors through multiprocessing queue."""

        # Unpack list of tuples into seperate lists.
        s_t, a_t, logprob_a_t, r_t, s_tp1, done_tp1 = map(list, zip(*unroll_sequences))

        returns_t, advantage_t = self._compute_returns_and_advantages(s_t, r_t, s_tp1, done_tp1)

        # Zip multiple lists into list of tuples, only keep relevant data
        zipped_sequence = zip(s_t, a_t, logprob_a_t, returns_t, advantage_t, s_tp1)

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

        # Get output from old policy
        output_t = self._old_policy_network(stacked_s_t)
        v_t = output_t.baseline.squeeze(-1)

        v_tp1 = self._old_policy_network(stacked_s_tp1).baseline.squeeze(-1)
        advantage_t = multistep.truncated_generalized_advantage_estimation(stacked_r_t, v_t, v_tp1, discount_tp1, self._lambda)

        returns_t = advantage_t + v_t

        # Normalize advantages
        advantage_t = (advantage_t - advantage_t.mean()) / advantage_t.std()

        returns_t = returns_t.cpu().numpy()
        advantage_t = advantage_t.cpu().numpy()

        return (returns_t, advantage_t)

    def _learn(self) -> Iterable[Mapping[Text, float]]:

        # Run update for K times
        for _ in range(self._update_k):
            # For each update epoch, split indices into 'bins' with batch_size.
            bined_indices = utils.split_indices_into_bins(self._batch_size, len(self._storage), shuffle=True)
            for indices in bined_indices:
                transitions = [self._storage[i] for i in indices]

                # Stack list of transitions, follow our code convention.
                s_t, a_t, logprob_a_t, returns_t, advantage_t, s_tp1 = map(list, zip(*transitions))
                stacked_transitions = Transition(
                    s_t=np.stack(s_t, axis=0),
                    a_t=np.stack(a_t, axis=0),
                    logprob_a_t=np.stack(logprob_a_t, axis=0),
                    returns_t=np.stack(returns_t, axis=0),
                    advantage_t=np.stack(advantage_t, axis=0),
                    s_tp1=np.stack(s_tp1, axis=0),
                )

                icm_output = self._update_icm(stacked_transitions)
                self._update_policy(stacked_transitions, icm_output)
                self._update_t += 1
                yield self.statistics

        self._storage = []  # discard old samples after using it
        self._update_old_policy()

    def _update_icm(self, transitions: Transition) -> None:
        self._icm_optimizer.zero_grad()
        loss, icm_output = self._calc_icm_loss(transitions=transitions)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._icm_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._icm_optimizer.step()

        return icm_output

    def _update_policy(self, transitions: Transition, icm_output: IcmModuleOutput) -> None:
        self._policy_optimizer.zero_grad()
        loss = self._calc_policy_loss(transitions, icm_output)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._policy_network.parameters(),
                max_norm=self._max_grad_norm,
                error_if_nonfinite=True,
            )

        self._policy_optimizer.step()

    def _calc_icm_loss(self, transitions: Transition) -> Tuple[torch.Tensor, IcmModuleOutput]:
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        a_t = torch.from_numpy(transitions.a_t).to(device=self._device, dtype=torch.int64)  # [batch_size]
        s_tp1 = torch.from_numpy(transitions.s_tp1).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(s_tp1, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_t, 1, torch.long)

        # Calculate ICM module inverse model and forward model losses
        icm_output = self._icm_network(s_t, a_t, s_tp1)
        pred_features_t = icm_output.pred_features
        features_t = icm_output.features
        pred_pi_logits_a_t = icm_output.pi_logits

        # Using F.cross_entropy with input logits, target with class indices
        inverse_losses = F.cross_entropy(pred_pi_logits_a_t, a_t, reduction='none')  # eq 3, [batch_size,]

        # Calculate forward losses, TODO investigate the impact on performance with and without detach features_t
        forward_losses = torch.mean(0.5 * torch.square(pred_features_t - features_t), dim=1)  # eq 5, [batch_size,]

        if len(forward_losses.shape) != 1:
            raise RuntimeError(f'Expect forward_losses tensor to be a vector, got {forward_losses.shape}')

        # Compute intrinsic reward
        intrinsic_reward = self._intrinsic_lambda * forward_losses.clone().detach()  # eq 6, [batch_size,]

        # Update intrinsic reward normalization statistics
        self._intrinsic_reward_normalizer.update(intrinsic_reward)
        # Normalize intrinsic_reward
        intrinsic_reward = self._intrinsic_reward_normalizer(intrinsic_reward)

        inverse_loss = inverse_losses.mean()
        forward_loss = forward_losses.mean()

        # Calculate intrinsic reward from ICM module forward prediction error
        if len(inverse_loss.shape) != 0 or len(forward_loss.shape) != 0:
            raise RuntimeError(
                f'Expect inverse_loss tensor and forward_loss tensor to be a scalar, got {inverse_loss.shape} and {forward_loss.shape}'
            )

        # Combine losses from inverse model and forward model
        icm_loss = inverse_loss + forward_loss

        if len(icm_loss.shape) != 0:
            raise RuntimeError(f'Expect icm_loss tensor to be a scalar, got {icm_loss.shape}')
        if len(intrinsic_reward.shape) != 1:
            raise RuntimeError(f'Expect intrinsic_reward tensor to be a vector, got {intrinsic_reward.shape}')

        return icm_loss, IcmModuleOutput(
            inverse_loss=inverse_loss.detach(), forward_loss=forward_loss.detach(), intrinsic_reward=intrinsic_reward
        )

    def _calc_policy_loss(self, transitions: Transition, icm_output: IcmModuleOutput) -> torch.Tensor:
        """Calculate loss for a batch transitions"""

        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        a_t = torch.from_numpy(transitions.a_t).to(device=self._device, dtype=torch.int64)  # [batch_size]
        behavior_logprob_a_t = torch.from_numpy(transitions.logprob_a_t).to(
            device=self._device, dtype=torch.float32
        )  # [batch_size, num_actions]
        returns_t = torch.from_numpy(transitions.returns_t).to(device=self._device, dtype=torch.float32)  # [batch_size]
        advantage_t = torch.from_numpy(transitions.advantage_t).to(device=self._device, dtype=torch.float32)  # [batch_size]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_t, 1, torch.long)
        base.assert_rank_and_dtype(returns_t, 1, torch.float32)
        base.assert_rank_and_dtype(advantage_t, 1, torch.float32)
        base.assert_rank_and_dtype(behavior_logprob_a_t, 1, torch.float32)

        # ICM module inverse and forward model losses, and intrinsic reward
        icm_inverse_loss = icm_output.inverse_loss
        icm_forward_loss = icm_output.forward_loss
        icm_intrinsic_reward = icm_output.intrinsic_reward

        # Make sure not mixing up ICM module gradients into policy network
        if icm_inverse_loss.requires_grad or icm_forward_loss.requires_grad or icm_intrinsic_reward.requires_grad:
            raise RuntimeError('Expect tensors from icm moduel do not require gradients')

        # Get policy action logits and baseline for s_tm1.
        policy_output = self._policy_network(s_t)
        pi_logits_t = policy_output.pi_logits
        v_t = policy_output.baseline.squeeze(-1)  # [batch_size]

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

        # Compute baseline state-value loss.
        baseline_loss = rl.baseline_loss(returns_t - v_t).loss

        # Averaging over batch dimension.
        policy_loss = torch.mean(policy_loss, dim=0)
        entropy_loss = self._entropy_coef * torch.mean(entropy_loss, dim=0)
        baseline_loss = self._baseline_coef * torch.mean(baseline_loss, dim=0)

        # Combine policy loss, baseline loss, entropy loss.
        # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
        loss = -(policy_loss + entropy_loss) + baseline_loss

        # Re-weight policy loss, add ICM module inverse model loss, forward model loss.
        loss = self._policy_loss_coef * loss + (1.0 - self._icm_beta) * icm_inverse_loss + self._icm_beta * icm_forward_loss

        # For logging only.
        self._policy_loss_t = policy_loss.detach().cpu().item()
        self._baseline_loss_t = baseline_loss.detach().cpu().item()
        self._entropy_loss_t = entropy_loss.detach().cpu().item()
        self._icm_inverse_loss_t = icm_inverse_loss.detach().cpu().item()
        self._icm_forward_loss_t = icm_forward_loss.detach().cpu().item()

        return loss

    def _update_old_policy(self):
        self._old_policy_network.load_state_dict(self._policy_network.state_dict())

    @property
    def clip_epsilon(self):
        """Call external clip epsilon scheduler"""
        return self._clip_epsilon(self._step_t)

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            'icm_learning_rate': self._icm_optimizer.param_groups[0]['lr'],
            'policy_loss': self._policy_loss_t,
            'baseline_loss': self._baseline_loss_t,
            'entropy_loss': self._entropy_loss_t,
            'icm_inverse_loss': self._icm_inverse_loss_t,
            'icm_forward_loss': self._icm_forward_loss_t,
            'discount': self._discount,
            'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
        }
