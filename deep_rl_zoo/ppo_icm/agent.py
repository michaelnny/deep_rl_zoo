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

From the paper "Curiosity-driven Exploration by Self-supervised Prediction"
https://arxiv.org/abs/1705.05363

From the paper "Proximal Policy Optimization Algorithms"
https://arxiv.org/abs/1707.06347.
"""
from typing import NamedTuple, Mapping, Tuple, Optional, Text
import itertools
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
from deep_rl_zoo import distributions
from deep_rl_zoo import base
from deep_rl_zoo import normalizer

# torch.autograd.set_detect_anomaly(True)


class IcmModuleOutput(NamedTuple):
    """ICM module output"""

    inverse_loss: Optional[torch.Tensor]
    forward_loss: Optional[torch.Tensor]
    intrinsic_reward: Optional[torch.Tensor]


class Actor(types_lib.Agent):
    """PPO-ICM actor"""

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
        self.agent_name = f'PPO-ICM-actor{rank}'
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
        n_step: int,
        batch_size: int,
        update_k: int,
        unroll_length: int,
        intrinsic_eta: float,
        icm_beta: float,
        policy_lambda: float,
        extrinsic_reward_coef: float,
        intrinsic_reward_coef: float,
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
            n_step: TD n-step returns.
            batch_size: sample batch_size of transitions.
            update_k: update k times when it's time to do learning.
            unroll_length: worker rollout horizon.
            intrinsic_eta: scaling facotr for intrinsic reward when calculate using equaltion 6.
            icm_beta: weights inverse model loss against the forward model loss.
            policy_lambda: weights policy loss against the importance of learning the intrinsic reward.
            extrinsic_reward_coef: weights intrinsic reward from environment.
            intrinsic_reward_coef: weights intrinsic reward from ICM module.
            entropy_coef: the coefficient of entryopy loss.
            baseline_coef: the coefficient of state-value loss.
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
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k to be integer geater than or equal to 1, got {update_k}')
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer geater than or equal to 1, got {unroll_length}')
        if not 0.0 <= intrinsic_eta:
            raise ValueError(f'Expect intrinsic_eta to be greater than or equal to 0.0, got {intrinsic_eta}')
        if not 0.0 <= icm_beta <= 1.0:
            raise ValueError(f'Expect icm_beta to in the range [0.0, 1.0], got {icm_beta}')
        if not 0.0 <= policy_lambda <= 1.0:
            raise ValueError(f'Expect policy_lambda to in the range [0.0, 1.0], got {policy_lambda}')
        if not 0.0 <= extrinsic_reward_coef <= 10.0:
            raise ValueError(f'Expect extrinsic_reward_coef to in the range [0.0, 10.0], got {extrinsic_reward_coef}')
        if not 0.0 <= intrinsic_reward_coef <= 10.0:
            raise ValueError(f'Expect intrinsic_reward_coef to in the range [0.0, 10.0], got {intrinsic_reward_coef}')
        if not 0.0 < entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to (0.0, 1.0], got {entropy_coef}')
        if not 0.0 < baseline_coef <= 1.0:
            raise ValueError(f'Expect baseline_coef to (0.0, 1.0], got {baseline_coef}')

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

        self._eta = intrinsic_eta
        self._beta = icm_beta
        self._lambda = policy_lambda
        self._int_reward_coef = intrinsic_reward_coef
        self._ext_reward_coef = extrinsic_reward_coef

        self._storage = []
        self._discount = discount
        self._n_step = n_step
        self._batch_size = batch_size
        self._unroll_length = unroll_length
        self._update_k = update_k

        self._entropy_coef = entropy_coef
        self._baseline_coef = baseline_coef
        self._clip_epsilon = clip_epsilon

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        # Counters
        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._icm_loss_t = np.nan

    def step(self) -> Mapping[Text, float]:
        """Increment learner step, and potentially do a update when called.

        Returns:
            learner statistics if network parameters update occurred, otherwise returns None.
        """
        self._step_t += 1

        if len(self._storage) < self._batch_size:
            return None

        self._learn()
        return self.statistics

    def reset(self) -> None:
        """Should be called at the begining of every iteration."""
        self._storage = []

    def received_item_from_queue(self, item) -> None:
        """Received item send by actors through multiprocessing queue."""
        self._storage.append(item)  # This will store a list (length unroll_length) of  transitions.

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
                icm_output = self._update_icm(stacked_transition)
                self._update_policy(stacked_transition, icm_output)
        self._storage = []  # discard old samples after using it
        self._update_old_policy()

    def _update_icm(self, transitions: replay_lib.OffPolicyTransition) -> IcmModuleOutput:
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

        # For logging only.
        self._icm_loss_t = loss.detach().cpu().item()

        return icm_output

    def _update_policy(self, transitions: replay_lib.OffPolicyTransition, icm_output: IcmModuleOutput) -> None:
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
        self._update_t += 1

        # For logging only.
        self._policy_loss_t = loss.detach().cpu().item()

    def _calc_icm_loss(self, transitions: replay_lib.OffPolicyTransition) -> Tuple[torch.Tensor, IcmModuleOutput]:
        s_tm1 = torch.from_numpy(transitions.s_tm1).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        a_tm1 = torch.from_numpy(transitions.a_tm1).to(device=self._device, dtype=torch.int64)  # [batch_size]
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_tm1, (2, 4), torch.float32)
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_tm1, 1, torch.long)

        # Calculate ICM module inverse model and forward model losses
        icm_output = self._icm_network(s_tm1, a_tm1, s_t)
        pred_features_t = icm_output.pred_features
        features_t = icm_output.features
        pred_pi_logits_a_tm1 = icm_output.pi_logits

        # Using F.cross_entropy with input logits, target with class indices
        inverse_losses = F.cross_entropy(pred_pi_logits_a_tm1, a_tm1, reduction='none')  # eq 3, (batch_size, )

        # Calculate forward losses, TODO investigate the impact on performance with and without detach features_t
        forward_losses = torch.mean(0.5 * torch.square(pred_features_t - features_t), dim=1)  # eq 5, (batch_size, )

        if len(forward_losses.shape) != 1:
            raise RuntimeError(f'Expect forward_losses tensor to be a vector, got {forward_losses.shape}')

        # Compute intrinsic reward
        intrinsic_reward = self._eta * forward_losses.detach()  # eq 6, (batch_size, )

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

    def _calc_policy_loss(self, transitions: replay_lib.OffPolicyTransition, icm_output: IcmModuleOutput) -> torch.Tensor:
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

        # ICM module inverse and forward model losses, and intrinsic reward
        icm_inverse_loss = icm_output.inverse_loss
        icm_forward_loss = icm_output.forward_loss
        icm_intrinsic_reward = icm_output.intrinsic_reward

        # Make sure not mixing up ICM module gradients into policy network
        if icm_inverse_loss.requires_grad or icm_forward_loss.requires_grad or icm_intrinsic_reward.requires_grad:
            raise RuntimeError('Expect tensors from icm moduel do not require gradients')

        discount_t = (~done).float() * self._discount**self._n_step

        # Get policy action logits and baseline for s_tm1.
        policy_output = self._policy_network(s_tm1)
        logits_tm1 = policy_output.pi_logits
        baseline_tm1 = policy_output.baseline.squeeze(-1)  # [batch_size]

        # Calculates TD n-step target and advantages.
        with torch.no_grad():
            baseline_s_t = self._old_policy_network(s_t).baseline.squeeze(1)  # [batch_size]
            # Add intrisic reward, re-weights with coefficients
            r_t = self._ext_reward_coef * r_t + self._int_reward_coef * icm_intrinsic_reward
            target_baseline = r_t + discount_t * baseline_s_t
            advantages = target_baseline - baseline_tm1

        # Calculate importance sampling ratio
        ratio = distributions.categorical_importance_sampling_ratios(logits_tm1, behavior_logits_tm1, a_tm1)

        if ratio.shape != advantages.shape:
            raise RuntimeError(f'Expect ratio and advantages have same shape, got {ratio.shape} and {advantages.shape}')

        # Compute PPO policy gradient loss.
        policy_loss = rl.clipped_surrogate_gradient_loss(ratio, advantages, self.clip_epsilon).loss

        # Compute entropy loss.
        entropy_loss = rl.entropy_loss(logits_tm1).loss

        # Compute baseline state-value loss.
        baseline_loss = rl.baseline_loss(baseline_tm1 - target_baseline).loss

        # Average over batch dimension.
        policy_loss = torch.mean(policy_loss, dim=0)
        entropy_loss = torch.mean(entropy_loss, dim=0)
        baseline_loss = torch.mean(baseline_loss, dim=0)

        # Combine policy loss, baseline loss, entropy loss.
        loss = policy_loss + self._baseline_coef * baseline_loss + self._entropy_coef * entropy_loss

        # Re-weight policy loss, add ICM module inverse model loss, forward model loss.
        loss = self._lambda * loss + (1.0 - self._beta) * icm_inverse_loss + self._beta * icm_forward_loss

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
            'icm_loss': self._icm_loss_t,
            'discount': self._discount,
            'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
        }
