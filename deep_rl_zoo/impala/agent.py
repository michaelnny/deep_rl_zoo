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
"""IMPALA agent class.

From the paper "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
https://arxiv.org/abs/1802.01561.


Code based on a combination of the following sources:
https://github.com/deepmind/scalable_agent
https://github.com/facebookresearch/torchbeast
"""
import collections
from typing import Mapping, Optional, Tuple, NamedTuple
import queue
import multiprocessing
import numpy as np
import torch
from torch import nn

# pylint: disable=import-error
from deep_rl_zoo import vtrace
from deep_rl_zoo import base
from deep_rl_zoo import distributions
from deep_rl_zoo.networks.policy import ImpalaActorCriticNetworkInputs
import deep_rl_zoo.replay as replay_lib
import deep_rl_zoo.types as types_lib
import deep_rl_zoo.policy_gradient as rl

# torch.autograd.set_detect_anomaly(True)

HiddenState = Tuple[torch.Tensor, torch.Tensor]


class ImpalaTransition(NamedTuple):
    """
    s_t, r_t, done are the tuple from env.step().

    a_t, logits_t are from actor.act(), with input s_t.

    last_action is the last agent the agent took, before in s_t.
    """

    s_t: Optional[np.ndarray]
    r_t: Optional[float]
    done: Optional[bool]
    a_t: Optional[int]
    logits_t: Optional[np.ndarray]
    last_action: Optional[int]
    init_h: Optional[np.ndarray]  # nn.LSTM initial hidden state
    init_c: Optional[np.ndarray]  # nn.LSTM initial cell state


TransitionStructure = ImpalaTransition(
    s_t=None,
    r_t=None,
    done=None,
    a_t=None,
    logits_t=None,
    last_action=None,
    init_h=None,
    init_c=None,
)


class Actor(types_lib.Agent):
    """IMPALA actor"""

    def __init__(
        self,
        rank: int,
        unroll_length: int,
        data_queue: multiprocessing.Queue,
        policy_network: torch.nn.Module,
        device: torch.device,
    ) -> None:
        """
        Args:
            rank: the rank for the actor.
            unroll_length: how many agent time step to unroll.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            policy_network: the policy network for worker to make action choice.
            device: PyTorch runtime device.
        """
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer geater than or equal to 1, got {unroll_length}')

        self.rank = rank
        self.agent_name = f'IMPALA-actor{rank}'

        self._policy_network = policy_network.to(device=device)
        self._device = device

        self._queue = data_queue

        # Store unroll trajectory, unroll_length + 1
        # because we reuse last step transition of current unroll at the begining of next unroll
        self._unroll = replay_lib.Unroll(
            unroll_length=unroll_length,
            overlap=1,
            structure=TransitionStructure,
            cross_episode=True,
        )

        self._last_action = None
        self._lstm_state = None  # Stores nn.LSTM hidden state and cell state

        self._step_t = -1

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given current timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1

        a_t, logits_t, hidden_s = self.act(timestep)

        # Note the reward is for s_tm1, a_tm1, because it's only available one agent step after,
        # and the done mark is for current timestep s_t.
        transition = ImpalaTransition(
            s_t=timestep.observation,
            r_t=timestep.reward,
            done=timestep.done,
            a_t=a_t,
            logits_t=logits_t,
            last_action=self._last_action,
            init_h=False if not self._lstm_state else self._lstm_state[0].squeeze(1).numpy(),  # remove batch dimension
            init_c=False if not self._lstm_state else self._lstm_state[1].squeeze(1).numpy(),
        )
        unrolled_transition = self._unroll.add(transition, timestep.done)
        self._last_action, self._lstm_state = a_t, hidden_s

        if unrolled_transition is not None:
            # To save memory, only use hidden states for first step tansition in the unroll,
            # we also remove the time dimension, we only do it if network is using LSTM.
            if self._lstm_state:
                unrolled_transition = unrolled_transition._replace(
                    init_h=np.squeeze(unrolled_transition.init_h[0:1], axis=0),  # [num_lstm_layer, lstm_hidden_size]
                    init_c=np.squeeze(unrolled_transition.init_c[0:1], axis=0),  # [num_lstm_layer, lstm_hidden_size]
                )

            self._queue.put(unrolled_transition)

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode before take any action."""
        self._last_action = 0  # During the first step of a new episode, use 'fake' previous action for network pass
        self._lstm_state = self._policy_network.get_initial_hidden_state(batch_size=1)

    def act(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action, np.ndarray, HiddenState]:
        'Given timestep, return an action.'
        return self._choose_action(timestep)

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action, np.ndarray, HiddenState]:
        """Given timestep, choose action a_t"""
        pi_output = self._policy_network(self._prepare_network_input(timestep))
        logits_t = pi_output.pi_logits

        # Sample an action
        a_t = distributions.categorical_distribution(logits_t).sample()

        # Remove T and B dimensions.
        logits_t = logits_t.squeeze(0).squeeze(0)  # [num_actions]
        return (a_t.cpu().item(), logits_t.cpu().numpy(), pi_output.hidden_s)

    def _prepare_network_input(self, timestep: types_lib.TimeStep) -> ImpalaActorCriticNetworkInputs:
        # IMPALA network requires more than just the state input, but also last action, and reward for last action
        # optionally the last hidden state from LSTM, and done mask if using LSTM
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(self._last_action).to(device=self._device, dtype=torch.int64)
        r_t = torch.tensor(timestep.reward).to(device=self._device, dtype=torch.float32)
        done = torch.tensor(timestep.done).to(device=self._device, dtype=torch.bool)
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)

        return ImpalaActorCriticNetworkInputs(
            s_t=s_t[None, ...],  # [T, B, state_shape]
            a_tm1=a_tm1[None, ...],  # [T, B]
            r_t=r_t[None, ...],  # [T, B]
            done=done[None, ...],  # [T, B]
            hidden_s=hidden_s,
        )

    @property
    def statistics(self) -> Mapping[str, float]:
        """Returns current actor's statistics as a dictionary."""
        return {}


class Learner:
    """IMPALA learner"""

    def __init__(
        self,
        data_queue: multiprocessing.Queue,
        policy_network: nn.Module,
        actor_policy_network: nn.Module,
        policy_optimizer: torch.optim.Optimizer,
        discount: float,
        unroll_length: int,
        batch_size: int,
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
            actor_policy_network: the policy network shared with actors.
            policy_optimizer: the optimizer for policy network.
            discount: the gamma discount for future rewards.
            unroll_length: actor unroll time step.
            batch_size: sample batch_size of transitions.
            num_actors: number of actor processes.
            entropy_coef: the coefficient of entryopy loss.
            baseline_coef: the coefficient of state-value loss.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to in the range [0.0, 1.0], got {discount}')
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer geater than or equal to 1, got {unroll_length}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to in the range [1, 512], got {batch_size}')
        if not 1 <= num_actors:
            raise ValueError(f'Expect num_actors to be integer geater than or equal to 1, got {num_actors}')
        if not 0.0 < entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to (0.0, 1.0], got {entropy_coef}')
        if not 0.0 < baseline_coef <= 1.0:
            raise ValueError(f'Expect baseline_coef to (0.0, 1.0], got {baseline_coef}')

        self.agent_name = 'IMPALA-learner'
        self._device = device
        self.actor_policy_network = actor_policy_network
        self._policy_network = policy_network.to(device=device)
        self._policy_optimizer = policy_optimizer

        self._discount = discount
        self._unroll_length = unroll_length
        self._batch_size = batch_size

        self._storage = collections.deque(maxlen=batch_size)

        self._entropy_coef = entropy_coef
        self._baseline_coef = baseline_coef

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        self._queue = data_queue
        self._num_actors = num_actors

        # Counters
        self._step_t = -1
        self._update_t = -1
        self._done_actors = 0

    def run_train_loop(
        self,
    ) -> None:
        """Start the learner training loop, only break if all actor processes are done."""
        self.reset()
        while True:
            self._step_t += 1
            # Pull one item off queue
            try:
                item = self._queue.get()
                if item == 'PROCESS_DONE':  # actor process is done
                    self._done_actors += 1
                else:
                    self._storage.append(item)
            except queue.Empty:
                pass
            except EOFError:
                pass

            # Only break if all actor processes are done
            if self._done_actors == self._num_actors:
                break

            if len(self._storage) < self._storage.maxlen:
                continue

            self._learn()

    def reset(self) -> None:
        """Should be called at the begining of every iteration."""
        self._done_actors = 0
        self._storage.clear()

    def _learn(self) -> None:
        transitions = list(self._storage)
        # Stack on batch dimension (1), so we get [T, B]
        transitions = replay_lib.np_stack_list_of_transitions(transitions, TransitionStructure, 1)
        self._update(transitions)
        # This is important to keep it 'on-policy' by clean up old samples.
        self._storage.clear()

    def _update(self, transitions: ImpalaTransition) -> torch.Tensor:
        self._policy_optimizer.zero_grad()
        loss = self._calc_loss(transitions)
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), self._max_grad_norm)

        self._policy_optimizer.step()
        self._update_t += 1
        self.actor_policy_network.load_state_dict(self._policy_network.state_dict())

    def _calc_loss(self, transitions: ImpalaTransition) -> torch.Tensor:
        """Calculate loss for a batch transitions"""
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [T+1, B, state_shape]
        a_t = torch.from_numpy(transitions.a_t).squeeze(-1).to(device=self._device, dtype=torch.int64)  # [T+1, B]
        behaviour_logits_t = torch.from_numpy(transitions.logits_t).to(
            device=self._device, dtype=torch.float32
        )  # [T+1, B, num_actions]
        last_actions = (
            torch.from_numpy(transitions.last_action).squeeze(-1).to(device=self._device, dtype=torch.int64)
        )  # [T+1, B]
        r_t = torch.from_numpy(transitions.r_t).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        done = torch.from_numpy(transitions.done).to(device=self._device, dtype=torch.bool)  # [T+1, B]

        # [num_lstm_layers, batch_size, lstm_hidden_size]
        init_h = torch.from_numpy(transitions.init_h).to(device=self._device, dtype=torch.float32)
        init_c = torch.from_numpy(transitions.init_c).to(device=self._device, dtype=torch.float32)
        hidden_states = (init_h, init_c)

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        base.assert_rank_and_dtype(s_t, (3, 5), torch.float32)
        base.assert_rank_and_dtype(a_t, 2, torch.long)
        base.assert_rank_and_dtype(last_actions, 2, torch.long)
        base.assert_rank_and_dtype(r_t, 2, torch.float32)
        base.assert_rank_and_dtype(done, 2, torch.bool)
        base.assert_rank_and_dtype(behaviour_logits_t, 3, torch.float32)

        # Only have valid data when use_lstm is set to True.
        if self._policy_network.use_lstm:
            base.assert_rank_and_dtype(init_h, 3, torch.float32)
            base.assert_rank_and_dtype(init_c, 3, torch.float32)

        network_output = self._policy_network(
            ImpalaActorCriticNetworkInputs(
                s_t=s_t,
                a_tm1=last_actions,
                r_t=r_t,
                done=done,
                hidden_s=hidden_states,
            )
        )

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = network_output.baseline[-1]

        # We have unrolled T + 1 steps. The last step is only used as bootstrap value, so it's removed.
        target_logits, baseline = network_output.pi_logits[:-1], network_output.baseline[:-1]
        action, behaviour_logits = a_t[:-1], behaviour_logits_t[:-1]
        reward, done = r_t[1:], done[1:]

        # Compute policy log probabilitiy for action the agent taken.
        target_action_log_probs = distributions.categorical_distribution(target_logits).log_prob(action)
        behaviour_action_log_probs = distributions.categorical_distribution(behaviour_logits).log_prob(action)
        discount = (~done).float() * self._discount

        with torch.no_grad():
            vtrace_returns = vtrace.from_importance_weights(
                target_action_log_probs=target_action_log_probs,
                behaviour_action_log_probs=behaviour_action_log_probs,
                discounts=discount,
                rewards=reward,
                values=baseline,
                bootstrap_value=bootstrap_value,
            )

        # Compute policy gradient loss.
        policy_loss = rl.policy_gradient_loss(target_logits, action, vtrace_returns.pg_advantages).loss

        # Compute entropy loss.
        entropy_loss = rl.entropy_loss(target_logits).loss

        # Compute baseline state-value loss.
        baseline_loss = rl.baseline_loss(baseline - vtrace_returns.vs).loss

        # Average over batch dimension.
        policy_loss = torch.mean(policy_loss, dim=0)
        entropy_loss = torch.mean(entropy_loss, dim=0)
        baseline_loss = torch.mean(baseline_loss, dim=0)

        loss = policy_loss + self._baseline_coef * baseline_loss + self._entropy_coef * entropy_loss

        return loss

    @property
    def statistics(self) -> Mapping[str, float]:
        """Returns current learner's statistics as a dictionary."""
        return {
            'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            'discount': self._discount,
            'updates': self._update_t,
        }
