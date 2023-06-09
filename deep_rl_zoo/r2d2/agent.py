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
"""R2D2 agent class.

From the paper "Recurrent Experience Replay in Distributed Reinforcement Learning"
https://openreview.net/pdf?id=r1lyTjAqYX.

The code for value function rescaling, inverse value function rescaling, and n-step bellman targets are from seed-rl:
https://github.com/google-research/seed_rl/blob/66e8890261f09d0355e8bf5f1c5e41968ca9f02b/agents/r2d2/learner.py

This agent supports store hidden state (only first step in a unroll) in replay, and burn in.
In fact, even if we use burn in, we're still going to store the hidden state (only first step in a unroll) in the replay.
"""
from typing import Mapping, Optional, Tuple, NamedTuple, Iterable, Text
import copy
import multiprocessing
import numpy as np
import torch
from torch import nn


# pylint: disable=import-error
import deep_rl_zoo.replay as replay_lib
import deep_rl_zoo.types as types_lib
from deep_rl_zoo import base
from deep_rl_zoo import multistep
from deep_rl_zoo import distributed
from deep_rl_zoo import transforms
from deep_rl_zoo.networks.value import RnnDqnNetworkInputs

torch.autograd.set_detect_anomaly(True)

HiddenState = Tuple[torch.Tensor, torch.Tensor]


class R2d2Transition(NamedTuple):
    """
    s_t, r_t, done are the tuple from env.step().

    last_action is the last agent the agent took, before in s_t.
    """

    s_t: Optional[np.ndarray]
    r_t: Optional[float]
    done: Optional[bool]
    a_t: Optional[int]
    q_t: Optional[np.ndarray]  # q values for s_t
    last_action: Optional[int]
    init_h: Optional[np.ndarray]  # nn.LSTM initial hidden state
    init_c: Optional[np.ndarray]  # nn.LSTM initial cell state


TransitionStructure = R2d2Transition(
    s_t=None,
    r_t=None,
    done=None,
    a_t=None,
    q_t=None,
    last_action=None,
    init_h=None,
    init_c=None,
)


def no_autograd(net: torch.nn.Module):
    """Disable autograd for a network."""
    for p in net.parameters():
        p.requires_grad = False


def calculate_losses_and_priorities(
    q_value: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    target_qvalue: torch.Tensor,
    target_action: torch.Tensor,
    gamma: float,
    n_step: int,
    eps: float = 0.001,
    eta: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculate loss and priority for given samples.

    T is the unrolled length, B the batch size, N is number of actions.

    Args:
        q_value: (T+1, B, action_dim) the predicted q values for a given state 's_t' from online Q network.
        action: [T+1, B] the actual action the agent take in state 's_t'.
        reward: [T+1, B] the reward the agent received at timestep t, this is for (s_tm1, a_tm1).
        done: [T+1, B] terminal mask for timestep t, state 's_t'.
        target_qvalue: (T+1, B, N) the estimated TD n-step target values from target Q network,
            this could also be the same q values when just calculate priorities to insert into replay.
        target_action: [T+1, B] the best action to take in t+n timestep target state.
        gamma: discount rate.
        n_step: TD n-step size.
        eps: constant for value function rescaling and inverse value function rescaling.
        eta: constant for calculate mixture priorities.

    Returns:
        losses: the losses for given unrolled samples, shape (B, )
        priorities: the priority for given samples, shape (B, )
    """

    base.assert_rank_and_dtype(q_value, 3, torch.float32)
    base.assert_rank_and_dtype(target_qvalue, 3, torch.float32)
    base.assert_rank_and_dtype(reward, 2, torch.float32)
    base.assert_rank_and_dtype(action, 2, torch.long)
    base.assert_rank_and_dtype(target_action, 2, torch.long)
    base.assert_rank_and_dtype(done, 2, torch.bool)

    q_value = q_value.gather(-1, action[..., None]).squeeze(-1)  # [T, B]

    target_q_max = target_qvalue.gather(-1, target_action[..., None]).squeeze(-1)  # [T, B]
    # Apply invertible value rescaling to TD target.
    target_q_max = transforms.signed_parabolic(target_q_max, eps)

    # Note the input rewards into 'n_step_bellman_target' should be non-discounted, non-summed.
    target_q = multistep.n_step_bellman_target(r_t=reward, done=done, q_t=target_q_max, gamma=gamma, n_steps=n_step)

    # q_value is actually Q(s_t, a_t), but rewards is for 's_tm1', 'a_tm1',
    # that means our 'target_q' value is one step behind 'q_value',
    # so we need to shift them to make it in the same timestep.
    q_value = q_value[:-1, ...]
    target_q = target_q[1:, ...]

    # Apply value rescaling to TD target.
    target_q = transforms.signed_hyperbolic(target_q, eps)

    if q_value.shape != target_q.shape:
        raise RuntimeError(f'Expect q_value and target_q have the same shape, got {q_value.shape} and {target_q.shape}')

    td_error = target_q - q_value

    with torch.no_grad():
        priorities = distributed.calculate_dist_priorities_from_td_error(td_error, eta)

    # Sums over time dimension.
    losses = 0.5 * torch.sum(torch.square(td_error), dim=0)  # [B]

    return losses, priorities


class Actor(types_lib.Agent):
    """R2D2 actor"""

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        network: torch.nn.Module,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        num_actors: int,
        action_dim: int,
        unroll_length: int,
        burn_in: int,
        actor_update_interval: int,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        """
        Args:
            rank: the rank number for the actor.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            network: the Q network for actor to make action choice.
            random_state: used to sample random actions for e-greedy policy.
            num_actors: the number actors for calculating e-greedy epsilon.
            action_dim: the number of valid actions in the environment.
            unroll_length: how many agent time step to unroll transitions before put on to queue.
            burn_in: two consecutive unrolls will overlap on burn_in+1 steps.
            actor_update_interval: the frequency to update actor local Q network.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """
        if not 0 < num_actors:
            raise ValueError(f'Expect num_actors to be positive integer, got {num_actors}')
        if not 0 < action_dim:
            raise ValueError(f'Expect action_dim to be positive integer, got {action_dim}')
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer greater than or equal to 1, got {unroll_length}')
        if not 0 <= burn_in < unroll_length:
            raise ValueError(f'Expect burn_in to be integer between [0, {unroll_length}), got {burn_in}')
        if not 1 <= actor_update_interval:
            raise ValueError(
                f'Expect actor_update_interval to be integer greater than or equal to 1, got {actor_update_interval}'
            )

        self.rank = rank
        self.agent_name = f'R2D2-actor{rank}'

        self._network = network.to(device=device)

        # Disable autograd for actor's network
        no_autograd(self._network)

        self._shared_params = shared_params

        self._queue = data_queue

        self._device = device
        self._random_state = random_state
        self._action_dim = action_dim
        self._actor_update_interval = actor_update_interval

        self._unroll = replay_lib.Unroll(
            unroll_length=unroll_length,
            overlap=burn_in + 1,  # Plus 1 to add room for shift during learning
            structure=TransitionStructure,
            cross_episode=False,
        )

        epsilons = distributed.get_actor_exploration_epsilon(num_actors)
        self._exploration_epsilon = epsilons[self.rank]

        self._last_action = None
        self._lstm_state = None  # Stores nn.LSTM hidden state and cell state

        self._step_t = -1

    @torch.no_grad()
    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1

        if self._step_t % self._actor_update_interval == 0:
            self._update_actor_network()

        q_t, a_t, hidden_s = self.act(timestep)

        # Note the reward is for s_tm1, a_tm1, because it's only available one agent step after,
        # and the done mark is for current timestep s_t.
        transition = R2d2Transition(
            s_t=timestep.observation,
            a_t=a_t,
            q_t=q_t,
            r_t=timestep.reward,
            done=timestep.done,
            last_action=self._last_action,
            init_h=self._lstm_state[0].squeeze(1).cpu().numpy(),  # remove batch dimension
            init_c=self._lstm_state[1].squeeze(1).cpu().numpy(),
        )
        unrolled_transition = self._unroll.add(transition, timestep.done)
        self._last_action, self._lstm_state = a_t, hidden_s

        if unrolled_transition is not None:
            self._put_unroll_onto_queue(unrolled_transition)

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode before take any action."""
        self._unroll.reset()
        self._last_action = self._random_state.randint(0, self._action_dim)  # Initialize a_tm1 randomly
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)

    def act(self, timestep: types_lib.TimeStep) -> Tuple[np.ndarray, types_lib.Action, Tuple[torch.Tensor]]:
        'Given state s_t and done marks, return an action.'
        return self._choose_action(timestep, self._exploration_epsilon)

    @torch.no_grad()
    def _choose_action(
        self, timestep: types_lib.TimeStep, epsilon: float
    ) -> Tuple[np.ndarray, types_lib.Action, Tuple[torch.Tensor]]:
        """Given state s_t, choose action a_t"""
        pi_output = self._network(self._prepare_network_input(timestep))
        q_t = pi_output.q_values.squeeze()
        a_t = torch.argmax(q_t, dim=-1).cpu().item()

        # To make sure every actors generates the same amount of samples, we apply e-greedy after the network pass,
        # otherwise the actor with higher epsilons will generate more samples,
        # while the actor with lower epsilon will generate less samples.
        if self._random_state.rand() <= epsilon:
            # randint() return random integers from low (inclusive) to high (exclusive).
            a_t = self._random_state.randint(0, self._action_dim)

        return (q_t.cpu().numpy(), a_t, pi_output.hidden_s)

    def _prepare_network_input(self, timestep: types_lib.TimeStep) -> RnnDqnNetworkInputs:
        # R2D2 network expect input shape [T, B, state_shape],
        # and additionally 'last action', 'reward for last action', and hidden state from previous timestep.
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(self._last_action, device=self._device, dtype=torch.int64)
        r_t = torch.tensor(timestep.reward, device=self._device, dtype=torch.float32)
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)

        return RnnDqnNetworkInputs(
            s_t=s_t.unsqueeze(0),  # [T, B, state_shape]
            a_tm1=a_tm1.unsqueeze(0),  # [T, B]
            r_t=r_t.unsqueeze(0),  # [T, B]
            hidden_s=hidden_s,
        )

    def _put_unroll_onto_queue(self, unrolled_transition):
        # Important note, store hidden states for every step in the unroll will consume HUGE memory.
        self._queue.put(unrolled_transition)

    def _update_actor_network(self):
        state_dict = self._shared_params['network']

        if state_dict is not None:
            if self._device != 'cpu':
                state_dict = {k: v.to(device=self._device) for k, v in state_dict.items()}
            self._network.load_state_dict(state_dict)

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current actor's statistics as a dictionary."""
        return {'exploration_epsilon': self._exploration_epsilon}


class Learner(types_lib.Learner):
    """R2D2 learner"""

    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay: replay_lib.PrioritizedReplay,
        target_net_update_interval: int,
        min_replay_size: int,
        batch_size: int,
        n_step: int,
        discount: float,
        burn_in: int,
        priority_eta: float,
        rescale_epsilon: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        """
        Args:
            network: the Q network we want to train and optimize.
            optimizer: the optimizer for Q network.
            replay: prioritized recurrent experience replay.
            target_net_update_interval: how often to copy online network parameters to target.
            min_replay_size: wait till experience replay buffer this number before start to learn.
            batch_size: sample batch_size of transitions.
            n_step: TD n-step bootstrap.
            discount: the gamma discount for future rewards.
            burn_in: burn n transitions to generate initial hidden state before learning.
            priority_eta: coefficient to mix the max and mean absolute TD errors.
            rescale_epsilon: rescaling factor for n-step targets in the invertible rescaling function.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """
        if not 1 <= target_net_update_interval:
            raise ValueError(f'Expect target_net_update_interval to be positive integer, got {target_net_update_interval}')
        if not 1 <= min_replay_size:
            raise ValueError(f'Expect min_replay_size to be integer greater than or equal to 1, got {min_replay_size}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to in the range [1, 512], got {batch_size}')
        if not 1 <= n_step:
            raise ValueError(f'Expect n_step to be integer greater than or equal to 1, got {n_step}')
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to in the range [0.0, 1.0], got {discount}')
        if not 0.0 <= priority_eta <= 1.0:
            raise ValueError(f'Expect priority_eta to in the range [0.0, 1.0], got {priority_eta}')
        if not 0.0 <= rescale_epsilon <= 1.0:
            raise ValueError(f'Expect rescale_epsilon to in the range [0.0, 1.0], got {rescale_epsilon}')

        self.agent_name = 'R2D2-learner'
        self._device = device
        self._network = network.to(device=device)
        self._optimizer = optimizer
        # Lazy way to create target Q network
        self._target_network = copy.deepcopy(self._network).to(device=self._device)

        # Disable autograd for target network
        no_autograd(self._target_network)

        self._shared_params = shared_params

        self._batch_size = batch_size
        self._n_step = n_step
        self._burn_in = burn_in
        self._target_net_update_interval = target_net_update_interval
        self._discount = discount
        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm
        self._rescale_epsilon = rescale_epsilon

        self._replay = replay
        self._min_replay_size = min_replay_size
        self._priority_eta = priority_eta

        self._max_seen_priority = 1.0  # New unroll will use this as priority

        # Counters
        self._step_t = -1
        self._update_t = 0
        self._target_update_t = 0
        self._loss_t = np.nan

    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.

        Yields:
            learner statistics if network parameters update occurred, otherwise returns None.
        """
        self._step_t += 1

        if self._replay.size < self._min_replay_size or self._step_t % max(4, int(self._batch_size * 0.25)) != 0:
            return

        self._learn()
        yield self.statistics

    def reset(self) -> None:
        """Should be called at the beginning of every iteration."""

    def received_item_from_queue(self, item) -> None:
        """Received item send by actors through multiprocessing queue."""
        self._replay.add(item, self._max_seen_priority)

    def get_network_state_dict(self):
        # To keep things consistent, we move the parameters to CPU
        return {k: v.cpu() for k, v in self._network.state_dict().items()}

    def _learn(self) -> None:
        transitions, indices, weights = self._replay.sample(self._batch_size)
        priorities = self._update(transitions, weights)
        self._update_t += 1

        if priorities.shape != (self._batch_size,):
            raise RuntimeError(f'Expect priorities has shape ({self._batch_size},), got {priorities.shape}')
        priorities = np.abs(priorities)
        self._max_seen_priority = np.max([self._max_seen_priority, np.max(priorities)])
        self._replay.update_priorities(indices, priorities)

        self._shared_params['network'] = self.get_network_state_dict()

        # Copy online Q network parameters to target Q network, every m updates
        if self._update_t > 1 and self._update_t % self._target_net_update_interval == 0:
            self._update_target_network()

    def _update(self, transitions: R2d2Transition, weights: np.ndarray) -> np.ndarray:
        """Update online Q network."""
        weights = torch.from_numpy(weights).to(device=self._device, dtype=torch.float32)  # [batch_size]
        base.assert_rank_and_dtype(weights, 1, torch.float32)

        # Get initial hidden state, handle possible burn in.
        init_hidden_s = self._extract_first_step_hidden_state(transitions)
        burn_transitions, learn_transitions = replay_lib.split_structure(transitions, TransitionStructure, self._burn_in)
        if burn_transitions is not None:
            hidden_s, target_hidden_s = self._burn_in_unroll_q_networks(burn_transitions, init_hidden_s)
        else:
            hidden_s = tuple(s.clone().to(device=self._device) for s in init_hidden_s)
            target_hidden_s = tuple(s.clone().to(device=self._device) for s in init_hidden_s)

        self._optimizer.zero_grad()
        # [batch_size]
        loss, priorities = self._calc_loss(learn_transitions, hidden_s, target_hidden_s)

        # Multiply loss by sampling weights, averaging over batch dimension
        loss = torch.mean(loss * weights.detach())
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), self._max_grad_norm)

        self._optimizer.step()

        # For logging only.
        self._loss_t = loss.detach().cpu().item()
        return priorities

    def _calc_loss(
        self,
        transitions: R2d2Transition,
        hidden_s: HiddenState,
        target_hidden_s: HiddenState,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Calculate loss and priorities for given unroll sequence transitions."""
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [T+1, B, state_shape]
        a_t = torch.from_numpy(transitions.a_t).to(device=self._device, dtype=torch.int64)  # [T+1, B]
        last_action = torch.from_numpy(transitions.last_action).to(device=self._device, dtype=torch.int64)  # [T+1, B]
        r_t = torch.from_numpy(transitions.r_t).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        done = torch.from_numpy(transitions.done).to(device=self._device, dtype=torch.bool)  # [T+1, B]

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        base.assert_rank_and_dtype(s_t, (3, 5), torch.float32)
        base.assert_rank_and_dtype(a_t, 2, torch.long)
        base.assert_rank_and_dtype(last_action, 2, torch.long)
        base.assert_rank_and_dtype(r_t, 2, torch.float32)
        base.assert_rank_and_dtype(done, 2, torch.bool)

        # Get q values from online Q network
        q_t = self._network(RnnDqnNetworkInputs(s_t=s_t, a_tm1=last_action, r_t=r_t, hidden_s=hidden_s)).q_values

        # Computes raw target q values, use double Q
        with torch.no_grad():
            # Get best actions a* for 's_t' from online Q network.
            best_a_t = torch.argmax(q_t, dim=-1)  # [T, B]

            # Get estimated q values for 's_t' from target Q network, using above best action a*.
            target_q_t = self._target_network(
                RnnDqnNetworkInputs(s_t=s_t, a_tm1=last_action, r_t=r_t, hidden_s=target_hidden_s)
            ).q_values

        losses, priorities = calculate_losses_and_priorities(
            q_value=q_t,
            action=a_t,
            reward=r_t,
            done=done,
            target_qvalue=target_q_t,
            target_action=best_a_t,
            gamma=self._discount,
            n_step=self._n_step,
            eps=self._rescale_epsilon,
            eta=self._priority_eta,
        )

        return (losses, priorities)

    @torch.no_grad()
    def _burn_in_unroll_q_networks(
        self, transitions: R2d2Transition, init_hidden_s: HiddenState
    ) -> Tuple[HiddenState, HiddenState]:
        """Unroll both online and target q networks to generate hidden states for LSTM."""
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [burn_in, B, state_shape]
        last_action = torch.from_numpy(transitions.last_action).to(device=self._device, dtype=torch.int64)  # [burn_in, B]
        r_t = torch.from_numpy(transitions.r_t).to(device=self._device, dtype=torch.float32)  # [burn_in, B]

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        base.assert_rank_and_dtype(s_t, (3, 5), torch.float32)
        base.assert_rank_and_dtype(last_action, 2, torch.long)
        base.assert_rank_and_dtype(r_t, 2, torch.float32)

        _hidden_s = tuple(s.clone().to(device=self._device) for s in init_hidden_s)
        _target_hidden_s = tuple(s.clone().to(device=self._device) for s in init_hidden_s)

        # Burn in to generate hidden states for LSTM, we unroll both online and target Q networks
        hidden_s = self._network(RnnDqnNetworkInputs(s_t=s_t, a_tm1=last_action, r_t=r_t, hidden_s=_hidden_s)).hidden_s
        target_hidden_s = self._target_network(
            RnnDqnNetworkInputs(s_t=s_t, a_tm1=last_action, r_t=r_t, hidden_s=_target_hidden_s)
        ).hidden_s

        return (hidden_s, target_hidden_s)

    def _extract_first_step_hidden_state(self, transitions: R2d2Transition) -> HiddenState:
        # We only need the first step hidden states in replay, shape [batch_size, num_lstm_layers, lstm_hidden_size]
        init_h = torch.from_numpy(transitions.init_h[0:1]).squeeze(0).to(device=self._device, dtype=torch.float32)
        init_c = torch.from_numpy(transitions.init_c[0:1]).squeeze(0).to(device=self._device, dtype=torch.float32)

        # Rank and dtype checks.
        base.assert_rank_and_dtype(init_h, 3, torch.float32)
        base.assert_rank_and_dtype(init_c, 3, torch.float32)

        # Swap batch and num_lstm_layers axis.
        init_h = init_h.swapaxes(0, 1)
        init_c = init_c.swapaxes(0, 1)

        # Batch dimension checks.
        base.assert_batch_dimension(init_h, self._batch_size, 1)
        base.assert_batch_dimension(init_c, self._batch_size, 1)

        return (init_h, init_c)

    def _update_target_network(self):
        self._target_network.load_state_dict(self._network.state_dict())
        self._target_update_t += 1

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            # 'learning_rate': self._optimizer.param_groups[0]['lr'],
            'loss': self._loss_t,
            # 'discount': self._discount,
            'updates': self._update_t,
            'target_updates': self._target_update_t,
        }
