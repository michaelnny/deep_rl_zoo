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
"""NGU agent class.

From the paper "Never Give Up: Learning Directed Exploration Strategies"
https://arxiv.org/abs/2002.06038.
"""

from typing import Iterable, Mapping, Optional, Tuple, NamedTuple, Text
import copy
import multiprocessing
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# pylint: disable=import-error
import deep_rl_zoo.replay as replay_lib
import deep_rl_zoo.types as types_lib
from deep_rl_zoo import normalizer
from deep_rl_zoo import nonlinear_bellman
from deep_rl_zoo import base
from deep_rl_zoo import distributed
from deep_rl_zoo.curiosity import EpisodicBonusModule, RndLifeLongBonusModule
from deep_rl_zoo.networks.value import NguNetworkInputs

torch.autograd.set_detect_anomaly(True)

HiddenState = Tuple[torch.Tensor, torch.Tensor]


class NguTransition(NamedTuple):
    """
    s_t, r_t, done are the tuple from env.step().

    last_action is the last agent the agent took, before in s_t.
    """

    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    q_t: Optional[np.ndarray]  # q values for s_t
    prob_a_t: Optional[np.ndarray]  # probability of choose a_t in s_t
    last_action: Optional[int]  # for network input only
    ext_r_t: Optional[float]  # extrinsic reward for (s_tm1, a_tm1)
    int_r_t: Optional[float]  # intrinsic reward for (s_tm1)
    policy_index: Optional[int]  # intrinsic reward scale beta index
    beta: Optional[float]  # intrinsic reward scale beta value
    discount: Optional[float]
    done: Optional[bool]
    init_h: Optional[np.ndarray]  # nn.LSTM initial hidden state
    init_c: Optional[np.ndarray]  # nn.LSTM initial cell state


TransitionStructure = NguTransition(
    s_t=None,
    a_t=None,
    q_t=None,
    prob_a_t=None,
    last_action=None,
    ext_r_t=None,
    int_r_t=None,
    policy_index=None,
    beta=None,
    discount=None,
    done=None,
    init_h=None,
    init_c=None,
)


def no_autograd(net: torch.nn.Module):
    """Disable autograd for a network."""

    for p in net.parameters():
        p.requires_grad = False


class Actor(types_lib.Agent):
    """NGU actor"""

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        network: torch.nn.Module,
        rnd_target_network: torch.nn.Module,
        rnd_predictor_network: torch.nn.Module,
        embedding_network: torch.nn.Module,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        ext_discount: float,
        int_discount: float,
        num_actors: int,
        action_dim: int,
        unroll_length: int,
        burn_in: int,
        num_policies: int,
        policy_beta: float,
        episodic_memory_capacity: int,
        reset_episodic_memory: bool,
        num_neighbors: int,
        cluster_distance: float,
        kernel_epsilon: float,
        max_similarity: float,
        actor_update_interval: int,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        """
        Args:
            rank: the rank number for the actor.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            network: the Q network for actor to make action choice.
            rnd_target_network: RND random target network.
            rnd_predictor_network: RND predictor target network.
            embedding_network: NGU action prediction network.
            random_state: random state.
            ext_discount: extrinsic reward discount.
            int_discount: intrinsic reward discount.
            num_actors: number of actors.
            action_dim: number of valid actions in the environment.
            unroll_length: how many agent time step to unroll transitions before put on to queue.
            burn_in: two consecutive unrolls will overlap on burn_in+1 steps.
            num_policies: number of exploring and exploiting policies.
            policy_beta: intrinsic reward scale beta.
            episodic_memory_capacity: maximum capacity of episodic memory.
            reset_episodic_memory: Reset the episodic_memory on every episode.
            num_neighbors: number of K-NN neighbors for compute episodic bonus.
            cluster_distance: K-NN neighbors cluster distance for compute episodic bonus.
            kernel_epsilon: K-NN kernel epsilon for compute episodic bonus.
            max_similarity: maximum similarity for compute episodic bonus.
            actor_update_interval: the frequency to update actor's Q network.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """
        if not 0.0 <= ext_discount <= 1.0:
            raise ValueError(f'Expect ext_discount to be [0.0, 1.0], got {ext_discount}')
        if not 0.0 <= int_discount <= 1.0:
            raise ValueError(f'Expect int_discount to be [0.0, 1.0], got {int_discount}')
        if not 0 < num_actors:
            raise ValueError(f'Expect num_actors to be positive integer, got {num_actors}')
        if not 0 < action_dim:
            raise ValueError(f'Expect action_dim to be positive integer, got {action_dim}')
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer greater than or equal to 1, got {unroll_length}')
        if not 0 <= burn_in < unroll_length:
            raise ValueError(f'Expect burn_in length to be [0, {unroll_length}), got {burn_in}')
        if not 1 <= num_policies:
            raise ValueError(f'Expect num_policies to be integer greater than or equal to 1, got {num_policies}')
        if not 0.0 <= policy_beta <= 1.0:
            raise ValueError(f'Expect policy_beta to be [0.0, 1.0], got {policy_beta}')
        if not 1 <= episodic_memory_capacity:
            raise ValueError(
                f'Expect episodic_memory_capacity to be integer greater than or equal to 1, got {episodic_memory_capacity}'
            )
        if not 1 <= num_neighbors:
            raise ValueError(f'Expect num_neighbors to be integer greater than or equal to 1, got {num_neighbors}')
        if not 0.0 <= cluster_distance <= 1.0:
            raise ValueError(f'Expect cluster_distance to be [0.0, 1.0], got {cluster_distance}')
        if not 0.0 <= kernel_epsilon <= 1.0:
            raise ValueError(f'Expect kernel_epsilon to be [0.0, 1.0], got {kernel_epsilon}')
        if not 1 <= actor_update_interval:
            raise ValueError(
                f'Expect actor_update_interval to be integer greater than or equal to 1, got {actor_update_interval}'
            )

        self.rank = rank  # Needs to make sure rank always start from 0
        self.agent_name = f'NGU-actor{rank}'

        self._network = network.to(device=device)
        self._rnd_target_network = rnd_target_network.to(device=device)
        self._rnd_predictor_network = rnd_predictor_network.to(device=device)
        self._embedding_network = embedding_network.to(device=device)

        # Disable autograd for actor's local networks
        no_autograd(self._network)
        no_autograd(self._rnd_target_network)
        no_autograd(self._rnd_predictor_network)
        no_autograd(self._embedding_network)

        self._shared_params = shared_params

        self._queue = data_queue

        self._device = device
        self._random_state = random_state
        self._num_actors = num_actors
        self._action_dim = action_dim
        self._actor_update_q_network_interval = actor_update_interval
        self._num_policies = num_policies

        self._unroll = replay_lib.Unroll(
            unroll_length=unroll_length,
            overlap=burn_in + 1,  # Plus 1 to add room for shift during learning
            structure=TransitionStructure,
            cross_episode=False,
        )

        self._betas, self._gammas = distributed.get_ngu_policy_betas_and_discounts(
            num_policies=num_policies,
            beta=policy_beta,
            gamma_max=ext_discount,
            gamma_min=int_discount,
        )

        self._policy_index = None
        self._policy_beta = None
        self._policy_discount = None
        self._sample_policy()

        self._reset_episodic_memory = reset_episodic_memory

        # E-greedy policy epsilon, rank 0 has the lowest noise, while rank N-1 has the highest noise.
        epsilons = distributed.get_actor_exploration_epsilon(num_actors)
        self._exploration_epsilon = epsilons[self.rank]

        # Episodic intrinsic bonus module
        self._episodic_module = EpisodicBonusModule(
            embedding_network=self._embedding_network,
            device=device,
            capacity=episodic_memory_capacity,
            num_neighbors=num_neighbors,
            kernel_epsilon=kernel_epsilon,
            cluster_distance=cluster_distance,
            max_similarity=max_similarity,
        )

        # Lifelong intrinsic bonus module
        self._lifelong_module = RndLifeLongBonusModule(
            target_network=self._rnd_target_network,
            predictor_network=self._rnd_predictor_network,
            device=device,
            discount=int_discount,
        )

        self._last_action = None
        self._episodic_bonus_t = None
        self._lifelong_bonus_t = None
        self._lstm_state = None  # Stores nn.LSTM hidden state and cell state

        self._step_t = -1

    @torch.no_grad()
    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1

        if self._step_t % self._actor_update_q_network_interval == 0:
            self._update_actor_network(False)

        q_t, a_t, prob_a_t, hidden_s = self.act(timestep)

        transition = NguTransition(
            s_t=timestep.observation,
            a_t=a_t,
            q_t=q_t,
            prob_a_t=prob_a_t,
            last_action=self._last_action,
            ext_r_t=timestep.reward,
            int_r_t=self.intrinsic_reward,
            policy_index=self._policy_index,
            beta=self._policy_beta,
            discount=self._policy_discount,
            done=timestep.done,
            init_h=self._lstm_state[0].squeeze(1).cpu().numpy(),  # remove batch dimension
            init_c=self._lstm_state[1].squeeze(1).cpu().numpy(),
        )

        unrolled_transition = self._unroll.add(transition, timestep.done)

        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)

        # Compute lifelong intrinsic bonus
        self._lifelong_bonus_t = self._lifelong_module.compute_bonus(s_t)

        # Compute episodic intrinsic bonus
        self._episodic_bonus_t = self._episodic_module.compute_bonus(s_t)

        # Update local state
        self._last_action, self._lstm_state = a_t, hidden_s

        if unrolled_transition is not None:
            self._put_unroll_onto_queue(unrolled_transition)

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode before take any action."""
        self._unroll.reset()

        # From NGU Paper on MONTEZUMA’S REVENGE:
        """
        Instead of resetting the memory after every episode, we do it after a small number of
        consecutive episodes, which we call a meta-episode. This structure plays an important role when the
        agent faces irreversible choices.
        """

        if self._reset_episodic_memory:
            self._episodic_module.reset()

        self._update_actor_network(True)

        self._sample_policy()

        # During the first step of a new episode,
        # use 'fake' previous action and 'intrinsic' reward for network pass
        self._last_action = self._random_state.randint(0, self._action_dim)  # Initialize a_tm1 randomly
        self._episodic_bonus_t = 0.0
        self._lifelong_bonus_t = 0.0
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)

    def act(self, timestep: types_lib.TimeStep) -> Tuple[np.ndarray, types_lib.Action, float, HiddenState]:
        'Given state s_t and done marks, return an action.'
        return self._choose_action(timestep)

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[np.ndarray, types_lib.Action, float, HiddenState]:
        """Given state s_t, choose action a_t"""
        pi_output = self._network(self._prepare_network_input(timestep))
        q_t = pi_output.q_values.squeeze()

        a_t = torch.argmax(q_t, dim=-1).cpu().item()

        # Policy probability for a_t, the detailed equation is mentioned in Agent57 paper.
        prob_a_t = 1 - (self._exploration_epsilon * ((self._action_dim - 1) / self._action_dim))

        # To make sure every actors generates the same amount of samples, we apply e-greedy after the network pass,
        # otherwise the actor with higher epsilons will generate more samples,
        # while the actor with lower epsilon will generate less samples.
        if self._random_state.rand() < self._exploration_epsilon:
            # randint() return random integers from low (inclusive) to high (exclusive).
            a_t = self._random_state.randint(0, self._action_dim)
            prob_a_t = self._exploration_epsilon / self._action_dim

        return (q_t.cpu().numpy(), a_t, prob_a_t, pi_output.hidden_s)

    def _prepare_network_input(self, timestep: types_lib.TimeStep) -> NguNetworkInputs:
        # NGU network expect input shape [T, B, state_shape],
        # and additionally 'last action', 'extrinsic reward for last action', last intrinsic reward, and intrinsic reward scale beta index.
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        last_action = torch.tensor(self._last_action, device=self._device, dtype=torch.int64)
        ext_r_t = torch.tensor(timestep.reward, device=self._device, dtype=torch.float32)
        int_r_t = torch.tensor(self.intrinsic_reward, device=self._device, dtype=torch.float32)
        policy_index = torch.tensor(self._policy_index, device=self._device, dtype=torch.int64)
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)
        return NguNetworkInputs(
            s_t=s_t.unsqueeze(0),  # [T, B, state_shape]
            a_tm1=last_action.unsqueeze(0),  # [T, B]
            ext_r_t=ext_r_t.unsqueeze(0),  # [T, B]
            int_r_t=int_r_t.unsqueeze(0),  # [T, B]
            policy_index=policy_index.unsqueeze(0),  # [T, B]
            hidden_s=hidden_s,
        )

    def _put_unroll_onto_queue(self, unrolled_transition):
        # Important note, store hidden states for every step in the unroll will consume HUGE memory.
        self._queue.put(unrolled_transition)

    def _update_actor_network(self, update_embed: bool = False):
        q_state_dict = self._shared_params['network']
        embed_state_dict = self._shared_params['embedding_network']
        rnd_state_dict = self._shared_params['rnd_predictor_network']

        if update_embed:
            state_net_pairs = zip(
                (q_state_dict, embed_state_dict, rnd_state_dict),
                (self._network, self._embedding_network, self._rnd_predictor_network),
            )
        else:
            state_net_pairs = zip(
                (q_state_dict, rnd_state_dict),
                (self._network, self._rnd_predictor_network),
            )

        for state_dict, network in state_net_pairs:
            if state_dict is not None:
                if self._device != 'cpu':
                    state_dict = {k: v.to(device=self._device) for k, v in state_dict.items()}
                network.load_state_dict(state_dict)

    def _sample_policy(self):
        self._policy_index = np.random.randint(0, self._num_policies)
        self._policy_beta = self._betas[self._policy_index]
        self._policy_discount = self._gammas[self._policy_index]

    @property
    def intrinsic_reward(self) -> float:
        """Returns intrinsic reward for last state s_tm1."""
        # Equation 1 of the paper.
        return self._episodic_bonus_t * min(max(self._lifelong_bonus_t, 1.0), 5.0)

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current actor's statistics as a dictionary."""
        return {
            # 'policy_index': self._policy_index,
            'policy_discount': self._policy_discount,
            'policy_beta': self._policy_beta,
            'exploration_epsilon': self._exploration_epsilon,
            'intrinsic_reward': self.intrinsic_reward,
            # 'episodic_bonus': self._episodic_bonus_t,
            # 'lifelong_bonus': self._lifelong_bonus_t,
        }


class Learner(types_lib.Learner):
    """NGU learner"""

    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        embedding_network: nn.Module,
        rnd_target_network: nn.Module,
        rnd_predictor_network: nn.Module,
        intrinsic_optimizer: torch.optim.Optimizer,
        replay: replay_lib.PrioritizedReplay,
        target_net_update_interval: int,
        min_replay_size: int,
        batch_size: int,
        unroll_length: int,
        burn_in: int,
        retrace_lambda: float,
        transformed_retrace: bool,
        priority_eta: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        """
        Args:
            network: the Q network we want to train and optimize.
            optimizer: the optimizer for Q network.
            embedding_network: NGU action prediction network.
            rnd_target_network: RND random network.
            rnd_predictor_network: RND predictor network.
            intrinsic_optimizer: the optimizer for action prediction and RND predictor networks.
            replay: prioritized recurrent experience replay.
            target_net_update_interval: how often to copy online network parameters to target.
            min_replay_size: wait till experience replay buffer this number before start to learn.
            batch_size: sample batch_size of transitions.
            burn_in: burn n transitions to generate initial hidden state before learning.
            unroll_length: transition sequence length.
            retrace_lambda: coefficient of the retrace lambda.
            transformed_retrace: if True, use transformed retrace.
            priority_eta: coefficient to mix the max and mean absolute TD errors.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """
        if not 1 <= target_net_update_interval:
            raise ValueError(f'Expect target_net_update_interval to be positive integer, got {target_net_update_interval}')
        if not 1 <= min_replay_size:
            raise ValueError(f'Expect min_replay_size to be integer greater than or equal to 1, got {min_replay_size}')
        if not 1 <= batch_size <= 128:
            raise ValueError(f'Expect batch_size to in the range [1, 128], got {batch_size}')
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be greater than or equal to 1, got {unroll_length}')
        if not 0 <= burn_in < unroll_length:
            raise ValueError(f'Expect burn_in length to be [0, {unroll_length}), got {burn_in}')
        if not 0.0 <= retrace_lambda <= 1.0:
            raise ValueError(f'Expect retrace_lambda to in the range [0.0, 1.0], got {retrace_lambda}')
        if not 0.0 <= priority_eta <= 1.0:
            raise ValueError(f'Expect priority_eta to in the range [0.0, 1.0], got {priority_eta}')

        self.agent_name = 'NGU-learner'
        self._device = device
        self._network = network.to(device=device)
        self._network.train()
        self._optimizer = optimizer
        self._embedding_network = embedding_network.to(device=self._device)
        self._embedding_network.train()
        self._rnd_predictor_network = rnd_predictor_network.to(device=self._device)
        self._rnd_predictor_network.train()
        self._intrinsic_optimizer = intrinsic_optimizer

        self._rnd_target_network = rnd_target_network.to(device=self._device)
        # Lazy way to create target Q network
        self._target_network = copy.deepcopy(self._network).to(device=self._device)

        # Disable autograd for target Q network and RND target network
        no_autograd(self._target_network)
        no_autograd(self._rnd_target_network)

        self._shared_params = shared_params

        self._batch_size = batch_size
        self._burn_in = burn_in
        self._unroll_length = unroll_length
        self._total_unroll_length = unroll_length + 1
        self._target_net_update_interval = target_net_update_interval
        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        # Accumulate running statistics to calculate mean and std
        self._rnd_obs_normalizer = normalizer.TorchRunningMeanStd(shape=(1, 84, 84), device=self._device)

        self._replay = replay
        self._min_replay_size = min_replay_size
        self._priority_eta = priority_eta

        self._max_seen_priority = 1.0  # New unroll will use this as priority

        self._retrace_lambda = retrace_lambda
        self._transformed_retrace = transformed_retrace

        self._step_t = -1
        self._update_t = 0
        self._target_update_t = 0
        self._retrace_loss_t = np.nan
        self._rnd_loss_t = np.nan
        self._embed_loss_t = np.nan

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

    def get_embedding_network_state_dict(self):
        # To keep things consistent, we move the parameters to CPU
        return {k: v.cpu() for k, v in self._embedding_network.state_dict().items()}

    def get_rnd_predictor_network_state_dict(self):
        # To keep things consistent, we move the parameters to CPU
        return {k: v.cpu() for k, v in self._rnd_predictor_network.state_dict().items()}

    def _learn(self) -> None:
        transitions, indices, weights = self._replay.sample(self._batch_size)
        priorities = self._update_q_network(transitions, weights)
        self._update_embed_and_rnd_predictor_networks(transitions, weights)
        self._update_t += 1

        if priorities.shape != (self._batch_size,):
            raise RuntimeError(f'Expect priorities has shape ({self._batch_size},), got {priorities.shape}')
        priorities = np.abs(priorities)
        self._max_seen_priority = np.max([self._max_seen_priority, np.max(priorities)])
        self._replay.update_priorities(indices, priorities)

        self._shared_params['network'] = self.get_network_state_dict()
        self._shared_params['embedding_network'] = self.get_embedding_network_state_dict()
        self._shared_params['rnd_predictor_network'] = self.get_rnd_predictor_network_state_dict()

        # Copy Q network parameters to target Q network, every m updates
        if self._update_t > 1 and self._update_t % self._target_net_update_interval == 0:
            self._update_target_network()

    def _update_q_network(self, transitions: NguTransition, weights: np.ndarray) -> np.ndarray:
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

        # Compute predicted q values using online and target Q networks.
        q_t = self._get_predicted_q_values(learn_transitions, self._network, hidden_s)
        with torch.no_grad():
            target_q_t = self._get_predicted_q_values(learn_transitions, self._target_network, target_hidden_s)

        # [batch_size]
        retrace_loss, priorities = self._calc_retrace_loss(learn_transitions, q_t, target_q_t.detach())
        # Multiply loss by sampling weights, averaging over batch dimension
        loss = torch.mean(retrace_loss * weights.detach())

        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), self._max_grad_norm)

        self._optimizer.step()

        # For logging only.
        self._retrace_loss_t = loss.detach().cpu().item()

        return priorities

    def _get_predicted_q_values(
        self, transitions: NguTransition, q_network: torch.nn.Module, hidden_state: HiddenState
    ) -> torch.Tensor:
        """Returns the predicted q values from the 'q_network' for a given batch of sampled unrolls.

        Args:
            transitions: sampled batch of unrolls, this should not include the burn_in part.
            q_network: this could be one of the online or target Q networks.
            hidden_state: initial hidden states for the 'q_network'.
        """

        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [T+1, B, state_shape]
        last_action = torch.from_numpy(transitions.last_action).to(device=self._device, dtype=torch.int64)  # [T+1, B]
        ext_r_t = torch.from_numpy(transitions.ext_r_t).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        int_r_t = torch.from_numpy(transitions.int_r_t).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        policy_index = torch.from_numpy(transitions.policy_index).to(device=self._device, dtype=torch.int64)  # [T+1, B]

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        base.assert_rank_and_dtype(s_t, (3, 5), torch.float32)
        base.assert_rank_and_dtype(last_action, 2, torch.long)
        base.assert_rank_and_dtype(ext_r_t, 2, torch.float32)
        base.assert_rank_and_dtype(int_r_t, 2, torch.float32)
        base.assert_rank_and_dtype(policy_index, 2, torch.long)

        assert not torch.any(torch.isnan(s_t))
        assert not torch.any(torch.isnan(last_action))
        assert not torch.any(torch.isnan(ext_r_t))
        assert not torch.any(torch.isnan(int_r_t))
        assert not torch.any(torch.isnan(policy_index))

        # Get q values from Q network
        q_t = q_network(
            NguNetworkInputs(
                s_t=s_t,
                a_tm1=last_action,
                ext_r_t=ext_r_t,
                int_r_t=int_r_t,
                policy_index=policy_index,
                hidden_s=hidden_state,
            )
        ).q_values

        assert not torch.any(torch.isnan(q_t))

        return q_t

    def _calc_retrace_loss(
        self,
        transitions: NguTransition,
        q_t: torch.Tensor,
        target_q_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Calculate loss and priorities for given unroll sequence transitions."""
        a_t = torch.from_numpy(transitions.a_t).to(device=self._device, dtype=torch.int64)  # [T+1, B]
        behavior_prob_a_t = torch.from_numpy(transitions.prob_a_t).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        ext_r_t = torch.from_numpy(transitions.ext_r_t).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        int_r_t = torch.from_numpy(transitions.int_r_t).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        beta = torch.from_numpy(transitions.beta).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        discount = torch.from_numpy(transitions.discount).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        done = torch.from_numpy(transitions.done).to(device=self._device, dtype=torch.bool)  # [T+1, B]

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        base.assert_rank_and_dtype(behavior_prob_a_t, 2, torch.float32)
        base.assert_rank_and_dtype(a_t, 2, torch.long)
        base.assert_rank_and_dtype(ext_r_t, 2, torch.float32)
        base.assert_rank_and_dtype(int_r_t, 2, torch.float32)
        base.assert_rank_and_dtype(beta, 2, torch.float32)
        base.assert_rank_and_dtype(discount, 2, torch.float32)
        base.assert_rank_and_dtype(done, 2, torch.bool)

        r_t = ext_r_t + beta * int_r_t  # Augmented rewards
        discount_t = (~done).float() * discount  # (T+1, B)

        # Derive target policy probabilities from q values.
        target_policy_probs = F.softmax(target_q_t, dim=-1)  # [T+1, B, action_dim]

        if self._transformed_retrace:
            transform_tx_pair = nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR
        else:
            transform_tx_pair = nonlinear_bellman.IDENTITY_PAIR  # No transform

        # Compute retrace loss.
        retrace_out = nonlinear_bellman.transformed_retrace(
            q_tm1=q_t[:-1],
            q_t=target_q_t[1:],
            a_tm1=a_t[:-1],
            a_t=a_t[1:],
            r_t=r_t[1:],
            discount_t=discount_t[1:],
            pi_t=target_policy_probs[1:],
            mu_t=behavior_prob_a_t[1:],
            lambda_=self._retrace_lambda,
            tx_pair=transform_tx_pair,
        )

        # Compute priority.
        with torch.no_grad():
            priorities = distributed.calculate_dist_priorities_from_td_error(retrace_out.extra.td_error, self._priority_eta)
        # Sums over time dimension.
        loss = torch.sum(retrace_out.loss, dim=0)
        return (loss, priorities)

    def _update_embed_and_rnd_predictor_networks(self, transitions: NguTransition, weights: np.ndarray) -> None:
        """Update the embedding action prediction and RND predictor networks."""
        b = self._batch_size
        weights = torch.from_numpy(weights[-b:]).to(device=self._device, dtype=torch.float32)  # [B]
        base.assert_rank_and_dtype(weights, 1, torch.float32)

        self._intrinsic_optimizer.zero_grad()
        # [batch_size]
        rnd_loss = self._calc_rnd_loss(transitions)
        embed_loss = self._calc_embed_inverse_loss(transitions)

        # Multiply loss by sampling weights, averaging over batch dimension
        loss = torch.mean((rnd_loss + embed_loss) * weights.detach())

        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(self._rnd_predictor_network.parameters(), self._max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._embedding_network.parameters(), self._max_grad_norm)

        self._intrinsic_optimizer.step()

        # For logging only.
        self._rnd_loss_t = rnd_loss.detach().mean().cpu().item()
        self._embed_loss_t = embed_loss.detach().mean().cpu().item()

    def _calc_rnd_loss(self, transitions: NguTransition) -> torch.Tensor:
        s_t = torch.from_numpy(transitions.s_t[-5:]).to(device=self._device, dtype=torch.float32)  # [5, B, state_shape]
        # Rank and dtype checks.
        base.assert_rank_and_dtype(s_t, (3, 5), torch.float32)
        # Merge batch and time dimension.
        s_t = torch.flatten(s_t, 0, 1)

        normed_s_t = self._normalize_rnd_obs(s_t)

        pred_s_t = self._rnd_predictor_network(normed_s_t)
        with torch.no_grad():
            target_s_t = self._rnd_target_network(normed_s_t)

        rnd_loss = torch.square(pred_s_t - target_s_t).mean(dim=1)
        # Reshape loss into [5, B].
        rnd_loss = rnd_loss.view(5, -1)

        # Sums over time dimension. shape [B]
        loss = torch.sum(rnd_loss, dim=0)

        return loss

    def _calc_embed_inverse_loss(self, transitions: NguTransition) -> torch.Tensor:
        s_t = torch.from_numpy(transitions.s_t[-6:]).to(device=self._device, dtype=torch.float32)  # [6, B, state_shape]
        a_t = torch.from_numpy(transitions.a_t[-6:]).to(device=self._device, dtype=torch.int64)  # [6, B]

        # Rank and dtype checks.
        base.assert_rank_and_dtype(s_t, (3, 5), torch.float32)
        base.assert_rank_and_dtype(a_t, 2, torch.long)

        s_tm1 = s_t[0:-1, ...]  # [5, B, state_shape]
        s_t = s_t[1:, ...]  # [5, B, state_shape]
        a_tm1 = a_t[:-1, ...]  # [5, B]

        # Merge batch and time dimension.
        s_tm1 = torch.flatten(s_tm1, 0, 1)
        s_t = torch.flatten(s_t, 0, 1)
        a_tm1 = torch.flatten(a_tm1, 0, 1)

        # Compute action prediction loss.
        embedding_s_tm1 = self._embedding_network(s_tm1)  # [5*B, latent_dim]
        embedding_s_t = self._embedding_network(s_t)  # [5*B, latent_dim]
        embeddings = torch.cat([embedding_s_tm1, embedding_s_t], dim=-1)
        pi_logits = self._embedding_network.inverse_prediction(embeddings)  # [5*B, action_dim]

        loss = F.cross_entropy(pi_logits, a_tm1, reduction='none')  # [5*B,]
        # Reshape loss into [5, B].
        loss = loss.view(5, -1)

        # Sums over time dimension. shape [B]
        loss = torch.sum(loss, dim=0)
        return loss

    @torch.no_grad()
    def _normalize_rnd_obs(self, rnd_obs):
        rnd_obs = rnd_obs.to(device=self._device, dtype=torch.float32)

        normed_obs = self._rnd_obs_normalizer.normalize(rnd_obs)
        normed_obs = normed_obs.clamp(-5, 5)

        self._rnd_obs_normalizer.update(rnd_obs)

        return normed_obs

    @torch.no_grad()
    def _burn_in_unroll_q_networks(
        self,
        transitions: NguTransition,
        init_hidden_s: HiddenState,
    ) -> Tuple[HiddenState, HiddenState]:
        """Unroll both online and target q networks to generate hidden states for LSTM."""
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [burn_in, B, state_shape]
        last_action = torch.from_numpy(transitions.last_action).to(device=self._device, dtype=torch.int64)  # [burn_in, B]
        ext_r_t = torch.from_numpy(transitions.ext_r_t).to(device=self._device, dtype=torch.float32)  # [burn_in, B]
        int_r_t = torch.from_numpy(transitions.int_r_t).to(device=self._device, dtype=torch.float32)  # [burn_in, B]
        policy_index = torch.from_numpy(transitions.policy_index).to(device=self._device, dtype=torch.int64)  # [burn_in, B]

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        base.assert_rank_and_dtype(s_t, (3, 5), torch.float32)
        base.assert_rank_and_dtype(last_action, 2, torch.long)
        base.assert_rank_and_dtype(ext_r_t, 2, torch.float32)
        base.assert_rank_and_dtype(int_r_t, 2, torch.float32)
        base.assert_rank_and_dtype(policy_index, 2, torch.long)

        _hidden_s = tuple(s.clone().to(device=self._device) for s in init_hidden_s)
        _target_hidden_s = tuple(s.clone().to(device=self._device) for s in init_hidden_s)

        # Burn in to generate hidden states for LSTM, we unroll both online and target Q networks
        hidden_s = self._network(
            NguNetworkInputs(
                s_t=s_t,
                a_tm1=last_action,
                ext_r_t=ext_r_t,
                int_r_t=int_r_t,
                policy_index=policy_index,
                hidden_s=_hidden_s,
            )
        ).hidden_s
        target_hidden_s = self._target_network(
            NguNetworkInputs(
                s_t=s_t,
                a_tm1=last_action,
                ext_r_t=ext_r_t,
                int_r_t=int_r_t,
                policy_index=policy_index,
                hidden_s=_target_hidden_s,
            )
        ).hidden_s

        return (hidden_s, target_hidden_s)

    def _extract_first_step_hidden_state(self, transitions: NguTransition) -> HiddenState:
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
            # 'ext_lr': self._optimizer.param_groups[0]['lr'],
            # 'int_lr': self._intrinsic_optimizer.param_groups[0]['lr'],
            'retrace_loss': self._retrace_loss_t,
            'rnd_loss': self._rnd_loss_t,
            'embed_loss': self._embed_loss_t,
            'updates': self._update_t,
            'target_updates': self._target_update_t,
        }
