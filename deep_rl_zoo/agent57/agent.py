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
"""Agent57 agent class.

From the paper "Agent57: Outperforming the Atari Human Benchmark"
https://arxiv.org/pdf/2003.13350.
"""

from typing import Mapping, Optional, Tuple, NamedTuple
import queue
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
from deep_rl_zoo import transforms
from deep_rl_zoo import bandit
from deep_rl_zoo.curiosity import EpisodicBonusModule, RndLifeLongBonusModule
from deep_rl_zoo.networks.dqn import NguDqnNetworkInputs

# torch.autograd.set_detect_anomaly(True)

HiddenState = Tuple[torch.Tensor, torch.Tensor]


class Agent57Transition(NamedTuple):
    r"""Ideally we want to construct transition in the manner of (s, a, r), this is not the case for gym env.
    when the agent takes action 'a' in state 's', it does not receive rewards immediately, only at the next time step.
    so there's going to be one timestep lag between the (s, a, r).

    For simplicity reasons, we choose to lag rewards, so (s, a, r) is actually (s', a', r).

    In our case 'a_t', 'done' is for 's_t', while
    'int_r_t' is for previous state-action (s_tm1, a_tm1) but received at current timestep.

    Note 'a_tm1' is only used for network pass during learning,
    'q_values' is only for calculating priority for the unroll sequence when adding into replay."""

    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    q_t: Optional[np.ndarray]  # q values for s_t, computed from both ext_q_network and int_q_network
    prob_a_t: Optional[np.ndarray]  # probability of choose a_t in s_t
    a_tm1: Optional[int]
    ext_r_t: Optional[float]  # extrinsic reward for (s_tm1, a_tm1)
    int_r_t: Optional[float]  # intrinsic reward for (s_tm1)
    policy_index: Optional[int]  # intrinsic reward scale beta index
    beta: Optional[float]  # intrinsic reward scale beta value
    discount: Optional[float]
    done: Optional[bool]
    ext_init_h: Optional[np.ndarray]  # nn.LSTM initial hidden state, from ext_q_network
    ext_init_c: Optional[np.ndarray]  # nn.LSTM initial cell state, from ext_q_network
    int_init_h: Optional[np.ndarray]  # nn.LSTM initial hidden state, from int_q_network
    int_init_c: Optional[np.ndarray]  # nn.LSTM initial cell state, from int_q_network


TransitionStructure = Agent57Transition(
    s_t=None,
    a_t=None,
    q_t=None,
    prob_a_t=None,
    a_tm1=None,
    ext_r_t=None,
    int_r_t=None,
    policy_index=None,
    beta=None,
    discount=None,
    done=None,
    ext_init_h=None,
    ext_init_c=None,
    int_init_h=None,
    int_init_c=None,
)


def compute_transformed_q(ext_q: torch.Tensor, int_q: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Returns transformed state-action values from ext_q and int_q."""
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta).expand_as(int_q).to(device=ext_q.device)

    if len(beta.shape) < len(int_q.shape):
        beta = beta[..., None].expand_as(int_q)

    return transforms.signed_hyperbolic(transforms.signed_parabolic(ext_q) + beta * transforms.signed_parabolic(int_q))


def no_autograd(net: torch.nn.Module):
    """Disable autograd for a network."""
    for p in net.parameters():
        p.requires_grad = False


class Actor(types_lib.Agent):
    """Agent57 actor"""

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        ext_q_network: torch.nn.Module,
        int_q_network: torch.nn.Module,
        learner_ext_q_network: torch.nn.Module,
        learner_int_q_network: torch.nn.Module,
        rnd_target_network: torch.nn.Module,
        rnd_predictor_network: torch.nn.Module,
        learner_rnd_predictor_network: torch.nn.Module,
        embedding_network: torch.nn.Module,
        learner_embedding_network: torch.nn.Module,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        ext_discount: float,
        int_discount: float,
        num_actors: int,
        num_actions: int,
        unroll_length: int,
        burn_in: int,
        num_policies: int,
        policy_beta: float,
        ucb_window_size: int,
        ucb_beta: float,
        ucb_epsilon: float,
        episodic_memory_capacity: int,
        num_neighbors: int,
        cluster_distance: float,
        kernel_epsilon: float,
        max_similarity: float,
        actor_update_frequency: int,
        device: torch.device,
    ) -> None:
        """
        Args:
            rank: the rank number for the actor.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            network: the Q network for actor to make action choice.
            learner_network: the Q networks with updated weights.
            rnd_target_network: RND random target network.
            rnd_predictor_network: RND predictor target network.
            learner_rnd_predictor_network: RND predictor target network with updated weights.
            embedding_network: NGU action prediction network.
            learner_embedding_network: NGU action prediction network with updated weights.
            random_state: random state.
            ext_discount: extrinsic reward discount.
            int_discount: intrinsic reward discount.
            num_actors: number of actors.
            num_actions: number of valid actions in the environment.
            unroll_length: how many agent time step to unroll transitions before put on to queue.
            burn_in: two consecutive unrolls will overlap on burn_in+1 steps.
            num_policies: number of exploring and exploiting policies.
            policy_beta: intrinsic reward scale beta.
            ucb_window_size: window size of the sliding window UCB algorithm.
            ucb_beta: beta for the sliding window UCB algorithm.
            ucb_epsilon: exploration epsilon for sliding window UCB algorithm.
            episodic_memory_capacity: maximum capacity of episodic memory.
            num_neighbors: number of K-NN neighbors for compute episodic bonus.
            cluster_distance: K-NN neighbors cluster distance for compute episodic bonus.
            kernel_epsilon: K-NN kernel epsilon for compute episodic bonus.
            max_similarity: maximum similarity for compute episodic bonus.
            actor_update_frequency: the frequence to update actor's Q network.
            device: PyTorch runtime device.
        """
        if not 0.0 <= ext_discount <= 1.0:
            raise ValueError(f'Expect ext_discount to be [0.0, 1.0], got {ext_discount}')
        if not 0.0 <= int_discount <= 1.0:
            raise ValueError(f'Expect int_discount to be [0.0, 1.0], got {int_discount}')
        if not 0 < num_actors:
            raise ValueError(f'Expect num_actors to be positive integer, got {num_actors}')
        if not 0 < num_actions:
            raise ValueError(f'Expect num_actions to be positive integer, got {num_actions}')
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer geater than or equal to 1, got {unroll_length}')
        if not 0 <= burn_in < unroll_length:
            raise ValueError(f'Expect unroll_burn_inlength to be [0, {unroll_length}), got {burn_in}')
        if not 1 <= num_policies:
            raise ValueError(f'Expect num_policies to be integer geater than or equal to 1, got {num_policies}')
        if not 0.0 <= policy_beta <= 1.0:
            raise ValueError(f'Expect policy_beta to be [0.0, 1.0], got {policy_beta}')
        if not 1 <= ucb_window_size:
            raise ValueError(f'Expect ucb_window_size to be integer geater than or equal to 1, got {ucb_window_size}')
        if not 0.0 <= ucb_beta <= 100.0:
            raise ValueError(f'Expect ucb_beta to be [0.0, 100.0], got {ucb_beta}')
        if not 0.0 <= ucb_epsilon <= 1.0:
            raise ValueError(f'Expect ucb_epsilon to be [0.0, 1.0], got {ucb_epsilon}')
        if not 1 <= episodic_memory_capacity:
            raise ValueError(
                f'Expect episodic_memory_capacity to be integer geater than or equal to 1, got {episodic_memory_capacity}'
            )
        if not 1 <= num_neighbors:
            raise ValueError(f'Expect num_neighbors to be integer geater than or equal to 1, got {num_neighbors}')
        if not 0.0 <= cluster_distance <= 1.0:
            raise ValueError(f'Expect cluster_distance to be [0.0, 1.0], got {cluster_distance}')
        if not 0.0 <= kernel_epsilon <= 1.0:
            raise ValueError(f'Expect kernel_epsilon to be [0.0, 1.0], got {kernel_epsilon}')
        if not 1 <= actor_update_frequency:
            raise ValueError(
                f'Expect actor_update_frequency to be integer geater than or equal to 1, got {actor_update_frequency}'
            )

        self.rank = rank  # Needs to make sure rank always start from 0
        self.agent_name = f'Agent57-actor{rank}'

        self._ext_q_network = ext_q_network.to(device=device)
        self._int_q_network = int_q_network.to(device=device)
        self._learner_ext_q_network = learner_ext_q_network.to(device=device)
        self._learner_int_q_network = learner_int_q_network.to(device=device)
        self._rnd_target_network = rnd_target_network.to(device=device)
        self._rnd_predictor_network = rnd_predictor_network.to(device=device)
        self._embedding_network = embedding_network.to(device=device)
        self._learner_rnd_predictor_network = learner_rnd_predictor_network.to(device=device)
        self._learner_embedding_network = learner_embedding_network.to(device=device)

        self._update_actor_q_network()

        # Disable autograd for actor's Q networks, embedding, and RND networks.
        no_autograd(self._ext_q_network)
        no_autograd(self._int_q_network)
        no_autograd(self._rnd_target_network)
        no_autograd(self._rnd_predictor_network)
        no_autograd(self._embedding_network)

        self._queue = data_queue
        self._device = device
        self._random_state = random_state
        self._num_actors = num_actors
        self._num_actions = num_actions
        self._actor_update_frequency = actor_update_frequency
        self._num_policies = num_policies

        self._unroll = replay_lib.Unroll(
            unroll_length=unroll_length,
            overlap=burn_in + 1,  # Plus 1 to add room for shift during learning
            structure=TransitionStructure,
            cross_episode=False,
        )

        # Meta-collector
        self._meta_coll = bandit.SimplifiedSlidingWindowUCB(
            self._num_policies, ucb_window_size, self._random_state, ucb_beta, ucb_epsilon
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

        # E-greedy policy epsilon, rank 0 has the lowest noise, while rank N-1 has the highes noise.
        epsilons = distributed.get_actor_exploration_epsilon(num_actors)
        self._exploration_epsilon = epsilons[self.rank]

        # Episodic intrinsic bonus module
        self._episodic_module = EpisodicBonusModule(
            embedding_network=embedding_network,
            device=device,
            capacity=episodic_memory_capacity,
            num_neighbors=num_neighbors,
            kernel_epsilon=kernel_epsilon,
            cluster_distance=cluster_distance,
            max_similarity=max_similarity,
        )

        # Lifelong intrinsic bonus module
        self._lifelong_module = RndLifeLongBonusModule(
            target_network=rnd_target_network,
            predictor_network=rnd_predictor_network,
            device=device,
        )

        self._episode_returns = 0.0
        self._a_tm1 = None
        self._episodic_bonus_t = None
        self._lifelong_bonus_t = None
        self._ext_lstm_state = None  # Stores nn.LSTM hidden state and cell state. for extrinsic Q network
        self._int_lstm_state = None  # Stores nn.LSTM hidden state and cell state. for intrinsic Q network

        self._step_t = -1

    @torch.no_grad()
    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1
        self._episode_returns += timestep.reward

        if self._step_t % self._actor_update_frequency == 0:
            self._update_actor_q_network()

        q_t, a_t, prob_a_t, ext_hidden_s, int_hidden_s = self.act(timestep)

        transition = Agent57Transition(
            s_t=timestep.observation,
            a_t=a_t,
            q_t=q_t,
            prob_a_t=prob_a_t,
            a_tm1=self._a_tm1,
            ext_r_t=timestep.reward,
            int_r_t=self.intrinsic_reward,
            policy_index=self._policy_index,
            beta=self._policy_beta,
            discount=self._policy_discount,
            done=timestep.done,
            ext_init_h=self._ext_lstm_state[0].squeeze(1).cpu().numpy(),  # remove batch dimension
            ext_init_c=self._ext_lstm_state[1].squeeze(1).cpu().numpy(),
            int_init_h=self._int_lstm_state[0].squeeze(1).cpu().numpy(),  # remove batch dimension
            int_init_c=self._int_lstm_state[1].squeeze(1).cpu().numpy(),
        )

        unrolled_transition = self._unroll.add(transition, timestep.done)

        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)

        # Compute lifelong intrinsic bonus
        self._lifelong_bonus_t = self._lifelong_module.compute_bonus(s_t)

        # Compute episodic intrinsic bonus
        self._episodic_bonus_t = self._episodic_module.compute_bonus(s_t)

        # Update local state
        self._a_tm1, self._ext_lstm_state, self._int_lstm_state = a_t, ext_hidden_s, int_hidden_s

        if unrolled_transition is not None:
            self._put_unroll_onto_queue(unrolled_transition)

        # Update Sliding Window UCB statistics.
        if timestep.done:
            self._meta_coll.update(self._policy_index, self._episode_returns)

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode before take any action."""
        self._unroll.reset()
        self._episodic_module.reset()
        self._episode_returns = 0.0

        # Update embedding and RND predictor network weights at beginning of every episode.
        self._update_embedding_and_rnd_networks()

        # Agent57 actor samples a policy using the Sliding Window UCB algorithm, then play a single episode.
        self._sample_policy()

        # During the first step of a new episode,
        # use 'fake' previous action and 'intrinsic' reward for network pass
        self._a_tm1 = self._random_state.randint(0, self._num_actions)  # Initialize a_tm1 randomly
        self._episodic_bonus_t = 0.0
        self._lifelong_bonus_t = 0.0
        self._ext_lstm_state = self._ext_q_network.get_initial_hidden_state(batch_size=1)
        self._int_lstm_state = self._int_q_network.get_initial_hidden_state(batch_size=1)

    def act(self, timestep: types_lib.TimeStep) -> Tuple[np.ndarray, types_lib.Action, float, HiddenState, HiddenState]:
        'Given state s_t and done marks, return an action.'
        return self._choose_action(timestep, self._exploration_epsilon)

    @torch.no_grad()
    def _choose_action(
        self, timestep: types_lib.TimeStep, epsilon: float
    ) -> Tuple[np.ndarray, types_lib.Action, float, HiddenState, HiddenState]:
        """Given state s_t, choose action a_t"""
        q_ext_input_ = self._prepare_network_input(timestep, self._ext_lstm_state)
        q_int_input_ = self._prepare_network_input(timestep, self._int_lstm_state)

        pi_ext_output = self._ext_q_network(q_ext_input_)
        pi_int_output = self._int_q_network(q_int_input_)
        ext_q_t = pi_ext_output.q_values.squeeze()
        int_q_t = pi_int_output.q_values.squeeze()

        q_t = compute_transformed_q(ext_q_t, int_q_t, self._policy_beta)

        a_t = torch.argmax(q_t, dim=-1).cpu().item()

        # Policy probability for a_t, the detailed equation is mentioned in Agent57 paper.
        prob_a_t = 1 - (self._exploration_epsilon * ((self._num_actions - 1) / self._num_actions))

        # To make sure every actors generates the same amount of samples, we apply e-greedy after the network pass,
        # otherwise the actor with higher epsilons will generate more samples,
        # while the actor with lower epsilon will geneate less samples.
        if self._random_state.rand() <= epsilon:
            # randint() return random integers from low (inclusive) to high (exclusive).
            a_t = self._random_state.randint(0, self._num_actions)
            prob_a_t = self._exploration_epsilon / self._num_actions

        return (
            q_t.cpu().numpy(),
            a_t,
            prob_a_t,
            pi_ext_output.hidden_s,
            pi_int_output.hidden_s,
        )

    def _prepare_network_input(self, timestep: types_lib.TimeStep, hidden_state: Tuple[torch.Tensor]) -> NguDqnNetworkInputs:
        # NGU network expect input shape [T, B, state_shape],
        # and additionally 'last action', 'extrinsic reward for last action', last intrinsic reward, and intrinsic reward scale beta index.
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(self._a_tm1).to(device=self._device, dtype=torch.int64)
        ext_r_t = torch.tensor(timestep.reward).to(device=self._device, dtype=torch.float32)
        int_r_t = torch.tensor(self.intrinsic_reward).to(device=self._device, dtype=torch.float32)
        policy_index = torch.tensor(self._policy_index).to(device=self._device, dtype=torch.int64)
        hidden_s = tuple(s.to(device=self._device) for s in hidden_state)
        return NguDqnNetworkInputs(
            s_t=s_t[None, ...],  # [T, B, state_shape]
            a_tm1=a_tm1[None, ...],  # [T, B]
            ext_r_t=ext_r_t[None, ...],  # [T, B]
            int_r_t=int_r_t[None, ...],  # [T, B]
            policy_index=policy_index[None, ...],  # [T, B]
            hidden_s=hidden_s,
        )

    def _put_unroll_onto_queue(self, unrolled_transition):
        # Important note, store hidden states for every step in the unroll will consume HUGE memory.
        self._queue.put(unrolled_transition)

    def _sample_policy(self):
        """Sample new policy from meta collector."""
        self._policy_index = self._meta_coll.sample()
        self._policy_beta = self._betas[self._policy_index]
        self._policy_discount = self._gammas[self._policy_index]

    def _update_actor_q_network(self):
        self._ext_q_network.load_state_dict(self._learner_ext_q_network.state_dict())
        self._int_q_network.load_state_dict(self._learner_int_q_network.state_dict())

    def _update_embedding_and_rnd_networks(self):
        self._lifelong_module.update_predictor_network(self._learner_rnd_predictor_network.state_dict())
        self._episodic_module.update_embedding_network(self._learner_embedding_network.state_dict())

    @property
    def intrinsic_reward(self) -> float:
        """Returns intrinsic reward for last state s_tm1."""
        # Equation 1 of the NGU paper.
        return self._episodic_bonus_t * min(max(self._lifelong_bonus_t, 1.0), 5.0)

    @property
    def statistics(self) -> Mapping[str, float]:
        """Returns current actor's statistics as a dictionary."""
        return {
            'policy_index': self._policy_index,
            'policy_discount': self._policy_discount,
            'policy_beta': self._policy_beta,
            'exploration_epsilon': self._exploration_epsilon,
            'intrinsic_reward': self.intrinsic_reward,
            'episodic_bonus': self._episodic_bonus_t,
            'lieflong_bonus': self._lifelong_bonus_t,
        }


class Learner:
    """Agent57 learner"""

    def __init__(
        self,
        data_queue: multiprocessing.Queue,
        ext_q_network: nn.Module,
        ext_q_optimizer: torch.optim.Optimizer,
        int_q_network: nn.Module,
        int_q_optimizer: torch.optim.Optimizer,
        embedding_network: nn.Module,
        rnd_target_network: nn.Module,
        rnd_predictor_network: nn.Module,
        intrinsic_optimizer: torch.optim.Optimizer,
        replay: replay_lib.PrioritizedReplay,
        target_network_update_frequency: int,
        min_replay_size: int,
        batch_size: int,
        unroll_length: int,
        burn_in: int,
        retrace_lambda: float,
        transformed_retrace: bool,
        priority_eta: float,
        num_actors: int,
        clip_grad: bool,
        max_grad_norm: float,
        device: torch.device,
    ) -> None:
        """
        Args:
            data_queue: a multiprocessing.Queue to get collected transitions from actor processes.
            network: the Q network we want to train and optimize.
            optimizer: the optimizer for Q network.
            embedding_network: NGU action prediction network.
            rnd_target_network: RND random network.
            rnd_predictor_network: RND predictor network.
            intrinsic_optimizer: the optimizer for action prediction and RND predictor networks.
            replay: prioritized recurrent experience replay.
            target_network_update_frequency: how often to copy online network weights to target.
            min_replay_size: wait till experience replay buffer this number before start to learn.
            batch_size: sample batch_size of transitions.
            burn_in: burn n transitions to generate initial hidden state before learning.
            unroll_length: transition sequence length.
            retrace_lambda: coefficient of the retrace lambda.
            transformed_retrace: if True, use transformed retrace.
            priority_eta: coefficient to mix the max and mean absolute TD errors.
            num_actors: number of actor processes.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """
        if not 1 <= target_network_update_frequency:
            raise ValueError(
                f'Expect target_network_update_frequency to be positive integer, got {target_network_update_frequency}'
            )
        if not 1 <= min_replay_size:
            raise ValueError(f'Expect min_replay_size to be integer geater than or equal to 1, got {min_replay_size}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to in the range [1, 512], got {batch_size}')
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be geater than or equal to 1, got {unroll_length}')
        if not 0 <= burn_in < unroll_length:
            raise ValueError(f'Expect unroll_burn_inlength to be [0, {unroll_length}), got {burn_in}')
        if not 0.0 <= retrace_lambda <= 1.0:
            raise ValueError(f'Expect retrace_lambda to in the range [0.0, 1.0], got {retrace_lambda}')
        if not 0.0 <= priority_eta <= 1.0:
            raise ValueError(f'Expect priority_eta to in the range [0.0, 1.0], got {priority_eta}')
        if not 1 <= num_actors:
            raise ValueError(f'Expect num_actors to be integer geater than or equal to 1, got {num_actors}')

        self.agent_name = 'Agent57-learner'
        self._device = device
        self._online_ext_q_network = ext_q_network.to(device=device)
        self._ext_q_optimizer = ext_q_optimizer
        self._online_int_q_network = int_q_network.to(device=device)
        self._int_q_optimizer = int_q_optimizer
        # Lazy way to create target Q networks
        self._target_ext_q_network = copy.deepcopy(self._online_ext_q_network).to(device=self._device)
        self._target_int_q_network = copy.deepcopy(self._online_int_q_network).to(device=self._device)
        self._embedding_network = embedding_network.to(device=self._device)
        self._rnd_target_network = rnd_target_network.to(device=self._device)
        self._rnd_predictor_network = rnd_predictor_network.to(device=self._device)
        self._intrinsic_optimizer = intrinsic_optimizer

        self._update_target_network()

        # Disable autograd for target Q networks, and RND target networks.
        no_autograd(self._target_ext_q_network)
        no_autograd(self._target_int_q_network)
        no_autograd(self._rnd_target_network)

        self._batch_size = batch_size
        self._burn_in = burn_in
        self._unroll_length = unroll_length
        self._total_unroll_length = unroll_length + 1
        self._target_network_update_frequency = target_network_update_frequency
        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        self._observation_normalizer = normalizer.Normalizer(eps=0.0001, clip_range=(-5, 5), device=self._device)

        self._replay = replay
        self._min_replay_size = min_replay_size
        self._priority_eta = priority_eta

        self._retrace_lambda = retrace_lambda
        self._transformed_retrace = transformed_retrace

        self._queue = data_queue
        self._num_actors = num_actors

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
                    priority = self._compute_priority_for_unroll(item)
                    self._replay.add(item, priority)
            except queue.Empty:
                pass
            except EOFError:
                pass

            # Only break if all actor processes are done
            if self._done_actors == self._num_actors:
                break

            if self._replay.size < self._min_replay_size:
                continue

            # Pull a batch before learning
            if self._step_t % self._batch_size == 0:
                self._learn()

    def reset(self) -> None:
        """Should be called at the begining of every iteration."""
        self._done_actors = 0

    def _learn(self) -> None:
        transitions, indices, weights = self._replay.sample(self._batch_size)
        priorities = self._update(transitions, weights)
        self._update_action_prediction_and_rnd_predictor_networks(transitions, weights)

        if priorities.shape != (self._batch_size,):
            raise RuntimeError(f'Expect priorities has shape ({self._batch_size},), got {priorities.shape}')
        priorities = np.abs(priorities)
        self._replay.update_priorities(indices, priorities)

        # Copy online Q network weights to target Q network, every m updates
        if self._update_t % self._target_network_update_frequency == 0:
            self._update_target_network()

    def _update(self, transitions: Agent57Transition, weights: np.ndarray) -> np.ndarray:
        weights = torch.from_numpy(weights).to(device=self._device, dtype=torch.float32)  # [batch_size]
        base.assert_rank_and_dtype(weights, 1, torch.float32)

        # Get initial hidden state for both extrinsic and intrinsic Q networks, handle possible burn in.
        init_ext_q_hidden_state, init_int_q_hidden_state = self._extract_first_step_hidden_state(transitions)
        burn_transitions, learn_transitions = replay_lib.split_structure(transitions, TransitionStructure, self._burn_in)
        if burn_transitions is not None:
            # Burn in for extinsic Q networks.
            hidden_ext_online_q, hidden_ext_target_q = self._burn_in_unroll_q_networks(
                burn_transitions,
                self._online_ext_q_network,
                self._target_ext_q_network,
                init_ext_q_hidden_state,
            )
            # Burn in for intinsic Q networks.
            hidden_int_online_q, hidden_int_target_q = self._burn_in_unroll_q_networks(
                burn_transitions,
                self._online_int_q_network,
                self._target_int_q_network,
                init_int_q_hidden_state,
            )
        else:
            # Make copy of hidden state for extinsic Q networks.
            hidden_ext_online_q = tuple(s.clone().to(device=self._device) for s in init_ext_q_hidden_state)
            hidden_ext_target_q = tuple(s.clone().to(device=self._device) for s in init_ext_q_hidden_state)
            # Make copy of hidden state for intinsic Q networks.
            hidden_int_online_q = tuple(s.clone().to(device=self._device) for s in init_int_q_hidden_state)
            hidden_int_target_q = tuple(s.clone().to(device=self._device) for s in init_int_q_hidden_state)

        # Do network pass for all four Q networks to get estimated q values.
        ext_q_t = self._get_predicted_q_values(learn_transitions, self._online_ext_q_network, hidden_ext_online_q)
        int_q_t = self._get_predicted_q_values(learn_transitions, self._online_int_q_network, hidden_int_online_q)
        with torch.no_grad():
            target_ext_q_t = self._get_predicted_q_values(learn_transitions, self._target_ext_q_network, hidden_ext_target_q)
            target_int_q_t = self._get_predicted_q_values(learn_transitions, self._target_int_q_network, hidden_int_target_q)

        # Update extrinsic online Q network.
        self._ext_q_optimizer.zero_grad()
        ext_q_loss, ext_priorities = self._calc_retrace_loss(learn_transitions, ext_q_t, target_ext_q_t)
        # Multiply loss by sampling weights, averages over batch dimension
        ext_q_loss = torch.mean(ext_q_loss * weights.detach())
        ext_q_loss.backward()
        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(self._online_ext_q_network.parameters(), self._max_grad_norm)
        self._ext_q_optimizer.step()

        # Update intrinsic online Q network.
        self._int_q_optimizer.zero_grad()
        int_q_loss, int_priorities = self._calc_retrace_loss(learn_transitions, int_q_t, target_int_q_t)
        # Multiply loss by sampling weights, averages over batch dimension
        int_q_loss = torch.mean(int_q_loss * weights.detach())
        int_q_loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(self._online_int_q_network.parameters(), self._max_grad_norm)
        self._int_q_optimizer.step()

        priorities = 0.8 * ext_priorities + 0.2 * int_priorities
        self._update_t += 1
        return priorities

    def _get_predicted_q_values(
        self,
        transitions: Agent57Transition,
        q_network: torch.nn.Module,
        hidden_state: HiddenState,
    ) -> torch.Tensor:
        """Returns the predicted q values from the 'q_network' for a given batch of sampled unrolls.

        Args:
            transitions: sampled batch of unrolls, this should not include the burn_in part.
            q_network: this could be any one of the extrinsic and intrinsic (online or target) networks.
            hidden_state: initial hidden states for the 'q_network'.
        """
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [T+1, B, state_shape]
        a_tm1 = torch.from_numpy(transitions.a_tm1).to(device=self._device, dtype=torch.int64)  # [T+1, B]
        ext_r_t = torch.from_numpy(transitions.ext_r_t).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        int_r_t = torch.from_numpy(transitions.int_r_t).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        policy_index = torch.from_numpy(transitions.policy_index).to(device=self._device, dtype=torch.int64)  # [T+1, B]

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        base.assert_rank_and_dtype(s_t, (3, 5), torch.float32)
        base.assert_rank_and_dtype(a_tm1, 2, torch.long)
        base.assert_rank_and_dtype(ext_r_t, 2, torch.float32)
        base.assert_rank_and_dtype(int_r_t, 2, torch.float32)
        base.assert_rank_and_dtype(policy_index, 2, torch.long)

        # Rank and dtype checks for hidden state.
        base.assert_rank_and_dtype(hidden_state[0], 3, torch.float32)
        base.assert_rank_and_dtype(hidden_state[1], 3, torch.float32)
        base.assert_batch_dimension(hidden_state[0], self._batch_size, 1)
        base.assert_batch_dimension(hidden_state[1], self._batch_size, 1)

        # Get q values from Q network,
        q_t = q_network(
            NguDqnNetworkInputs(
                s_t=s_t,
                a_tm1=a_tm1,
                ext_r_t=ext_r_t,
                int_r_t=int_r_t,
                policy_index=policy_index,
                hidden_s=hidden_state,
            )
        ).q_values

        return q_t

    def _calc_retrace_loss(
        self,
        transitions: Agent57Transition,
        q_t: torch.Tensor,
        target_q_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, np.ndarray]:
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
        target_policy_probs = F.softmax(q_t, dim=-1)  # [T+1, B, num_actions]

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
            r_t=r_t[:-1],
            discount_t=discount_t[:-1],
            pi_t=target_policy_probs[1:],
            mu_t=behavior_prob_a_t[1:],
            lambda_=self._retrace_lambda,
            tx_pair=transform_tx_pair,
        )

        # Compute priority.
        priorities = distributed.calculate_dist_priorities_from_td_error(retrace_out.extra.td_error, self._priority_eta)

        # Sums over time dimension.
        loss = torch.sum(retrace_out.loss, dim=0)

        return loss, priorities

    def _update_action_prediction_and_rnd_predictor_networks(
        self, transitions: Agent57Transition, weights: np.ndarray
    ) -> None:
        """Use last 5 frames to update the embedding and RND predictor networks."""
        b = self._batch_size
        weights = torch.from_numpy(weights[-b:]).to(device=self._device, dtype=torch.float32)  # [B]
        base.assert_rank_and_dtype(weights, 1, torch.float32)

        self._intrinsic_optimizer.zero_grad()
        # [batch_size]
        rnd_pred_loss = self._calc_rnd_predictor_loss(transitions)
        act_pred_loss = self._calc_action_prediction_loss(transitions)
        loss = rnd_pred_loss + act_pred_loss
        # Multiply loss by sampling weights, averages over batch dimension
        loss = torch.mean(loss * weights.detach())

        loss.backward()
        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(self._rnd_predictor_network.parameters(), self._max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._embedding_network.parameters(), self._max_grad_norm)

        self._intrinsic_optimizer.step()

    def _calc_rnd_predictor_loss(self, transitions: Agent57Transition) -> torch.Tensor:
        s_t = torch.from_numpy(transitions.s_t[-5:]).to(device=self._device, dtype=torch.float32)  # [5, B, state_shape]
        # Rank and dtype checks.
        base.assert_rank_and_dtype(s_t, (3, 5), torch.float32)
        # Merge batch and time dimension.
        s_t = torch.flatten(s_t, 0, 1)

        # Compute RND predictor loss.
        # Update normalize statistics and normalize observations before pass to RND networks.
        if len(s_t.shape) > 3:
            # Make channel last, we normalize images by channel.
            s_t = s_t.swapaxes(1, -1)
            self._observation_normalizer.update(s_t)
            s_t = self._observation_normalizer(s_t)
            # Make channel first so PyTorch Conv2D works.
            s_t = s_t.swapaxes(1, -1)
        else:
            self._observation_normalizer.update(s_t)
            s_t = self._observation_normalizer(s_t)

        pred_s_t = self._rnd_predictor_network(s_t)
        with torch.no_grad():
            target_s_t = self._rnd_target_network(s_t)

        # Compute L2 loss, shape [5*B,]
        loss = torch.sum(torch.square(pred_s_t - target_s_t), dim=-1)
        # Reshape loss into [5, B].
        loss = loss.view(5, -1)
        # Sums over time dimension. shape [B]
        loss = torch.sum(loss, dim=0)
        return loss

    def _calc_action_prediction_loss(self, transitions: Agent57Transition) -> torch.Tensor:
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
        pi_logits = self._embedding_network.inverse_prediction(embeddings)  # [5*B, num_actions]

        # [5*B,]
        loss = F.cross_entropy(pi_logits, a_tm1, reduction='none')
        # Reshape loss into [5, B].
        loss = loss.view(5, -1)
        # Sums over time dimension. shape [B]
        loss = torch.sum(loss, dim=0)
        return loss

    def _burn_in_unroll_q_networks(
        self,
        transitions: Agent57Transition,
        online_q_network: torch.nn.Module,
        target_q_network: torch.nn.Module,
        hidden_state: HiddenState,
    ) -> Tuple[HiddenState, HiddenState]:
        """Unroll both online and target q networks to generate hidden states for LSTM."""
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [burn_in, B, state_shape]
        a_tm1 = torch.from_numpy(transitions.a_tm1).to(device=self._device, dtype=torch.int64)  # [burn_in, B]
        ext_r_t = torch.from_numpy(transitions.ext_r_t).to(device=self._device, dtype=torch.float32)  # [burn_in, B]
        int_r_t = torch.from_numpy(transitions.int_r_t).to(device=self._device, dtype=torch.float32)  # [burn_in, B]
        policy_index = torch.from_numpy(transitions.policy_index).to(device=self._device, dtype=torch.int64)  # [burn_in, B]

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        base.assert_rank_and_dtype(s_t, (3, 5), torch.float32)
        base.assert_rank_and_dtype(a_tm1, 2, torch.long)
        base.assert_rank_and_dtype(ext_r_t, 2, torch.float32)
        base.assert_rank_and_dtype(int_r_t, 2, torch.float32)
        base.assert_rank_and_dtype(policy_index, 2, torch.long)

        hidden_online = tuple(s.clone().to(device=self._device) for s in hidden_state)
        hidden_target = tuple(s.clone().to(device=self._device) for s in hidden_state)

        # Burn in to generate hidden states for LSTM, we unroll both online and target Q networks
        with torch.no_grad():
            hidden_online_q = online_q_network(
                NguDqnNetworkInputs(
                    s_t=s_t,
                    a_tm1=a_tm1,
                    ext_r_t=ext_r_t,
                    int_r_t=int_r_t,
                    policy_index=policy_index,
                    hidden_s=hidden_online,
                )
            ).hidden_s
            hidden_target_q = target_q_network(
                NguDqnNetworkInputs(
                    s_t=s_t,
                    a_tm1=a_tm1,
                    ext_r_t=ext_r_t,
                    int_r_t=int_r_t,
                    policy_index=policy_index,
                    hidden_s=hidden_target,
                )
            ).hidden_s

        return (hidden_online_q, hidden_target_q)

    def _extract_first_step_hidden_state(self, transitions: Agent57Transition) -> Tuple[HiddenState, HiddenState]:
        """Returns ext_hidden_state and int_hidden_state."""
        # We only need the first step hidden states in replay, shape [batch_size, num_lstm_layers, lstm_hidden_size]
        ext_init_h = torch.from_numpy(transitions.ext_init_h[0:1]).squeeze(0).to(device=self._device, dtype=torch.float32)
        ext_init_c = torch.from_numpy(transitions.ext_init_c[0:1]).squeeze(0).to(device=self._device, dtype=torch.float32)
        int_init_h = torch.from_numpy(transitions.int_init_h[0:1]).squeeze(0).to(device=self._device, dtype=torch.float32)
        int_init_c = torch.from_numpy(transitions.int_init_c[0:1]).squeeze(0).to(device=self._device, dtype=torch.float32)

        # Rank and dtype checks.
        base.assert_rank_and_dtype(ext_init_h, 3, torch.float32)
        base.assert_rank_and_dtype(ext_init_c, 3, torch.float32)
        base.assert_rank_and_dtype(int_init_h, 3, torch.float32)
        base.assert_rank_and_dtype(int_init_c, 3, torch.float32)

        # Swap batch and num_lstm_layers axis.
        ext_init_h = ext_init_h.swapaxes(0, 1)
        ext_init_c = ext_init_c.swapaxes(0, 1)
        int_init_h = int_init_h.swapaxes(0, 1)
        int_init_c = int_init_c.swapaxes(0, 1)

        # Batch dimension checks.
        base.assert_batch_dimension(ext_init_h, self._batch_size, 1)
        base.assert_batch_dimension(ext_init_c, self._batch_size, 1)
        base.assert_batch_dimension(int_init_h, self._batch_size, 1)
        base.assert_batch_dimension(int_init_c, self._batch_size, 1)

        return ((ext_init_h, ext_init_c), (int_init_h, int_init_c))

    @torch.no_grad()
    def _compute_priority_for_unroll(self, transitions: Agent57Transition) -> float:
        """Returns priority for a single unroll, no network pass and gradients are required."""
        # Note we skip the burn in part, and use the same q values for target.
        _, learn_transitions = replay_lib.split_structure(transitions, TransitionStructure, self._burn_in)

        q_t = torch.from_numpy(learn_transitions.q_t).to(device=self._device, dtype=torch.float32)  # [T+1, num_actions]
        a_t = torch.from_numpy(learn_transitions.a_t).to(device=self._device, dtype=torch.int64)  # [T+1, ]
        ext_r_t = torch.from_numpy(learn_transitions.ext_r_t).to(device=self._device, dtype=torch.float32)  # [T+1, ]
        int_r_t = torch.from_numpy(learn_transitions.int_r_t).to(device=self._device, dtype=torch.float32)  # [T+1, ]
        beta = torch.from_numpy(learn_transitions.beta).to(device=self._device, dtype=torch.float32)  # [T+1, ]
        discount = torch.from_numpy(learn_transitions.discount).to(device=self._device, dtype=torch.float32)  # [T+1, ]
        done = torch.from_numpy(learn_transitions.done).to(device=self._device, dtype=torch.bool)  # [T+1, ]
        behavior_prob_a_t = torch.from_numpy(learn_transitions.prob_a_t).to(
            device=self._device, dtype=torch.float32
        )  # [T+1, ]

        # Rank and dtype checks, single unroll should not have batch dimension.
        base.assert_rank_and_dtype(q_t, 2, torch.float32)
        base.assert_rank_and_dtype(a_t, 1, torch.long)
        base.assert_rank_and_dtype(ext_r_t, 1, torch.float32)
        base.assert_rank_and_dtype(int_r_t, 1, torch.float32)
        base.assert_rank_and_dtype(beta, 1, torch.float32)
        base.assert_rank_and_dtype(discount, 1, torch.float32)
        base.assert_rank_and_dtype(done, 1, torch.bool)
        base.assert_rank_and_dtype(behavior_prob_a_t, 1, torch.float32)

        r_t = ext_r_t + beta * int_r_t  # Augmented rewards
        discount_t = (~done).float() * discount

        # Derive policy probabilities from q values
        target_policy_probs = F.softmax(q_t, dim=-1)  # [T+1, num_actions]

        # Compute retrace loss, add a batch dimension before retrace ops.
        if self._transformed_retrace:
            transform_tx_pair = nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR
        else:
            transform_tx_pair = nonlinear_bellman.IDENTITY_PAIR  # No transform
        retrace_out = nonlinear_bellman.transformed_retrace(
            q_tm1=q_t[:-1].unsqueeze(1),
            q_t=q_t[1:].unsqueeze(1),
            a_tm1=a_t[:-1].unsqueeze(1),
            a_t=a_t[1:].unsqueeze(1),
            r_t=r_t[:-1].unsqueeze(1),
            discount_t=discount_t[:-1].unsqueeze(1),
            pi_t=target_policy_probs[1:].unsqueeze(1),
            mu_t=behavior_prob_a_t[1:].unsqueeze(1),
            lambda_=self._retrace_lambda,
            tx_pair=transform_tx_pair,
        )

        prioritiy = distributed.calculate_dist_priorities_from_td_error(retrace_out.extra.td_error, self._priority_eta)
        return prioritiy.item()

    def _update_target_network(self):
        self._target_ext_q_network.load_state_dict(self._online_ext_q_network.state_dict())
        self._target_int_q_network.load_state_dict(self._online_int_q_network.state_dict())

    @property
    def statistics(self):
        """Returns current agent statistics as a dictionary."""
        return {}
