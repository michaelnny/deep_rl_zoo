# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
#
# The classs 'EpsilonGreedyActor' has been modified
# by The Deep RL Zoo Authors to support PyTorch opeartion.
#
# ==============================================================================
"""Greedy actors for testing and evaluation."""
from typing import Mapping, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# pylint: disable=import-error
import deep_rl_zoo.types as types_lib
from deep_rl_zoo.networks.policy import ImpalaActorCriticNetworkInputs
from deep_rl_zoo.networks.dqn import RnnDqnNetworkInputs, NguDqnNetworkInputs
from deep_rl_zoo.curiosity import EpisodicBonusModule, RndLifeLongBonusModule
from deep_rl_zoo.agent57.agent import compute_transformed_q


HiddenState = Tuple[torch.Tensor, torch.Tensor]


def apply_egreedy_policy(
    q_values: torch.Tensor,
    epsilon: float,
    random_state: np.random.RandomState,  # pylint: disable=no-member
) -> types_lib.Action:
    """Apply e-greedy policy."""
    num_actions = q_values.shape[-1]
    if random_state.rand() <= epsilon:
        a_t = random_state.randint(0, num_actions)
    else:
        a_t = q_values.argmax(-1).cpu().item()
    return a_t


class EpsilonGreedyActor(types_lib.Agent):
    """DQN e-greedy actor."""

    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        device: torch.device,
        name: str = 'DQN-greedy',
    ):
        self.agent_name = name
        self._device = device
        self._network = network.to(device=device)
        self._exploration_epsilon = exploration_epsilon
        self._random_state = random_state

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Give current timestep, return best action"""
        return self._select_action(timestep)

    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat.
        This method should be called at the beginning of every episode.
        """

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        q_t = self._network(s_t).q_values
        return apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)

    @property
    def statistics(self) -> Mapping[str, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            'exploration_epsilon': self._exploration_epsilon,
        }


class IqnEpsilonGreedyActor(EpsilonGreedyActor):
    """IQN e-greedy actor."""

    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        device: torch.device,
        tau_samples: int,
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            device,
            'IQN-greedy',
        )
        self._tau_samples = tau_samples

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        q_t = self._network(s_t, self._tau_samples).q_values
        return apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)


class DrqnEpsilonGreedyActor(EpsilonGreedyActor):
    """DRQN e-greedy actor."""

    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        device: torch.device,
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            device,
            'DRQN-greedy',
        )
        self._lstm_state = None

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        s_t = torch.tensor(timestep.observation[None, None, ...]).to(device=self._device, dtype=torch.float32)
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)
        network_output = self._network(s_t, hidden_s)
        q_t = network_output.q_values
        self._lstm_state = network_output.hidden_s
        return apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)

    def reset(self) -> None:
        """Reset hidden state to zeros at new episodes."""
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)


class R2d2EpsilonGreedyActor(EpsilonGreedyActor):
    """R2D2 e-greedy actor."""

    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        device: torch.device,
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            device,
            'R2D2-greedy',
        )
        self._a_tm1 = None
        self._lstm_state = None

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(self._a_tm1).to(device=self._device, dtype=torch.int64)
        r_t = torch.tensor(timestep.reward).to(device=self._device, dtype=torch.float32)
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)

        network_output = self._network(
            RnnDqnNetworkInputs(
                s_t=s_t[None, ...],
                a_tm1=a_tm1[None, ...],
                r_t=r_t[None, ...],
                hidden_s=hidden_s,
            )
        )
        q_t = network_output.q_values
        self._lstm_state = network_output.hidden_s

        return apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)

    def reset(self) -> None:
        """Reset hidden state to zeros at new episodes."""
        self._a_tm1 = 0  # During the first step of a new episode, use 'fake' previous action for network pass
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)


class NguEpsilonGreedyActor(EpsilonGreedyActor):
    """NGU e-greedy actor."""

    def __init__(
        self,
        network: torch.nn.Module,
        embedding_network: torch.nn.Module,
        rnd_target_network: torch.nn.Module,
        rnd_predictor_network: torch.nn.Module,
        episodic_memory_capacity: int,
        num_neighbors: int,
        cluster_distance: float,
        kernel_epsilon: float,
        max_similarity: float,
        exploration_epsilon: float,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        device: torch.device,
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            device,
            'NGU-greedy',
        )

        self._policy_index = 0
        self._policy_beta = 0

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

        self._a_tm1 = None
        self._episodic_bonus_t = None
        self._lifelong_bonus_t = None
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)

    @torch.no_grad()
    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Give current timestep, return best action"""
        a_t = self._select_action(timestep)

        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)

        # Compute lifelong intrinsic bonus
        self._lifelong_bonus_t = self._lifelong_module.compute_bonus(s_t)

        # Compute episodic intrinsic bonus
        self._episodic_bonus_t = self._episodic_module.compute_bonus(s_t)

        return a_t

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(self._a_tm1).to(device=self._device, dtype=torch.int64)
        ext_r_t = torch.tensor(timestep.reward).to(device=self._device, dtype=torch.float32)
        int_r_t = torch.tensor(self.intrinsic_reward).to(device=self._device, dtype=torch.float32)
        policy_index = torch.tensor(self._policy_index).to(device=self._device, dtype=torch.int64)
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)

        pi_output = self._network(
            NguDqnNetworkInputs(
                s_t=s_t[None, ...],  # [T, B, state_shape]
                a_tm1=a_tm1[None, ...],  # [T, B]
                ext_r_t=ext_r_t[None, ...],  # [T, B]
                int_r_t=int_r_t[None, ...],  # [T, B]
                policy_index=policy_index[None, ...],  # [T, B]
                hidden_s=hidden_s,
            )
        )

        q_t = pi_output.q_values
        self._lstm_state = pi_output.hidden_s

        return apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)

    def reset(self) -> None:
        """Reset hidden state to zeros at new episodes."""
        self._episodic_module.reset()
        self._a_tm1 = 0  # Initialize a_tm1 to 0.
        self._episodic_bonus_t = 0.0
        self._lifelong_bonus_t = 0.0
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)

    @property
    def intrinsic_reward(self) -> float:
        """Returns intrinsic reward for last state s_tm1."""
        # Equation 1 of the NGU paper.
        return self._episodic_bonus_t * min(max(self._lifelong_bonus_t, 1.0), 5.0)


class Agent57EpsilonGreedyActor(types_lib.Agent):
    """Agent57 e-greedy actor."""

    def __init__(
        self,
        ext_q_network: torch.nn.Module,
        int_q_network: torch.nn.Module,
        embedding_network: torch.nn.Module,
        rnd_target_network: torch.nn.Module,
        rnd_predictor_network: torch.nn.Module,
        episodic_memory_capacity: int,
        num_neighbors: int,
        cluster_distance: float,
        kernel_epsilon: float,
        max_similarity: float,
        exploration_epsilon: float,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        device: torch.device,
    ):

        self.agent_name = 'Agent57-greedy'
        self._ext_q_network = ext_q_network.to(device=device)
        self._int_q_network = int_q_network.to(device=device)
        self._device = device

        self._random_state = random_state
        self._exploration_epsilon = exploration_epsilon

        self._policy_index = 0
        self._policy_beta = 0

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

        self._a_tm1 = None
        self._episodic_bonus_t = None
        self._lifelong_bonus_t = None
        self._ext_lstm_state = None  # Stores nn.LSTM hidden state and cell state. for extrinsic Q network
        self._int_lstm_state = None  # Stores nn.LSTM hidden state and cell state. for intrinsic Q network

    @torch.no_grad()
    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Give current timestep, return best action"""
        a_t = self._select_action(timestep)

        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)

        # Compute lifelong intrinsic bonus
        self._lifelong_bonus_t = self._lifelong_module.compute_bonus(s_t)

        # Compute episodic intrinsic bonus
        self._episodic_bonus_t = self._episodic_module.compute_bonus(s_t)

        return a_t

    def reset(self) -> None:
        """Reset hidden state to zeros at new episodes."""
        self._episodic_module.reset()
        self._a_tm1 = 0  # Initialize a_tm1 to 0.
        self._episodic_bonus_t = 0.0
        self._lifelong_bonus_t = 0.0
        self._ext_lstm_state = self._ext_q_network.get_initial_hidden_state(batch_size=1)
        self._int_lstm_state = self._int_q_network.get_initial_hidden_state(batch_size=1)

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        q_ext_input_ = self._prepare_network_input(timestep, self._ext_lstm_state)
        q_int_input_ = self._prepare_network_input(timestep, self._int_lstm_state)

        pi_ext_output = self._ext_q_network(q_ext_input_)
        pi_int_output = self._int_q_network(q_int_input_)
        ext_q_t = pi_ext_output.q_values.squeeze()
        int_q_t = pi_int_output.q_values.squeeze()

        q_t = compute_transformed_q(ext_q_t, int_q_t, self._policy_beta)

        self._ext_lstm_state = pi_ext_output.hidden_s
        self._int_lstm_state = pi_int_output.hidden_s

        return apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)

    def _prepare_network_input(self, timestep: types_lib.TimeStep, hidden_state: HiddenState) -> NguDqnNetworkInputs:
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

    @property
    def intrinsic_reward(self) -> float:
        """Returns intrinsic reward for last state s_tm1."""
        # Equation 1 of the NGU paper.
        return self._episodic_bonus_t * min(max(self._lifelong_bonus_t, 1.0), 5.0)

    @property
    def statistics(self) -> Mapping[str, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            'exploration_epsilon': self._exploration_epsilon,
        }


class PolicyGreedyActor(types_lib.Agent):
    """Agent that acts with a given set of policy network parameters."""

    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
        name: str = '',
    ):
        self.agent_name = name
        self._device = device
        self._network = network.to(device=device)

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Give current timestep, return best action"""
        return self.act(timestep)

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Selects action given a timestep."""
        return self._select_action(timestep)

    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat.

        This method should be called at the beginning of every episode.
        """

    @torch.no_grad()
    def _select_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Samples action from policy at given state."""
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        pi_logits = self._network(s_t).pi_logits
        # Don't sample when testing.
        prob_t = F.softmax(pi_logits, dim=1)
        a_t = torch.argmax(prob_t, dim=1)
        return a_t.cpu().item()

    @property
    def statistics(self) -> Mapping[str, float]:
        """Empty statistics"""
        return {}


class ImpalaGreedyActor(PolicyGreedyActor):
    """IMPALA greedy actor to do evaluation during training"""

    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
    ) -> None:
        super().__init__(
            network,
            device,
            'IMPALA',
        )

        self._a_tm1 = None
        self._hidden_s = self._network.get_initial_hidden_state(batch_size=1)

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, return action a_t"""
        a_t = self.act(timestep)

        # Update local states after create the transition
        self._a_tm1 = a_t

        return a_t

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        'Given state s_t and done marks, return an action.'
        a_t = self._choose_action(timestep)
        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode before take any action."""
        self._a_tm1 = 0  # During the first step of a new episode, use 'fake' previous action for network pass
        self._hidden_s = self._network.get_initial_hidden_state(batch_size=1)

    @torch.no_grad()
    def _choose_action(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given state s_t, choose action a_t"""
        # IMPALA network requires more than just the state input, but also last action, and reward for last action
        # optionally the last hidden state from LSTM and done mask if using LSTM
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(self._a_tm1).to(device=self._device, dtype=torch.int64)
        r_t = torch.tensor(timestep.reward).to(device=self._device, dtype=torch.float32)
        done = torch.tensor(timestep.done).to(device=self._device, dtype=torch.bool)

        hidden_s = tuple(s.to(device=self._device) for s in self._hidden_s)

        network_output = self._network(
            ImpalaActorCriticNetworkInputs(
                s_t=s_t[None, ...],
                a_tm1=a_tm1[None, ...],
                r_t=r_t[None, ...],
                done=done[None, ...],
                hidden_s=hidden_s,
            )
        )
        pi_logits = network_output.pi_logits.squeeze(0)  # Remove T dimension
        prob_t = F.softmax(pi_logits, dim=-1)
        a_t = torch.argmax(prob_t, dim=-1)
        self._hidden_s = network_output.hidden_s  # Save last hidden state for next pass
        return a_t.cpu().item()

    @property
    def statistics(self) -> Mapping[str, float]:
        """Returns current actor's statistics as a dictionary."""
        return {}
