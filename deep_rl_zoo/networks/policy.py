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
"""Policy networks"""
import torch
from torch import nn
import torch.nn.functional as F
from typing import NamedTuple, Optional, Tuple

# pylint: disable=import-error
from deep_rl_zoo.networks import common


class ActorNetworkOutputs(NamedTuple):
    pi_logits: torch.Tensor


class CriticNetworkOutputs(NamedTuple):
    baseline: torch.Tensor


class ActorCriticNetworkOutputs(NamedTuple):
    pi_logits: torch.Tensor
    baseline: torch.Tensor


class ImpalaActorCriticNetworkOutputs(NamedTuple):
    pi_logits: torch.Tensor
    baseline: torch.Tensor
    hidden_s: torch.Tensor


class ImpalaActorCriticNetworkInputs(NamedTuple):
    s_t: torch.Tensor
    a_tm1: torch.Tensor
    r_t: torch.Tensor  # reward for (s_tm1, a_tm1), but received at current timestep.
    done: torch.Tensor
    hidden_s: Optional[Tuple[torch.Tensor]]


class RndActorCriticNetworkOutputs(NamedTuple):
    """Random Network Distillation"""

    pi_logits: torch.Tensor
    int_baseline: torch.Tensor  # intrinsic baseline head
    ext_baseline: torch.Tensor  # extrinsic baseline head


class ActorMlpNet(nn.Module):
    """Actor MLP network."""

    def __init__(self, input_shape: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> ActorNetworkOutputs:
        """Given raw state x, predict the action probability distribution."""
        # Predict action distributions wrt policy
        pi_logits = self.net(x)

        return ActorNetworkOutputs(pi_logits=pi_logits)


class CriticMlpNet(nn.Module):
    """Critic MLP network."""

    def __init__(self, input_shape: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> CriticNetworkOutputs:
        """Given raw state x, predict the state-value."""
        baseline = self.net(x)
        return CriticNetworkOutputs(baseline=baseline)


class ActorCriticMlpNet(nn.Module):
    """Actor-Critic MLP network."""

    def __init__(self, input_shape: int, num_actions: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, num_actions)
        self.baseline_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> ActorCriticNetworkOutputs:
        """Given raw state x, predict the action probability distribution
        and baseline state-value."""
        # Extract features from raw input state
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(features)

        # Predict state-value
        baseline = self.baseline_head(features)

        return ActorCriticNetworkOutputs(pi_logits=pi_logits, baseline=baseline)


class GaussianActorMlpNet(nn.Module):
    """Gaussian Actor MLP network for continuous action space."""

    def __init__(self, input_shape: int, num_actions: int, hidden_size: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),
        )

        self.sigma_head = nn.Linear(hidden_size, num_actions)
        self.mu_head = nn.Linear(hidden_size, num_actions)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Given raw state x, predict the action probability distribution
        and baseline state-value."""
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_mu = self.mu_head(features)
        pi_sigma = torch.exp(self.sigma_head(features))

        return pi_mu, pi_sigma


class GaussianCriticMlpNet(nn.Module):
    """Gaussian Critic MLP network for continuous action space."""

    def __init__(self, input_shape: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given raw state x, predict the baseline state-value."""

        # Predict state-value
        baseline = self.net(x)

        return baseline


class ImpalaActorCriticMlpNet(nn.Module):
    """IMPALA Actor-Critic MLP network, with LSTM."""

    def __init__(self, input_shape: int, num_actions: int, use_lstm: bool = False) -> None:
        """
        Args:
            input_shape: state space size of environment state dimension
            num_actions: action space size of number of actions of the environment
            feature_size: number of units of the last feature representation linear layer
        """
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.use_lstm = use_lstm

        self.body = nn.Sequential(
            nn.Linear(self.input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # Feature representation output size + one-hot of last action + last reward.
        core_output_size = 128 + self.num_actions + 1

        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=core_output_size, hidden_size=128, num_layers=1)
            core_output_size = 128

        self.policy_head = nn.Linear(core_output_size, num_actions)
        self.baseline_head = nn.Linear(core_output_size, 1)

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        """Get initial LSTM hidden state, which is all zeros,
        shoul call at the begining of new episode."""
        if self.use_lstm:
            # Shape should be num_layers, batch_size, hidden_size, note lstm expects two hidden states.
            return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))
        else:
            return tuple()

    def forward(self, input_: ImpalaActorCriticNetworkInputs) -> ImpalaActorCriticNetworkOutputs:
        """
        Given state, predict the action probability distribution and state-value,
        T refers to the time dimension ranging from 0 to T-1. B refers to the batch size

        If self.use_lstm is set to True, and no hidden_s is given, will use zero start method.

        Args:
            input_: the ImpalaActorCriticNetworkInputs which contains the follow attributes:
                s_t: timestep t environment state, shape [T, B, state_shape].
                a_tm1: action taken in t-1 timestep, shape [T, B].
                r_t: reward for state-action pair (stm1, a_tm1), shape [T, B].
                done: current timestep state s_t is done state, shape [T, B].
                hidden_s: (optional) LSTM layer hidden state from t-1 timestep, tuple of two tensors each shape (num_lstm_layers, B, lstm_hidden_size).

        Returns:
            ImpalaActorCriticNetworkOutputs object with the following attributes:
                pi_logits: action probability logits.
                baseline: baseline state-value.
                hidden_s: (optional) hidden state from LSTM layer output.
        """
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        r_t = input_.r_t
        done = input_.done
        hidden_s = input_.hidden_s

        T, B, *_ = s_t.shape  # [T, B, state_shape]
        x = torch.flatten(s_t, 0, 1)  # Merge time and batch.

        # Extract features from raw input state
        x = self.body(x)

        # Append clipped last reward and one hot last action.
        one_hot_a_tm1 = F.one_hot(a_tm1.view(T * B), self.num_actions).float().to(device=x.device)
        rewards = torch.clamp(r_t, -1, 1).view(T * B, 1)  # Clip reward [-1, 1]
        core_input = torch.cat([x, rewards, one_hot_a_tm1], dim=-1)

        if self.use_lstm:
            assert done.dtype == torch.bool
            # Pass through RNN LSTM layer
            core_input = core_input.view(T, B, -1)
            lstm_output_list = []
            notdone = (~done).float()

            # Use zero start if not given
            if hidden_s is None:
                hidden_s = self.get_initial_hidden_state(B)
                hidden_s = tuple(s.to(device=x.device) for s in hidden_s)

            for inpt, n_d in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                n_d = n_d.view(1, -1, 1)
                hidden_s = tuple(n_d * s for s in hidden_s)
                output, hidden_s = self.lstm(inpt.unsqueeze(0), hidden_s)  # LSTM takes input x and previous hidden units
                lstm_output_list.append(output)

            core_output = torch.flatten(torch.cat(lstm_output_list), 0, 1)
        else:
            core_output = core_input
            hidden_s = tuple()

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(core_output)

        # Predict state-value baseline
        baseline = self.baseline_head(core_output)

        # Reshape to matching original shape
        pi_logits = pi_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        return ImpalaActorCriticNetworkOutputs(pi_logits=pi_logits, baseline=baseline, hidden_s=hidden_s)


class RndActorCriticMlpNet(nn.Module):
    """Actor-Critic MLP network with two baseline heads.

    From the paper "Exploration by Random Network Distillation"
    https://arxiv.org/abs/1810.12894
    """

    def __init__(self, input_shape: int, num_actions: int) -> None:
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, num_actions)

        self.ext_baseline_head = nn.Linear(128, 1)
        self.int_baseline_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> RndActorCriticNetworkOutputs:
        """Given raw state x, predict the action probability distribution,
        and extrinsic and intrinsic baseline values."""
        # Extract features from raw input state
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(features)

        # Predict state-value
        ext_baseline = self.ext_baseline_head(features)
        int_baseline = self.int_baseline_head(features)

        return RndActorCriticNetworkOutputs(pi_logits=pi_logits, ext_baseline=ext_baseline, int_baseline=int_baseline)


class ActorConvNet(nn.Module):
    """Actor Conv2d network."""

    def __init__(self, input_shape: int, num_actions: int) -> None:
        super().__init__()

        self.body = common.NatureCnnBackboneNet(input_shape)
        self.policy_head = nn.Linear(self.body.out_features, num_actions)

        # Initialize weights
        common.initialize_weights(self)

    def forward(self, x: torch.Tensor) -> ActorNetworkOutputs:
        """Given raw state x, predict the action probability distribution."""
        # Extract features from raw input state
        x = x.float() / 255.0
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(features)
        return ActorNetworkOutputs(pi_logits=pi_logits)


class CriticConvNet(nn.Module):
    """Critic Conv2d network."""

    def __init__(self, input_shape: int) -> None:
        super().__init__()

        self.body = common.NatureCnnBackboneNet(input_shape)
        self.baseline_head = nn.Linear(self.body.out_features, 1)

        # Initialize weights
        common.initialize_weights(self)

    def forward(self, x: torch.Tensor) -> CriticNetworkOutputs:
        """Given raw state x, predict the state-value."""
        # Extract features from raw input state
        x = x.float() / 255.0
        features = self.body(x)

        # Predict state-value
        baseline = self.baseline_head(features)
        return CriticNetworkOutputs(baseline=baseline)


class ActorCriticConvNet(nn.Module):
    """Actor-Critic Conv2d network."""

    def __init__(self, input_shape: tuple, num_actions: int) -> None:
        super().__init__()

        self.body = common.NatureCnnBackboneNet(input_shape)

        self.policy_head = nn.Linear(self.body.out_features, num_actions)
        self.baseline_head = nn.Linear(self.body.out_features, 1)

        # Initialize weights
        common.initialize_weights(self)

    def forward(self, x: torch.Tensor) -> ActorCriticNetworkOutputs:
        """Given raw state x, predict the action probability distribution
        and baseline state-value."""
        # Extract features from raw input state
        x = x.float() / 255.0
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(features)

        # Predict state-value
        baseline = self.baseline_head(features)

        return ActorCriticNetworkOutputs(pi_logits=pi_logits, baseline=baseline)


class ImpalaActorCriticConvNet(nn.Module):
    """IMPALA Actor-Critic Conv2d network, with LSTM.

    Reference code from Facebook Torchbeast:
    https://github.com/facebookresearch/torchbeast/blob/0af07b051a2176a8f9fd20c36891ba2bba6bae68/torchbeast/polybeast_learner.py#L135
    """

    def __init__(self, input_shape: tuple, num_actions: int, use_lstm: bool = False) -> None:
        super().__init__()

        self.num_actions = num_actions
        self.use_lstm = use_lstm

        assert input_shape[1] == input_shape[2] == 84

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = input_shape[0]
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(3872, 256)

        # Feature representation output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + self.num_actions + 1

        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=core_output_size, hidden_size=256, num_layers=1)
            core_output_size = 256

        self.policy_head = nn.Linear(core_output_size, num_actions)
        self.baseline_head = nn.Linear(core_output_size, 1)

        # Initialize weights
        common.initialize_weights(self)

        # BUG
        # Scenario: when using nature CNN instead of above res-blocks and using a single linear layer (without activation function) for the model's output heads.
        # Issues: after few params updates, the CNN output becomes NaNs.
        # Things tried so far:
        #   - using two linear layers (with relu activation function) for the model's output heads, the model still act randomly after 4*5 millions frames on Pong.

        # self.policy_head = nn.Sequential(
        #     nn.Linear(core_output_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_actions),
        # )

        # self.baseline_head = nn.Sequential(
        #     nn.Linear(core_output_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1),
        # )

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        """Get initial LSTM hidden state, which is all zeros,
        shoul call at the begining of new episode.
        """
        if self.use_lstm:
            # Shape should be num_layers, batch_size, hidden_size, note lstm expects two hidden states.
            return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))
        else:
            return tuple()

    def forward(self, input_: ImpalaActorCriticNetworkInputs) -> ImpalaActorCriticNetworkOutputs:
        """
        Given state, predict the action probability distribution and state-value,
        T refers to the time dimension ranging from 0 to T-1. B refers to the batch size

        If self.use_lstm is set to True, and no hidden_s is given, will use zero start method.

        Args:
            input_: the ImpalaActorCriticNetworkInputs which contains the follow attributes:
                s_t: timestep t environment state, shape [T, B, state_shape].
                a_tm1: action taken in t-1 timestep, shape [T, B].
                r_t: reward for state-action pair (stm1, a_tm1), shape [T, B].
                done: current timestep state s_t is done state, shape [T, B].
                hidden_s: (optional) LSTM layer hidden state from t-1 timestep, tuple of two tensors each shape (num_lstm_layers, B, lstm_hidden_size).

        Returns:
            ImpalaActorCriticNetworkOutputs object with the following attributes:
                pi_logits: action probability logits.
                baseline: baseline state-value.
                hidden_s: (optional) hidden state from LSTM layer output.
        """
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        r_t = input_.r_t
        done = input_.done
        hidden_s = input_.hidden_s

        T, B, *_ = s_t.shape  # [T, B, input_shape].
        x = torch.flatten(s_t, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        # Extract features from raw input state
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        # Append clipped last reward and one hot last action.
        one_hot_a_tm1 = F.one_hot(a_tm1.view(T * B), self.num_actions).float().to(device=x.device)
        rewards = torch.clamp(r_t, -1, 1).view(T * B, 1)  # Clip reward [-1, 1]
        core_input = torch.cat([x, rewards, one_hot_a_tm1], dim=-1)

        if self.use_lstm:
            assert done.dtype == torch.bool

            # Pass through RNN LSTM layer
            core_input = core_input.view(T, B, -1)
            lstm_output_list = []
            notdone = (~done).float()

            # Use zero start if not given
            if hidden_s is None:
                hidden_s = self.get_initial_hidden_state(B)
                hidden_s = tuple(s.to(device=x.device) for s in hidden_s)

            for inpt, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                hidden_s = tuple(nd * s for s in hidden_s)
                output, hidden_s = self.lstm(inpt.unsqueeze(0), hidden_s)  # LSTM takes input x and previous hidden units
                lstm_output_list.append(output)
            core_output = torch.flatten(torch.cat(lstm_output_list), 0, 1)
        else:
            core_output = core_input
            hidden_s = tuple()

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(core_output)

        # Predict state-value baseline
        baseline = self.baseline_head(core_output)

        # Reshape to matching original shape
        pi_logits = pi_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        return ImpalaActorCriticNetworkOutputs(pi_logits=pi_logits, baseline=baseline, hidden_s=hidden_s)


class RndActorCriticConvNet(nn.Module):
    """Actor-Critic Conv2d network with two baseline heads.

    From the paper "Exploration by Random Network Distillation"
    https://arxiv.org/abs/1810.12894
    """

    def __init__(self, input_shape: tuple, num_actions: int) -> None:
        super().__init__()

        self.body = common.NatureCnnBackboneNet(input_shape)

        self.policy_head = nn.Linear(self.body.out_features, num_actions)

        self.ext_baseline_head = nn.Linear(self.body.out_features, 1)
        self.int_baseline_head = nn.Linear(self.body.out_features, 1)

        # Initialize weights
        common.initialize_weights(self)

    def forward(self, x: torch.Tensor) -> RndActorCriticNetworkOutputs:
        """Given raw state x, predict the action probability distribution,
        and extrinsic and intrinsic baseline values."""
        # Extract features from raw input state
        x = x.float() / 255.0
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(features)

        # Predict state-value
        ext_baseline = self.ext_baseline_head(features)
        int_baseline = self.int_baseline_head(features)

        return RndActorCriticNetworkOutputs(pi_logits=pi_logits, ext_baseline=ext_baseline, int_baseline=int_baseline)
