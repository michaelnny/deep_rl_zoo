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
"""Networks for curiosity-driven-exploration."""


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import NamedTuple

# pylint: disable=import-error
from deep_rl_zoo import base
from deep_rl_zoo.networks import common


class IcmNetworkOutput(NamedTuple):
    """ICM module"""

    pi_logits: torch.Tensor
    features: torch.Tensor
    pred_features: torch.Tensor


class IcmMlpNet(nn.Module):
    """ICM module of curiosity driven exploration for Mlp networks.

    From the paper "Curiosity-driven Exploration by Self-supervised Prediction"
    https://arxiv.org/abs/1705.05363.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            action_dim: the number of units for the output liner layer.
        """
        super().__init__()

        self.action_dim = action_dim

        feature_vector_size = 128

        # Feature representations
        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_vector_size),
            nn.ReLU(),
        )

        # Forward model, predict feature vector of s_t from s_tm1 and a_t
        self.forward_net = nn.Sequential(
            nn.Linear(feature_vector_size + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_vector_size),
            nn.ReLU(),
        )

        # Inverse model, predict a_tm1 from feature vectors of s_tm1, s_t
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_vector_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, s_tm1: torch.Tensor, a_tm1: torch.Tensor, s_t: torch.Tensor) -> IcmNetworkOutput:
        """Given raw state s_tm1, s_t, and action a_tm1,
        call forward model and inverse model to predict a_tm1 and feature vector of s_t"""
        base.assert_rank(s_tm1, 2)
        base.assert_rank(s_t, 2)
        base.assert_rank(a_tm1, 1)

        a_tm1_onehot = F.one_hot(a_tm1, self.action_dim).float()

        # Get feature vectors of s_tm1 and s_t
        features_tm1 = self.body(s_tm1)
        features_t = self.body(s_t)

        # Predict feature vector of s_t
        forward_input = torch.cat([features_tm1, a_tm1_onehot], dim=-1)
        pred_features_t = self.forward_net(forward_input)

        # Predict actions a_tm1 from feature vectors s_tm1 and s_t
        inverse_input = torch.cat([features_tm1, features_t], dim=-1)
        pi_logits_a_tm1 = self.inverse_net(inverse_input)  # Returns logits not probability distribution

        return IcmNetworkOutput(pi_logits=pi_logits_a_tm1, pred_features=pred_features_t, features=features_t)


class IcmNatureConvNet(nn.Module):
    """ICM module of curiosity driven exploration for Conv networks.

    From the paper "Curiosity-driven Exploration by Self-supervised Prediction"
    https://arxiv.org/abs/1705.05363.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            action_dim: the number of units for the output liner layer.
        """
        super().__init__()

        self.action_dim = action_dim

        # Compute the output shape of final conv2d layer
        c, h, w = state_dim
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        conv2d_out_size = 32 * h * w  # output size 288

        self.body = self.net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Forward model, predict feature vector of s_t from s_tm1 and a_t
        self.forward_net = nn.Sequential(
            nn.Linear(conv2d_out_size + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, conv2d_out_size),
            nn.ReLU(),
        )

        # Inverse model, predict a_tm1 from feature vectors of s_tm1, s_t
        self.inverse_net = nn.Sequential(
            nn.Linear(conv2d_out_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        common.initialize_weights(self)

    def forward(self, s_tm1: torch.Tensor, a_tm1: torch.Tensor, s_t: torch.Tensor) -> IcmNetworkOutput:
        """Given raw state s_tm1, s_t, and action a_tm1,
        call forward model and inverse model to predict a_tm1 and feature vector of s_t"""
        base.assert_rank(s_tm1, (2, 4))
        base.assert_rank(s_t, (2, 4))
        base.assert_rank(a_tm1, 1)

        a_tm1_onehot = F.one_hot(a_tm1, self.action_dim).float()

        # Get feature vectors of s_tm1 and s_t
        s_tm1 = s_tm1.float() / 255.0
        s_t = s_t.float() / 255.0
        features_tm1 = self.body(s_tm1)
        features_t = self.body(s_t)

        # Predict feature vector of s_t
        forward_input = torch.cat([features_tm1, a_tm1_onehot], dim=-1)
        pred_features_t = self.forward_net(forward_input)

        # Predict actions a_tm1 from feature vectors s_tm1 and s_t
        inverse_input = torch.cat([features_tm1, features_t], dim=-1)
        pi_logits_a_tm1 = self.inverse_net(inverse_input)  # Returns logits not probability distribution

        return IcmNetworkOutput(pi_logits=pi_logits_a_tm1, pred_features=pred_features_t, features=features_t)


class RndConvNet(nn.Module):
    """RND Conv2d network.

    From the paper "Exploration by Random Network Distillation"
    https://arxiv.org/abs/1810.12894
    """

    def __init__(self, state_dim: int, is_target: bool = False, latent_dim: int = 256) -> None:
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            is_target: if True, use one single linear layer at the head, default False.
            latent_dim: the embedding latent dimension, default 256.
        """
        super().__init__()

        # Compute the output shape of final conv2d layer
        c, h, w = state_dim
        h, w = common.calc_conv2d_output((h, w), 8, 4)
        h, w = common.calc_conv2d_output((h, w), 4, 2)
        h, w = common.calc_conv2d_output((h, w), 3, 1)
        conv2d_out_size = 64 * h * w

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        if is_target:
            self.head = nn.Linear(conv2d_out_size, latent_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(conv2d_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, latent_dim),
            )

        # Initialize weights.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                module.bias.data.zero_()

        if is_target:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given raw state x, returns the feature embedding."""
        # RND normalizes state using a running mean and std instead of divide by 255.
        x = self.body(x)
        return self.head(x)


class NguEmbeddingConvNet(nn.Module):
    """Conv2d Embedding networks for NGU.

    From the paper "Never Give Up: Learning Directed Exploration Strategies"
    https://arxiv.org/abs/2002.06038
    """

    def __init__(self, state_dim: tuple, action_dim: int):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            action_dim: the number of units for the output liner layer.
        """
        super().__init__()

        self.embed_size = 32

        self.net = common.NatureCnnBackboneNet(state_dim)

        self.fc = nn.Linear(self.net.out_features, 32)

        # *2 because the input to inverse head is two embeddings [s_t, s-tp1]
        self.inverse_head = nn.Sequential(
            nn.Linear(32 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        common.initialize_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given state x, return the embedding."""
        x = x.float() / 255.0
        x = self.net(x)

        return F.relu(self.fc(x))

    def inverse_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Given combined embedding features of (s_tm1 + s_t), returns the raw logits of predicted action a_tm1."""
        pi_logits = self.inverse_head(x)  # [batch_size, action_dim]
        return pi_logits


class NGURndConvNet(nn.Module):
    """RND Conv2d network for NGU agent.

    From the paper "Never Give Up: Learning Directed Exploration Strategies"
    https://arxiv.org/abs/2002.06038
    """

    def __init__(self, state_dim: int, latent_dim: int = 128, is_target: bool = False) -> None:
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            latent_dim: the latent vector dimension, default 128.
        """
        super().__init__()

        # Compute the output shape of final conv2d layer
        c, h, w = state_dim
        h, w = common.calc_conv2d_output((h, w), 8, 4)
        h, w = common.calc_conv2d_output((h, w), 4, 2)
        h, w = common.calc_conv2d_output((h, w), 3, 1)
        conv2d_out_size = 64 * h * w

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.head = nn.Linear(conv2d_out_size, latent_dim)

        # Initialize weights.
        common.initialize_weights(self)

        if is_target:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given raw state x, returns the laten feature vector."""
        # RND normalizes state using a running mean and std instead of divide by 255.
        x = self.body(x)
        return self.head(x)
