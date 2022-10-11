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

    def __init__(self, input_shape: int, num_actions: int) -> None:
        """
        Args:
            input_shape: the shape of the input tensor to the neural network.
            num_actions: the number of units for the output liner layer.
        """
        super().__init__()

        self.num_actions = num_actions

        feature_vector_size = 128

        # Feature representations
        self.body = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_vector_size),
            nn.ReLU(),
        )

        # Forward model, predict feature vector of s_t from s_tm1 and a_t
        self.forward_net = nn.Sequential(
            nn.Linear(feature_vector_size + num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, feature_vector_size),
            nn.ReLU(),
        )

        # Inverse model, predict a_tm1 from feature vectors of s_tm1, s_t
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_vector_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, s_tm1: torch.Tensor, a_tm1: torch.Tensor, s_t: torch.Tensor) -> IcmNetworkOutput:
        """Given raw state s_tm1, s_t, and action a_tm1,
        call forward model and inverse model to predict a_tm1 and feature vector of s_t"""
        base.assert_rank(s_tm1, 2)
        base.assert_rank(s_t, 2)
        base.assert_rank(a_tm1, 1)

        a_tm1_onehot = F.one_hot(a_tm1, self.num_actions).float()

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

    def __init__(self, input_shape: int, num_actions: int) -> None:
        """
        Args:
            input_shape: the shape of the input tensor to the neural network.
            num_actions: the number of units for the output liner layer.
        """
        super().__init__()

        self.num_actions = num_actions

        # Compute the output shape of final conv2d layer
        c, h, w = input_shape
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
            nn.Linear(conv2d_out_size + self.num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, conv2d_out_size),
            nn.ReLU(),
        )

        # Inverse model, predict a_tm1 from feature vectors of s_tm1, s_t
        self.inverse_net = nn.Sequential(
            nn.Linear(conv2d_out_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, s_tm1: torch.Tensor, a_tm1: torch.Tensor, s_t: torch.Tensor) -> IcmNetworkOutput:
        """Given raw state s_tm1, s_t, and action a_tm1,
        call forward model and inverse model to predict a_tm1 and feature vector of s_t"""
        base.assert_rank(s_tm1, (2, 4))
        base.assert_rank(s_t, (2, 4))
        base.assert_rank(a_tm1, 1)

        a_tm1_onehot = F.one_hot(a_tm1, self.num_actions).float()

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


class RndMlpNet(nn.Module):
    """RND MLP network.

    From the paper "Exploration by Random Network Distillation"
    https://arxiv.org/abs/1810.12894
    """

    def __init__(self, input_shape: int, is_target: bool = False, latent_dim: int = 128) -> None:
        """
        Args:
            input_shape: the shape of the input tensor to the neural network.
            is_target: if True, use one single linear layer at the head, default False.
            latent_dim: the embedding latent dimension, default 128.
        """
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
        )

        if is_target:
            self.head = nn.Linear(128, latent_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim),
            )

        # Initialize weights.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given raw state x, returns the feature embedding."""
        x = self.body(x)
        return self.head(x)


class RndConvNet(nn.Module):
    """RND Conv2d network.

    From the paper "Exploration by Random Network Distillation"
    https://arxiv.org/abs/1810.12894
    """

    def __init__(self, input_shape: int, is_target: bool = False, latent_dim: int = 256) -> None:
        """
        Args:
            input_shape: the shape of the input tensor to the neural network.
            is_target: if True, use one single linear layer at the head, default False.
            latent_dim: the embedding latent dimension, default 256.
        """
        super().__init__()

        # Compute the output shape of final conv2d layer
        c, h, w = input_shape
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
                nn.Linear(512, latent_dim),
            )

        # Initialize weights.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given raw state x, returns the feature embedding."""
        # RND normalizes state using a running mean and std instead of devide by 255.
        x = self.body(x)
        return self.head(x)


class NguEmbeddingMlpNet(nn.Module):
    """MLP Embedding networks for NGU.

    From the paper "Never Give Up: Learning Directed Exploration Strategies"
    https://arxiv.org/abs/2002.06038
    """

    def __init__(self, input_shape: tuple, num_actions: int, latent_dim: int = 64):
        """
        Args:
            input_shape: the shape of the input tensor to the neural network.
            num_actions: the number of units for the output liner layer.
            latent_dim: the embedding latent dimension, default 64.
        """
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

        self.inverse_head = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given state x, return the embedding."""
        return self.body(x)

    def inverse_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Given combined embedding features of (s_tm1 + s_t), returns the raw logits of predicted action a_tm1."""
        pi_logits = self.inverse_head(x)  # [batch_size, num_actions]
        return pi_logits


class NguEmbeddingConvNet(nn.Module):
    """Conv2d Embedding networks for NGU.

    From the paper "Never Give Up: Learning Directed Exploration Strategies"
    https://arxiv.org/abs/2002.06038
    """

    def __init__(self, input_shape: tuple, num_actions: int, latent_dim: int = 128):
        """
        Args:
            input_shape: the shape of the input tensor to the neural network.
            num_actions: the number of units for the output liner layer.
            latent_dim: the embedding latent dimension, default 128.
        """
        super().__init__()

        self.net = common.NatureCnnBackboneNet(input_shape, out_features=latent_dim)

        self.inverse_head = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given state x, return the embedding."""
        x = x.float() / 255.0
        return self.net(x)

    def inverse_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Given combined embedding features of (s_tm1 + s_t), returns the raw logits of predicted action a_tm1."""
        pi_logits = self.inverse_head(x)  # [batch_size, num_actions]
        return pi_logits
