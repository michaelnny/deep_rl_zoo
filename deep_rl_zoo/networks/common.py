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
"""Common components for network."""
import math
from typing import Iterable, NamedTuple, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F


class Conv2dLayerConfig(NamedTuple):
    in_channels: Optional[int]
    out_channels: Optional[int]
    kernel_size: Optional[int]
    stride: Optional[int]
    padding: Optional[int]


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def make_conv2d_modules(
    input_shape: tuple, layers_config: Iterable[Conv2dLayerConfig], activation: str, out_features: int
) -> Tuple[Iterable[torch.nn.Module], int]:
    """Make conv2d layers according to the input layers_config, will also flatten the output of last conv2d layer,
    and calculate the output dimension of the last conv2d layer.
    """

    _, h, w = input_shape  # Get input shape height and width

    input_hw = (h, w)  # Set to input shape
    conv_out_dim = 0

    modules = []
    for layer in layers_config:
        modules.append(
            nn.Conv2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
            )
        )
        if activation == 'relu':
            modules.append(nn.ReLU())
        elif activation == 'elu':
            modules.append(nn.ELU())
        elif activation == 'leaky_relu':
            modules.append(nn.LeakyReLU())
        elif activation == 'none':
            pass

        input_hw = calc_conv2d_output(input_hw, layer.kernel_size, layer.stride, layer.padding)

    # Calculate the output dimension of last conv2d layer
    conv_out_dim = input_hw[0] * input_hw[1] * layers_config[-1].out_channels

    # Flatten conv2d output into a vector
    modules.append(nn.Flatten())

    # Add one fc layer
    modules.append(nn.Linear(in_features=conv_out_dim, out_features=out_features))
    if activation == 'relu':
        modules.append(nn.ReLU())
    elif activation == 'elu':
        modules.append(nn.ELU())
    elif activation == 'leaky_relu':
        modules.append(nn.LeakyReLU())
    elif activation == 'none':
        pass

    return modules


class NatureCnnBodyNet(nn.Module):
    """DQN Nature paper conv2d layers backbone, returns feature representation vector"""

    def __init__(self, input_shape: tuple, out_features: int = 512) -> None:
        super().__init__()
        if out_features < 1:
            raise ValueError(f'Expect out_features to be positive integer, got {out_features}')

        c, h, w = input_shape
        self.out_features = out_features

        # Nature DQN configurations
        layers_config = [
            Conv2dLayerConfig(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=0),
            Conv2dLayerConfig(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            Conv2dLayerConfig(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
        ]

        modules = make_conv2d_modules(
            input_shape=input_shape,
            layers_config=layers_config,
            out_features=self.out_features,
            activation='relu',
        )
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given raw state images, returns feature representation vector"""
        return self.body(x)


class NoisyLinear(nn.Module):
    """Factorised NoisyLinear layer with bias.
    https://github.com/Kaixhin/Rainbow/blob/master/model.py
    """

    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Only call this during initialization"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Should call this after doing backpropagation"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)
