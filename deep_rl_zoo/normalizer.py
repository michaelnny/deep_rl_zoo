# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
# Copyright 2019 The SEED Authors
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
# The file has been modified by The Deep RL Zoo Authors
# to support PyTorch opeartion.
#
# ==============================================================================
"""Components for normalize tensor."""
import torch
import numpy as np


class Normalizer:

    """Normalizes tensors by tracking their element-wise mean and variance.
    code adapted from https://github.com/google-research/seed_rl/blob/master/common/normalizer.py
    """

    def __init__(self, eps: float = 0.001, clip_range: tuple = (-5, 5), device: str = 'cpu') -> None:
        """Initialize the normalizer.
        Args:
            eps: A constant added to the standard deviation of data beforenormalization.
            clip_range: Normalized values are clipped to this range.
        """
        super().__init__()
        self.eps = eps
        self.clip_range = clip_range
        self.initialized = False
        self.device = device

    def _build(self, input_shape: tuple) -> None:
        assert not self.initialized
        self.initialized = True

        size = input_shape[-1]

        # Statistics accumulators
        self.steps = torch.zeros(1, dtype=torch.float32, device=self.device).requires_grad_(False)
        self.sum = torch.zeros(size, dtype=torch.float32, device=self.device).requires_grad_(False)
        self.sumsq = torch.zeros(size, dtype=torch.float32, device=self.device).requires_grad_(False)
        self.mean = torch.zeros(size, dtype=torch.float32, device=self.device).requires_grad_(False)
        self.std = torch.zeros(size, dtype=torch.float32, device=self.device).requires_grad_(False)

    def update(self, input_: torch.Tensor) -> None:
        """Update normalization statistics.
        Args:
            input_: A tensor. All dimensions apart from the last one are treated
            as batch dimensions.
        """
        assert len(input_.shape) >= 1

        # Add a new dimension if input is a 1d vector
        if len(input_.shape) == 1:
            input_ = input_[:, None]

        if not self.initialized:
            self._build(input_.shape)

        # Reshape to 2 dimensions
        shape = input_.shape
        input_ = torch.reshape(input_, (np.prod(shape[:-1]), shape[-1]))

        assert len(input_.shape) == 2

        # Update statistics accumulators
        self.steps += float(input_.shape[0])
        self.sum += torch.sum(input_, dim=0)
        self.sumsq += torch.sum(torch.square(input_), dim=0)
        self.mean = self.sum / self.steps
        self.std = torch.sqrt(torch.maximum(torch.tensor(0.0), (self.sumsq / self.steps) - torch.square(self.mean)))

    def __call__(self, input_: torch.Tensor) -> torch.Tensor:
        """Update normalization statistics and return normalized tensor.
        Args:
            input_: A tensor. All dimensions apart from the last one are treated
            as batch dimensions.
        Returns:
            a nomalized tensor with zero mean, unit deviation of the same shape and dtype as input_.
        """
        if not self.initialized:
            return input_

        # Don't want to do it in place
        copy_input_ = input_.clone()

        # Rember original shape
        orig_shape = copy_input_.shape

        # Add a new dimension if input is a 1d vector
        if len(input_.shape) == 1:
            copy_input_ = copy_input_[:, None]

        # Reshape to 2 dimensions
        shape = copy_input_.shape

        copy_input_ = torch.reshape(copy_input_, (np.prod(shape[:-1]), shape[-1]))

        assert len(copy_input_.shape) == 2

        # Normalize to mean 0, unit deviation
        copy_input_ -= self.mean[None, :]
        copy_input_ /= self.std[None, :] + self.eps

        # Clip value range
        copy_input_ = torch.clamp(copy_input_, *self.clip_range)

        # Reshape to the original shape
        return torch.reshape(copy_input_, orig_shape)
