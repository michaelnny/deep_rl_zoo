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
# to support PyTorch operation.
#
# ==============================================================================
"""Components for normalize tensor."""
import torch
import numpy as np


class TorchRunningMeanStd:
    """For RND networks"""

    def __init__(self, shape=(), device='cpu'):
        self.device = device
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = 0

        self.deltas = []
        self.min_size = 10

    @torch.no_grad()
    def update(self, x):
        x = x.to(self.device)
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)

        # update count and moments
        n = x.shape[0]
        self.count += n
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count
        m_a = self.var * (self.count - n)
        m_b = batch_var * n
        M2 = m_a + m_b + torch.square(delta) * self.count * n / self.count
        self.var = M2 / self.count

    @torch.no_grad()
    def update_single(self, x):
        self.deltas.append(x)

        if len(self.deltas) >= self.min_size:
            batched_x = torch.concat(self.deltas, dim=0)
            self.update(batched_x)

            del self.deltas[:]

    @torch.no_grad()
    def normalize(self, x):
        return (x.to(self.device) - self.mean) / torch.sqrt(self.var + 1e-8)


class RunningMeanStd:
    """For RND networks"""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = 0

        self.deltas = []
        self.min_size = 10

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)

        # update count and moments
        n = x.shape[0]
        self.count += n
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count
        m_a = self.var * (self.count - n)
        m_b = batch_var * n
        M2 = m_a + m_b + np.square(delta) * self.count * n / self.count
        self.var = M2 / self.count

    def update_single(self, x):
        self.deltas.append(x)

        if len(self.deltas) >= self.min_size:
            batched_x = np.stack(self.deltas, axis=0)
            self.update(batched_x)

            del self.deltas[:]

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
