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
# The functions 'knn_query' and '_cdist' has been modified
# by The Deep RL Zoo Authors to support PyTorch operation.
#
# ==============================================================================
"""Implementing functions and class for curiosity driven exploration."""

import collections
from typing import NamedTuple, Dict

import torch
from deep_rl_zoo import base
from deep_rl_zoo import normalizer


class KNNQueryResult(NamedTuple):
    neighbors: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_distances: torch.Tensor


def _cdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Returns the squared Euclidean distance between the two inputs.

    A simple equation on Euclidean distance for one-dimension:
        d(p,q)=sqrt[(p-q)^2]
    """
    return torch.sum(torch.square(a - b))


def knn_query(data: torch.Tensor, memory: torch.Tensor, num_neighbors: int) -> KNNQueryResult:
    """Finds closest neighbors in data to the query points & their squared euclidean distances.

    Args:
      data: tensor of embedding, shape [embedding_size].
      memory: tensor of previous embedded data, shape [m, embedding_size],
        where m is the number of previous embeddings.
      num_neighbors: number of neighbors to find.

    Returns:
      KNNQueryResult with (all sorted by squared euclidean distance):
        - neighbors, shape [num_neighbors, feature size].
        - neighbor_indices, shape [num_neighbors].
        - neighbor_distances, shape [num_neighbors].
    """
    base.assert_rank_and_dtype(data, 1, torch.float32)
    base.assert_rank_and_dtype(memory, 2, torch.float32)
    base.assert_batch_dimension(data, memory.shape[-1], -1)

    assert memory.shape[0] >= num_neighbors

    distances = torch.stack([_cdist(memory[i], data) for i in range(memory.shape[0])], dim=0)

    distances, indices = distances.topk(num_neighbors, largest=False)
    neighbors = torch.stack([memory[i] for i in indices], dim=0)
    return KNNQueryResult(neighbors=neighbors, neighbor_indices=indices, neighbor_distances=distances)


class EpisodicBonusModule:
    """Episodic memory for calculate intrinsic bonus, used in NGU and Agent57."""

    def __init__(
        self,
        embedding_network: torch.nn.Module,
        device: torch.device,
        capacity: int,
        num_neighbors: int,
        kernel_epsilon: float = 0.0001,
        cluster_distance: float = 0.008,
        max_similarity: float = 8.0,
        c_constant: float = 0.001,
    ) -> None:

        self._embedding_network = embedding_network.to(device=device)
        self._device = device

        self._memory = collections.deque(maxlen=capacity)

        # Compute the running mean dₘ².
        self._cdist_normalizer = normalizer.Normalizer(eps=0.0001, clip_range=(-10, 10), device=self._device)

        self._num_neighbors = num_neighbors
        self._kernel_epsilon = kernel_epsilon
        self._cluster_distance = cluster_distance
        self._max_similarity = max_similarity
        self._c_constant = c_constant

    @torch.no_grad()
    def compute_bonus(self, s_t: torch.Tensor) -> float:
        """Compute episodic intrinsic bonus for given state."""
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)

        embedding = self._embedding_network(s_t).squeeze(0)

        memory = list(self._memory)

        # Insert single embedding into memory.
        self._memory.append(embedding)

        if len(memory) < self._num_neighbors:
            return 0.0

        memory = torch.stack(memory, dim=0)
        knn_query_result = knn_query(embedding, memory, self._num_neighbors)
        # neighbor_distances from knn_query is the squared Euclidean distances.
        nn_distances_sq = knn_query_result.neighbor_distances

        # Update the running mean dₘ².
        self._cdist_normalizer.update(nn_distances_sq[..., None])

        # Normalize distances with running mean dₘ².
        distance_rate = nn_distances_sq / self._cdist_normalizer.mean

        # The distance rate becomes 0 if already small: r <- max(r-ξ, 0).
        distance_rate = torch.min((distance_rate - self._cluster_distance), torch.tensor(0.0))

        # Compute the Kernel value K(xₖ, x) = ε/(rate + ε).
        kernel_output = self._kernel_epsilon / (distance_rate + self._kernel_epsilon)

        # Compute the similarity for the embedding x:
        # s = √(Σ_{xₖ ∈ Nₖ} K(xₖ, x)) + c
        similarity = torch.sqrt(torch.sum(kernel_output)) + self._c_constant

        if torch.isnan(similarity):
            return 0.0

        # Compute the intrinsic reward:
        # r = 1 / s.
        if similarity > self._max_similarity:
            return 0.0

        return (1 / similarity).cpu().item()

    def reset(self) -> None:
        """Resets episodic memory"""
        self._memory.clear()

    def update_embedding_network(self, state_dict: Dict) -> None:
        """Update embedding network."""
        self._embedding_network.load_state_dict(state_dict)


class RndLifeLongBonusModule:
    """RND lifelong intrinsic bonus module, used in NGU and Agent57."""

    def __init__(self, target_network: torch.nn.Module, predictor_network: torch.nn.Module, device: torch.device) -> None:
        self._target_network = target_network.to(device=device)
        self._predictor_network = predictor_network.to(device=device)
        self._device = device

        # RND module observation and lifeline intrinsic reward normalizers
        self._observation_normalizer = normalizer.Normalizer(eps=0.0001, clip_range=(-5, 5), device=self._device)
        self._int_reward_normalizer = normalizer.Normalizer(eps=0.0001, clip_range=(-10, 10), device=self._device)

    @torch.no_grad()
    def compute_bonus(self, s_t: torch.Tensor) -> float:
        """Compute lifelong bonus for a given state."""
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)

        # Update observation normalizer statistics and normalize observation
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

        # Pass state into RND target and predictor networks
        target = self._target_network(s_t)
        pred = self._predictor_network(s_t)

        lifelong_int_t = torch.sum(torch.square(pred - target), dim=1)  # Sums over latent dimension

        # Update lifelong intrinsic reward normalization statistics
        self._int_reward_normalizer.update(lifelong_int_t[..., None])

        # Normalize lifelong intrinsic reward, add some constant to avoid divide by zero
        norm_lifelong_int_t = 1 + (lifelong_int_t - self._int_reward_normalizer.mean) / (
            self._int_reward_normalizer.std + 1e-8
        )

        return norm_lifelong_int_t.cpu().item()

    def update_predictor_network(self, state_dict: Dict) -> None:
        """Update RND predictor network."""
        self._predictor_network.load_state_dict(state_dict)
