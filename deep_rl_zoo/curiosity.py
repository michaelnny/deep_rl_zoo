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

from typing import NamedTuple, Dict
import numpy as np
import torch
from deep_rl_zoo import base
from deep_rl_zoo import normalizer


class KNNQueryResult(NamedTuple):
    neighbors: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_distances: torch.Tensor


def knn_query(current: torch.Tensor, memory: torch.Tensor, num_neighbors: int) -> KNNQueryResult:
    """Finds closest neighbors and their squared euclidean distances.

    Args:
      current: tensor of current embedding, shape [embedding_size].
      memory: tensor of previous embedded data, shape [m, embedding_size],
        where m is the number of previous embeddings.
      num_neighbors: number of neighbors to find.

    Returns:
      KNNQueryResult with (all sorted by squared euclidean distance):
        - neighbors, shape [num_neighbors, feature size].
        - neighbor_indices, shape [num_neighbors].
        - neighbor_distances, shape [num_neighbors].
    """
    base.assert_rank_and_dtype(current, 1, torch.float32)
    base.assert_rank_and_dtype(memory, 2, torch.float32)
    base.assert_batch_dimension(current, memory.shape[-1], -1)

    assert memory.shape[0] >= num_neighbors

    distances = torch.cdist(current.unsqueeze(0), memory).squeeze(0).pow(2)

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

        self._memory = torch.zeros(
            capacity, self._embedding_network.embed_size, device=self._device
        )  # Initialize memory tensor
        self._mask = torch.zeros(capacity, dtype=torch.bool, device=self._device)  # Initialize mask

        self._capacity = capacity
        self._counter = 0

        # Compute the running mean dₘ².
        self._cdist_normalizer = normalizer.TorchRunningMeanStd(shape=(1,), device=self._device)

        self._num_neighbors = num_neighbors
        self._kernel_epsilon = kernel_epsilon
        self._cluster_distance = cluster_distance
        self._max_similarity = max_similarity
        self._c_constant = c_constant

    def _add_to_memory(self, embedding: torch.Tensor) -> None:
        # Insert new embedding
        idx = self._counter % self._capacity
        self._memory[idx] = embedding
        self._mask[idx] = True
        self._counter += 1

    @torch.no_grad()
    def compute_bonus(self, s_t: torch.Tensor) -> float:
        """Compute episodic intrinsic bonus for given state."""
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)

        embedding = self._embedding_network(s_t).squeeze(0)

        # Make a copy of mask because we don't want to use the current embedding when compute the distance
        prev_mask = self._mask.clone()

        self._add_to_memory(embedding)

        if self._counter <= self._num_neighbors:
            return 0.0

        knn_query_result = knn_query(embedding, self._memory[prev_mask], self._num_neighbors)

        # neighbor_distances from knn_query is the squared Euclidean distances.
        nn_distances_sq = knn_query_result.neighbor_distances

        # Update the running mean dₘ².
        self._cdist_normalizer.update_single(nn_distances_sq)

        # Normalize distances with running mean dₘ².
        distance_rate = nn_distances_sq / (self._cdist_normalizer.mean + 1e-8)

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

    def reset(self):
        self._mask = torch.zeros(self._capacity, dtype=torch.bool, device=self._device)  # Initialize mask
        self._counter = 0

    def update_embedding_network(self, state_dict: Dict) -> None:
        """Update embedding network."""
        self._embedding_network.load_state_dict(state_dict)


class RndLifeLongBonusModule:
    """RND lifelong intrinsic bonus module, used in NGU and Agent57."""

    def __init__(
        self, target_network: torch.nn.Module, predictor_network: torch.nn.Module, device: torch.device, discount: float
    ) -> None:
        self._target_network = target_network.to(device=device)
        self._predictor_network = predictor_network.to(device=device)
        self._device = device
        self._discount = discount

        # RND module observation and lifeline intrinsic reward normalizers
        self._int_reward_normalizer = normalizer.RunningMeanStd(shape=(1,))
        self._rnd_obs_normalizer = normalizer.TorchRunningMeanStd(shape=(1, 84, 84), device=self._device)

    @torch.no_grad()
    def _normalize_rnd_obs(self, rnd_obs):
        rnd_obs = rnd_obs.to(device=self._device, dtype=torch.float32)

        normed_obs = self._rnd_obs_normalizer.normalize(rnd_obs)

        normed_obs = normed_obs.clamp(-5, 5)

        self._rnd_obs_normalizer.update_single(rnd_obs)

        return normed_obs

    def _normalize_int_rewards(self, int_rewards):
        """Compute returns then normalize the intrinsic reward based on these returns"""

        self._int_reward_normalizer.update_single(int_rewards)

        normed_int_rewards = int_rewards / np.sqrt(self._int_reward_normalizer.var + 1e-8)

        return normed_int_rewards.item()

    @torch.no_grad()
    def compute_bonus(self, s_t: torch.Tensor) -> float:
        """Compute lifelong bonus for a given state."""
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)

        normed_s_t = self._normalize_rnd_obs(s_t)

        pred = self._predictor_network(normed_s_t)
        target = self._target_network(normed_s_t)

        int_r_t = torch.square(pred - target).mean(dim=1).detach().cpu().numpy()

        # Normalize intrinsic reward
        normed_int_r_t = self._normalize_int_rewards(int_r_t)

        return normed_int_r_t

    def update_predictor_network(self, state_dict: Dict) -> None:
        """Update RND predictor network."""
        self._predictor_network.load_state_dict(state_dict)
