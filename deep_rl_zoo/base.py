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
"""Utilities for Reinforcement Learning ops."""
from typing import NamedTuple, Optional, Tuple, Union
import torch
import torch.nn.functional as F


class LossOutput(NamedTuple):
    loss: torch.Tensor
    extra: Optional[NamedTuple]


def assert_rank_and_dtype(tensor: torch.Tensor, rank: Union[int, Tuple[int]], dtype: Union[torch.dtype, Tuple[torch.dtype]]):
    """Asserts that the tensor have the correct rank and dtype.

    Args:
      tensor: tensor.
      rank: A scalar or tuple of scalars specifying the rank that the tensors passed need to have,
        if is a tuple, 'OR' condition is checked.
      dtype: A single torch tensor dtype or tuple of dtypes. If is a tuple, 'OR' condition is checked.

    Raises:
      ValueError: If the tensor is empty or fail the rank and dtype checks.
    """
    assert_rank(tensor, rank)
    assert_dtype(tensor, dtype)


def assert_rank(tensor: torch.Tensor, rank: Union[int, Tuple[int]]) -> None:
    """Asserts that the tensor have the correct rank.

    Args:
      tensor: tensor.
      rank: A scalar or tuple of scalars specifying the rank that the tensors passed need to have,
        if is a tuple, 'OR' condition is checked.

    Raises:
      ValueError: If the tensor is empty or fail the rank checks.
    """

    if not isinstance(tensor, torch.Tensor):
        raise ValueError('Error in rank and/or compatibility check. The input tensor should be a valid torch.Tensor.')
    supported_rank = []
    if isinstance(rank, tuple):
        supported_rank = rank
    else:
        supported_rank.append(rank)
    if len(tensor.shape) not in supported_rank:
        raise ValueError(
            f'Error in rank and/or compatibility check. The input tensor should be rank {rank} torch.Tensor, got {tensor.shape}.'
        )


def assert_dtype(tensor: torch.Tensor, dtype: Union[torch.dtype, Tuple[torch.dtype]]) -> None:
    """Asserts that the tensor have the correct dtype.

    Args:
      tensor: tensor.
      dtype: A single torch tensor dtype or tuple of dtypes. If is a tuple, 'OR' condition is checked.

    Raises:
      ValueError: If the list of tensors is empty or fail the dtype checks.
    """

    if not isinstance(tensor, torch.Tensor):
        raise ValueError('Error in rank and/or compatibility check. The input tensor should be a valid torch.Tensor.')
    supported_dtype = []
    if isinstance(dtype, tuple):
        supported_dtype = dtype
    else:
        supported_dtype.append(dtype)
    if tensor.dtype not in supported_dtype:
        raise ValueError(f'Error in rank and/or compatibility check. The input tensor should be {dtype}, got {tensor.dtype}.')


def assert_batch_dimension(tensor: torch.Tensor, batch_dize: int, dim: int = 0) -> None:
    """Asserts that the tensor have the correct rank.

    Args:
      tensor: tensor.
      batch_dize: A scalar specifying the batch size that the tensors passed need to have.
      dim: A scalar specifying which dimension to perform the check.

    Raises:
      ValueError: If the list of tensors is empty or fail the batch dimension checks.
    """

    if not isinstance(tensor, torch.Tensor):
        raise ValueError('Error in rank and/or compatibility check. The input tensor should be a valid torch.Tensor.')

    if tensor.shape[dim] != batch_dize:
        raise ValueError(
            f'Error in rank and/or compatibility check. The input tensor should have {batch_dize} entry on batch dimension {dim}, got {tensor.shape}.'
        )


def batched_index(values: torch.Tensor, indices: torch.Tensor, dim: int = -1, keepdims: bool = False) -> torch.Tensor:
    """Equivalent to `values[:, indices]`.

    Performs indexing on batches and sequence-batches by reducing over
    zero-masked values.

    Args:
      values: tensor of shape `[B, num_values]` or `[T, B, num_values]`
      indices: tensor of shape `[B]` or `[T, B]` containing indices.
      dim: indexing dimension to perform the selection, defauot -1.
      keepdims: If `True`, the returned tensor will have an added 1 dimension at
        the end (e.g. `[B, 1]` or `[T, B, 1]`).

    Returns:
      Tensor of shape `[B]` or `[T, B]` containing values for the given indices.

    Raises: ValueError if values and indices have sizes that are known
      statically (i.e. during graph construction), and those sizes are not
      compatible (see shape descriptions in Args list above).
    """

    assert_rank(values, (2, 3))
    assert_rank_and_dtype(indices, (1, 2), torch.long)

    assert_batch_dimension(indices, values.shape[0], 0)

    if len(indices.shape) == 2:
        assert_batch_dimension(indices, values.shape[1], 1)

    one_hot_indices = F.one_hot(indices, values.shape[dim]).to(dtype=values.dtype)

    # Incase values have rank 3, add a new dimension to one_hot_indices, for quantile_q_learning.
    if len(values.shape) == 3 and len(one_hot_indices.shape) == 2:
        one_hot_indices = one_hot_indices.unsqueeze(1)
    return torch.sum(values * one_hot_indices, dim=dim, keepdims=keepdims)
