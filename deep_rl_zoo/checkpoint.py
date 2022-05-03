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
"""Checkpoint class for Deep RL Zoo."""
import os
from pathlib import Path
from absl import logging
import torch


class PyTorchCheckpoint:
    """Simple checkpoint implementation for PyTorch.
    Supports multiple networks, must provide a 'environment_name',
    as most the agents are trained for a specific game.
    When try to restore from checkpoint, it'll also check if 'environment_name' matches.

    Usage:
        Case 1: create checkpoint.
            checkpoint = PyTorchCheckpoint('/tmp/checkpoint/')
            state = checkpoint.state
            state.environment_name = 'Pong'
            state.iteration = 0
            state.network = network
            state.rnd_target_network = rnd_target_network
            state.rnd_predictor_network = rnd_predictor_network
            state.embedding_network = embedding_network

            checkpoint.save()

        Case 2: restore checkpoint from file.
            checkpoint = PyTorchCheckpoint('/tmp/checkpoint/Pong_iteration_0.ckpt')
            state = checkpoint.state
            state.environment_name = 'Pong'
            state.network = network
            state.rnd_target_network = rnd_target_network
            state.rnd_predictor_network = rnd_predictor_network
            state.embedding_network = embedding_network

            checkpoint.restore(runtime_device)

    """

    def __init__(self, checkpoint_dir: str, is_training: bool = True):
        self._file_ext = 'ckpt'
        if is_training and f'.{self._file_ext}' in checkpoint_dir:
            raise ValueError(f'Expect checkpoint a dir not a file when in training mode, got {checkpoint_dir}')

        self.state = AttributeDict()
        self._checkpoint_dir = checkpoint_dir

        self._base_path = Path(checkpoint_dir)
        self._abs_base_path = self._base_path.resolve()
        # Only create directory if it's in training mode
        if is_training and not self._base_path.exists():
            self._base_path.mkdir(parents=True, exist_ok=True)

        self._symlink_path = None
        self._abs_symlink_path = None

    def save(self) -> None:
        """Save pytorch model"""
        # Only continue if checkpoint dir is given.
        if not self._checkpoint_dir or self._checkpoint_dir == '':
            return

        if self.state.environment_name is None:
            raise RuntimeError(f'Expect state.environment_name to be string, got "{self.state.environment_name}"')
        if self.state.iteration is None:
            raise RuntimeError(f'Expect state.iteration to be integer, got "{self.state.iteration}"')

        saved_file = self._get_file_name()
        ckpt_file_path = self._base_path / saved_file

        state = self._prepare_state_for_checkpoint()

        torch.save(state, ckpt_file_path)
        logging.info(f'Created checkpoint file at "{ckpt_file_path}"')

        self._create_symlink_for_file(saved_file)

    def restore(self, device: torch.device) -> None:
        """Try to restore checkpoint from a valid checkpoint file.
        if the given checkpoint file is a dir, will try to find the latest checkpoint file,
        otherwise if the given checkpoint is a file with extension '.chkpt', then will try to load the checkpoint file.
        """

        if not self._symlink_path:
            self._init_symlink()

        if not self._can_be_restored():
            raise RuntimeError(f'Except a valid check point file, but not found at "{self._checkpoint_dir}"')

        file_path = self._get_checkpoint_files()
        # Load state into memory from checkpoint file
        loaded_state = torch.load(file_path, map_location=torch.device(device))

        # Needs to match environment_name
        if loaded_state['environment_name'] != self.state.environment_name:
            raise RuntimeError(
                f'Expect environment_name to match from saved checkpoint, '
                f'got "{loaded_state["environment_name"]}" and "{self.state.environment_name}"'
            )

        loaded_keys = [k for k in loaded_state.keys()]

        for key, item in self.state.items():
            # Only restore state with object key in loaded_state
            if key not in loaded_keys:
                continue

            if self._is_pytorch_object(item):
                self.state[key].load_state_dict(loaded_state[key])
            else:
                self.state[key] = loaded_state[key]

        logging.info(f'Loaded checkpoint file from "{file_path}"')

    def _get_file_name(self):
        return f'{self.state.environment_name}_iteration_{self.state.iteration}.{self._file_ext}'

    def _init_symlink(self):
        self._symlink_path = Path(f'{self._checkpoint_dir}/{self.state.environment_name}-latest')

    def _create_symlink_for_file(self, file_path):
        # Add 'latest' as symlink to the most recent checkpoint file.
        try:
            if not self._symlink_path:
                self._init_symlink()
            if self._symlink_path.is_symlink():
                os.remove(self._symlink_path)
            if not self._symlink_path.exists():
                self._symlink_path.symlink_to(file_path)
        except OSError:
            pass

    def _prepare_state_for_checkpoint(self):
        obj = {}
        # Find what ever is available to save, PyTorch models need to use state_dict()
        for key, item in self.state.items():
            if self._is_pytorch_object(item):
                obj[key] = item.state_dict()
            else:
                obj[key] = item

        return obj

    def _can_be_restored(self) -> bool:
        """Check if can be restored either from specific checkpoint file,
        or latest checkpoint file from base dir"""
        return self._get_checkpoint_files() is not None

    def _get_checkpoint_files(self):
        """
        Check file first, then the latest symlink, if none exits, return.
        """
        if self._base_path.exists() and self._base_path.is_file():
            return self._base_path  # .resolve()
        if self._symlink_path.is_symlink():
            return self._symlink_path  # .resolve()
        return None

    def _is_pytorch_object(self, obj):
        return isinstance(obj, (torch.nn.Module, torch.optim.Optimizer))


class AttributeDict(dict):
    """A `dict` that supports getting, setting, deleting keys via attributes."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]
