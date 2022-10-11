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
from typing import Mapping, Tuple, Text, Any
import os
from pathlib import Path
import torch


class PyTorchCheckpoint:
    """Simple checkpoint class implementation for PyTorch.

    Example create checkpoint:

    Create checkpoint instance and register network to the checkpoint internal state

    ```
    checkpoint = PyTorchCheckpoint(environment_name=FLAGS.environment_name, agent_name='NGU', save_dir=FLAGS.checkpoint_dir)
    checkpoint.register_pair(('network', network))

    for iteration in range(num_iterations):
        ...
        checkpoint.set_iteration(checkpoint)
        checkpoint.save()
        ...
    ```


    Example restore checkpoint:

    Create checkpoint instance and register network to the checkpoint internal state

    ```
    checkpoint = PyTorchCheckpoint(environment_name=FLAGS.environment_name, agent_name='NGU', save_dir=FLAGS.checkpoint_dir)
    checkpoint.register_pair(('network', network))

    checkpoint.restore(FLAGS.load_checkpoint_file)
    network.eval()

    ```

    """

    def __init__(
        self,
        environment_name: str,
        agent_name: str = 'RLAgent',
        save_dir: str = None,
        iteration: int = 0,
        file_ext: str = 'ckpt',
        restore_only: bool = False,
    ) -> None:
        """
        Args:
            environment_name: the environment name for the running agent.
            agent_name: agent name, default RLAgent.
            save_dir: checkpoint files save directory, default None.
            file_ext: checkpoint file extension.
            iteration: current iteration, default 0.
            restore_only: only used for evaluation, will not able to create checkpoints, default off.
        """

        # if not restore_only and (self.save_dir is None or self.save_dir == ''):
        #     raise ValueError('Invalid save_dir for checkpoint instance.')

        self.save_dir = save_dir
        self.file_ext = file_ext
        self.base_path = None

        if not restore_only and self.save_dir is not None and self.save_dir != '':
            self.base_path = Path(self.save_dir)
            if not self.base_path.exists():
                self.base_path.mkdir(parents=True, exist_ok=True)

        # Stores internal state for checkpoint.
        self.state = AttributeDict()
        self.state.iteration = iteration
        self.state.environment_name = environment_name
        self.state.agent_name = agent_name

    def register_pair(self, pair: Tuple[Text, Any]) -> None:
        """
        Add a pair of (key, item) to internal state so that later we can save as checkpoint.
        """
        assert isinstance(pair, Tuple)

        key, item = pair
        self.state[key] = item

    def save(self) -> str:
        """
        Save pytorch model as checkpoint, default file name is {agent_name}_{env_name}_{iteration}.ckpt, for example A2C_CartPole-v1_0.ckpt

        Returns:
            the full path of checkpoint file.
        """
        if self.base_path is None:
            return

        file_name = f'{self.state.agent_name}_{self.state.environment_name}_{self.state.iteration}.{self.file_ext}'
        save_path = self.base_path / file_name

        states = self._get_states_dict()
        torch.save(states, save_path)
        return save_path

    def restore(self, file_to_restore: str) -> None:
        """Try to restore checkpoint from a given checkpoint file."""
        if not file_to_restore or not os.path.isfile(file_to_restore) or not os.path.exists(file_to_restore):
            raise ValueError(f'"{file_to_restore}" is not a valid checkpoint file.')

        # Always load checkpoint state to CPU.
        loaded_state = torch.load(file_to_restore, map_location=torch.device('cpu'))

        # Needs to match environment_name and agent name
        if loaded_state['environment_name'] != self.state.environment_name:
            err_msg = f'environment_name "{loaded_state["environment_name"]}" and "{self.state.environment_name}" mismatch.'
            raise RuntimeError(err_msg)
        if 'agent_name' in loaded_state and loaded_state['agent_name'] != self.state.agent_name:
            err_msg = f'agent_name "{loaded_state["agent_name"]}" and "{self.state.agent_name}" mismatch.'
            raise RuntimeError(err_msg)

        # Ready to restore the states.
        loaded_keys = [k for k in loaded_state.keys()]

        for key, item in self.state.items():
            # Only restore state with object key in loaded_state
            if key not in loaded_keys:
                continue

            if self._is_torch_model(item):
                self.state[key].load_state_dict(loaded_state[key])
            else:
                self.state[key] = loaded_state[key]

    def set_iteration(self, iteration) -> None:
        self.state.iteration = iteration

    def get_iteration(self) -> int:
        return self.state.iteration

    def _get_states_dict(self) -> Mapping[Text, Any]:
        states_dict = {}
        # Find whatever is available to save, PyTorch models and optimizers need to use state_dict()
        for key, item in self.state.items():
            if self._is_torch_model(item):
                states_dict[key] = item.state_dict()
            else:
                states_dict[key] = item

        return states_dict

    def _is_torch_model(self, obj) -> bool:
        return isinstance(obj, (torch.nn.Module, torch.optim.Optimizer))


class AttributeDict(dict):
    """A `dict` that supports getting, setting, deleting keys via attributes."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]
