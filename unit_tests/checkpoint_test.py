# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for checkpoint.py."""
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from pathlib import Path
import shutil
import os
import torch

from deep_rl_zoo import checkpoint as checkpoint_lib

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', '/tmp/unit_test_checkpoint', '')
flags.DEFINE_string('environment_name', 'DummyEnv-v1', '')
flags.DEFINE_string('agent_name', 'UNIT-TEST-RL', '')


class PytorchCheckpointTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_dir = Path(FLAGS.checkpoint_dir)

        # Remove existing directory
        if self.checkpoint_dir.exists() and self.checkpoint_dir.is_dir():
            shutil.rmtree(self.checkpoint_dir)

    @parameterized.named_parameters(('iteration_0', 0), ('iteration_1', 1))
    def test_save_checkpoint(self, iteration):
        """Checks can save checkpoint"""

        model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        checkpoint = checkpoint_lib.PyTorchCheckpoint(
            environment_name=FLAGS.environment_name, agent_name=FLAGS.agent_name, save_dir=FLAGS.checkpoint_dir
        )
        checkpoint.register_pair(('model', model))
        checkpoint.register_pair(('optimizer', optimizer))

        checkpoint.set_iteration(iteration)
        checkpoint.save()

        expected_ckpt_file = f'{FLAGS.checkpoint_dir}/{FLAGS.agent_name}_{FLAGS.environment_name}_{iteration}.ckpt'

        self.assertTrue(self.checkpoint_dir.exists())
        self.assertTrue(self.checkpoint_dir.is_dir())
        self.assertTrue(os.path.exists(expected_ckpt_file))

    @parameterized.named_parameters(('iteration_0', 0), ('iteration_1', 1))
    def test_load_and_restore_checkpoint_specific_file(self, iteration):
        """Checks can load and restore a specific checkpoint file"""

        model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        checkpoint = checkpoint_lib.PyTorchCheckpoint(
            environment_name=FLAGS.environment_name, agent_name=FLAGS.agent_name, save_dir=FLAGS.checkpoint_dir
        )
        checkpoint.register_pair(('model', model))
        checkpoint.register_pair(('optimizer', optimizer))

        # Generate 2 checkpoint files
        for i in range(2):
            checkpoint.set_iteration(i)
            checkpoint.save()

        ckpt_file_to_restore = f'{FLAGS.checkpoint_dir}/{FLAGS.agent_name}_{FLAGS.environment_name}_{iteration}.ckpt'

        restore_checkpoint = checkpoint_lib.PyTorchCheckpoint(
            environment_name=FLAGS.environment_name, agent_name=FLAGS.agent_name, restore_only=True, iteration=999
        )
        restore_checkpoint.register_pair(('model', model))
        restore_checkpoint.register_pair(('optimizer', optimizer))

        restore_checkpoint.restore(ckpt_file_to_restore)

        model.to(self.runtime_device)

        self.assertEqual(restore_checkpoint.state.iteration, iteration)

    @parameterized.named_parameters(('dummy_file_null', ''), ('dummy_file_not_found', '/tmp/1234.abcd'))
    def test_load_and_restore_invalid_path(self, dummy_file):
        """Checks can not load and restore latest checkpoint dur to invalid file path"""

        model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        checkpoint = checkpoint_lib.PyTorchCheckpoint(
            environment_name=FLAGS.environment_name, agent_name=FLAGS.agent_name, save_dir=FLAGS.checkpoint_dir
        )
        checkpoint.register_pair(('model', model))
        checkpoint.register_pair(('optimizer', optimizer))

        # Generate 2 checkpoint files
        for i in range(2):
            checkpoint.set_iteration(i)
            checkpoint.save()

        restore_checkpoint = checkpoint_lib.PyTorchCheckpoint(
            environment_name=FLAGS.environment_name, agent_name=FLAGS.agent_name, restore_only=True, iteration=999
        )
        restore_checkpoint.register_pair(('model', model))
        restore_checkpoint.register_pair(('optimizer', optimizer))

        with self.assertRaisesRegex(ValueError, f'"{dummy_file}" is not a valid checkpoint file.'):
            checkpoint.restore(dummy_file)

    @parameterized.named_parameters(('env_name_cartpole', 'CartPole-v1'), ('env_name_pong', 'Pong'))
    def test_load_and_restore_env_name_mismatch(self, env_name):
        """Checks can not load and restore latest checkpoint dur to env_name mismatch"""

        model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        checkpoint = checkpoint_lib.PyTorchCheckpoint(
            environment_name=FLAGS.environment_name, agent_name=FLAGS.agent_name, save_dir=FLAGS.checkpoint_dir
        )
        checkpoint.register_pair(('model', model))
        checkpoint.register_pair(('optimizer', optimizer))

        # Generate 2 checkpoint files
        ckpt_file_to_restore = checkpoint.save()

        restore_checkpoint = checkpoint_lib.PyTorchCheckpoint(
            environment_name=env_name, agent_name=FLAGS.agent_name, restore_only=True, iteration=999
        )
        restore_checkpoint.register_pair(('model', model))
        restore_checkpoint.register_pair(('optimizer', optimizer))

        with self.assertRaisesRegex(
            RuntimeError,
            f'environment_name "{FLAGS.environment_name}" and "{env_name}" mismatch',
        ):
            restore_checkpoint.restore(ckpt_file_to_restore)

    def tearDown(self) -> None:
        # Clean up
        try:
            shutil.rmtree(self.checkpoint_dir)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    absltest.main()
