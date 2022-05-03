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
flags.DEFINE_string('checkpoint_path', '', '')
flags.DEFINE_string('environment_name', '', '')


class PytorchCheckpointTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        FLAGS.checkpoint_path = '/tmp/unit_test_checkpoint'
        FLAGS.environment_name = 'Dummy-v1'
        self.checkpoint_dir = Path(FLAGS.checkpoint_path)

        # Remove existing directory
        if self.checkpoint_dir.exists() and self.checkpoint_dir.is_dir():
            shutil.rmtree(self.checkpoint_dir)

    @parameterized.named_parameters(('iteration_0', 0), ('iteration_1', 1))
    def test_save_checkpoint(self, iteration):
        """Checks can save checkpoint"""

        model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        checkpoint = checkpoint_lib.PyTorchCheckpoint(FLAGS.checkpoint_path)
        checkpoint.state.environment_name = FLAGS.environment_name
        checkpoint.state.iteration = iteration
        checkpoint.state.model = model
        checkpoint.state.optimizer = optimizer

        checkpoint.save()

        ckpt_file = f'{FLAGS.checkpoint_path}/{FLAGS.environment_name}_iteration_{iteration}.ckpt'
        latest_file = f'{FLAGS.checkpoint_path}/{FLAGS.environment_name}-latest'

        self.assertTrue(self.checkpoint_dir.exists())
        self.assertTrue(self.checkpoint_dir.is_dir())
        self.assertTrue(self.checkpoint_dir.exists())
        self.assertTrue(os.path.exists(ckpt_file))
        self.assertTrue(os.path.exists(latest_file))
        self.assertTrue(os.path.islink(latest_file))

    def test_load_and_restore_latest_checkpoint(self):
        """Checks can load and restore latest checkpoint"""

        model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        checkpoint = checkpoint_lib.PyTorchCheckpoint(FLAGS.checkpoint_path)
        checkpoint.state.environment_name = FLAGS.environment_name
        checkpoint.state.iteration = 0
        checkpoint.state.model = model
        checkpoint.state.optimizer = optimizer

        # Generate 5 checkpoint files
        for i in range(5):
            checkpoint.state.iteration = i
            checkpoint.save()

        # Simulate a new run by clean up existing checkpoint instance
        del checkpoint
        checkpoint = checkpoint_lib.PyTorchCheckpoint(FLAGS.checkpoint_path, False)
        checkpoint.state.environment_name = FLAGS.environment_name
        checkpoint.state.iteration = 0  # deliberately set iteration to zero
        checkpoint.state.model = model
        checkpoint.state.optimizer = optimizer

        checkpoint.restore(self.runtime_device)
        self.assertEqual(checkpoint.state.iteration, 4)

    @parameterized.named_parameters(('iteration_0', 0), ('iteration_1', 3))
    def test_load_and_restore_checkpoint_specific_file(self, iteration):
        """Checks can load and restore a specific checkpoint file"""

        model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        checkpoint = checkpoint_lib.PyTorchCheckpoint(FLAGS.checkpoint_path)
        checkpoint.state.environment_name = FLAGS.environment_name
        checkpoint.state.iteration = 0
        checkpoint.state.model = model
        checkpoint.state.optimizer = optimizer

        # Generate 5 checkpoint files
        for i in range(5):
            checkpoint.state.iteration = i
            checkpoint.save()

        # Simulate a new run by clean up existing checkpoint instance
        del checkpoint

        ckpt_file = f'{FLAGS.checkpoint_path}/{FLAGS.environment_name}_iteration_{iteration}.ckpt'
        checkpoint = checkpoint_lib.PyTorchCheckpoint(ckpt_file, False)

        checkpoint.state.environment_name = FLAGS.environment_name
        checkpoint.state.iteration = 0  # deliberately set iteration to zero
        checkpoint.state.model = model
        checkpoint.state.optimizer = optimizer

        checkpoint.restore(self.runtime_device)
        self.assertEqual(checkpoint.state.iteration, iteration)

    @parameterized.named_parameters(('dummy_file_null', ''), ('dummy_file_not_found', '/tmp/1234.abcd'))
    def test_load_and_restore_invalid_path(self, dummy_file):
        """Checks can not load and restore latest checkpoint dur to invalid file path"""

        model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        checkpoint = checkpoint_lib.PyTorchCheckpoint(FLAGS.checkpoint_path)
        checkpoint.state.environment_name = FLAGS.environment_name
        checkpoint.state.iteration = 0
        checkpoint.state.model = model
        checkpoint.state.optimizer = optimizer

        # Generate 5 checkpoint files
        for i in range(5):
            checkpoint.state.iteration = i
            checkpoint.save()

        # Simulate a new run by clean up existing checkpoint instance
        del checkpoint
        checkpoint = checkpoint_lib.PyTorchCheckpoint(dummy_file, False)
        checkpoint.state.environment_name = FLAGS.environment_name
        checkpoint.state.iteration = 0  # deliberately set iteration to zero
        checkpoint.state.model = model
        checkpoint.state.optimizer = optimizer

        with self.assertRaisesRegex(RuntimeError, f'Except a valid check point file, but not found at "{dummy_file}"'):
            checkpoint.restore(self.runtime_device)

    @parameterized.named_parameters(('env_name_cartpole', 'CartPole-v1'), ('env_name_pong', 'Pong'))
    def test_load_and_restore_env_name_mismatch(self, env_name):
        """Checks can not load and restore latest checkpoint dur to env_name mismatch"""

        model = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        checkpoint = checkpoint_lib.PyTorchCheckpoint(FLAGS.checkpoint_path)
        checkpoint.state.environment_name = FLAGS.environment_name
        checkpoint.state.iteration = 0
        checkpoint.state.model = model
        checkpoint.state.optimizer = optimizer

        # Generate 5 checkpoint files
        for i in range(5):
            checkpoint.state.iteration = i
            checkpoint.save()

        # Simulate a new run by clean up existing checkpoint instance
        del checkpoint
        checkpoint = checkpoint_lib.PyTorchCheckpoint(FLAGS.checkpoint_path, False)
        checkpoint.state.environment_name = env_name
        checkpoint.state.iteration = 0  # deliberately set iteration to zero
        checkpoint.state.model = model
        checkpoint.state.optimizer = optimizer

        with self.assertRaisesRegex(
            RuntimeError,
            f'Except a valid check point file, but not found at "{FLAGS.checkpoint_path}"',
        ):
            checkpoint.restore(self.runtime_device)

    def tearDown(self) -> None:
        # Clean up
        try:
            shutil.rmtree(self.checkpoint_dir)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    absltest.main()
