# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
#
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
"""Tests for DQN."""

from pathlib import Path
import os
import shutil
from absl import flags
from absl.testing import flagsaver
from absl.testing import absltest
from deep_rl_zoo.dqn import run_classic

FLAGS = flags.FLAGS
FLAGS.checkpoint_dir = '/tmp/e2e_test_checkpoint'
FLAGS.results_csv_path = ''
FLAGS.tensorboard = False
FLAGS.num_train_frames = 500
FLAGS.num_eval_frames = 200
FLAGS.num_iterations = 1


class RunClassicGameTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.checkpoint_dir = Path(FLAGS.checkpoint_dir)

    @flagsaver.flagsaver
    def test_can_run_agent_and_save_csv_log(self):

        _log_file = '/tmp/e2e-test-dqn-results.csv'
        log_file = Path(_log_file)

        if log_file.exists():
            os.remove(log_file)

        FLAGS.environment_name = 'CartPole-v1'
        FLAGS.batch_size = 4
        FLAGS.replay_capacity = 100
        FLAGS.min_replay_size = 4
        FLAGS.clip_grad = True
        FLAGS.results_csv_path = _log_file
        run_classic.main(None)

        log_file_exists = log_file.exists() and log_file.is_file()

        self.assertTrue(log_file_exists)
        os.remove(log_file)  # clean up

    def tearDown(self) -> None:
        # Clean up
        try:
            shutil.rmtree(self.checkpoint_dir)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    absltest.main()
