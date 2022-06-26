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
"""Tests for Rainbow DQN."""
from absl import flags
from absl.testing import flagsaver
from absl.testing import absltest
from deep_rl_zoo.rainbow import run_atari

FLAGS = flags.FLAGS


class RunAtariTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        FLAGS.checkpoint_path = ''
        FLAGS.results_csv_path = ''
        FLAGS.tensorboard = False
        FLAGS.max_episode_steps = 500

    @flagsaver.flagsaver
    def test_can_run_agent(self):
        FLAGS.environment_name = 'Pong'
        FLAGS.num_train_steps = 500
        FLAGS.num_eval_steps = 200
        FLAGS.num_iterations = 1
        FLAGS.batch_size = 4
        FLAGS.replay_capacity = 100
        FLAGS.min_replay_size = 4
        FLAGS.clip_grad = True
        run_atari.main(None)


if __name__ == '__main__':
    absltest.main()