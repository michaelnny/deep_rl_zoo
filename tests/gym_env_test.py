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
"""Tests for gym_env.py."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import gym
from deep_rl_zoo import gym_env


class BumpUpReward(gym.RewardWrapper):
    'Bump up rewards so later can use it to test clip reward'

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        noise = np.random.randint(10, 100)
        if np.random.rand() < 0.5:
            return reward + noise
        return reward - noise


class AtariEnvironmentTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.frame_height = 96
        self.frame_width = 84

    @parameterized.named_parameters(('environment_pong', 'Pong'), ('environment_breakout', 'Breakout'))
    def test_run_step(self, environment_name):
        seed = 1
        env = gym_env.create_atari_environment(env_name=environment_name, seed=seed)
        env.reset()
        env.step(0)
        env.close()

    def test_environment_name_exception(self):
        environment_name = 'PongNoFrameskip-v4'
        seed = 1

        with self.assertRaisesRegex(ValueError, 'Environment name should not include NoFrameskip, got PongNoFrameskip-v4'):
            env = gym_env.create_atari_environment(env_name=environment_name, seed=seed)
            env.reset()
            env.step(0)
            env.close()

    @parameterized.named_parameters(
        ('sizes_84x84x1', (84, 84, 1)), ('sizes_84x84x4', (84, 84, 4)), ('sizes_96x72x8', (96, 72, 8))
    )
    def test_env_channel_last_different_sizes(self, sizes):
        seed = 1

        env = gym_env.create_atari_environment(
            env_name='Pong',
            seed=seed,
            frame_height=sizes[0],
            frame_width=sizes[1],
            frame_stack=sizes[2],
            channel_first=False,
            scale_obs=False,
        )

        obs = env.reset()
        expected_dtype = np.uint8
        self.assertEqual(env.observation_space.shape, sizes)
        self.assertEqual(env.observation_space.dtype, expected_dtype)
        self.assertEqual(obs.shape, sizes)
        self.assertEqual(obs.dtype, expected_dtype)
        # self.assertTrue(obs.flags['C_CONTIGUOUS'])

        for _ in range(3):  # 3 games
            obs = env.reset()
            for _ in range(20):  # each game 20 steps
                obs, r, done, _ = env.step(env.action_space.sample())
                self.assertEqual(obs.shape, sizes)
                self.assertEqual(obs.dtype, expected_dtype)
                if done:
                    break
        env.close()

    @parameterized.named_parameters(
        ('sizes_1x84x84', (1, 84, 84)), ('sizes_4x84x84', (4, 84, 84)), ('sizes_8x96x72', (8, 96, 72))
    )
    def test_env_channel_first_different_sizes(self, sizes):
        seed = 1

        env = gym_env.create_atari_environment(
            env_name='Pong',
            seed=seed,
            frame_height=sizes[1],
            frame_width=sizes[2],
            frame_stack=sizes[0],
            channel_first=True,
            scale_obs=False,
        )

        obs = env.reset()
        expected_dtype = np.uint8
        self.assertEqual(env.observation_space.shape, sizes)
        self.assertEqual(env.observation_space.dtype, expected_dtype)
        self.assertEqual(obs.shape, sizes)
        self.assertEqual(obs.dtype, expected_dtype)
        # self.assertTrue(obs.flags['C_CONTIGUOUS'])

        for _ in range(3):  # 3 games
            obs = env.reset()
            for _ in range(20):  # each game 20 steps
                obs, r, done, _ = env.step(env.action_space.sample())
                self.assertEqual(obs.shape, sizes)
                self.assertEqual(obs.dtype, expected_dtype)
                if done:
                    break
        env.close()

    @parameterized.named_parameters(('not_clip_reward', False), ('clip_reward', True))
    def test_clip_reward(self, clip_reward):
        full_env_name = 'PongNoFrameskip-v4'
        seed = 1

        env = env = gym.make(full_env_name)
        env.seed(seed)
        env = BumpUpReward(env)
        env = gym_env.ClipRewardWithBound(env, 1.0)

        for _ in range(3):  # 3 games
            obs = env.reset()
            for _ in range(20):  # each game 20 steps
                obs, r, done, _ = env.step(env.action_space.sample())
                if clip_reward:
                    self.assertBetween(r, -1, 1)
                else:
                    self.assertBetween(r, -400, 400)
                self.assertEqual(obs.dtype, np.uint8)
                if done:
                    break
        env.close()

    def test_scale_observation(self):
        environment_name = 'Pong'
        seed = 1
        env = gym_env.create_atari_environment(
            env_name=environment_name,
            seed=seed,
            frame_height=self.frame_height,
            frame_width=self.frame_width,
            frame_skip=4,
            frame_stack=4,
            scale_obs=True,
            channel_first=False,
        )
        obs = env.reset()

        expected_dtype = np.float32
        self.assertEqual(env.observation_space.dtype, expected_dtype)
        self.assertEqual(obs.dtype, expected_dtype)
        self.assertLessEqual(np.max(obs), 1.0)
        self.assertGreaterEqual(np.min(obs), 0.0)

        for _ in range(3):  # 3 games
            obs = env.reset()
            for _ in range(20):  # each game 20 steps
                obs, r, done, _ = env.step(env.action_space.sample())
                self.assertEqual(obs.dtype, expected_dtype)
                self.assertLessEqual(np.max(obs), 1.0)
                self.assertGreaterEqual(np.min(obs), 0.0)
                if done:
                    break
        env.close()


class SimpleEnvironmentTest(parameterized.TestCase):
    @parameterized.parameters(gym_env.CLASSIC_ENV_NAMES)
    def test_run_step(self, environment_name):
        seed = 1
        env = gym_env.create_classic_environment(env_name=environment_name, seed=seed)
        env.reset()
        env.step(0)
        env.close()


if __name__ == '__main__':
    absltest.main()
