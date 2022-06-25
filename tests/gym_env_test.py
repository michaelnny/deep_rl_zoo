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

    @parameterized.named_parameters(('frame_stack_1', 1), ('frame_stack_4', 4))
    def test_observation_shape_channel_first(self, frame_stack):
        environment_name = 'Pong'
        seed = 1
        env = gym_env.create_atari_environment(
            env_name=environment_name,
            seed=seed,
            screen_height=210,
            screen_width=160,
            frame_skip=4,
            frame_stack=frame_stack,
            channel_first=True,
        )
        for _ in range(5):  # 5 games
            obs = env.reset()
            for _ in range(100):  # each game 100 steps
                obs, r, done, _ = env.step(env.action_space.sample())
                # obs = np.asarray(obs)
                self.assertEqual(obs.dtype, np.uint8)
                self.assertEqual(obs.shape, (frame_stack, 210, 160))
                self.assertTrue(obs.flags['C_CONTIGUOUS'])
                if done:
                    break
        env.close()

    @parameterized.named_parameters(('frame_stack_1', 1), ('frame_stack_4', 4))
    def test_observation_shape_channel_last(self, frame_stack):
        environment_name = 'Pong'
        seed = 1
        env = gym_env.create_atari_environment(
            env_name=environment_name,
            seed=seed,
            screen_height=210,
            screen_width=160,
            frame_skip=4,
            frame_stack=frame_stack,
            channel_first=False,
        )

        for _ in range(5):  # 5 games
            obs = env.reset()
            for _ in range(100):  # each game 100 steps
                obs, r, done, _ = env.step(env.action_space.sample())
                # obs = np.asarray(obs)
                self.assertEqual(obs.dtype, np.uint8)
                self.assertEqual(obs.shape, (210, 160, frame_stack))
                self.assertTrue(obs.flags['C_CONTIGUOUS'])
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

        for _ in range(5):  # 5 games
            obs = env.reset()
            for _ in range(100):  # each game 100 steps
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
            screen_height=210,
            screen_width=160,
            frame_skip=4,
            frame_stack=4,
            scale_obs=True,
            channel_first=False,
        )

        for _ in range(5):  # 5 games
            obs = env.reset()
            for _ in range(30):  # each game 100 steps
                obs, r, done, _ = env.step(env.action_space.sample())
                self.assertEqual(obs.dtype, np.float32)
                self.assertLessEqual(np.max(obs), 1.0)
                self.assertGreaterEqual(np.min(obs), 0.0)
                self.assertEqual(obs.shape, (210, 160, 4))
                self.assertTrue(obs.flags['C_CONTIGUOUS'])
                if done:
                    break
        env.close()

    def test_obscure_observation(self):
        environment_name = 'Pong'
        seed = 1
        env = gym_env.create_atari_environment(
            env_name=environment_name,
            screen_height=210,
            screen_width=160,
            seed=seed,
            frame_skip=4,
            frame_stack=1,
            channel_first=True,
            obscure_epsilon=0.5,
        )
        for _ in range(5):  # 5 games
            obs = env.reset()
            for _ in range(100):  # each game 100 steps
                obs, r, done, _ = env.step(env.action_space.sample())
                # obs = np.asarray(obs)
                self.assertEqual(obs.dtype, np.uint8)
                self.assertEqual(obs.shape, (1, 210, 160))
                self.assertEqual(len(obs.shape), 3)
                self.assertTrue(obs.flags['C_CONTIGUOUS'])
                if done:
                    break
        env.close()

    @parameterized.parameters([1, 1.8])
    def test_obscure_epsilon_exception(self, obscure_epsilon):
        environment_name = 'Pong'
        seed = 1
        with self.assertRaisesRegex(ValueError, 'Expect obscure epsilon should be between'):
            env = gym_env.create_atari_environment(
                env_name=environment_name,
                screen_height=210,
                screen_width=160,
                seed=seed,
                frame_skip=4,
                frame_stack=1,
                channel_first=True,
                obscure_epsilon=obscure_epsilon,
            )


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
