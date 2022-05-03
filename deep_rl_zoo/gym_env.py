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
"""gym environment processing components."""

# Temporally suppress annoy DeprecationWarning from gym.
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

from collections import deque
import datetime
import gym
from gym.spaces import Box
import numpy as np
import cv2
from absl import logging

# pylint: disable=import-error
import deep_rl_zoo.types as types_lib

# Not so complete classic env name lists.
CLASSIC_ENV_NAMES = ['CartPole-v1', 'LunarLander-v2', 'MountainCar-v0']


class AtariPreprocessing(gym.Wrapper):
    r"""Atari 2600 preprocessings.
    This class follows the guidelines in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".

    Specifically:
        * NoopReset: obtain initial state by taking random number of no-ops on reset.
        * Frame skipping (defaults to 4).
        * Terminal signal when a life is lost (off by default).
        * Grayscale and max-pooling of the last two frames.
        * Downsample the screen to a square image (defaults to 84x84).
        * Clip reward into in the range [-1, 1] (optional)

    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 0,
        screen_height: int = 84,
        screen_width: int = 84,
        frame_skip: int = 4,
        done_on_life_loss: bool = False,
        clip_reward: bool = False,
        channel_first: bool = False,
    ):
        """
        Args:
            env: Gym environment whose observations are preprocessed.
            noop_max: maximum number of no-ops to apply at the beginning
                of each episode to reduce determinism. These no-ops are applied at a
                low-level, before frame skipping.
            screen_height: resize height of Atari frame
            screen_width: resize width of Atari frame
            frame_skip: the frequency at which the agent experiences the game.
            done_on_life_loss: if True, then step() returns done=True whenever a
                life is lost.
            clip_reward: if True, use np.sign() to scale summed reward into in the range [-1, 1].
            channel_first: if True, put channel at the first dimension, this is used for PyTorch only.

        Raises:
            RuntimeError: if opencv not installed.
            ValueError: if frame_skip or screen_height, screen_width, frame_skip, noop_max are not strictly positive.
        """
        super().__init__(env)
        if cv2 is None:
            raise RuntimeError(
                'opencv-python package not installed! Try running pip install gym[atari] to get dependencies  for atari'
            )

        if frame_skip <= 0:
            raise ValueError(f'Expect frame_skip to be positive integer, got {frame_skip}')
        if screen_height <= 0:
            raise ValueError(f'Expect screen_height to be positive integer, got {screen_height}')
        if screen_width <= 0:
            raise ValueError(f'Expect screen_width to be positive integer, got {screen_width}')
        if frame_skip <= 0:
            raise ValueError(f'Expect frame_skip to be positive integer, got {frame_skip}')

        if frame_skip > 1:
            if 'NoFrameskip' not in env.spec.id and getattr(env.unwrapped, '_frameskip', None) != 1:
                raise ValueError(
                    'Disable frame-skipping in the original env. Otherwise, more than one'
                    ' frame-skip will happen as through this wrapper'
                )
        self.noop_max = noop_max
        if env.unwrapped.get_action_meanings()[0] != 'NOOP':
            raise RuntimeError(f'Expect first action to be NOOP, got {env.unwrapped.get_action_meanings()[0]}')

        self.screen_height = screen_height
        self.screen_width = screen_width
        self.frame_skip = frame_skip
        self.done_on_life_loss = done_on_life_loss
        self.clip_reward = clip_reward
        self.channel_first = channel_first

        # buffer of most recent two observations for max pooling
        self.obs_buffer = [
            np.empty(env.observation_space.shape[:2], dtype=np.uint8),
            np.empty(env.observation_space.shape[:2], dtype=np.uint8),
        ]

        self.lives = 0
        self.game_over = False

        # The channel of observation shape is 1 as we always grayscale and max-pool last two frames
        if self.channel_first:
            obs_shape = (1, screen_height, screen_width)
        else:
            obs_shape = (screen_height, screen_width, 1)
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def step(self, action):
        accumulated_reward = 0.0

        for t in range(self.frame_skip):
            _, reward, done, info = self.env.step(action)

            accumulated_reward += reward
            self.game_over = done

            if self.done_on_life_loss:
                new_lives = self.ale.lives()
                done = done or new_lives < self.lives
                self.lives = new_lives

            if done:
                break
            if t == self.frame_skip - 2:
                self.ale.getScreenGrayscale(self.obs_buffer[1])
            elif t == self.frame_skip - 1:
                self.ale.getScreenGrayscale(self.obs_buffer[0])

        # Pool the last two observations.
        observation = self._pool_and_resize()

        # Clip summed rewards.
        if self.clip_reward:
            accumulated_reward = np.sign(accumulated_reward)

        return observation, accumulated_reward, done, info

    def reset(self, **kwargs):
        # NoopReset
        if kwargs.get('return_info', False):
            _, reset_info = self.env.reset(**kwargs)
        else:
            _ = self.env.reset(**kwargs)
            reset_info = {}

        reset_info = self._apply_random_noops(kwargs, reset_info)

        self.lives = self.ale.lives()
        # We bypass the Gym observation altogether and directly fetch the
        # grayscale image from the ALE. This is a little faster.
        self.ale.getScreenGrayscale(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)

        if kwargs.get('return_info', False):
            return self._pool_and_resize(), reset_info
        else:
            return self._pool_and_resize()

    @property
    def ale(self):
        """Fix cannot pickle 'ale_py._ale_py.ALEInterface' object error in multiprocessing."""
        return self.env.unwrapped.ale

    def _apply_random_noops(self, kwargs, reset_info):
        """Steps environment with random no-ops.
        No-op is assumed to be action 0."""
        if self.noop_max <= 0:
            return reset_info

        noops = self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            _, _, done, step_info = self.env.step(0)
            reset_info.update(step_info)
            if done:
                if kwargs.get('return_info', False):
                    _, reset_info = self.env.reset(**kwargs)
                else:
                    _ = self.env.reset(**kwargs)
                    reset_info = {}
        return reset_info

    def _pool_and_resize(self):
        """Returns resized observation with shape (height, width)"""
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])

        # cv2.resize() takes (width, height)
        # pylint: disable=no-member
        transformed_image = cv2.resize(
            self.obs_buffer[0],
            (self.screen_width, self.screen_height),
            interpolation=cv2.INTER_AREA,
        )
        # pylint: disable=no-member

        obs = np.asarray(transformed_image, dtype=np.uint8)
        # Add a channel axis
        if self.channel_first:
            return np.expand_dims(obs, axis=0)  # shape (1, height, width)
        else:
            return np.expand_dims(obs, axis=2)  # shape (height, width, 1)


class FrameStackWrapper(gym.ObservationWrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.

    Example::
        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = gym.wrappers.AtariPreprocessing(env)
        >>> env = FrameStackWrapper(env, 4)
        >>> env.observation_space
        Box(210, 160, 4)
    Args:
        env (Env): environment object
        num_stack (int): number of stacks
    """

    def __init__(self, env, num_stack, channel_first):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)
        self.channel_first = channel_first
        old_obs_shape = env.observation_space.shape

        if self.channel_first:
            obs_shape = (old_obs_shape[0] * num_stack, old_obs_shape[1], old_obs_shape[2])
        else:
            obs_shape = (old_obs_shape[0], old_obs_shape[1], old_obs_shape[-1] * num_stack)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=env.observation_space.dtype,
        )

    def observation(self):  # pylint: disable=arguments-differ
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.channel_first)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        if kwargs.get('return_info', False):
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
            info = None  # Unused
        for _ in range(self.num_stack):
            self.frames.append(obs)

        if kwargs.get('return_info', False):
            return self.observation(), info
        else:
            return self.observation()


class LazyFrames:
    """This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    This object should only be converted to numpy array before being passed to the model.
    You'd not believe how complex the previous solution was."""

    def __init__(self, frames, channel_first):

        self.dtype = frames[0].dtype
        self.channel_first = channel_first
        if self.channel_first:
            self.shape = (len(frames), frames[0].shape[1], frames[0].shape[2])
        else:
            self.shape = (frames[0].shape[0], frames[0].shape[1], len(frames))
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            if self.channel_first:
                self._out = np.concatenate(self._frames, axis=0)
            else:
                self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class ObscureObservationWrapper(gym.ObservationWrapper):
    """Make the environment POMDP by obscure the state with probability epsilon.
    this should be used as the very first"""

    def __init__(self, env, epsilon: float = 0.0):
        super().__init__(env)
        if not 0.0 <= epsilon < 1.0:
            raise ValueError(f'Expect obscure epsilon should be between [0.0, 1), got {epsilon}')
        self._eps = epsilon

    def observation(self, observation):
        if self.env.unwrapped.np_random.random() <= self._eps:
            observation = np.zeros_like(observation, dtype=self.observation_space.dtype)
        return observation


class FireOnResetWrapper(gym.Wrapper):
    """Some environments requires the agent to press the 'FIRE' button to start the game,
    this wrapper will automatically take the 'FIRE' action when calls reset().
    """

    def __init__(self, env):

        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        """Try to take the 'FIRE' action."""
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class ObservationChannelFirstWrapper(gym.ObservationWrapper):
    """Make observation image channel first, this is for PyTorch only."""

    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        # permute [H, W, C] array to in the range [C, H, W]
        obs = np.array(observation).transpose(2, 0, 1)
        # make sure it's C-contiguous for compress state
        return np.ascontiguousarray(obs, dtype=obs.dtype)


class ClipRewardWithBoundWrapper(gym.RewardWrapper):
    'Clip reward to in the range [-bound, bound]'

    def __init__(self, env, bound):
        super().__init__(env)
        self._bound = bound

    def reward(self, reward):
        return None if reward is None else max(min(reward, self._bound), -self._bound)


class ObservationToNumpyWrapper(gym.ObservationWrapper):
    """Make the observation into numpy ndarrays."""

    def observation(self, observation):
        return np.asarray(observation, dtype=observation.dtype)


def create_atari_environment(
    env_name: str,
    seed: int = 1,
    frame_skip: int = 4,
    frame_stack: int = 4,
    screen_height: int = 84,
    screen_width: int = 84,
    noop_max: int = 30,
    max_episode_steps: int = 108000,
    obscure_epsilon: float = 0.0,
    sticky_actions: bool = False,
    done_on_life_loss: bool = False,
    clip_reward: bool = False,
    channel_first: bool = True,
) -> gym.Env:
    """
    Process gym env for Atari games

    Args:
        env_name: the environment name without 'NoFrameskip' and version.
        seed: seed the runtime.
        frame_skip: the frequency at which the agent experiences the game,
                the environment will also repeat action.
        frame_stack: stack n last frames.
        screen_height: height of the resized frame.
        screen_width: width of the resized frame.
        noop_max: maximum number of no-ops to apply at the beginning
                of each episode to reduce determinism. These no-ops are applied at a
                low-level, before frame skipping.
        max_episode_steps: maximum steps for an episode.
        obscure_epsilon: with epsilon probability [0.0, 1.0) obscure the state to make it POMDP.
        sticky_actions: if True, use sticky version of the atari game,
                which will repeat the last action with 0.25 probability.
        done_on_life_loss: if True, mark end of game when loss a life, default off.
        clip_reward: clip reward in the range of [-1, 1], default off.
        channel_first: if True, change observation image from shape [H, W, C] to in the range [C, H, W], this is for PyTorch only, default on.

    Returns:
        preprocessed gym.Env for Atari games, note the obersevations are not scaled to in the range [0, 1].
    """
    if 'NoFrameskip' in env_name:
        raise ValueError(f'Environment name should not include NoFrameskip, got {env_name}')

    version = 'v0' if sticky_actions else 'v4'
    full_env_name = f'{env_name}NoFrameskip-{version}'

    env = gym.make(full_env_name)
    env.reset(seed=seed)

    # Change TimeLimit wrapper to 108,000 steps (30 min) as default in the
    # litterature instead of OpenAI Gym's default of 100,000 steps.
    env = gym.wrappers.TimeLimit(env.env, max_episode_steps=max_episode_steps)

    # Obscure observation with obscure_epsilon probability
    if obscure_epsilon > 0.0:
        env = ObscureObservationWrapper(env, obscure_epsilon)

    # Returns processes env with observation space shape (height, width)
    env = AtariPreprocessing(
        env=env,
        screen_height=screen_height,
        screen_width=screen_width,
        frame_skip=frame_skip,
        noop_max=noop_max,
        done_on_life_loss=done_on_life_loss,
        clip_reward=clip_reward,
        channel_first=channel_first,
    )

    if frame_stack > 1:
        env = FrameStackWrapper(env, num_stack=frame_stack, channel_first=channel_first)
    env = ObservationToNumpyWrapper(env)

    return env


def create_classic_environment(
    env_name: str,
    seed: int = 1,
    max_abs_reward: int = None,
    obscure_epsilon: float = 0.0,
) -> gym.Env:
    """
    Process gym env for classic games like CartPole, LunarLander, MountainCar

    Args:
        env_name: the environment name with version attached.
        seed: seed the runtime.
        max_abs_reward: clip reward in the range of [-max_abs_reward, max_abs_reward], default off.
        obscure_epsilon: with epsilon probability [0.0, 1.0) obscure the state to make it POMDP.

    Returns:
        gym.Env for classic games
    """

    env = gym.make(env_name)
    env.reset(seed=seed)

    # Clip reward to max absolute reward bound
    if max_abs_reward is not None:
        env = ClipRewardWithBoundWrapper(env, abs(max_abs_reward))

    # Obscure observation with obscure_epsilon probability
    if obscure_epsilon > 0.0:
        env = ObscureObservationWrapper(env, obscure_epsilon)

    return env


def play_and_record_video(
    agent: types_lib.Agent,
    env: gym.Env,
    max_episode_steps: int = 0,
    save_dir: str = 'recordings',
    auto_fire: bool = True,
) -> None:
    """Self-play and record a video for a single game.

    Args:
        env: the gym environment to play.
        agent: the agent which should have step() method to return action for a given state.
        max_episode_steps: maximun steps per episode, default 0.
        save_dir: the recording video file directory, default save to 'recording/some_time_stamp'.
        auto_fire: if True, take 'FIRE' action after loss a life, default off.
    """
    if not (hasattr(agent, 'step') and callable(getattr(agent, 'step'))):
        raise RuntimeError('Expect agent to have a callable step() method.')

    # Create a sub folder with name env.id + timestamp
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    full_save_dir = f'{save_dir}/{env.spec.id}_{ts}'
    logging.info(f'Recording self-play video at "{full_save_dir}"')

    def take_fire_action(env):
        """Some games requires the agent to press 'FIRE' to start the game once loss a life."""
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        s_t, _, _, _ = env.step(1)
        return s_t

    def check_atari_env(env):
        """Check if is atari env and has fire action."""
        has_fire_action = False
        lives = 0
        try:
            lives = env.ale.lives()
            if env.unwrapped.get_action_meanings()[1] == 'FIRE':
                has_fire_action = True
        except AttributeError:
            pass

        return has_fire_action, lives

    env = gym.wrappers.RecordVideo(env, full_save_dir)

    observation = env.reset()
    reward = 0.0
    done = False
    first_step = True

    t = 0
    should_fire, lives = check_atari_env(env)
    if not auto_fire:
        should_fire = False

    while True:
        timestep_t = types_lib.TimeStep(observation=observation, reward=reward, done=done, first=first_step)
        a_t = agent.step(timestep_t)
        observation, reward, done, info = env.step(a_t)
        t += 1
        first_step = False

        # Take fire action after loss a life
        if should_fire and not done and lives != info['lives']:
            lives = info['lives']
            observation = take_fire_action(env)

        if max_episode_steps > 0 and t >= max_episode_steps:
            assert t == max_episode_steps
            done = True

        if done:
            break

    env.close()
