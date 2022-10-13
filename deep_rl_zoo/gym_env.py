# The MIT License

# Copyright (c) 2017 OpenAI (http://openai.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Code adapted from openAI baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
# Made changes to adapting coding styles, also add new function to support recording video.
#
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
CLASSIC_ENV_NAMES = ['CartPole-v1', 'LunarLander-v2', 'MountainCar-v0', 'Acrobot-v1']


class NoopReset(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """

    def __init__(self, env, noop_max=30):

        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class FireOnReset(gym.Wrapper):
    """Take fire action on reset for environments like Breakout."""

    def __init__(self, env):

        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
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


class DoneOnLifeLoss(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    """

    def __init__(self, env):

        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkip(gym.Wrapper):
    """Return only every `skip`-th frame"""

    def __init__(self, env, skip=4):

        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ResizeAndGrayscaleFrame(gym.ObservationWrapper):
    """
    Resize frames to 84x84, and grascale image as done in the Nature paper.
    If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
    observation should be warped.
    """

    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):

        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]  # pylint: disable=no-member
            self.observation_space.spaces[self._key] = new_space  # pylint: disable=no-member
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, observation):
        if self._key is None:
            frame = observation
        else:
            frame = observation[self._key]

        # pylint: disable=no-member
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        # pylint: disable=no-member

        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    """Stack k last frames.
    Returns lazy array, which is much more memory efficient.
    See Also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """

    def __init__(self, env, k):

        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    """This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    This object should only be converted to numpy array before being passed to the model.
    You'd not believe how complex the previous solution was."""

    def __init__(self, frames):
        self.dtype = frames[0].dtype
        self.shape = (frames[0].shape[0], frames[0].shape[1], len(frames))
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class ScaledFloatFrame(gym.ObservationWrapper):
    """Scale frame by devide 255."""

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class ObscureObservation(gym.ObservationWrapper):
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


class ClipRewardWithBound(gym.RewardWrapper):
    'Clip reward to in the range [-bound, bound]'

    def __init__(self, env, bound):
        super().__init__(env)
        self._bound = bound

    def reward(self, reward):
        return None if reward is None else max(min(reward, self._bound), -self._bound)


class ClipRewardWithSign(gym.RewardWrapper):
    """Clip reward to {+1, 0, -1} by using np.sign() function."""

    def reward(self, reward):
        return np.sign(reward)


class ObservationChannelFirst(gym.ObservationWrapper):
    """Make observation image channel first, this is for PyTorch only."""

    def __init__(self, env, scale_obs):
        super().__init__(env)
        old_shape = env.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        _low, _high = (0.0, 255) if not scale_obs else (0.0, 1.0)
        new_dtype = env.observation_space.dtype if not scale_obs else np.float32
        self.observation_space = Box(low=_low, high=_high, shape=new_shape, dtype=new_dtype)

    def observation(self, observation):
        # permute [H, W, C] array to in the range [C, H, W]
        return np.transpose(observation, axes=(2, 0, 1)).astype(self.observation_space.dtype)
        # obs = np.asarray(observation, dtype=self.observation_space.dtype).transpose(2, 0, 1)
        # make sure it's C-contiguous for compress state
        # return np.ascontiguousarray(obs, dtype=self.observation_space.dtype)


class ObservationToNumpy(gym.ObservationWrapper):
    """Make the observation into numpy ndarrays."""

    def observation(self, observation):
        return np.asarray(observation, dtype=self.observation_space.dtype)


class ClipObservationWithBound(gym.ObservationWrapper):
    """Make the observation into [-max_abs_value, max_abs_value]."""

    def __init__(self, env, max_abs_value):
        super().__init__(env)
        self._max_abs_value = max_abs_value

    def observation(self, observation):
        return np.clip(observation, -self._max_abs_value, self._max_abs_value)


class RecordRawReward(gym.Wrapper):
    """This wrapper will add unclipped/unscaled raw reward to the info dict."""

    def step(self, action):
        """Take action and add unclipped/unscaled raw reward to the info dict."""

        obs, reward, done, info = self.env.step(action)
        info['raw_reward'] = reward

        return obs, reward, done, info


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
    terminal_on_life_loss: bool = False,
    clip_reward: bool = True,
    scale_obs: bool = False,
    channel_first: bool = True,
) -> gym.Env:
    """
    Process gym env for Atari games according to the Nature DQN paper.

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
        obscure_epsilon: with epsilon probability [0.0, 1.0), obscure the state to make it POMDP.
        terminal_on_life_loss: if True, mark end of game when loss a life, default off.
        clip_reward: clip reward in the range of [-1, 1], default on.
        scale_obs: scale the frame by devide 255, turn this on may require 4-5x more RAM when using experience replay, default off.
        channel_first: if True, change observation image from shape [H, W, C] to in the range [C, H, W], this is for PyTorch only, default on.

    Returns:
        preprocessed gym.Env for Atari games.
    """
    if 'NoFrameskip' in env_name:
        raise ValueError(f'Environment name should not include NoFrameskip, got {env_name}')

    env = gym.make(f'{env_name}NoFrameskip-v4')
    env.seed(seed)
    # env.reset(seed=seed)

    # Change TimeLimit wrapper to 108,000 steps (30 min) as default in the
    # litterature instead of OpenAI Gym's default of 100,000 steps.
    env = gym.wrappers.TimeLimit(env.env, max_episode_steps=None if max_episode_steps <= 0 else max_episode_steps)

    env = RecordRawReward(env)

    env = NoopReset(env, noop_max=noop_max)
    env = MaxAndSkip(env, skip=frame_skip)

    # Obscure observation with obscure_epsilon probability
    if obscure_epsilon > 0.0:
        env = ObscureObservation(env, obscure_epsilon)
    if terminal_on_life_loss:
        env = DoneOnLifeLoss(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireOnReset(env)
    env = ResizeAndGrayscaleFrame(env, width=screen_width, height=screen_height)
    if scale_obs:
        env = ScaledFloatFrame(env)
    if clip_reward:
        env = ClipRewardWithBound(env, 1.0)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    if channel_first:
        env = ObservationChannelFirst(env, scale_obs)
    else:
        # This is required as LazeFrame object is not numpy.array.
        env = ObservationToNumpy(env)
    return env


def create_classic_environment(
    env_name: str,
    seed: int = 1,
    max_abs_reward: int = None,
    obscure_epsilon: float = 0.0,
) -> gym.Env:
    """
    Process gym env for classic control tasks like CartPole, LunarLander, MountainCar

    Args:
        env_name: the environment name with version attached.
        seed: seed the runtime.
        max_abs_reward: clip reward in the range of [-max_abs_reward, max_abs_reward], default off.
        obscure_epsilon: with epsilon probability [0.0, 1.0) obscure the state to make it POMDP.

    Returns:
        gym.Env for classic control tasks
    """

    env = gym.make(env_name)
    env.seed(seed)
    # env.reset(seed=seed)

    env = RecordRawReward(env)

    # Clip reward to max absolute reward bound
    if max_abs_reward is not None:
        env = ClipRewardWithBound(env, abs(max_abs_reward))

    # Obscure observation with obscure_epsilon probability
    if obscure_epsilon > 0.0:
        env = ObscureObservation(env, obscure_epsilon)

    return env


def create_continuous_environment(
    env_name: str,
    seed: int = 1,
    max_abs_obs: int = 10,
    max_abs_reward: int = 10,
) -> gym.Env:
    """
    Process gym env for classic control tasks like CartPole, LunarLander, MountainCar

    Args:
        env_name: the environment name with version attached.
        seed: seed the runtime.
        max_abs_obs: clip observation in the range of [-max_abs_obs, max_abs_obs], default 10.
        max_abs_reward: clip reward in the range of [-max_abs_reward, max_abs_reward], default 10.

    Returns:
        gym.Env for classic control tasks
    """

    env = gym.make(env_name)
    env = RecordRawReward(env)

    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env)

    # Using lambda function does not work with python multiprocessing
    # env = gym.wrappers.TransformObservation(env, lambda reward: np.clip(obs, -max_abs_obs, max_abs_obs))
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -max_abs_reward, max_abs_reward))

    # env = ClipObservationWithBound(env, max_abs_obs)
    # env = ClipRewardWithBound(env, max_abs_reward)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def play_and_record_video(
    agent: types_lib.Agent,
    env: gym.Env,
    save_dir: str = 'recordings',
    auto_fire: bool = True,
) -> None:
    """Self-play and record a video for a single game.

    Args:
        env: the gym environment to play.
        agent: the agent which should have step() method to return action for a given state.
        save_dir: the recording video file directory, default save to 'recordings/{agent_name}_{env.spec.id}_{timestamp}'.
        auto_fire: if True, take 'FIRE' action after loss a life, default on.

    Raises:
        if agent is not an instance of types_lib.Agent.
    """

    if not isinstance(agent, types_lib.Agent):
        raise RuntimeError('Expect agent to have a callable step() method.')

    # Create a sub folder with name env.id + timestamp
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    full_save_dir = f'{save_dir}/{agent.agent_name}_{env.spec.id}_{ts}'
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
    agent.reset()

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

        if done:
            break

    env.close()
