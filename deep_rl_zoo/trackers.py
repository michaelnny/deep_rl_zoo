# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
#
# The file has been modified by The Deep RL Zoo Authors
# to add tensorboard support, and tracking additional agent statistics.
#
# ==============================================================================
"""Components for statistics and tensorboard monitoring."""
# Temporally suppress annoy DeprecationWarning from torch tensorboard.
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import timeit
from pathlib import Path
import shutil
import collections
from typing import Any, Text, Iterable, Mapping, Optional, Tuple, Union
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# pylint: disable=import-error
import deep_rl_zoo.replay as replay_lib


class EpisodeTracker:
    """Tracks episode return and other statistics."""

    def __init__(self):
        self._num_steps_since_reset = None
        self._episode_returns = None
        self._episode_steps = None
        self._current_episode_rewards = None
        self._current_episode_step = None

    def step(self, env, timestep_t, agent, a_t) -> None:
        """Accumulates statistics from timestep."""
        del (env, agent, a_t)

        if timestep_t.first:
            if self._current_episode_rewards:
                raise ValueError('Current episode reward list should be empty.')
            if self._current_episode_step != 0:
                raise ValueError('Current episode step should be zero.')
        else:
            # First reward is invalid, all other rewards are appended.
            self._current_episode_rewards.append(timestep_t.reward)

        self._num_steps_since_reset += 1
        self._current_episode_step += 1

        if timestep_t.done:
            self._episode_returns.append(sum(self._current_episode_rewards))
            self._episode_steps.append(self._current_episode_step)
            self._current_episode_rewards = []
            self._current_episode_step = 0

    def reset(self) -> None:
        """Resets all gathered statistics, not to be called between episodes."""
        self._num_steps_since_reset = 0
        self._episode_returns = []
        self._episode_steps = []
        self._current_episode_step = 0
        self._current_episode_rewards = []

    def get(self) -> Mapping[str, Union[int, float, None]]:
        """Aggregates statistics and returns as a dictionary.

        Here the convention is `episode_return` is set to `current_episode_return`
        if a full episode has not been encountered. Otherwise it is set to
        `mean_episode_return` which is the mean return of complete episodes only. If
        no steps have been taken at all, `episode_return` is set to `NaN`.

        Returns:
          A dictionary of aggregated statistics.
        """

        if self._episode_returns:
            mean_episode_return = np.array(self._episode_returns).mean()
            current_episode_return = sum(self._current_episode_rewards)
            episode_return = mean_episode_return
        else:
            mean_episode_return = np.nan
            if self._num_steps_since_reset > 0:
                current_episode_return = sum(self._current_episode_rewards)
            else:
                current_episode_return = np.nan
            episode_return = current_episode_return

        return {
            'mean_episode_return': mean_episode_return,
            'current_episode_return': current_episode_return,
            'episode_return': episode_return,
            'num_episodes': len(self._episode_returns),
            'current_episode_step': self._current_episode_step,
            'num_steps_since_reset': self._num_steps_since_reset,
        }


class StepRateTracker:
    """Tracks step rate, number of steps taken and duration since last reset."""

    def __init__(self):
        self._num_steps_since_reset = None
        self._start = None

    def step(self, env, timestep_t, agent, a_t) -> None:
        """Accumulates statistics from timestep."""
        del (env, timestep_t, agent, a_t)

        self._num_steps_since_reset += 1

    def reset(self) -> None:
        """Reset statistics."""
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()

    def get(self) -> Mapping[Text, float]:
        """Returns statistics as a dictionary."""
        duration = timeit.default_timer() - self._start
        if self._num_steps_since_reset > 0:
            step_rate = self._num_steps_since_reset / duration
        else:
            step_rate = np.nan
        return {
            'step_rate': step_rate,
            'num_steps': self._num_steps_since_reset,
            'duration': duration,
        }


class TensorboardEpisodTracker(EpisodeTracker):
    """Extend EpisodeTracker to write to tensorboard"""

    # def __init__(self, log_dir: str):
    def __init__(self, writer: SummaryWriter):
        super().__init__()
        self._writer = writer

    def step(self, env, timestep_t, agent, a_t) -> None:
        super().step(env, timestep_t, agent, a_t)

        # To improve performance, only logging at end of an episode.
        if timestep_t.done:
            tb_steps = self._num_steps_since_reset
            num_episodes = len(self._episode_returns)

            # tracker per episode
            episode_return = self._episode_returns[-1]
            episode_step = self._episode_steps[-1]

            # tracker per step
            self._writer.add_scalar('performance(env_steps)/num_episodes', num_episodes, tb_steps)
            self._writer.add_scalar('performance(env_steps)/episode_return', episode_return, tb_steps)
            self._writer.add_scalar('performance(env_steps)/episode_steps', episode_step, tb_steps)


class TensorboardStepRateTracker(StepRateTracker):
    """Extend StepRateTracker to write to tensorboard"""

    def __init__(self, writer: SummaryWriter):
        super().__init__()
        self._writer = writer

    def step(self, env, timestep_t, agent, a_t) -> None:
        """Accumulates statistics from timestep."""
        super().step(env, timestep_t, agent, a_t)

        # To improve performance, only logging at end of an episode.
        if timestep_t.done:
            time_stats = self.get()

            # tracker per step
            tb_steps = self._num_steps_since_reset
            self._writer.add_scalar('performance(env_steps)/run_duration(minutes)', time_stats['duration'] / 60, tb_steps)
            self._writer.add_scalar('performance(env_steps)/step_rate(second)', time_stats['step_rate'], tb_steps)


class TensorboardScreenshotTracker:
    """Take screenshots of the environment and add to tensorboard every N episodes.
    This should be used for debugging only."""

    def __init__(self, writer: SummaryWriter, log_interval: int = 100):
        self._num_steps_since_reset = None
        self._log_interval = log_interval
        self._writer = writer

    def step(self, env, timestep_t, agent, a_t) -> None:
        """Accumulates statistics from timestep."""
        del (agent, a_t)

        if timestep_t.done:
            self._num_steps_since_reset += 1

            if self._num_steps_since_reset % self._log_interval == 0:
                try:
                    img = env.render(mode='rgb_array')
                    self._writer.add_image(
                        f'debug(episode)/done_episode_{self._num_steps_since_reset}',
                        img,
                        self._num_steps_since_reset,
                        dataformats='HWC',
                    )
                except Exception:
                    pass

    def reset(self) -> None:
        """Reset statistics."""
        self._num_steps_since_reset = 0

    def get(self) -> Mapping[Text, float]:
        """Returns statistics as a dictionary."""
        return {}


class TensorboardAgentStatisticsTracker:
    """Write agent statistics to tensorboard"""

    def __init__(self, writer: SummaryWriter):
        self._num_steps_since_reset = None
        self._writer = writer

    def step(self, env, timestep_t, agent, a_t) -> None:
        """Accumulates statistics from timestep."""
        del (env, a_t)
        self._num_steps_since_reset += 1

        # To improve performance, only logging at end of an episode.
        # This should not block the training loop if there's any exception.
        if timestep_t.done:
            try:
                stats = agent.statistics
                if stats:
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            self._writer.add_scalar(f'agent_statistics(env_steps)/{k}', v, self._num_steps_since_reset)
            except Exception:
                pass

    def reset(self) -> None:
        """Reset statistics."""
        self._num_steps_since_reset = 0

    def get(self) -> Mapping[Text, float]:
        """Returns statistics as a dictionary."""
        return {}


class TensorboardLearnerStatisticsTracker:
    """Write learner statistics to tensorboard in actor-learner scheme"""

    def __init__(self, writer: SummaryWriter):
        self._num_steps_since_reset = None
        self._writer = writer

    def step(self, stats) -> None:
        """Accumulates statistics from timestep."""
        # Log every N train steps. This should not block the training loop if there's any exception.
        if self._num_steps_since_reset % 10 == 0:
            try:
                if stats:
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            self._writer.add_scalar(f'learner_statistics(learner_steps)/{k}', v, self._num_steps_since_reset)
            except Exception:
                pass

        self._num_steps_since_reset += 1

    def reset(self) -> None:
        """Reset statistics."""
        self._num_steps_since_reset = 0

    def get(self) -> Mapping[Text, float]:
        """Returns statistics as a dictionary."""
        return {}


def make_default_trackers(log_dir=None, debug_screenshots_frequency=0):
    """
    Create trackers for the training/evaluation run.

    Args:
        log_dir: tensorboard runtime log directory.
        debug_screenshots_frequency: the frequency to take screenshots and add to tensorboard, default 0 no screenshots.
    """

    if log_dir:
        log_dir = Path(f'runs/{log_dir}')

        # Remove existing log directory
        if log_dir.exists() and log_dir.is_dir():
            shutil.rmtree(log_dir)

        writer = SummaryWriter(log_dir)

        trackers = [
            TensorboardEpisodTracker(writer),
            TensorboardStepRateTracker(writer),
            TensorboardAgentStatisticsTracker(writer),
        ]

        if debug_screenshots_frequency > 0:
            trackers.append(TensorboardScreenshotTracker(writer, debug_screenshots_frequency))

        return trackers

    else:
        return [EpisodeTracker(), StepRateTracker()]


def make_learner_trackers(run_log_dir=None):
    """
    Create trackers for learner for parallel training (actor-learner) run.

    Args:
        run_log_dir: tensorboard run log directory.
    """

    if run_log_dir:
        tb_log_dir = Path(f'runs/{run_log_dir}')

        # Remove existing log directory
        if tb_log_dir.exists() and tb_log_dir.is_dir():
            shutil.rmtree(tb_log_dir)

        writer = SummaryWriter(tb_log_dir)

        return [TensorboardLearnerStatisticsTracker(writer)]

    else:
        return []


def generate_statistics(
    trackers: Iterable[Any],
    timestep_action_sequence: Iterable[Tuple[Optional[replay_lib.Transition]]],
) -> Mapping[str, Any]:
    """Generates statistics from a sequence of timestep and actions."""
    # Only reset at the start, not between episodes.
    for tracker in trackers:
        tracker.reset()

    for env, timestep_t, agent, a_t in timestep_action_sequence:
        for tracker in trackers:
            tracker.step(env, timestep_t, agent, a_t)

    # Merge all statistics dictionaries into one.
    statistics_dicts = (tracker.get() for tracker in trackers)
    return dict(collections.ChainMap(*statistics_dicts))
