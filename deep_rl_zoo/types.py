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
# The functions has been modified by The Deep RL Zoo Authors
# to support gym environment without DeepMind's dm.env wrapper.
#
# ==============================================================================
"""Common types"""
import abc

from typing import NamedTuple, Text, Mapping, Iterable, Optional, Any
import numpy as np

Action = int


class TimeStep(NamedTuple):
    """Environment timestep"""

    observation: Optional[np.ndarray]
    reward: Optional[float]  # reward couble be clipped or scaled
    raw_reward: Optional[float]  # non-clipped/unscaled reward, only used by the trackers
    done: Optional[bool]  # This may not be the real episode termination, since Atari we often use done-on-loss-life
    first: Optional[bool]  # first step of an episode


class Agent(abc.ABC):
    """Agent interface."""

    agent_name: str  # agent name
    step_t: int  # runtime steps

    @abc.abstractmethod
    def step(self, timestep: TimeStep) -> Action:
        """Selects action given timestep and potentially learns."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat.

        This method should be called at the beginning of every episode.
        """

    @property
    @abc.abstractmethod
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""


class Learner(abc.ABC):
    """Learner interface."""

    agent_name: str  # agent name
    step_t: int  # learner steps

    @abc.abstractmethod
    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.

        Yields:
            learner statistics if network parameters update occurred, otherwise None.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Should be called at the beginning of every iteration."""

    @abc.abstractmethod
    def received_item_from_queue(self, item: Any) -> None:
        """Received item send by actors through multiprocessing queue."""

    @property
    @abc.abstractmethod
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
