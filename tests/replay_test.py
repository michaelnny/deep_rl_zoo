# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
#
# The functions has been modified by The Deep RL Zoo Authors
# to support PyTorch opeartion.
#
# ============================================================================
"""Tests for replay.py."""
from absl.testing import absltest
from absl.testing import parameterized
import collections
import itertools
import numpy as np
from typing import NamedTuple, Any, Mapping, Sequence, Text
import torch

from deep_rl_zoo import replay as replay_lib
from deep_rl_zoo import types as types_lib


Pair = collections.namedtuple('Pair', ['a', 'b'])
ReplayStructure = collections.namedtuple('ReplayStructure', ['value'])


class UniformReplayTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.capacity = 10
        self.replay = replay_lib.UniformReplay(
            capacity=self.capacity, structure=Pair(a=None, b=None), random_state=np.random.RandomState(1)
        )
        self.items = [
            Pair(a=1, b=2),
            Pair(a=11, b=22),
            Pair(a=111, b=222),
            Pair(a=1111, b=2222),
        ]
        for item in self.items:
            self.replay.add(item)

    def test_size(self):
        self.assertLen(self.items, self.replay.size)

    def test_capacity(self):
        self.assertEqual(self.capacity, self.replay.capacity)

    def test_sample(self):
        num_samples = 2
        samples = self.replay.sample(num_samples)
        self.assertEqual(samples.a.shape, (num_samples,))


class NStepTransitionAccumulatorTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.n = 3
        self.discount = 0.9
        self.accumulator = replay_lib.NStepTransitionAccumulator(self.n, self.discount)

        self.num_timesteps = 10
        self.states = list(range(self.num_timesteps))
        self.discounts = np.array(
            [self.discount for _ in range(self.num_timesteps)]
        )  # np.linspace(0.9, 1.0, self.num_timesteps, endpoint=False)
        self.rewards = np.linspace(-5, 5, self.num_timesteps, endpoint=False)
        self.actions = [i % 4 for i in range(self.num_timesteps)]

        self.accumulator_output = []
        for i in range(self.num_timesteps):
            timestep = types_lib.TimeStep(
                observation=self.states[i],
                reward=self.rewards[i],
                done=True if i == self.num_timesteps else False,
                first=True if i == 0 else False,
            )
            self.accumulator_output.append(list(self.accumulator.step(timestep, self.actions[i])))

    def test_no_transitions_returned_for_first_n_steps(self):
        self.assertEqual([[]] * self.n, self.accumulator_output[: self.n])
        self.assertNotEqual([], self.accumulator_output[self.n])

    def test_states_accumulation(self):
        actual_s_tm1 = [tr.s_tm1 for tr in itertools.chain(*self.accumulator_output)]
        actual_s_t = [tr.s_t for tr in itertools.chain(*self.accumulator_output)]

        expected_s_tm1 = self.states[: -self.n]
        expected_s_t = self.states[self.n :]

        np.testing.assert_array_equal(expected_s_tm1, actual_s_tm1)
        np.testing.assert_array_equal(expected_s_t, actual_s_t)

    # def test_discount_accumulation(self):
    #     expected = []
    #     for i in range(len(self.discounts) - self.n):
    #         # Offset by 1 since first discount is unused.
    #         expected.append(np.prod(self.discounts[i + 1 : i + 1 + self.n]))

    #     actual = [tr.discount_t for tr in itertools.chain(*self.accumulator_output)]

    #     np.testing.assert_allclose(expected, actual)

    def test_reward_accumulation(self):
        expected = []
        for i in range(len(self.discounts) - self.n):
            # Offset by 1 since first discount and reward is unused.
            discounts = np.concatenate([[1.0], self.discounts[i + 1 : i + 1 + self.n - 1]])
            cumulative_discounts = np.cumprod(discounts)
            rewards = self.rewards[i + 1 : i + 1 + self.n]
            expected.append(np.sum(cumulative_discounts * rewards))

        actual = [tr.r_t for tr in itertools.chain(*self.accumulator_output)]

        np.testing.assert_allclose(expected, actual)

    def test_correct_action_is_stored_in_transition(self):
        expected = self.actions[: -self.n]
        actual = [tr.a_tm1 for tr in itertools.chain(*self.accumulator_output)]
        np.testing.assert_array_equal(expected, actual)

    def test_reset(self):
        self.accumulator.reset()
        transitions = self.accumulator.step(
            timestep_t=types_lib.TimeStep(first=True, observation=-1, reward=3, done=False), a_t=1
        )
        self.assertEqual([], list(transitions))

    def test_consistent_with_transition_accumulator(self):
        n_step_transition_accumulator = replay_lib.NStepTransitionAccumulator(1, self.discount)
        transition_accumulator = replay_lib.TransitionAccumulator()

        # Add the same timesteps to both accumulators.
        for i in range(self.num_timesteps):
            timestep = types_lib.TimeStep(
                observation=self.states[i],
                reward=self.rewards[i],
                done=True if i == self.num_timesteps else False,
                first=True if i == 0 else False,
            )
            transitions = list(transition_accumulator.step(timestep, self.actions[i]))
            n_step_transitions = list(n_step_transition_accumulator.step(timestep, self.actions[i]))
            self.assertEqual(transitions, n_step_transitions)

    def test_all_remaining_transitions_yielded_when_timestep_is_last(self):
        f = 'first'
        m = 'mid'
        l = 'last'

        n = 3
        accumulator = replay_lib.NStepTransitionAccumulator(n, self.discount)
        step_types = [f, m, m, m, m, m, l, f, m, m, m, m, f, m]
        num_timesteps = len(step_types)
        states = list(range(num_timesteps))
        rewards = np.ones(num_timesteps)
        actions = list(range(num_timesteps, 0, -1))

        accumulator_output = []
        for i in range(num_timesteps):
            timestep = types_lib.TimeStep(
                observation=states[i],
                reward=rewards[i],
                first=step_types[i] == f,
                done=step_types[i] == l,
            )
            accumulator_output.append(list(accumulator.step(timestep, actions[i])))

        output_lengths = [len(output) for output in accumulator_output]
        expected_output_lengths = [0, 0, 0, 1, 1, 1, n, 0, 0, 0, 1, 1, 0, 0]
        self.assertEqual(expected_output_lengths, output_lengths)

        # Get transitions yielded at the end of an episode.
        end_index = expected_output_lengths.index(n)
        episode_end_transitions = accumulator_output[end_index]

        # Check the start and end states are correct.
        # Normal n-step transition
        self.assertEqual(episode_end_transitions[0].s_t, end_index)
        self.assertEqual(episode_end_transitions[0].s_tm1, end_index - n)
        # (n - 1)-step transition.
        self.assertEqual(episode_end_transitions[1].s_t, end_index)
        self.assertEqual(episode_end_transitions[1].s_tm1, end_index - (n - 1))
        # (n - 2)-step transition.
        self.assertEqual(episode_end_transitions[2].s_t, end_index)
        self.assertEqual(episode_end_transitions[2].s_tm1, end_index - (n - 2))

    def test_transitions_returned_if_episode_length_less_than_n(self):
        f = 'first'
        m = 'mid'
        l = 'last'

        n = 4
        accumulator = replay_lib.NStepTransitionAccumulator(n, self.discount)
        step_types = [f, m, l]
        num_timesteps = len(step_types)
        states = list(range(num_timesteps))
        rewards = np.ones(num_timesteps)
        actions = np.ones(num_timesteps)

        accumulator_output = []
        for i in range(num_timesteps):
            timestep = types_lib.TimeStep(
                observation=states[i],
                reward=rewards[i],
                first=step_types[i] == f,
                done=step_types[i] == l,
            )
            accumulator_output.append(list(accumulator.step(timestep, actions[i])))
        output_lengths = [len(output) for output in accumulator_output]
        output_states = [[(tr.s_tm1, tr.s_t) for tr in output] for output in accumulator_output]

        # Expect a 1-step transition and a 2-step transition after LAST timestep.
        expected_output_lengths = [0, 0, 2]
        expected_output_states = [[], [], [(0, 2), (1, 2)]]
        self.assertEqual(expected_output_lengths, output_lengths)
        self.assertEqual(expected_output_states, output_states)


class Transition(NamedTuple):
    s_t: torch.Tensor
    a_t: torch.Tensor
    r_t: torch.Tensor


Structure = Transition(s_t=None, a_t=None, r_t=None)


class SplitStructureTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.s_t = torch.tensor(
            [
                [[1, 1, 1], [9, 0, 9], [1, 0, 0], [0, 0, 9]],
                [[9, 9, 0], [9, 0, 0], [1, 1, 1], [9, -9, 0]],
                [[1, 1, 1], [9, 0, 9], [0, 0, 9], [1, 0, 0]],
                [[9, 9, 0], [9, 0, 0], [1, 1, 1], [9, -9, 0]],
                [[9, 9, 0], [9, 0, 0], [0, 9, 9], [9, -9, 0]],
            ],
            dtype=torch.float32,
        )
        self.a_t = torch.tensor([2, 1, 3, 0, 1], dtype=torch.int64)
        self.r_t = torch.tensor([0.5, 0.0, 0.5, 0.8, -0.1], dtype=torch.float32)

        self.transition = Transition(self.s_t, self.a_t, self.r_t)

    def test_split_structure_with_size_0(self):
        """Checks split structure."""

        prefix, suffix = replay_lib.split_structure(self.transition, Structure, 0)

        self.assertEqual(prefix, None)

        self.assertTrue(torch.equal(self.transition.s_t, suffix.s_t))
        self.assertTrue(torch.equal(self.transition.a_t, suffix.a_t))
        self.assertTrue(torch.equal(self.transition.r_t, suffix.r_t))

    @parameterized.named_parameters(('size_1', 1), ('size_3', 3), ('size_5', 5))
    def test_split_structure_with_size_n(self, split_size):
        """Checks split structure."""

        prefix, suffix = replay_lib.split_structure(self.transition, Structure, split_size)

        self.assertTrue(torch.equal(self.transition.s_t[:split_size], prefix.s_t))
        self.assertTrue(torch.equal(self.transition.a_t[:split_size], prefix.a_t))
        self.assertTrue(torch.equal(self.transition.r_t[:split_size], prefix.r_t))
        self.assertTrue(torch.equal(self.transition.s_t[split_size:], suffix.s_t))
        self.assertTrue(torch.equal(self.transition.a_t[split_size:], suffix.a_t))
        self.assertTrue(torch.equal(self.transition.r_t[split_size:], suffix.r_t))

    def test_split_structure_with_size_error(self):
        """Checks split structure."""

        with self.assertRaisesRegex(ValueError, 'Expect prefix_length to be less or equal to'):
            prefix, suffix = replay_lib.split_structure(self.transition, Structure, self.transition.s_t.shape[0] + 1)


if __name__ == '__main__':
    absltest.main()
