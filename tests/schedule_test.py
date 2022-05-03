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
# ============================================================================
"""Tests for schedule.py."""
from absl import flags
from absl.testing import flagsaver
from absl.testing import absltest

from deep_rl_zoo import schedule as schedule_lib

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_path', '', '')
flags.DEFINE_string('environment_name', '', '')


class LinearScheduleTest(absltest.TestCase):
    def test_descent(self):
        """Checks basic linear decay schedule_lib."""
        schedule = schedule_lib.LinearSchedule(begin_t=5, decay_steps=7, begin_value=1.0, end_value=0.3)
        for step in range(20):
            val = schedule(step)
            if step <= 5:
                self.assertEqual(1.0, val)
            elif step >= 12:
                self.assertEqual(0.3, val)
            else:
                self.assertAlmostEqual(1.0 - ((step - 5) / 7) * 0.7, val)

    def test_ascent(self):
        """Checks basic linear ascent schedule_lib."""
        schedule = schedule_lib.LinearSchedule(begin_t=5, end_t=12, begin_value=-0.4, end_value=0.4)
        for step in range(20):
            val = schedule(step)
            if step <= 5:
                self.assertEqual(-0.4, val)
            elif step >= 12:
                self.assertEqual(0.4, val)
            else:
                self.assertAlmostEqual(-0.4 + ((step - 5) / 7) * 0.8, val)

    def test_constant(self):
        """Checks constant schedule_lib."""
        schedule = schedule_lib.LinearSchedule(begin_t=5, decay_steps=7, begin_value=0.5, end_value=0.5)
        for step in range(20):
            val = schedule(step)
            self.assertAlmostEqual(0.5, val)

    def test_error_wrong_end_args(self):
        """Checks error in case none or both of end_t, decay_steps are given."""
        with self.assertRaisesRegex(ValueError, 'Exactly one of'):
            _ = schedule_lib.LinearSchedule(begin_value=0.0, end_value=1.0, begin_t=5)
        with self.assertRaisesRegex(ValueError, 'Exactly one of'):
            _ = schedule_lib.LinearSchedule(begin_value=0.0, end_value=1.0, begin_t=5, end_t=12, decay_steps=7)


if __name__ == '__main__':
    absltest.main()
