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
"""Tests for distributed.py."""

from absl.testing import absltest
from deep_rl_zoo import distributed


class GetActorExplorationEpsilonTest(absltest.TestCase):
    def test_get_actor_epsilon(self):
        epsilons = distributed.get_actor_exploration_epsilon(10)

        # Training epsilons.
        self.assertAlmostEqual(epsilons[0], 0.4)
        self.assertAlmostEqual(epsilons[9], 0.4**8)


if __name__ == '__main__':
    absltest.main()
