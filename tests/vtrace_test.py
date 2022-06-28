# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
# Copyright 2018 The trfl Authors. All Rights Reserved.
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
"""Tests for vtrace.py."""
from absl.testing import absltest
from absl.testing import parameterized
import torch
import numpy as np

from deep_rl_zoo import vtrace


def _shaped_arange(*shape):
    """Runs np.arange, converts to float and reshapes."""
    return torch.tensor(np.arange(np.prod(shape), dtype=np.float32).reshape(*shape))


def _ground_truth_calculation(
    discounts,
    behavior_action_log_probs,
    target_action_log_probs,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold,
    clip_pg_rho_threshold,
):
    """Calculates the ground truth for V-trace in Python/Numpy."""
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vs = []
    seq_len = len(discounts)
    rhos = torch.exp(log_rhos)
    cs = torch.minimum(rhos, torch.tensor(1.0))
    clipped_rhos = rhos
    if clip_rho_threshold:
        clipped_rhos = torch.minimum(rhos, torch.tensor(clip_rho_threshold))
    clipped_pg_rhos = rhos
    if clip_pg_rho_threshold:
        clipped_pg_rhos = torch.minimum(rhos, torch.tensor(clip_pg_rho_threshold))

    # This is a very inefficient way to calculate the V-trace ground truth.
    # We calculate it this way because it is close to the mathematical notation of
    # V-trace.
    # v_s = V(x_s)
    #       + \sum^{T-1}_{t=s} \gamma^{t-s}
    #         * \prod_{i=s}^{t-1} c_i
    #         * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
    # Note that when we take the product over c_i, we write `s:t` as the notation
    # of the paper is inclusive of the `t-1`, but Python is exclusive.
    # Also note that np.prod([]) == 1.
    values_t_plus_1 = torch.concat([values, bootstrap_value[None, :]], dim=0)
    for s in range(seq_len):
        v_s = torch.clone(values[s])  # Very important copy.
        for t in range(s, seq_len):
            v_s += (
                torch.prod(discounts[s:t], dim=0)
                * torch.prod(cs[s:t], dim=0)
                * clipped_rhos[t]
                * (rewards[t] + discounts[t] * values_t_plus_1[t + 1] - values[t])
            )
        vs.append(v_s)
    vs = torch.stack(vs, dim=0)
    pg_advantages = clipped_pg_rhos * (rewards + discounts * torch.concat([vs[1:], bootstrap_value[None, :]], dim=0) - values)

    return vtrace.VTraceReturns(vs=vs, pg_advantages=pg_advantages)


class VtraceTest(parameterized.TestCase):
    def test_vtrace(self):
        """Tests V-trace against ground truth data calculated in python."""
        batch_size = 5
        seq_len = 5

        # Create log_rhos such that rho will span from near-zero to above the
        # clipping thresholds. In particular, calculate log_rhos in [-2.5, 2.5),
        # so that rho is in approx [0.08, 12.2).
        log_rhos = _shaped_arange(seq_len, batch_size) / (batch_size * seq_len)
        log_rhos = 5 * (log_rhos - 0.5)  # [0.0, 1.0) -> [-2.5, 2.5).
        values = {
            'behavior_action_log_probs': torch.zeros_like(log_rhos),
            'target_action_log_probs': log_rhos,
            # T, B where B_i: [0.9 / (i+1)] * T
            'discounts': torch.tensor(
                np.array([[0.9 / (b + 1) for b in range(batch_size)] for _ in range(seq_len)]), dtype=torch.float32
            ),
            'rewards': _shaped_arange(seq_len, batch_size),
            'values': _shaped_arange(seq_len, batch_size) / batch_size,
            'bootstrap_value': _shaped_arange(batch_size) + 1.0,
            'clip_rho_threshold': 3.7,
            'clip_pg_rho_threshold': 2.2,
        }

        output = vtrace.from_importance_weights(**values)
        ground_truth_v = _ground_truth_calculation(**values)

        torch.testing.assert_allclose(output.vs, ground_truth_v.vs)
        torch.testing.assert_allclose(output.pg_advantages, ground_truth_v.pg_advantages)


if __name__ == '__main__':
    absltest.main()
