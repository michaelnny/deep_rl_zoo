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
"""Functions for distributed training."""
from typing import List, Tuple
import numpy as np
import torch


def get_actor_exploration_epsilon(n: int) -> List[float]:
    """Returns exploration epsilon for actor. This follows the Ape-x, R2D2 papers.

    Example for 4 actors: [0.4, 0.04715560318259695, 0.005559127278786369, 0.0006553600000000003]

    """
    assert 1 <= n
    return np.power(0.4, np.linspace(1.0, 8.0, num=n)).flatten().tolist()


def calculate_dist_priorities_from_td_error(td_error: torch.Tensor, eta: float) -> np.ndarray:
    """Calculate priorities for distributed experience replay, follows Ape-x and R2D2 papers."""
    abs_td_errors = torch.abs(td_error).detach()

    priorities = eta * torch.max(abs_td_errors, dim=0)[0] + (1 - eta) * torch.mean(abs_td_errors, dim=0)
    priorities = torch.clamp(priorities, min=0.0001, max=1000)  # Avoid NaNs
    priorities = priorities.cpu().numpy()

    return priorities


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_ngu_policy_betas(
    n: int,
    beta: float = 0.3,
):
    """Returns list of intrinsic reward scale betas, following the 'Ditributed training' in the paper at Appendix.
    This is used in Never Give Up and Agent57.

    """
    results = []
    for i in range(n):
        if i == 0:
            results.append(0.0)
        elif i == n - 1:
            results.append(beta)
        else:
            _beta_i = beta * sigmoid(10 * ((2 * i - (n - 2)) / (n - 2)))
            results.append(_beta_i)

    return results


def get_ngu_discount_gammas(n: int, gamma_max: float, gamma_min: float) -> float:
    """Return list of discount gammas, following Equation 4. in Appendix.
    This is used in Never Give Up and Agent57.

    The first one (index 0) has the highest discount,
    while the last one has lowest discount.
    """
    results = []
    for i in range(n):
        _numerator = (n - 1 - i) * np.log(1 - gamma_max) + i * np.log(1 - gamma_min)
        _gamma_i = 1 - np.exp(_numerator / (n - 1))
        results.append(_gamma_i)
    return results


def get_ngu_policy_betas_and_discounts(
    num_policies: int,
    beta: float = 0.3,
    gamma_min: float = 0.99,
    gamma_max: float = 0.997,
) -> Tuple[List[float], List[float]]:
    """Returns intrinsic reward scale beta index, reward scale beta, discount gamma, and e-greedy exploration epsilon.
    This is used in Never Give Up and Agent57.

    Specifics:
    * i = 0 is the most exploitative actor (beta=0), while i = N-1 is the most explorative actor (beta=beta_max).
    * small values of beta_i is associated with high values of gamma_i
    * high values of beta_i is associated with small values of gamm_i
    """
    beta_list = get_ngu_policy_betas(num_policies, beta)
    gamma_list = get_ngu_discount_gammas(num_policies, gamma_max, gamma_min)
    return (beta_list, gamma_list)
