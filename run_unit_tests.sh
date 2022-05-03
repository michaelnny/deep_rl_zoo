#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Run unit tests

python3 -m tests.base_test
python3 -m tests.curiosity_test
python3 -m tests.distributed_test
python3 -m tests.distributions_test
python3 -m tests.nonlinear_bellman_test
python3 -m tests.multistep_test
python3 -m tests.policy_gradient_test
python3 -m tests.transforms_test
python3 -m tests.value_learning_test
python3 -m tests.vtrace_test

python3 -m tests.gym_env_test
python3 -m tests.replay_test
python3 -m tests.checkpoint_test
python3 -m tests.schedule_test
python3 -m tests.normalizer_test
