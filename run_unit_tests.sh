#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Run unit tests

python3 -m unit_tests.base_test
python3 -m unit_tests.curiosity_test
python3 -m unit_tests.distributed_test
python3 -m unit_tests.distributions_test
python3 -m unit_tests.nonlinear_bellman_test
python3 -m unit_tests.multistep_test
python3 -m unit_tests.policy_gradient_test
python3 -m unit_tests.transforms_test
python3 -m unit_tests.value_learning_test
python3 -m unit_tests.vtrace_test

python3 -m unit_tests.gym_env_test
python3 -m unit_tests.replay_test
python3 -m unit_tests.checkpoint_test
python3 -m unit_tests.schedule_test
