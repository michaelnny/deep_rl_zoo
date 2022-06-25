#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Run all agents on Pong

ENV_NAME="Pong"

python3 -m deep_rl_zoo.dqn.run_atari \
--environment_name="$ENV_NAME" \
--num_iterations=1 \
--num_train_steps=500000 \
--num_eval_steps=0 \
--exploration_epsilon_decay_step=500000 \
--min_replay_size=10000 \
--replay_capacity=100000 \
--target_network_update_frequency=1000 \
--tag=replay[10k-100k]-target-q[1k]


python3 -m deep_rl_zoo.dqn.run_atari \
--environment_name="$ENV_NAME" \
--num_iterations=1 \
--num_train_steps=500000 \
--num_eval_steps=0 \
--exploration_epsilon_decay_step=500000 \
--min_replay_size=10000 \
--replay_capacity=100000 \
--target_network_update_frequency=2000 \
--tag=replay[10k-100k]-target-q[2k]


python3 -m deep_rl_zoo.dqn.run_atari \
--environment_name="$ENV_NAME" \
--num_iterations=1 \
--num_train_steps=500000 \
--num_eval_steps=0 \
--exploration_epsilon_decay_step=500000 \
--min_replay_size=50000 \
--replay_capacity=100000 \
--target_network_update_frequency=1000 \
--tag=replay[50k-100k]-target-q[1k]


python3 -m deep_rl_zoo.dqn.run_atari \
--environment_name="$ENV_NAME" \
--num_iterations=1 \
--num_train_steps=500000 \
--num_eval_steps=0 \
--exploration_epsilon_decay_step=500000 \
--min_replay_size=50000 \
--replay_capacity=1000000 \
--target_network_update_frequency=10000 \
--tag=replay[50k-1000k]-target-q[10k]
