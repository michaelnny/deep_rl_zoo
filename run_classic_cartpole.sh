#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Run all agents on CartPole-v1

python3 -m deep_rl_zoo.dqn.run_classic \
--environment_name=CartPole-v1 \
--min_replay_size=10000 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.double_dqn.run_classic \
--environment_name=CartPole-v1 \
--min_replay_size=10000 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.prioritized_dqn.run_classic \
--environment_name=CartPole-v1 \
--min_replay_size=10000 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.c51_dqn.run_classic \
--environment_name=CartPole-v1 \
--min_replay_size=10000 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.rainbow.run_classic \
--environment_name=CartPole-v1 \
--min_replay_size=10000 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.qr_dqn.run_classic \
--environment_name=CartPole-v1 \
--min_replay_size=10000 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.iqn.run_classic \
--environment_name=CartPole-v1 \
--min_replay_size=10000 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.drqn.run_classic \
--environment_name=CartPole-v1 \
--min_replay_size=10000 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.r2d2.run_classic \
--environment_name=CartPole-v1 \
--min_replay_size=10000 \
--num_actors=2 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.ngu.run_classic \
--environment_name=CartPole-v1 \
--min_replay_size=10000 \
--num_actors=2 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.agent57.run_classic \
--environment_name=CartPole-v1 \
--min_replay_size=10000 \
--num_actors=2 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.reinforce.run_classic \
--environment_name=CartPole-v1 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.reinforce_baseline.run_classic \
--environment_name=CartPole-v1 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.actor_critic.run_classic \
--environment_name=CartPole-v1 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.a2c.run_classic \
--environment_name=CartPole-v1 \
--num_actors=2 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.a2c.run_classic_grad \
--environment_name=CartPole-v1 \
--num_actors=2 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.ppo.run_classic \
--environment_name=CartPole-v1 \
--num_actors=2 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.ppo_icm.run_classic \
--environment_name=CartPole-v1 \
--num_actors=2 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.ppo_rnd.run_classic \
--environment_name=CartPole-v1 \
--num_actors=2 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.sac.run_classic \
--environment_name=CartPole-v1 \
--num_actors=2 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''

python3 -m deep_rl_zoo.impala.run_classic \
--environment_name=CartPole-v1 \
--num_actors=2 \
--num_iterations=1 \
--num_train_steps=200000 \
--num_eval_steps=0 \
--results_csv_path='' \
--checkpoint_path=''
