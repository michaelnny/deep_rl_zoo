#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Run all agents on Pong

# python3 -m deep_rl_zoo.dqn.run_atari \
# --environment_name=Pong \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0 \

# python3 -m deep_rl_zoo.double_dqn.run_atari \
# --environment_name=Pong \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0 \

python3 -m deep_rl_zoo.prioritized_dqn.run_atari \
--environment_name=Pong \
--num_iterations=1 \
--num_train_steps=1000000 \
--num_eval_steps=0 \

python3 -m deep_rl_zoo.c51_dqn.run_atari \
--environment_name=Pong \
--num_iterations=1 \
--num_train_steps=1000000 \
--num_eval_steps=0 \

python3 -m deep_rl_zoo.rainbow.run_atari \
--environment_name=Pong \
--num_iterations=1 \
--num_train_steps=1000000 \
--num_eval_steps=0 \

# python3 -m deep_rl_zoo.qr_dqn.run_atari \
# --environment_name=Pong \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0 \

python3 -m deep_rl_zoo.iqn.run_atari \
--environment_name=Pong \
--num_iterations=1 \
--num_train_steps=1000000 \
--num_eval_steps=0 \

# python3 -m deep_rl_zoo.drqn.run_atari \
# --environment_name=Pong \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0

python3 -m deep_rl_zoo.r2d2.run_atari \
--environment_name=Pong \
--num_actors=8 \
--num_iterations=1 \
--num_train_steps=1000000 \
--num_eval_steps=0

# python3 -m deep_rl_zoo.ngu.run_atari \
# --environment_name=Pong \
# --num_actors=8 \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0

# python3 -m deep_rl_zoo.agent57.run_atari \
# --environment_name=Pong \
# --num_actors=8 \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0

# python3 -m deep_rl_zoo.reinforce.run_atari \
# --environment_name=Pong \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0

# python3 -m deep_rl_zoo.reinforce_baseline.run_atari \
# --environment_name=Pong \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0

# python3 -m deep_rl_zoo.actor_critic.run_atari \
# --environment_name=Pong \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0

# python3 -m deep_rl_zoo.a2c.run_atari \
# --environment_name=Pong \
# --num_actors=8 \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0

# python3 -m deep_rl_zoo.a2c.run_atari_grad \
# --environment_name=Pong \
# --num_actors=8 \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0

# python3 -m deep_rl_zoo.ppo.run_atari \
# --environment_name=Pong \
# --num_actors=8 \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0

# python3 -m deep_rl_zoo.ppo_icm.run_atari \
# --environment_name=Pong \
# --num_actors=8 \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0

# python3 -m deep_rl_zoo.ppo_rnd.run_atari \
# --environment_name=Pong \
# --num_actors=8 \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0

# python3 -m deep_rl_zoo.sac.run_atari \
# --environment_name=Pong \
# --num_actors=8 \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0

# python3 -m deep_rl_zoo.impala.run_atari \
# --environment_name=Pong \
# --num_actors=8 \
# --num_iterations=1 \
# --num_train_steps=1000000 \
# --num_eval_steps=0
