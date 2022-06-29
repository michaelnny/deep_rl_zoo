#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Run end-to-end tests

# # Deep Q learning agents
python3 -m tests.dqn.run_classic_test
python3 -m tests.dqn.run_atari_test
python3 -m tests.dqn.eval_agent_test
python3 -m tests.double_dqn.run_classic_test
python3 -m tests.double_dqn.run_atari_test
python3 -m tests.double_dqn.eval_agent_test
python3 -m tests.prioritized_dqn.run_classic_test
python3 -m tests.prioritized_dqn.run_atari_test
python3 -m tests.prioritized_dqn.eval_agent_test
python3 -m tests.c51_dqn.run_classic_test
python3 -m tests.c51_dqn.run_atari_test
python3 -m tests.c51_dqn.eval_agent_test
python3 -m tests.rainbow.run_classic_test
python3 -m tests.rainbow.run_atari_test
python3 -m tests.rainbow.eval_agent_test
python3 -m tests.qr_dqn.run_classic_test
python3 -m tests.qr_dqn.run_atari_test
python3 -m tests.qr_dqn.eval_agent_test
python3 -m tests.iqn.run_classic_test
python3 -m tests.iqn.run_atari_test
python3 -m tests.iqn.eval_agent_test
python3 -m tests.drqn.run_classic_test
python3 -m tests.drqn.run_atari_test
python3 -m tests.drqn.eval_agent_test
python3 -m tests.r2d2.run_classic_test
python3 -m tests.r2d2.run_atari_test
python3 -m tests.r2d2.eval_agent_test
python3 -m tests.ngu.run_classic_test
python3 -m tests.ngu.run_atari_test
python3 -m tests.ngu.eval_agent_test
python3 -m tests.agent57.run_classic_test
python3 -m tests.agent57.run_atari_test
python3 -m tests.agent57.eval_agent_test

# Policy gradient agents
python3 -m tests.reinforce.run_classic_test
python3 -m tests.reinforce.run_atari_test
python3 -m tests.reinforce.eval_agent_test
python3 -m tests.reinforce_baseline.run_classic_test
python3 -m tests.reinforce_baseline.run_atari_test
python3 -m tests.reinforce_baseline.eval_agent_test
python3 -m tests.actor_critic.run_classic_test
python3 -m tests.actor_critic.run_atari_test
python3 -m tests.actor_critic.eval_agent_test
python3 -m tests.a2c.run_classic_test
python3 -m tests.a2c.run_atari_test
python3 -m tests.a2c.eval_agent_test
python3 -m tests.a2c.run_classic_grad_test
python3 -m tests.a2c.run_atari_grad_test
python3 -m tests.ppo.run_classic_test
python3 -m tests.ppo.run_atari_test
python3 -m tests.ppo.eval_agent_test
python3 -m tests.ppo_icm.run_classic_test
python3 -m tests.ppo_icm.run_atari_test
python3 -m tests.ppo_icm.eval_agent_test
python3 -m tests.ppo_rnd.run_classic_test
python3 -m tests.ppo_rnd.run_atari_test
python3 -m tests.ppo_rnd.eval_agent_test
python3 -m tests.sac.run_classic_test
python3 -m tests.sac.run_atari_test
python3 -m tests.sac.eval_agent_test
python3 -m tests.impala.run_classic_test
python3 -m tests.impala.run_atari_test
python3 -m tests.impala.eval_agent_test
