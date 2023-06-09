#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Run end-to-end tests

# Value-based RL agents
python3 -m unit_tests.dqn.run_classic_test
python3 -m unit_tests.dqn.run_atari_test
python3 -m unit_tests.dqn.eval_agent_test
python3 -m unit_tests.double_dqn.run_classic_test
python3 -m unit_tests.double_dqn.run_atari_test
python3 -m unit_tests.double_dqn.eval_agent_test
python3 -m unit_tests.prioritized_dqn.run_classic_test
python3 -m unit_tests.prioritized_dqn.run_atari_test
python3 -m unit_tests.prioritized_dqn.eval_agent_test
python3 -m unit_tests.c51_dqn.run_classic_test
python3 -m unit_tests.c51_dqn.run_atari_test
python3 -m unit_tests.c51_dqn.eval_agent_test
python3 -m unit_tests.rainbow.run_classic_test
python3 -m unit_tests.rainbow.run_atari_test
python3 -m unit_tests.rainbow.eval_agent_test
python3 -m unit_tests.qr_dqn.run_classic_test
python3 -m unit_tests.qr_dqn.run_atari_test
python3 -m unit_tests.qr_dqn.eval_agent_test
python3 -m unit_tests.iqn.run_classic_test
python3 -m unit_tests.iqn.run_atari_test
python3 -m unit_tests.iqn.eval_agent_test
python3 -m unit_tests.drqn.run_classic_test
python3 -m unit_tests.drqn.run_atari_test
python3 -m unit_tests.drqn.eval_agent_test
python3 -m unit_tests.r2d2.run_classic_test
python3 -m unit_tests.r2d2.run_atari_test
python3 -m unit_tests.r2d2.eval_agent_test
python3 -m unit_tests.ngu.run_atari_test
python3 -m unit_tests.ngu.eval_agent_test
python3 -m unit_tests.agent57.run_atari_test
python3 -m unit_tests.agent57.eval_agent_test

# Policy-based RL agents
python3 -m unit_tests.reinforce.run_classic_test
python3 -m unit_tests.reinforce.run_atari_test
python3 -m unit_tests.reinforce.eval_agent_test
python3 -m unit_tests.reinforce_baseline.run_classic_test
python3 -m unit_tests.reinforce_baseline.run_atari_test
python3 -m unit_tests.reinforce_baseline.eval_agent_test
python3 -m unit_tests.actor_critic.run_classic_test
python3 -m unit_tests.actor_critic.run_atari_test
python3 -m unit_tests.actor_critic.eval_agent_test
python3 -m unit_tests.a2c.run_classic_test
python3 -m unit_tests.a2c.run_atari_test
python3 -m unit_tests.a2c.eval_agent_test
python3 -m unit_tests.a2c.run_classic_grad_test
python3 -m unit_tests.a2c.run_atari_grad_test
python3 -m unit_tests.sac.run_classic_test
python3 -m unit_tests.sac.run_atari_test
python3 -m unit_tests.sac.eval_agent_test
python3 -m unit_tests.impala.run_classic_test
python3 -m unit_tests.impala.run_atari_test
python3 -m unit_tests.impala.eval_agent_test
python3 -m unit_tests.ppo.run_classic_test
python3 -m unit_tests.ppo.run_atari_test
python3 -m unit_tests.ppo.run_continuous_test
python3 -m unit_tests.ppo.eval_agent_test
python3 -m unit_tests.ppo_icm.run_classic_test
python3 -m unit_tests.ppo_icm.run_atari_test
python3 -m unit_tests.ppo_icm.eval_agent_test
python3 -m unit_tests.ppo_rnd.run_atari_test
python3 -m unit_tests.ppo_rnd.eval_agent_test
