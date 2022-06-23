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
"""A PPO-ICM agent training on classic games like CartPole, MountainCar, or LunarLander.

From the paper "Curiosity-driven Exploration by Self-supervised Prediction"
https://arxiv.org/abs/1705.05363

From the paper "Proximal Policy Optimization Algorithms"
https://arxiv.org/abs/1707.06347.

To solve MountainCar:
python3 -m deep_rl.ppo_icm.run_classic --environment_name=MountainCar-v0 --num_train_steps=1000000 --num_iterations=1 --intrinsic_eta=20
"""

from absl import app
from absl import flags
from absl import logging
import os

os.environ['OMP_NUM_THREADS'] = '1'

import multiprocessing
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.policy import ActorCriticMlpNet
from deep_rl_zoo.networks.curiosity import IcmMlpNet
from deep_rl_zoo.ppo_icm import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo.schedule import LinearSchedule
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'CartPole-v1', 'Classic game name like CartPole-v1, MountainCar-v0, LunarLander-v2.')
flags.DEFINE_integer('num_actors', 8, 'Number of worker processes to use.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 40.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('entropy_coef', 0.001, 'Coefficient for the entropy loss.')
flags.DEFINE_float('baseline_coef', 0.5, 'Coefficient for the state-value loss.')
flags.DEFINE_float('clip_epsilon_begin_value', 0.2, 'PPO clip epsilon begin value.')
flags.DEFINE_float('clip_epsilon_end_value', 0.0, 'PPO clip epsilon final value.')

flags.DEFINE_float('icm_learning_rate', 0.0005, 'Learning rate for ICM module.')
flags.DEFINE_float('intrinsic_eta', 0.1, 'Scaling facotr for intrinsic reward when calculate using equaltion 6.')
flags.DEFINE_float('icm_beta', 0.2, 'Weights inverse model loss against the forward model loss.')
flags.DEFINE_float('policy_lambda', 1.0, 'Weights policy loss against the importance of learning the intrinsic reward.')
flags.DEFINE_float('extrinsic_reward_coef', 1.0, 'Weight of extrinsic reward from environment.')
flags.DEFINE_float('intrinsic_reward_coef', 1.0, 'Weight of intrinsic reward from ICM module.')

flags.DEFINE_integer('n_step', 2, 'TD n-step bootstrap.')
flags.DEFINE_integer('batch_size', 64, 'Learner batch size for learning.')
flags.DEFINE_integer('unroll_length', 128, 'Actor unroll length.')
flags.DEFINE_integer('update_k', 4, 'Run update k times when do learning.')
flags.DEFINE_integer('num_iterations', 2, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(2e5), 'Number of training steps per iteration.')
flags.DEFINE_integer('num_eval_steps', int(1e5), 'Number of evaluation steps per iteration.')
flags.DEFINE_integer('max_episode_steps', 0, 'Maximum steps per episode. 0 means no limit.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', 'logs/ppo_icm_classic_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_path', 'checkpoints/ppo_icm', 'Path for checkpoint directory.')


def main(argv):
    """Trains PPO-ICM agent on classic games."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Listen to signals to exit process.
    main_loop.handle_exit_signal()

    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Create environment.
    def environment_builder(random_int=0):
        return gym_env.create_classic_environment(env_name=FLAGS.environment_name, seed=FLAGS.seed + int(random_int))

    eval_env = environment_builder()

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', eval_env.action_space.n)
    logging.info('Observation spec: %s', eval_env.observation_space.shape)

    input_shape = eval_env.observation_space.shape[0]
    num_actions = eval_env.action_space.n

    # Test environment and state shape.
    obs = eval_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (input_shape,)

    # Create policy network, master will optimize this network
    policy_network = ActorCriticMlpNet(input_shape=input_shape, num_actions=num_actions)
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=FLAGS.learning_rate)

    # The 'old' policy for actors to act
    old_policy_network = ActorCriticMlpNet(input_shape=input_shape, num_actions=num_actions)
    old_policy_network.share_memory()

    # ICM module
    icm_network = IcmMlpNet(input_shape=input_shape, num_actions=num_actions)
    icm_optimizer = torch.optim.Adam(icm_network.parameters(), lr=FLAGS.icm_learning_rate)

    # Test network output.
    s = torch.from_numpy(obs[None, ...]).float()
    network_output = policy_network(s)
    pi_logits = network_output.pi_logits
    baseline = network_output.baseline
    assert pi_logits.shape == (1, num_actions)
    assert baseline.shape == (1, 1)

    # Create queue shared between actors and learner and log queue
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors)

    clip_epsilon_scheduler = LinearSchedule(
        begin_t=0,
        end_t=(FLAGS.num_iterations * int(FLAGS.num_train_steps * FLAGS.num_actors)),  # Learner step_t is fater than worker
        begin_value=FLAGS.clip_epsilon_begin_value,
        end_value=FLAGS.clip_epsilon_end_value,
    )

    # Create PPO-ICM learner agent instance
    learner_agent = agent.Learner(
        data_queue=data_queue,
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        old_policy_network=old_policy_network,
        icm_network=icm_network,
        icm_optimizer=icm_optimizer,
        clip_epsilon=clip_epsilon_scheduler,
        discount=FLAGS.discount,
        n_step=FLAGS.n_step,
        batch_size=FLAGS.batch_size,
        update_k=FLAGS.update_k,
        num_actors=FLAGS.num_actors,
        unroll_length=FLAGS.unroll_length,
        entropy_coef=FLAGS.entropy_coef,
        baseline_coef=FLAGS.baseline_coef,
        intrinsic_eta=FLAGS.intrinsic_eta,
        icm_beta=FLAGS.icm_beta,
        policy_lambda=FLAGS.policy_lambda,
        extrinsic_reward_coef=FLAGS.extrinsic_reward_coef,
        intrinsic_reward_coef=FLAGS.intrinsic_reward_coef,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
    )

    # Create actor environments, runtime devices, and actor instances.
    actor_envs = [environment_builder(i) for i in range(FLAGS.num_actors)]

    # TODO map to dedicated device if have multiple GPUs
    actor_devices = [runtime_device] * FLAGS.num_actors

    actors = [
        agent.Actor(
            rank=i,
            data_queue=data_queue,
            policy_network=old_policy_network,
            transition_accumulator=replay_lib.PgNStepTransitionAccumulator(n=FLAGS.n_step, discount=FLAGS.discount),
            unroll_length=FLAGS.unroll_length,
            device=actor_devices[i],
        )
        for i in range(FLAGS.num_actors)
    ]

    # Create evaluation agent instance
    eval_agent = greedy_actors.PolicyGreedyActor(
        network=policy_network,
        device=runtime_device,
        name='PPO-ICM-greedy',
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(FLAGS.checkpoint_path)
    state = checkpoint.state
    state.environment_name = FLAGS.environment_name
    state.iteration = 0
    state.policy_network = policy_network
    state.icm_network = icm_network

    # Run parallel traning N iterations.
    main_loop.run_parallel_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        network=policy_network,
        learner_agent=learner_agent,
        eval_agent=eval_agent,
        eval_env=eval_env,
        actors=actors,
        actor_envs=actor_envs,
        data_queue=data_queue,
        checkpoint=checkpoint,
        csv_file=FLAGS.results_csv_path,
        tensorboard=FLAGS.tensorboard,
        tag=FLAGS.tag,
    )


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)
