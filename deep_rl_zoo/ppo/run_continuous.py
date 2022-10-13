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
"""

From the paper "Proximal Policy Optimization Algorithms"
https://arxiv.org/abs/1707.06347.

"""
from absl import app
from absl import flags
from absl import logging
import multiprocessing
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.policy import GaussianActorMlpNet, GaussianCriticMlpNet
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo.schedule import LinearSchedule
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo.ppo import agent
from deep_rl_zoo import greedy_actors

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'Humanoid-v4',
    'Classic continuous control task name, like Hopper-v4, HalfCheetah-v4, Humanoid-v4, Swimmer-v4, Walker2d-v4.',
)
flags.DEFINE_integer('num_actors', 8, 'Number of worker processes to use.')
flags.DEFINE_bool('clip_grad', True, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_float('baseline_learning_rate', 0.0001, 'Learning rate for critic.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('gae_lambda', 0.95, 'Lambda for the GAE general advantage estimator.')
flags.DEFINE_float('entropy_coef', 0.01, 'Coefficient for the entropy loss.')
flags.DEFINE_float('clip_epsilon_begin_value', 0.2, 'PPO clip epsilon begin value.')
flags.DEFINE_float('clip_epsilon_end_value', 0.0, 'PPO clip epsilon final value.')
flags.DEFINE_integer('hidden_size', 64, 'Number of units in the hidden layer.')
flags.DEFINE_integer('batch_size', 64, 'Learner batch size for learning.')
flags.DEFINE_integer('unroll_length', 2048, 'Collect N transitions (cross episodes) before send to learner, per actor.')
flags.DEFINE_integer('update_k', 10, 'Run update k times when do learning.')
flags.DEFINE_integer('num_iterations', 1, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_frames', int(5e6), 'Number of training env steps to run per iteration, per actor.')
flags.DEFINE_integer('num_eval_frames', int(1e5), 'Number of evaluation env steps to run per iteration.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_integer(
    'debug_screenshots_frequency',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', 'logs/ppo_classic_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Path for checkpoint directory.')


def main(argv):
    """Trains PPO agent on continuous action control tasks."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs PPO agent on {runtime_device}')
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Create environment.
    def environment_builder():
        return gym_env.create_continuous_environment(
            env_name=FLAGS.environment_name,
            seed=random_state.randint(1, 2**32),
        )

    eval_env = environment_builder()

    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', action_dim)
    logging.info('Observation spec: %s', action_dim)

    # Create policy network, master will optimize this network
    policy_network = GaussianActorMlpNet(input_shape=state_dim, num_actions=action_dim, hidden_size=FLAGS.hidden_size)
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=FLAGS.learning_rate)

    critic_network = GaussianCriticMlpNet(input_shape=state_dim, hidden_size=FLAGS.hidden_size)
    critic_optimizer = torch.optim.Adam(critic_network.parameters(), lr=FLAGS.baseline_learning_rate)

    # The 'old' policy for actors to act
    old_policy_network = GaussianActorMlpNet(input_shape=state_dim, num_actions=action_dim, hidden_size=FLAGS.hidden_size)
    old_policy_network.share_memory()

    # Create queue shared between actors and learner
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors)

    clip_epsilon_scheduler = LinearSchedule(
        begin_t=0,
        end_t=int(
            (FLAGS.num_iterations * int(FLAGS.num_train_frames * FLAGS.num_actors)) / FLAGS.unroll_length
        ),  # Learner step_t is often faster than worker
        begin_value=FLAGS.clip_epsilon_begin_value,
        end_value=FLAGS.clip_epsilon_end_value,
    )

    # Create PPO learner agent instance
    learner_agent = agent.GaussianLearner(
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        old_policy_network=old_policy_network,
        critic_network=critic_network,
        critic_optimizer=critic_optimizer,
        clip_epsilon=clip_epsilon_scheduler,
        discount=FLAGS.discount,
        gae_lambda=FLAGS.gae_lambda,
        total_unroll_length=int(FLAGS.unroll_length * FLAGS.num_actors),
        batch_size=FLAGS.batch_size,
        update_k=FLAGS.update_k,
        entropy_coef=FLAGS.entropy_coef,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
    )

    # Create actor environments, runtime devices, and actor instances.
    actor_envs = [environment_builder() for _ in range(FLAGS.num_actors)]

    # TODO map to dedicated device if have multiple GPUs
    actor_devices = [runtime_device] * FLAGS.num_actors

    actors = [
        agent.GaussianActor(
            rank=i,
            data_queue=data_queue,
            policy_network=old_policy_network,
            unroll_length=FLAGS.unroll_length,
            device=actor_devices[i],
        )
        for i in range(FLAGS.num_actors)
    ]

    # Create evaluation agent instance
    eval_agent = greedy_actors.GaussianPolicyGreedyActor(
        network=policy_network,
        device=runtime_device,
        name='PPO-greedy',
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(environment_name=FLAGS.environment_name, agent_name='PPO', save_dir=FLAGS.checkpoint_dir)
    checkpoint.register_pair(('policy_network', policy_network))

    # Run parallel traning N iterations.
    main_loop.run_parallel_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_frames=FLAGS.num_train_frames,
        num_eval_frames=FLAGS.num_eval_frames,
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
        debug_screenshots_frequency=FLAGS.debug_screenshots_frequency,
    )


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)
