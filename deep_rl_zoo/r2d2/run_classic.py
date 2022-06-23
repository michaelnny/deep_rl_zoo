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
"""A R2D2 agent training on classic games like CartPole, MountainCar, or LunarLander.

From the paper "Recurrent Experience Replay in Distributed Reinforcement Learning"
https://openreview.net/pdf?id=r1lyTjAqYX.

For LunarLander:
python3 -m deep_rl.r2d2.run_classic --environment_name=LunarLander-v2 --unroll_length=40 --burn_in=20

Does not work at all on MountainCar.
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
from deep_rl_zoo.networks.dqn import R2d2DqnMlpNet, RnnDqnNetworkInputs
from deep_rl_zoo.r2d2 import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'CartPole-v1', 'Classic game name like CartPole-v1, LunarLander-v2.')
flags.DEFINE_integer('num_actors', 16, 'Number of actor processes to use.')
flags.DEFINE_integer('replay_capacity', 5000, 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 100, 'Minimum replay size before learning starts.')
flags.DEFINE_bool('clip_grad', True, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 40.0, 'Max gradients norm when do gradients clip.')

flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate for adam.')
flags.DEFINE_float('adam_eps', 0.001, 'Epsilon for adam.')
flags.DEFINE_float('discount', 0.997, 'Discount rate.')
flags.DEFINE_integer('unroll_length', 15, 'Sequence of transitions to unroll before add to replay.')
flags.DEFINE_integer(
    'burn_in',
    5,
    'Sequence of transitions used to pass RNN before actual learning.'
    'The effective length of unrolls will be burn_in + unroll_length, '
    'two consecutive unrolls will overlap on burn_in steps.',
)
flags.DEFINE_integer('batch_size', 8, 'Batch size for learning.')

flags.DEFINE_float('priority_exponent', 0.9, 'Priotiry exponent used in prioritized replay.')
flags.DEFINE_float('importance_sampling_exponent', 0.6, 'Importance sampling exponent value.')

flags.DEFINE_float('priority_eta', 0.9, 'Priotiry eta to mix the max and mean absolute TD errors.')
flags.DEFINE_float('rescale_epsilon', 0.001, 'Epsilon used in the invertible value rescaling for n-step targets.')
flags.DEFINE_integer('n_step', 4, 'TD n-step bootstrap.')

flags.DEFINE_integer('num_iterations', 2, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(2e5), 'Number of training steps per iteration.')
flags.DEFINE_integer('num_eval_steps', int(1e5), 'Number of evaluation steps per iteration.')
flags.DEFINE_integer('max_episode_steps', 0, 'Maximum steps per episode. 0 means no limit.')
flags.DEFINE_integer(
    'target_network_update_frequency',
    100,
    'Number of learner online Q network updates before update target Q networks.',
)
flags.DEFINE_integer('actor_update_frequency', 100, 'The frequency (measured in actor steps) to update actor local Q network.')
flags.DEFINE_float('eval_exploration_epsilon', 0.001, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', 'logs/r2d2_classic_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_path', 'checkpoints/r2d2', 'Path for checkpoint directory.')


def main(argv):
    """Trains R2D2 agent on classic games."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member

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

    # Create network for learner to optimize, actor will use the same network with share memory.
    network = R2d2DqnMlpNet(input_shape=input_shape, num_actions=num_actions)
    network.share_memory()
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate, eps=FLAGS.adam_eps)

    # Test network output.
    x = RnnDqnNetworkInputs(
        s_t=torch.from_numpy(obs[None, None, ...]).float(),
        a_tm1=torch.zeros(1, 1).long(),
        r_t=torch.zeros(1, 1).float(),
        hidden_s=network.get_initial_hidden_state(1),
    )
    network_output = network(x)
    q_values = network_output.q_values
    q_hidden_s = network_output.hidden_s

    assert q_values.shape == (1, 1, num_actions)
    assert len(q_hidden_s) == 2

    # Create prioritized transition replay, no importance_sampling_exponent decay
    importance_sampling_exponent = FLAGS.importance_sampling_exponent

    def importance_sampling_exponent_schedule(x):
        return importance_sampling_exponent

    replay = replay_lib.PrioritizedReplay(
        capacity=FLAGS.replay_capacity,
        structure=agent.TransitionStructure,
        priority_exponent=FLAGS.priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        time_major=True,
    )

    # Create queue shared between actors and learner
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors)

    # Create R2D2 learner instance
    learner_agent = agent.Learner(
        data_queue=data_queue,
        network=network,
        optimizer=optimizer,
        replay=replay,
        min_replay_size=FLAGS.min_replay_size,
        target_network_update_frequency=FLAGS.target_network_update_frequency,
        discount=FLAGS.discount,
        burn_in=FLAGS.burn_in,
        priority_eta=FLAGS.priority_eta,
        rescale_epsilon=FLAGS.rescale_epsilon,
        batch_size=FLAGS.batch_size,
        n_step=FLAGS.n_step,
        num_actors=FLAGS.num_actors,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
    )

    # Create actor environments, actor instances.
    actor_envs = [environment_builder(i) for i in range(FLAGS.num_actors)]
    # TODO map to dedicated device if have multiple GPUs
    actor_devices = [runtime_device] * FLAGS.num_actors

    # Rank 0 is the most explorative actor, while rank N-1 is the most exploitative actor.
    # Each actor has it's own network with different weights.
    actors = [
        agent.Actor(
            rank=i,
            data_queue=data_queue,
            network=R2d2DqnMlpNet(input_shape=input_shape, num_actions=num_actions),
            learner_network=network,
            random_state=np.random.RandomState(FLAGS.seed + int(i)),  # pylint: disable=no-member
            num_actors=FLAGS.num_actors,
            num_actions=num_actions,
            unroll_length=FLAGS.unroll_length,
            burn_in=FLAGS.burn_in,
            actor_update_frequency=FLAGS.actor_update_frequency,
            device=actor_devices[i],
        )
        for i in range(FLAGS.num_actors)
    ]

    # Create evaluation agent instance
    eval_agent = greedy_actors.R2d2EpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
        device=runtime_device,
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(FLAGS.checkpoint_path)
    state = checkpoint.state
    state.environment_name = FLAGS.environment_name
    state.iteration = 0
    state.network = network

    # Run parallel traning N iterations.
    main_loop.run_parallel_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        network=network,
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
