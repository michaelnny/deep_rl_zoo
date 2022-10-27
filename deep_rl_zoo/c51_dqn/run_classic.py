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
From the paper "A Distributional Perspective on Reinforcement Learning"
http://arxiv.org/abs/1707.06887.
"""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.dqn import C51DqnMlpNet
from deep_rl_zoo.c51_dqn import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo.schedule import LinearSchedule
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'CartPole-v1',
    'Classic control tasks name like CartPole-v1, LunarLander-v2, MountainCar-v0, Acrobot-v1.',
)
flags.DEFINE_integer('replay_capacity', 100000, 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 10000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 64, 'Sample batch size when do learning.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 10.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('exploration_epsilon_begin_value', 1.0, 'Begin value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_end_value', 0.05, 'End (decayed) value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_decay_step', 100000, 'Total steps to decay value of the exploration rate.')
flags.DEFINE_float('eval_exploration_epsilon', 0.001, 'Fixed exploration rate in e-greedy policy for evaluation.')

flags.DEFINE_float('priority_exponent', 0.6, 'Priority exponent used in prioritized replay.')
flags.DEFINE_float('importance_sampling_exponent_begin_value', 0.4, 'Importance sampling exponent begin value.')
flags.DEFINE_float('importance_sampling_exponent_end_value', 1.0, 'Importance sampling exponent end value after decay.')
flags.DEFINE_float('uniform_sample_probability', 1e-3, 'Add some noise when sampling from the prioritized replay.')
flags.DEFINE_bool('normalize_weights', True, 'Normalize sampling weights in prioritized replay.')

flags.DEFINE_integer('num_atoms', 51, 'Number of elements in the support of the categorical DQN.')
flags.DEFINE_float('v_min', -20.0, 'Minimum elements value in the support of the categorical DQN.')
flags.DEFINE_float('v_max', 20.0, 'Maximum elements value in the support of the categorical DQN.')

flags.DEFINE_integer('n_step', 2, 'TD n-step bootstrap.')
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_integer('num_iterations', 2, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_frames', int(5e5), 'Number of training env steps to run per iteration.')
flags.DEFINE_integer('num_eval_frames', int(1e5), 'Number of evaluation env steps to run per iteration.')
flags.DEFINE_integer('learn_frequency', 2, 'The frequency (measured in agent steps) to update parameters.')
flags.DEFINE_integer(
    'target_network_update_frequency',
    100,
    'The frequency (measured in number of Q network parameter updates) to update target networks.',
)
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_integer(
    'debug_screenshots_frequency',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', 'logs/c51_dqn_classic_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Path for checkpoint directory.')


def main(argv):
    """Trains C51-DQN agent on classic control tasks."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs C51 agent on {runtime_device}')
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Create environment.
    def environment_builder():
        return gym_env.create_classic_environment(
            env_name=FLAGS.environment_name,
            seed=random_state.randint(1, 2**32),
        )

    train_env = environment_builder()
    eval_env = environment_builder()

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', train_env.action_space.n)
    logging.info('Observation spec: %s', train_env.observation_space.shape[0])

    input_shape = train_env.observation_space.shape[0]
    num_actions = train_env.action_space.n

    atoms = torch.linspace(FLAGS.v_min, FLAGS.v_max, FLAGS.num_atoms).to(device=runtime_device, dtype=torch.float32)

    network = C51DqnMlpNet(input_shape=input_shape, num_actions=num_actions, atoms=atoms)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate)

    # Test network input and output
    obs = train_env.reset()
    s = torch.from_numpy(obs[None, ...]).float()
    network_output = network(s)
    assert network_output.q_logits.shape == (1, num_actions, FLAGS.num_atoms)
    assert network_output.q_values.shape == (1, num_actions)

    # Create e-greedy exploration epsilon schedule
    exploration_epsilon_schedule = LinearSchedule(
        begin_t=int(FLAGS.min_replay_size),
        decay_steps=int(FLAGS.exploration_epsilon_decay_step),
        begin_value=FLAGS.exploration_epsilon_begin_value,
        end_value=FLAGS.exploration_epsilon_end_value,
    )

    # Create prioritized transition replay
    # Note the t in the replay is not exactly aligned with the agent t.
    importance_sampling_exponent_schedule = LinearSchedule(
        begin_t=int(FLAGS.min_replay_size),
        end_t=(FLAGS.num_iterations * int(FLAGS.num_train_frames)),
        begin_value=FLAGS.importance_sampling_exponent_begin_value,
        end_value=FLAGS.importance_sampling_exponent_end_value,
    )
    replay = replay_lib.PrioritizedReplay(
        capacity=FLAGS.replay_capacity,
        structure=replay_lib.TransitionStructure,
        priority_exponent=FLAGS.priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        uniform_sample_probability=FLAGS.uniform_sample_probability,
        normalize_weights=FLAGS.normalize_weights,
        random_state=random_state,
    )

    # Create C51Dqn agent instance
    train_agent = agent.C51Dqn(
        network=network,
        optimizer=optimizer,
        atoms=atoms,
        transition_accumulator=replay_lib.NStepTransitionAccumulator(n=FLAGS.n_step, discount=FLAGS.discount),
        replay=replay,
        exploration_epsilon=exploration_epsilon_schedule,
        batch_size=FLAGS.batch_size,
        min_replay_size=FLAGS.min_replay_size,
        learn_frequency=FLAGS.learn_frequency,
        target_network_update_frequency=FLAGS.target_network_update_frequency,
        n_step=FLAGS.n_step,
        discount=FLAGS.discount,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        num_actions=num_actions,
        random_state=random_state,
        device=runtime_device,
    )

    # Create evaluation agent instance
    eval_agent = greedy_actors.EpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
        device=runtime_device,
        name='C51-DQN-greedy',
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(
        environment_name=FLAGS.environment_name, agent_name='C51-DQN', save_dir=FLAGS.checkpoint_dir
    )
    checkpoint.register_pair(('network', network))

    # Run the training and evaluation for N iterations.
    main_loop.run_single_thread_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_frames=FLAGS.num_train_frames,
        num_eval_frames=FLAGS.num_eval_frames,
        network=network,
        train_agent=train_agent,
        train_env=train_env,
        eval_agent=eval_agent,
        eval_env=eval_env,
        checkpoint=checkpoint,
        csv_file=FLAGS.results_csv_path,
        tensorboard=FLAGS.tensorboard,
        tag=FLAGS.tag,
        debug_screenshots_frequency=FLAGS.debug_screenshots_frequency,
    )


if __name__ == '__main__':
    app.run(main)
