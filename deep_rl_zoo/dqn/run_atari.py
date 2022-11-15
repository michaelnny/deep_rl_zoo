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
From the paper "Human Level Control Through Deep Reinforcement Learning"
http://www.nature.com/articles/nature14236.
"""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.dqn import DqnConvNet
from deep_rl_zoo.dqn import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo.schedule import LinearSchedule
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'Pong', 'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.')
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')
flags.DEFINE_bool('compress_state', True, 'Compress state images when store in experience replay.')
flags.DEFINE_integer('replay_capacity', int(1e6), 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 50000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 32, 'Sample batch size when do learning.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('exploration_epsilon_begin_value', 1.0, 'Begin value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_end_value', 0.01, 'End (decayed) value of the exploration rate in e-greedy policy.')
flags.DEFINE_float(
    'exploration_epsilon_decay_step',
    int(1e6),
    'Total steps (after frame skip) to decay value of the exploration rate.',
)
flags.DEFINE_float('eval_exploration_epsilon', 0.01, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_integer('num_iterations', 100, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_frames', int(1e6 / 4), 'Number of training frames (after frame skip) to run per iteration.')
flags.DEFINE_integer('num_eval_frames', int(1e5), 'Number of evaluation frames (after frame skip) to run per iteration.')
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps (before frame skip) per episode.')
flags.DEFINE_integer('learn_frequency', 4, 'The frequency (measured in agent steps) to update parameters.')
flags.DEFINE_integer(
    'target_network_update_frequency',
    2500,
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
flags.DEFINE_string('results_csv_path', 'logs/dqn_atari_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', '', 'Path for checkpoint directory.')


def main(argv):
    """Trains DQN agent on Atari."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs DQN agent on {runtime_device}')
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Create environment.
    def environment_builder():
        return gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            screen_height=FLAGS.environment_height,
            screen_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=random_state.randint(1, 2**32),
            noop_max=30,
            terminal_on_life_loss=True,
        )

    train_env = environment_builder()
    eval_env = environment_builder()

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', train_env.action_space.n)
    logging.info('Observation spec: %s', train_env.observation_space.shape)

    input_shape = train_env.observation_space.shape
    num_actions = train_env.action_space.n

    # Test environment and state shape.
    obs = train_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)

    network = DqnConvNet(input_shape=input_shape, num_actions=num_actions)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate)

    # Test network input and output
    q = network(torch.from_numpy(obs[None, ...])).q_values
    assert q.shape == (1, num_actions)

    # Create e-greedy exploration epsilon schedule
    exploration_epsilon_schedule = LinearSchedule(
        begin_t=int(FLAGS.min_replay_size),
        decay_steps=int(FLAGS.exploration_epsilon_decay_step),
        begin_value=FLAGS.exploration_epsilon_begin_value,
        end_value=FLAGS.exploration_epsilon_end_value,
    )

    # Create transition replay
    if FLAGS.compress_state:

        def encoder(transition):
            return transition._replace(
                s_tm1=replay_lib.compress_array(transition.s_tm1),
                s_t=replay_lib.compress_array(transition.s_t),
            )

        def decoder(transition):
            return transition._replace(
                s_tm1=replay_lib.uncompress_array(transition.s_tm1),
                s_t=replay_lib.uncompress_array(transition.s_t),
            )

    else:
        encoder = None
        decoder = None

    replay = replay_lib.UniformReplay(
        capacity=FLAGS.replay_capacity,
        structure=replay_lib.TransitionStructure,
        random_state=random_state,
        encoder=encoder,
        decoder=decoder,
    )

    # Create DQN agent instance
    train_agent = agent.Dqn(
        network=network,
        optimizer=optimizer,
        transition_accumulator=replay_lib.TransitionAccumulator(),
        replay=replay,
        exploration_epsilon=exploration_epsilon_schedule,
        batch_size=FLAGS.batch_size,
        min_replay_size=FLAGS.min_replay_size,
        learn_frequency=FLAGS.learn_frequency,
        target_network_update_frequency=FLAGS.target_network_update_frequency,
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
        name='DQN-greedy',
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(environment_name=FLAGS.environment_name, agent_name='DQN', save_dir=FLAGS.checkpoint_dir)
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
