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
"""A DRQN agent training on Atari.

From the paper "Deep Recurrent Q-Learning for Partially Observable MDPs"
https://arxiv.org/abs/1507.06527.
"""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.dqn import DrqnConvNet
from deep_rl_zoo.drqn import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo.schedule import LinearSchedule
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name', 'Breakout', 'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.'
)
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')
flags.DEFINE_integer('replay_capacity', 20000, 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 5000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 8, 'Sample batch size when do learning.')
flags.DEFINE_integer('unroll_length', 10, 'The number of agent steps to rollout.')
flags.DEFINE_bool('clip_grad', True, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 10.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('exploration_epsilon_begin_value', 1.0, 'Begin value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_end_value', 0.1, 'End (decayed) value of the exploration rate in e-greedy policy.')
flags.DEFINE_float('exploration_epsilon_decay_step', 500000, 'Total steps to decay value of the exploration rate.')
flags.DEFINE_float('eval_exploration_epsilon', 0.001, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_integer('n_step', 3, 'TD n-step bootstrap.')
flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('obscure_epsilon', 0.5, 'Make the problem POMDP by obsecure environment state with probability epsilon.')
flags.DEFINE_integer('num_iterations', 20, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(1e6), 'Number of training steps per iteration.')
flags.DEFINE_integer('num_eval_steps', int(2e5), 'Number of evaluation steps per iteration.')
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps per episode. 0 means no limit.')
flags.DEFINE_integer('learn_frequency', 4, 'The frequency (measured in agent steps) to do learning.')
flags.DEFINE_integer(
    'target_network_update_frequency',
    1000,
    'The frequency (measured in number of online Q parameter updates) to update target Q networks.',
)
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', 'logs/drqn_atari_results.csv', '')
flags.DEFINE_string('checkpoint_path', 'checkpoints/drqn', '')


def main(argv):
    """Trains DRQN agent on Atari."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Create environment.
    def environment_builder(random_int=0):
        return gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            screen_height=FLAGS.environment_height,
            screen_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            obscure_epsilon=FLAGS.obscure_epsilon,
            seed=FLAGS.seed + int(random_int),
            noop_max=30,
            terminal_on_life_loss=True,
        )

    env = environment_builder()
    eval_env = environment_builder()

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', env.action_space.n)
    logging.info('Observation spec: %s', env.observation_space.shape)

    input_shape = (FLAGS.environment_frame_skip, FLAGS.environment_height, FLAGS.environment_width)
    num_actions = env.action_space.n

    # Test environment and state shape.
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == input_shape

    network = DrqnConvNet(input_shape=input_shape, num_actions=num_actions)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate)

    # Test network input and output
    s = torch.from_numpy(obs[None, None, ...]).float()
    hidden_s = network.get_initial_hidden_state(batch_size=1)
    q = network(s, hidden_s).q_values
    assert q.shape == (1, 1, num_actions)

    # Create e-greedy exploration epsilon schdule
    exploration_epsilon_schedule = LinearSchedule(
        begin_t=int(FLAGS.min_replay_size),
        decay_steps=int(FLAGS.exploration_epsilon_decay_step),
        begin_value=FLAGS.exploration_epsilon_begin_value,
        end_value=FLAGS.exploration_epsilon_end_value,
    )

    # Create transition replay
    replay = replay_lib.UniformReplay(
        capacity=FLAGS.replay_capacity, structure=replay_lib.TransitionStructure, random_state=random_state
    )

    # Create DRQN agent instance
    train_agent = agent.Drqn(
        network=network,
        optimizer=optimizer,
        transition_accumulator=replay_lib.NStepTransitionAccumulator(n=FLAGS.n_step, discount=FLAGS.discount),
        replay=replay,
        exploration_epsilon=exploration_epsilon_schedule,
        batch_size=FLAGS.batch_size,
        min_replay_size=FLAGS.min_replay_size,
        learn_frequency=FLAGS.learn_frequency,
        target_network_update_frequency=FLAGS.target_network_update_frequency,
        n_step=FLAGS.n_step,
        discount=FLAGS.discount,
        unroll_length=FLAGS.unroll_length,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        num_actions=num_actions,
        random_state=random_state,
        device=runtime_device,
    )

    # Create evaluation agent instance
    eval_agent = greedy_actors.DrqnEpsilonGreedyActor(
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

    # Run the traning and evaluation for N iterations.
    main_loop.run_single_thread_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        network=network,
        train_agent=train_agent,
        train_env=env,
        eval_agent=eval_agent,
        eval_env=eval_env,
        checkpoint=checkpoint,
        csv_file=FLAGS.results_csv_path,
        tensorboard=FLAGS.tensorboard,
        tag=FLAGS.tag,
    )


if __name__ == '__main__':
    app.run(main)
