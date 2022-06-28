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
"""A Rainbow agent training on Atari.

From the paper "Rainbow: Combining Improvements in Deep Reinforcement Learning"
http://arxiv.org/abs/1710.02298.

This agent combines:

*   Double Q-learning
*   Prioritized experience replay (proportional variant)
*   Dueling networks
*   Multi-step learning
*   Distributional RL (C51)
*   Noisy networks
"""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.dqn import RainbowDqnConvNet
from deep_rl_zoo.rainbow import agent
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
flags.DEFINE_integer('replay_capacity', 200000, 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 50000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 32, 'Sample batch size when do learning.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 10.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('eval_exploration_epsilon', 0.001, 'Fixed exploration rate in e-greedy policy for evaluation.')

flags.DEFINE_float('priority_exponent', 0.6, 'Priotiry exponent used in prioritized replay.')
flags.DEFINE_float('importance_sampling_exponent_begin_value', 0.4, 'Importance sampling exponent begin value.')
flags.DEFINE_float('importance_sampling_exponent_end_value', 1.0, 'Importance sampling exponent end value after decay.')
flags.DEFINE_float('uniform_sample_probability', 1e-3, 'Add some noise when sampling from the prioritized replay.')
flags.DEFINE_bool('normalize_weights', True, 'Normalize sampling weights in prioritized replay.')

flags.DEFINE_integer('num_atoms', 51, 'Number of elements in the support of the categorical DQN.')
flags.DEFINE_float('v_min', -10.0, 'Minimum elements value in the support of the categorical DQN.')
flags.DEFINE_float('v_max', 10.0, 'Maximum elements value in the support of the categorical DQN.')

flags.DEFINE_integer('n_step', 3, 'TD n-step bootstrap.')
flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_integer('num_iterations', 20, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(1e6), 'Number of training steps per iteration.')
flags.DEFINE_integer('num_eval_steps', int(2e5), 'Number of evaluation steps per iteration.')
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps per episode. 0 means no limit.')
flags.DEFINE_integer('learn_frequency', 4, 'The frequency (measured in agent steps) to do learning.')
flags.DEFINE_integer(
    'target_network_update_frequency',
    2000,
    'The frequency (measured in number of online Q network parameter updates) to update target Q networks.',
)
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_integer(
    'debug_screenshots_frequency',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', 'logs/rainbow_atari_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_path', 'checkpoints/rainbow', 'Path for checkpoint directory.')


def main(argv):
    """Trains Rainbow agent on Atari."""
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

    atoms = torch.linspace(FLAGS.v_min, FLAGS.v_max, FLAGS.num_atoms).to(device=runtime_device, dtype=torch.float32)

    network = RainbowDqnConvNet(input_shape=input_shape, num_actions=num_actions, atoms=atoms)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate)

    # Test network input and output
    network_output = network(torch.from_numpy(obs[None, ...]).float())
    q_logits = network_output.q_logits
    q_values = network_output.q_values
    assert q_logits.shape == (1, num_actions, FLAGS.num_atoms)
    assert q_values.shape == (1, num_actions)

    # Create prioritized transition replay
    # Note the t in the replay is not exactly aligned with the agent t.
    importance_sampling_exponent_schedule = LinearSchedule(
        begin_t=int(FLAGS.min_replay_size),
        end_t=(FLAGS.num_iterations * int(FLAGS.num_train_steps)),
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

    # Create RainbowDqn agent instance
    train_agent = agent.RainbowDqn(
        network=network,
        optimizer=optimizer,
        atoms=atoms,
        transition_accumulator=replay_lib.NStepTransitionAccumulator(n=FLAGS.n_step, discount=FLAGS.discount),
        replay=replay,
        batch_size=FLAGS.batch_size,
        min_replay_size=FLAGS.min_replay_size,
        learn_frequency=FLAGS.learn_frequency,
        target_network_update_frequency=FLAGS.target_network_update_frequency,
        n_step=FLAGS.n_step,
        discount=FLAGS.discount,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
    )

    # Create evaluation agent instance
    eval_agent = greedy_actors.EpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
        device=runtime_device,
        name='Rainbow-greedy',
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
        debug_screenshots_frequency=FLAGS.debug_screenshots_frequency,
    )


if __name__ == '__main__':
    app.run(main)
