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
"""Tests and evaluates trained Implicit Quantile DQN agent
by loading network state from checkpoint file and using e-greedy actor."""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.dqn import IqnMlpNet, IqnConvNet
from deep_rl_zoo import main_loop
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'CartPole-v1',
    'Both classic game name like CartPole-v1, MountainCar-v0, LunarLander-v2, and Atari game like Pong, Breakout.',
)
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height, for atari only.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width, for atari only.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip, for atari only.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack, for atari only.')
flags.DEFINE_float('eval_exploration_epsilon', 0.001, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_integer('tau_latent_dim', 128, 'Embeding layer dimenstion.')
flags.DEFINE_integer('tau_samples_policy', 64, 'Number of samples to pull from the network when choose actions.')
flags.DEFINE_integer('num_iterations', 1, 'Number of evaluation iterations to run.')
flags.DEFINE_integer('num_eval_frames', int(2e5), 'Number of evaluation frames (or env steps) to run during per iteration.')
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps per episode, for atari only.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_string('load_checkpoint_file', '', 'Load a specific checkpoint file.')
flags.DEFINE_string(
    'recording_video_dir',
    'recordings/iqn',
    'Path for recording a video of agent self-play.',
)


def main(argv):
    """Tests IQN agent."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Create evaluation environments
    if FLAGS.environment_name in gym_env.CLASSIC_ENV_NAMES:
        eval_env = gym_env.create_classic_environment(env_name=FLAGS.environment_name, seed=FLAGS.seed)
        input_shape = eval_env.observation_space.shape[0]
        num_actions = eval_env.action_space.n
        network = IqnMlpNet(input_shape=input_shape, num_actions=num_actions, latent_dim=FLAGS.tau_latent_dim)
    else:
        eval_env = gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            screen_height=FLAGS.environment_height,
            screen_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=FLAGS.seed,
            noop_max=30,
            terminal_on_life_loss=False,
            clip_reward=False,
        )
        input_shape = (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)
        num_actions = eval_env.action_space.n
        network = IqnConvNet(input_shape=input_shape, num_actions=num_actions, latent_dim=FLAGS.tau_latent_dim)

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', num_actions)
    logging.info('Observation spec: %s', input_shape)

    # Setup checkpoint and load model weights from checkpoint.
    checkpoint = PyTorchCheckpoint(environment_name=FLAGS.environment_name, agent_name='IQN', restore_only=True)
    checkpoint.register_pair(('network', network))

    if FLAGS.load_checkpoint_file:
        checkpoint.restore(FLAGS.load_checkpoint_file)

    network.eval()

    # Create evaluation agent instance
    eval_agent = greedy_actors.IqnEpsilonGreedyActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
        device=runtime_device,
        tau_samples=FLAGS.tau_samples_policy,
    )

    # Run test N iterations.
    main_loop.run_evaluation_iterations(
        num_iterations=FLAGS.num_iterations,
        num_eval_frames=FLAGS.num_eval_frames,
        eval_agent=eval_agent,
        eval_env=eval_env,
        tensorboard=FLAGS.tensorboard,
        recording_video_dir=FLAGS.recording_video_dir,
    )


if __name__ == '__main__':
    app.run(main)
