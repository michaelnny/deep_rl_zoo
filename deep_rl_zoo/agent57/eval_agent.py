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
"""Tests trained Agent57 agent from checkpoint with a e-greedy actor.
on classic control tasks like CartPole, MountainCar, or LunarLander, and on Atari."""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.dqn import NguDqnMlpNet, NguDqnConvNet
from deep_rl_zoo.networks.curiosity import RndMlpNet, NguEmbeddingMlpNet, RndConvNet, NguEmbeddingConvNet
from deep_rl_zoo import main_loop
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'CartPole-v1',
    'Both Classic control tasks name like CartPole-v1, LunarLander-v2, MountainCar-v0, Acrobot-v1. and Atari game like Pong, Breakout.',
)
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height, for atari only.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width, for atari only.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip, for atari only.')
flags.DEFINE_integer('environment_frame_stack', 1, 'Number of frames to stack, for atari only.')
flags.DEFINE_float('eval_exploration_epsilon', 0.001, 'Fixed exploration rate in e-greedy policy for evaluation.')

flags.DEFINE_integer('num_policies', 32, 'Number of directed policies to learn, scaled by intrinsic reward scale beta.')

flags.DEFINE_integer('episodic_memory_capacity', 5000, 'Maximum size of episodic memory.')  # 10000
flags.DEFINE_integer('num_neighbors', 10, 'Number of K-nearest neighbors.')
flags.DEFINE_float('kernel_epsilon', 0.0001, 'K-nearest neighbors kernel epsilon.')
flags.DEFINE_float('cluster_distance', 0.008, 'K-nearest neighbors custer distance.')
flags.DEFINE_float('max_similarity', 8.0, 'K-nearest neighbors custer distance.')

flags.DEFINE_integer('num_iterations', 1, 'Number of evaluation iterations to run.')
flags.DEFINE_integer('num_eval_frames', int(1e5), 'Number of evaluation env steps to run per iteration.')
flags.DEFINE_integer('max_episode_steps', 58000, 'Maximum steps (before frame skip) per episode, for atari only.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_string('load_checkpoint_file', '', 'Load a specific checkpoint file.')
flags.DEFINE_string(
    'recording_video_dir',
    'recordings',
    'Path for recording a video of agent self-play.',
)


def main(argv):
    """Tests Agent57 agent."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Create evaluation environments
    if FLAGS.environment_name in gym_env.CLASSIC_ENV_NAMES:
        eval_env = gym_env.create_classic_environment(env_name=FLAGS.environment_name, seed=random_state.randint(1, 2**32))
        input_shape = eval_env.observation_space.shape[0]
        num_actions = eval_env.action_space.n
        ext_q_network = NguDqnMlpNet(input_shape=input_shape, num_actions=num_actions, num_policies=FLAGS.num_policies)
        int_q_network = NguDqnMlpNet(input_shape=input_shape, num_actions=num_actions, num_policies=FLAGS.num_policies)
        rnd_target_network = RndMlpNet(input_shape=input_shape)
        rnd_predictor_network = RndMlpNet(input_shape=input_shape)
        embedding_network = NguEmbeddingMlpNet(input_shape=input_shape, num_actions=num_actions)
    else:
        eval_env = gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            screen_height=FLAGS.environment_height,
            screen_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=random_state.randint(1, 2**32),
            noop_max=30,
            terminal_on_life_loss=False,
            clip_reward=False,
        )
        input_shape = (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)
        num_actions = eval_env.action_space.n
        ext_q_network = NguDqnConvNet(input_shape=input_shape, num_actions=num_actions, num_policies=FLAGS.num_policies)
        int_q_network = NguDqnConvNet(input_shape=input_shape, num_actions=num_actions, num_policies=FLAGS.num_policies)
        rnd_target_network = RndConvNet(input_shape=input_shape)
        rnd_predictor_network = RndConvNet(input_shape=input_shape)
        embedding_network = NguEmbeddingConvNet(input_shape=input_shape, num_actions=num_actions)

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', num_actions)
    logging.info('Observation spec: %s', input_shape)

    # Setup checkpoint and load model weights from checkpoint.
    checkpoint = PyTorchCheckpoint(environment_name=FLAGS.environment_name, agent_name='Agent57', restore_only=True)
    checkpoint.register_pair(('ext_q_network', ext_q_network))
    checkpoint.register_pair(('int_q_network', int_q_network))
    checkpoint.register_pair(('rnd_target_network', rnd_target_network))
    checkpoint.register_pair(('rnd_predictor_network', rnd_predictor_network))
    checkpoint.register_pair(('embedding_network', embedding_network))

    if FLAGS.load_checkpoint_file:
        checkpoint.restore(FLAGS.load_checkpoint_file)

    ext_q_network.eval()
    int_q_network.eval()
    rnd_target_network.eval()
    rnd_predictor_network.eval()
    embedding_network.eval()

    # Create evaluation agent instance
    eval_agent = greedy_actors.Agent57EpsilonGreedyActor(
        ext_q_network=ext_q_network,
        int_q_network=int_q_network,
        embedding_network=embedding_network,
        rnd_target_network=rnd_target_network,
        rnd_predictor_network=rnd_predictor_network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        episodic_memory_capacity=FLAGS.episodic_memory_capacity,
        num_neighbors=FLAGS.num_neighbors,
        kernel_epsilon=FLAGS.kernel_epsilon,
        cluster_distance=FLAGS.cluster_distance,
        max_similarity=FLAGS.max_similarity,
        random_state=random_state,
        device=runtime_device,
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
