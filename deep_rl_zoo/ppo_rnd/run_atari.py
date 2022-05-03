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
"""A PPO-RND agent training on Atari.

From the paper "Exploration by Random Network Distillation"
https://arxiv.org/abs/1810.12894

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
from deep_rl_zoo.networks.policy import RndActorCriticConvNet
from deep_rl_zoo.networks.curiosity import RndConvNet
from deep_rl_zoo.ppo_rnd import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo.schedule import LinearSchedule
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib
from deep_rl_zoo import normalizer


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name', 'Pong', 'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.'
)  # MontezumaRevenge
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')
flags.DEFINE_integer('num_actors', 8, 'Number of worker processes to use.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 40.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_float('discount', 0.999, 'Discount rate.')
flags.DEFINE_float('rnd_discount', 0.99, 'Discount rate.')
flags.DEFINE_float('entropy_coef', 0.001, 'Coefficient for the entropy loss.')
flags.DEFINE_float('baseline_coef', 0.5, 'Coefficient for the state-value loss.')
flags.DEFINE_float('clip_epsilon_begin_value', 0.2, 'PPO clip epsilon begin value.')
flags.DEFINE_float('clip_epsilon_end_value', 0.0, 'PPO clip epsilon final value.')
flags.DEFINE_float('extrinsic_reward_coef', 2.0, 'Weights extrinsic reward coming from environment.')
flags.DEFINE_float('intrinsic_reward_coef', 1.0, 'Weights intrinsic reward coming from RND bonus.')
flags.DEFINE_float('rnd_experience_proportion', 0.25, 'Proportion of experience used for training RND predictor.')
flags.DEFINE_integer(
    'observation_norm_steps',
    int(1e4),
    'Warm up random steps to take in order to generate statistics for observation, used for RND networks only.',
)
flags.DEFINE_integer('observation_norm_clip', 5, 'Observation normalization clip range for RND.')
flags.DEFINE_integer('n_step', 4, 'TD n-step bootstrap.')
flags.DEFINE_integer('batch_size', 64, 'Learner batch size for learning.')
flags.DEFINE_integer('unroll_length', 128, 'Actor unroll length.')
flags.DEFINE_integer('update_k', 4, 'Run update k times when do learning.')
flags.DEFINE_integer('num_iterations', 10, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(1e6), 'Number of training steps per iteration.')
flags.DEFINE_integer('num_eval_steps', int(1e5), 'Number of evaluation steps per iteration.')
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps per episode. 0 means no limit.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', 'logs/ppo_rnd_atari_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_path', 'checkpoints/ppo_rnd', 'Path for checkpoint directory.')


def main(argv):
    """Trains PPO-RND agent on Atari."""
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
        return gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            screen_height=FLAGS.environment_height,
            screen_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=FLAGS.seed + int(random_int),
            noop_max=30,
            done_on_life_loss=True,
            clip_reward=True,
        )

    eval_env = environment_builder()

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', eval_env.action_space.n)
    logging.info('Observation spec: %s', eval_env.observation_space.shape)

    input_shape = (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)
    num_actions = eval_env.action_space.n

    # Test environment and state shape..
    obs = eval_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == input_shape

    # Create observation normalizer and run random steps to generate statistics.
    obs_norm_clip = FLAGS.observation_norm_clip
    observation_normalizer = normalizer.Normalizer(
        eps=0.0001, clip_range=(-obs_norm_clip, obs_norm_clip), device=runtime_device
    )
    logging.info(f'Generating {FLAGS.observation_norm_steps} random obserations for normalizer')
    for i in range(FLAGS.observation_norm_steps):
        a_t = eval_env.action_space.sample()
        s_t, _, done, _ = eval_env.step(a_t)
        if done:
            eval_env.reset()
        # Unstack frames, RND normalize one frame.
        s_t_unstacked = torch.unbind(torch.from_numpy(s_t), dim=0)
        for observation in s_t_unstacked:
            # Add last dimension as channel inorder to do normalization by channel.
            observation_normalizer.update(observation[..., None].to(device=runtime_device))

    # Create policy network, master will optimize this network
    policy_network = RndActorCriticConvNet(input_shape=input_shape, num_actions=num_actions)

    # The 'old' policy for actors to act
    old_policy_network = RndActorCriticConvNet(input_shape=input_shape, num_actions=num_actions)
    old_policy_network.share_memory()

    # Create RND target and predictor networks.
    rnd_target_network = RndConvNet(input_shape=input_shape)
    rnd_predictor_network = RndConvNet(input_shape=input_shape)

    # Use a single optimizer for both policy and RND predictor networks.
    policy_optimizer = torch.optim.Adam(
        list(policy_network.parameters()) + list(rnd_predictor_network.parameters()),
        lr=FLAGS.learning_rate,
    )

    # Test network output..
    s = torch.from_numpy(obs[None, ...]).float()
    network_output = policy_network(s)
    pi_logits = network_output.pi_logits
    ext_baseline = network_output.ext_baseline
    int_baseline = network_output.int_baseline
    assert pi_logits.shape == (1, num_actions)
    assert ext_baseline.shape == int_baseline.shape == (1, 1)

    # Create queue shared between actors and learner and log queue.
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors)

    clip_epsilon_scheduler = LinearSchedule(
        begin_t=0,
        end_t=(FLAGS.num_iterations * int(FLAGS.num_train_steps * FLAGS.num_actors)),  # Learner step_t is fater than worker
        begin_value=FLAGS.clip_epsilon_begin_value,
        end_value=FLAGS.clip_epsilon_end_value,
    )

    # Create PPO-RND learner agent instance
    learner_agent = agent.Learner(
        data_queue=data_queue,
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        old_policy_network=old_policy_network,
        rnd_target_network=rnd_target_network,
        rnd_predictor_network=rnd_predictor_network,
        observation_normalizer=observation_normalizer,
        clip_epsilon=clip_epsilon_scheduler,
        discount=FLAGS.discount,
        rnd_discount=FLAGS.rnd_discount,
        n_step=FLAGS.n_step,
        batch_size=FLAGS.batch_size,
        update_k=FLAGS.update_k,
        unroll_length=FLAGS.unroll_length,
        extrinsic_reward_coef=FLAGS.extrinsic_reward_coef,
        intrinsic_reward_coef=FLAGS.intrinsic_reward_coef,
        rnd_experience_proportion=FLAGS.rnd_experience_proportion,
        num_actors=FLAGS.num_actors,
        entropy_coef=FLAGS.entropy_coef,
        baseline_coef=FLAGS.baseline_coef,
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
        name='PPO-RND-greedy',
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(FLAGS.checkpoint_path)
    state = checkpoint.state
    state.environment_name = FLAGS.environment_name
    state.iteration = 0
    state.policy_network = policy_network

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
        max_episode_steps=FLAGS.max_episode_steps,
    )


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)
