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
From the paper "Exploration by Random Network Distillation"
https://arxiv.org/abs/1810.12894

"""

from absl import app
from absl import flags
from absl import logging
import multiprocessing
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.policy import RndActorCriticMlpNet
from deep_rl_zoo.networks.curiosity import RndMlpNet
from deep_rl_zoo.ppo_rnd import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo.schedule import LinearSchedule
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import normalizer


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'CartPole-v1',
    'Classic control tasks name like CartPole-v1, LunarLander-v2, MountainCar-v0, Acrobot-v1.',
)
flags.DEFINE_integer('num_actors', 8, 'Number of worker processes to use.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('learning_rate', 0.0003, 'Learning rate.')
flags.DEFINE_float('discount', 0.999, 'Discount rate for entrinsic environment reward.')
flags.DEFINE_float('rnd_discount', 0.99, 'Discount rate intrinsic reward.')
flags.DEFINE_float('gae_lambda', 0.95, 'Lambda for the GAE general advantage estimator.')
flags.DEFINE_float('entropy_coef', 0.0025, 'Coefficient for the entropy loss.')
flags.DEFINE_float('baseline_coef', 0.5, 'Coefficient for the state-value loss.')
flags.DEFINE_float('clip_epsilon_begin_value', 0.2, 'PPO clip epsilon begin value.')
flags.DEFINE_float('clip_epsilon_end_value', 0.0, 'PPO clip epsilon final value.')
flags.DEFINE_float('rnd_experience_proportion', 0.25, 'Proportion of experience used for training RND predictor.')
flags.DEFINE_integer(
    'observation_norm_steps',
    int(1e4),
    'Warm up random steps to take in order to generate statistics for observation, used for RND networks only.',
)
flags.DEFINE_integer('observation_norm_clip', 10, 'Observation normalization clip range for RND.')

flags.DEFINE_integer('batch_size', 64, 'Learner batch size for learning.')
flags.DEFINE_integer('unroll_length', 1024, 'Collect N transitions (cross episodes) before send to learner, per actor.')
flags.DEFINE_integer('update_k', 4, 'Number of epochs to update network parameters.')
flags.DEFINE_integer('num_iterations', 2, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_frames', int(5e5), 'Number of training env steps to run per iteration, per actor.')
flags.DEFINE_integer('num_eval_frames', int(1e5), 'Number of evaluation env steps to run per iteration.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_integer(
    'debug_screenshots_frequency',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', 'logs/ppo_rnd_classic_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Path for checkpoint directory.')


def main(argv):
    """Trains PPO-RND agent on classic control tasks."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs PPO-RND agent on {runtime_device}')
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Listen to signals to exit process.
    main_loop.handle_exit_signal()

    # Create environment.
    def environment_builder():
        return gym_env.create_classic_environment(
            env_name=FLAGS.environment_name,
            seed=random_state.randint(1, 2**32),
        )

    eval_env = environment_builder()

    state_dim = eval_env.observation_space.shape[0]
    num_actions = eval_env.action_space.n

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', num_actions)
    logging.info('Observation spec: %s', state_dim)

    # Create observation normalizer and run random steps to generate statistics.
    obs = eval_env.reset()
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
        observation_normalizer.update(torch.from_numpy(s_t[None, ...]).to(device=runtime_device))

    # Create policy network, master will optimize this network
    policy_network = RndActorCriticMlpNet(input_shape=state_dim, num_actions=num_actions)

    # The 'old' policy for actors to act
    old_policy_network = RndActorCriticMlpNet(input_shape=state_dim, num_actions=num_actions)
    old_policy_network.share_memory()

    # Create RND target and predictor networks.
    rnd_target_network = RndMlpNet(input_shape=state_dim, is_target=True)
    rnd_predictor_network = RndMlpNet(input_shape=state_dim)

    # Use a single optimizer for both policy and RND predictor networks.
    policy_optimizer = torch.optim.Adam(
        list(policy_network.parameters()) + list(rnd_predictor_network.parameters()),
        lr=FLAGS.learning_rate,
    )

    # Test network output.
    obs = eval_env.reset()
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
        end_t=int(
            (FLAGS.num_iterations * int(FLAGS.num_train_frames * FLAGS.num_actors)) / FLAGS.unroll_length
        ),  # Learner step_t is often faster than worker
        begin_value=FLAGS.clip_epsilon_begin_value,
        end_value=FLAGS.clip_epsilon_end_value,
    )

    # Create PPO-RND learner agent instance
    learner_agent = agent.Learner(
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        old_policy_network=old_policy_network,
        rnd_target_network=rnd_target_network,
        rnd_predictor_network=rnd_predictor_network,
        observation_normalizer=observation_normalizer,
        clip_epsilon=clip_epsilon_scheduler,
        discount=FLAGS.discount,
        rnd_discount=FLAGS.discount,
        gae_lambda=FLAGS.gae_lambda,
        total_unroll_length=int(FLAGS.unroll_length * FLAGS.num_actors),
        batch_size=FLAGS.batch_size,
        update_k=FLAGS.update_k,
        rnd_experience_proportion=FLAGS.rnd_experience_proportion,
        entropy_coef=FLAGS.entropy_coef,
        baseline_coef=FLAGS.baseline_coef,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
    )

    # Create actor environments, runtime devices, and actor instances.
    actor_envs = [environment_builder() for _ in range(FLAGS.num_actors)]

    # TODO map to dedicated device if have multiple GPUs
    actor_devices = [runtime_device] * FLAGS.num_actors

    actors = [
        agent.Actor(
            rank=i,
            data_queue=data_queue,
            policy_network=old_policy_network,
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
    checkpoint = PyTorchCheckpoint(
        environment_name=FLAGS.environment_name, agent_name='PPO-RND', save_dir=FLAGS.checkpoint_dir
    )
    checkpoint.register_pair(('policy_network', policy_network))
    checkpoint.register_pair(('rnd_target_network', rnd_target_network))
    checkpoint.register_pair(('rnd_predictor_network', rnd_predictor_network))

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
