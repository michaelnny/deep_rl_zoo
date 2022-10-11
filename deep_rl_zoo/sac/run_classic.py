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
From the paper "Soft Actor-Critic for Discrete Action Settings"
https://arxiv.org/abs/1910.07207.

From the paper "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
https://arxiv.org/abs/1801.01290.
"""
from absl import app
from absl import flags
from absl import logging
import multiprocessing
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.dqn import DqnMlpNet
from deep_rl_zoo.networks.policy import ActorMlpNet
from deep_rl_zoo.sac import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
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
flags.DEFINE_integer('num_actors', 8, 'Number of worker processes to use.')
flags.DEFINE_integer('replay_capacity', 100000, 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 10000, 'Minimum replay size before learning starts.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 40.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate for policy network.')
flags.DEFINE_float('q_learning_rate', 0.001, 'Learning rate for Q networks.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('q_target_tau', 0.995, 'Target Q network parameters update ratio.')
flags.DEFINE_integer('n_step', 4, 'TD n-step bootstrap.')
flags.DEFINE_integer('batch_size', 64, 'Learner batch size for learning.')
flags.DEFINE_integer('learn_frequency', 1, 'The frequency (measured in agent steps) to update parameters.')
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
flags.DEFINE_string('results_csv_path', 'logs/sac_classic_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Path for checkpoint directory.')


def main(argv):
    """Trains SAC agent on classic control tasks."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs SAC agent on {runtime_device}')
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

    # Create policy network which is shared between actors and learner.
    policy_network = ActorMlpNet(input_shape=state_dim, num_actions=num_actions)
    policy_network.share_memory()
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=FLAGS.learning_rate)

    # Create Q networks, only used by learner process to do policy evaluation
    q1_network = DqnMlpNet(input_shape=state_dim, num_actions=num_actions)
    q1_optimizer = torch.optim.Adam(q1_network.parameters(), lr=FLAGS.q_learning_rate)

    q2_network = DqnMlpNet(input_shape=state_dim, num_actions=num_actions)
    q2_optimizer = torch.optim.Adam(q2_network.parameters(), lr=FLAGS.q_learning_rate)

    # Test network output.
    obs = eval_env.reset()
    s = torch.from_numpy(obs[None, ...]).float()
    pi_logits = policy_network(s).pi_logits
    q1_values = q1_network(s).q_values
    q2_values = q2_network(s).q_values
    assert pi_logits.shape == (1, num_actions)
    assert q1_values.shape == q2_values.shape == (1, num_actions)

    # Create queue shared between actors and learner.
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors)

    replay = replay_lib.UniformReplay(
        capacity=FLAGS.replay_capacity,
        structure=replay_lib.TransitionStructure,
        random_state=random_state,
    )

    # Create SAC learner agent instance
    learner_agent = agent.Learner(
        replay=replay,
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        q1_network=q1_network,
        q1_optimizer=q1_optimizer,
        q2_network=q2_network,
        q2_optimizer=q2_optimizer,
        num_actions=num_actions,
        q_target_tau=FLAGS.q_target_tau,
        discount=FLAGS.discount,
        n_step=FLAGS.n_step,
        batch_size=FLAGS.batch_size,
        min_replay_size=FLAGS.min_replay_size,
        learn_frequency=FLAGS.learn_frequency,
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
            policy_network=policy_network,
            num_actions=num_actions,
            min_replay_size=FLAGS.min_replay_size,
            transition_accumulator=replay_lib.NStepTransitionAccumulator(n=FLAGS.n_step, discount=FLAGS.discount),
            device=actor_devices[i],
        )
        for i in range(FLAGS.num_actors)
    ]

    # Create evaluation agent instance
    eval_agent = greedy_actors.PolicyGreedyActor(
        network=policy_network,
        device=runtime_device,
        name='SAC-greedy',
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(environment_name=FLAGS.environment_name, agent_name='SAC', save_dir=FLAGS.checkpoint_dir)
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
