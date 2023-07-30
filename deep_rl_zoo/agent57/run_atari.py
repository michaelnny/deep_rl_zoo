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
From the paper "Agent57: Outperforming the Atari Human Benchmark"
https://arxiv.org/pdf/2003.13350.
"""

from absl import app
from absl import flags
from absl import logging
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import multiprocessing
import numpy as np
import torch
import copy

# pylint: disable=import-error
from deep_rl_zoo.networks.value import Agent57Conv2dNet, Agent57NetworkInputs
from deep_rl_zoo.networks.curiosity import NGURndConvNet, NguEmbeddingConvNet
from deep_rl_zoo.agent57 import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name', 'Pong', 'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.'
)  # MontezumaRevenge, Pitfall, Solaris, Skiing
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 1, 'Number of frames to stack.')
flags.DEFINE_bool('compress_state', True, 'Compress state images when store in experience replay.')
flags.DEFINE_integer('num_actors', 8, 'Number of actor processes to run in parallel.')
flags.DEFINE_integer('replay_capacity', 20000, 'Maximum replay size (in number of unrolls stored).')  # watch for out of RAM
flags.DEFINE_integer(
    'min_replay_size', 1000, 'Minimum replay size before learning starts (in number of unrolls stored).'
)  # 6250
flags.DEFINE_bool('clip_grad', True, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 40.0, 'Max gradients norm when do gradients clip.')

flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for adam.')
flags.DEFINE_float(
    'int_learning_rate', 0.0005, 'Intrinsic learning rate for adam, this is for embedding and RND predictor networks.'
)
flags.DEFINE_float('ext_discount', 0.997, 'Extrinsic reward discount rate.')
flags.DEFINE_float('int_discount', 0.99, 'Intrinsic reward discount rate.')
flags.DEFINE_float('adam_eps', 0.0001, 'Epsilon for adam.')
flags.DEFINE_integer('unroll_length', 80, 'Sequence of transitions to unroll before add to replay.')
flags.DEFINE_integer(
    'burn_in',
    40,
    'Sequence of transitions used to pass RNN before actual learning.'
    'The effective length of unrolls will be burn_in + unroll_length, '
    'two consecutive unrolls will overlap on burn_in steps.',
)
flags.DEFINE_integer('batch_size', 32, 'Batch size for learning.')

flags.DEFINE_float('policy_beta', 0.3, 'Scalar for the intrinsic reward scale.')
flags.DEFINE_integer('num_policies', 32, 'Number of directed policies to learn, scaled by intrinsic reward scale beta.')
flags.DEFINE_integer('ucb_window_size', 90, 'Sliding window size of the UCB algorithm.')
flags.DEFINE_float('ucb_beta', 1.0, 'Beta for the UCB algorithm.')
flags.DEFINE_float('ucb_epsilon', 0.5, 'Exploration epsilon for the UCB algorithm.')

flags.DEFINE_integer('episodic_memory_capacity', 3000, 'Maximum size of episodic memory.')  # 30000
flags.DEFINE_bool(
    'reset_episodic_memory',
    True,
    'Reset the episodic_memory on every episode, only applicable to actors, default on.'
    'From NGU Paper on MontezumaRevenge, Instead of resetting the memory after every episode, we do it after a small number of '
    'consecutive episodes, which we call a meta-episode. This structure plays an important role when the'
    'agent faces irreversible choices.',
)
flags.DEFINE_integer('num_neighbors', 10, 'Number of K-nearest neighbors.')
flags.DEFINE_float('kernel_epsilon', 0.0001, 'K-nearest neighbors kernel epsilon.')
flags.DEFINE_float('cluster_distance', 0.008, 'K-nearest neighbors custer distance.')
flags.DEFINE_float('max_similarity', 8.0, 'K-nearest neighbors custer distance.')

flags.DEFINE_float('retrace_lambda', 0.95, 'Lambda coefficient for retrace.')
flags.DEFINE_bool('transformed_retrace', True, 'Transformed retrace loss, default on.')

flags.DEFINE_float('priority_exponent', 0.9, 'Priority exponent used in prioritized replay.')
flags.DEFINE_float('importance_sampling_exponent', 0.6, 'Importance sampling exponent value.')
flags.DEFINE_bool('normalize_weights', True, 'Normalize sampling weights in prioritized replay.')
flags.DEFINE_float('priority_eta', 0.9, 'Priority eta to mix the max and mean absolute TD errors.')

flags.DEFINE_integer('num_iterations', 100, 'Number of iterations to run.')
flags.DEFINE_integer(
    'num_train_steps', int(5e5), 'Number of training steps (environment steps or frames) to run per iteration, per actor.'
)
flags.DEFINE_integer(
    'num_eval_steps', int(2e4), 'Number of evaluation steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps (before frame skip) per episode.')
flags.DEFINE_integer(
    'target_net_update_interval',
    1500,
    'The interval (meassured in Q network updates) to update target Q networks.',
)
flags.DEFINE_integer('actor_update_interval', 100, 'The frequency (measured in actor steps) to update actor local Q network.')
flags.DEFINE_float('eval_exploration_epsilon', 0.01, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', True, 'Run actors on GPU, default on.')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', './logs/agent57_atari_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Path for checkpoint directory.')

flags.register_validator('environment_frame_stack', lambda x: x == 1)


def main(argv):
    """Trains Agent57 agent on Atari."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs Agent57 agent on {runtime_device}')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member

    # Create evaluation environment, like R2D2, we disable terminate-on-life-loss and clip reward.
    def environment_builder():
        return gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            frame_height=FLAGS.environment_height,
            frame_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=random_state.randint(1, 2**10),
            noop_max=30,
            terminal_on_life_loss=False,
            sticky_action=False,
            clip_reward=False,
        )

    eval_env = environment_builder()

    state_dim = eval_env.observation_space.shape
    action_dim = eval_env.action_space.n

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', action_dim)
    logging.info('Observation spec: %s', state_dim)

    # Test environment and state shape.
    obs = eval_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)

    # Create extrinsic and intrinsic reward Q networks for learner to optimize.
    network = Agent57Conv2dNet(state_dim=state_dim, action_dim=action_dim, num_policies=FLAGS.num_policies)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate, eps=FLAGS.adam_eps)

    # Create RND target and predictor networks.
    rnd_target_network = NGURndConvNet(state_dim=state_dim, is_target=True)
    rnd_predictor_network = NGURndConvNet(state_dim=state_dim, is_target=False)

    # Create embedding networks.
    embedding_network = NguEmbeddingConvNet(state_dim=state_dim, action_dim=action_dim)

    # Second Adam optimizer for embedding and RND predictor networks.
    intrinsic_optimizer = torch.optim.Adam(
        list(embedding_network.parameters()) + list(rnd_predictor_network.parameters()),
        lr=FLAGS.int_learning_rate,
        eps=FLAGS.adam_eps,
    )

    ext_state, int_state = network.get_initial_hidden_state(1)

    # Test network output.
    x = Agent57NetworkInputs(
        s_t=torch.from_numpy(obs[None, None, ...]).float(),
        a_tm1=torch.zeros(1, 1).long(),
        ext_r_t=torch.zeros(1, 1).float(),
        int_r_t=torch.zeros(1, 1).float(),
        policy_index=torch.zeros(1, 1).long(),
        ext_hidden_s=ext_state,
        int_hidden_s=int_state,
    )
    network_output = network(x)
    assert network_output.ext_q_values.shape == (1, 1, action_dim)
    assert network_output.int_q_values.shape == (1, 1, action_dim)
    assert len(network_output.ext_hidden_s) == 2
    assert len(network_output.int_hidden_s) == 2

    # Create prioritized transition replay, no importance_sampling_exponent decay
    importance_sampling_exponent = FLAGS.importance_sampling_exponent

    def importance_sampling_exponent_schedule(x):
        return importance_sampling_exponent

    # Create transition replay
    if FLAGS.compress_state:

        def encoder(transition):
            return transition._replace(
                s_t=replay_lib.compress_array(transition.s_t),
            )

        def decoder(transition):
            return transition._replace(
                s_t=replay_lib.uncompress_array(transition.s_t),
            )

    else:
        encoder = None
        decoder = None

    replay = replay_lib.PrioritizedReplay(
        capacity=FLAGS.replay_capacity,
        structure=agent.TransitionStructure,
        priority_exponent=FLAGS.priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        normalize_weights=FLAGS.normalize_weights,
        random_state=random_state,
        time_major=True,
        encoder=encoder,
        decoder=decoder,
    )

    # Create queue to shared transitions between actors and learner
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors * 2)
    # Create shared objects so all actor processes can access them
    manager = multiprocessing.Manager()

    # Store copy of latest parameters of the neural network in a shared dictionary, so actors can later access it
    shared_params = manager.dict(
        {
            'network': None,
            'embedding_network': None,
            'rnd_predictor_network': None,
        }
    )

    # Create Agent57 learner instance
    learner_agent = agent.Learner(
        network=network,
        optimizer=optimizer,
        embedding_network=embedding_network,
        rnd_target_network=rnd_target_network,
        rnd_predictor_network=rnd_predictor_network,
        intrinsic_optimizer=intrinsic_optimizer,
        replay=replay,
        min_replay_size=FLAGS.min_replay_size,
        target_net_update_interval=FLAGS.target_net_update_interval,
        unroll_length=FLAGS.unroll_length,
        burn_in=FLAGS.burn_in,
        retrace_lambda=FLAGS.retrace_lambda,
        transformed_retrace=FLAGS.transformed_retrace,
        priority_eta=FLAGS.priority_eta,
        batch_size=FLAGS.batch_size,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
        shared_params=shared_params,
    )

    # Create actor environments, actor instances.
    actor_envs = [environment_builder() for _ in range(FLAGS.num_actors)]

    actor_devices = ['cpu'] * FLAGS.num_actors
    # Evenly distribute the actors to all available GPUs
    if torch.cuda.is_available() and FLAGS.actors_on_gpu:
        num_gpus = torch.cuda.device_count()
        actor_devices = [torch.device(f'cuda:{i % num_gpus}') for i in range(FLAGS.num_actors)]

    # Each actor has it's own embedding and RND predictor networks,
    # because we don't want to update these network parameters in the middle of an episode,
    # it will only update these networks at the beginning of an episode.
    actors = [
        agent.Actor(
            rank=i,
            data_queue=data_queue,
            network=copy.deepcopy(network),
            rnd_target_network=copy.deepcopy(rnd_target_network),
            rnd_predictor_network=copy.deepcopy(rnd_predictor_network),
            embedding_network=copy.deepcopy(embedding_network),
            random_state=np.random.RandomState(FLAGS.seed + int(i)),  # pylint: disable=no-member
            ext_discount=FLAGS.ext_discount,
            int_discount=FLAGS.int_discount,
            num_policies=FLAGS.num_policies,
            policy_beta=FLAGS.policy_beta,
            ucb_window_size=FLAGS.ucb_window_size,
            ucb_beta=FLAGS.ucb_beta,
            ucb_epsilon=FLAGS.ucb_epsilon,
            episodic_memory_capacity=FLAGS.episodic_memory_capacity,
            reset_episodic_memory=FLAGS.reset_episodic_memory,
            num_neighbors=FLAGS.num_neighbors,
            kernel_epsilon=FLAGS.kernel_epsilon,
            cluster_distance=FLAGS.cluster_distance,
            max_similarity=FLAGS.max_similarity,
            num_actors=FLAGS.num_actors,
            action_dim=action_dim,
            unroll_length=FLAGS.unroll_length,
            burn_in=FLAGS.burn_in,
            actor_update_interval=FLAGS.actor_update_interval,
            device=actor_devices[i],
            shared_params=shared_params,
        )
        for i in range(FLAGS.num_actors)
    ]

    # Create evaluation agent instance
    eval_agent = greedy_actors.Agent57EpsilonGreedyActor(
        network=network,
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

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(
        environment_name=FLAGS.environment_name, agent_name='Agent57', save_dir=FLAGS.checkpoint_dir
    )
    checkpoint.register_pair(('network', network))
    checkpoint.register_pair(('rnd_target_network', rnd_target_network))
    checkpoint.register_pair(('rnd_predictor_network', rnd_predictor_network))
    checkpoint.register_pair(('embedding_network', embedding_network))

    # Run parallel training N iterations.
    main_loop.run_parallel_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        learner_agent=learner_agent,
        eval_agent=eval_agent,
        eval_env=eval_env,
        actors=actors,
        actor_envs=actor_envs,
        data_queue=data_queue,
        checkpoint=checkpoint,
        csv_file=FLAGS.results_csv_path,
        use_tensorboard=FLAGS.use_tensorboard,
        tag=FLAGS.tag,
        debug_screenshots_interval=FLAGS.debug_screenshots_interval,
    )


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)
