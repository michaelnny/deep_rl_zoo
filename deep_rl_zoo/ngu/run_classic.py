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
"""A NGU agent training on classic games like LunarLander.

From the paper "Never Give Up: Learning Directed Exploration Strategies"
https://arxiv.org/abs/2002.06038.
"""

from absl import app
from absl import flags
from absl import logging
import multiprocessing
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.dqn import NguDqnMlpNet, NguDqnNetworkInputs
from deep_rl_zoo.networks.curiosity import RndMlpNet, NguEmbeddingMlpNet
from deep_rl_zoo.ngu import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'CartPole-v1', 'Classic game name like CartPole-v1, LunarLander-v2, MountainCar-v0.')
flags.DEFINE_integer('num_actors', 8, 'Number of actor processes to use.')
flags.DEFINE_integer('replay_capacity', 10000, 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 1000, 'Minimum replay size before learning starts.')
flags.DEFINE_bool('clip_grad', True, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 40.0, 'Max gradients norm when do gradients clip.')

flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate for adam.')
flags.DEFINE_float(
    'int_learning_rate', 0.0005, 'Intrinsic learning rate for adam, this is for embedding and RND predictor networks.'
)
flags.DEFINE_float('ext_discount', 0.997, 'Extrinsic reward discount rate.')
flags.DEFINE_float('int_discount', 0.99, 'Intrinsic reward discount rate.')
flags.DEFINE_integer('unroll_length', 15, 'Sequence of transitions to unroll before add to replay.')
flags.DEFINE_integer(
    'burn_in',
    0,
    'Sequence of transitions used to pass RNN before actual learning.'
    'The effective length of unrolls will be burn_in + unroll_length, '
    'two consecutive unrolls will overlap on burn_in steps.',
)
flags.DEFINE_integer('batch_size', 64, 'Batch size for learning.')
flags.DEFINE_float('policy_beta', 0.5, 'Scalar for the intrinsic reward scale.')
flags.DEFINE_integer('num_policies', 32, 'Number of directed policies to learn, scaled by intrinsic reward scale beta.')

flags.DEFINE_integer('episodic_memory_capacity', 1000, 'Maximum size of episodic memory.')
flags.DEFINE_integer('num_neighbors', 10, 'Number of K-nearest neighbors.')
flags.DEFINE_float('kernel_epsilon', 0.01, 'K-nearest neighbors kernel epsilon.')
flags.DEFINE_float('cluster_distance', 0.008, 'K-nearest neighbors custer distance.')
flags.DEFINE_float('max_similarity', 8.0, 'K-nearest neighbors custer distance.')

flags.DEFINE_float('retrace_lambda', 0.97, 'Lambda coefficient for retrace.')
flags.DEFINE_bool('transformed_retrace', True, 'Transformed retrace loss, default on.')

flags.DEFINE_float('priority_exponent', 0.9, 'Priotiry exponent used in prioritized replay.')
flags.DEFINE_float('importance_sampling_exponent', 0.0, 'Importance sampling exponent value.')
flags.DEFINE_float('uniform_sample_probability', 1e-3, 'Add some noise when sampling from the prioritized replay.')
flags.DEFINE_bool('normalize_weights', True, 'Normalize sampling weights in prioritized replay.')
flags.DEFINE_float('priority_eta', 0.9, 'Priotiry eta to mix the max and mean absolute TD errors.')

flags.DEFINE_integer('num_iterations', 2, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(5e5), 'Number of training steps per iteration.')
flags.DEFINE_integer('num_eval_steps', int(2e5), 'Number of evaluation steps per iteration.')
flags.DEFINE_integer(
    'target_network_update_frequency',
    50,
    'Number of learner online Q network updates before update target Q networks.',
)
flags.DEFINE_integer('actor_update_frequency', 100, 'The frequency (measured in actor steps) to update actor local Q network.')
flags.DEFINE_float('eval_exploration_epsilon', 0.001, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_integer(
    'debug_screenshots_frequency',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', 'logs/ngu_classic_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_path', 'checkpoints/ngu', 'Path for checkpoint directory.')


def main(argv):
    """Trains NGU agent on classic games."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member

    # Listen to signals to exit process.
    main_loop.handle_exit_signal()

    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Create environment.
    def environment_builder(random_int=0):
        return gym_env.create_classic_environment(env_name=FLAGS.environment_name, seed=FLAGS.seed + int(random_int))

    eval_env = environment_builder()

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', eval_env.action_space.n)
    logging.info('Observation spec: %s', eval_env.observation_space.shape)

    input_shape = eval_env.observation_space.shape[0]
    num_actions = eval_env.action_space.n

    # Test environment and state shape.
    obs = eval_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (input_shape,)

    # Create Q network for learner to optimize.
    network = NguDqnMlpNet(input_shape=input_shape, num_actions=num_actions, num_policies=FLAGS.num_policies)
    network.share_memory()
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate)

    # Create RND target and predictor networks.
    rnd_target_network = RndMlpNet(input_shape=input_shape)
    rnd_target_network.share_memory()
    rnd_predictor_network = RndMlpNet(input_shape=input_shape)
    rnd_predictor_network.share_memory()

    # Create embedding networks.
    embedding_network = NguEmbeddingMlpNet(input_shape=input_shape, num_actions=num_actions)
    embedding_network.share_memory()

    # Second Adam optimizer for embedding and RND predictor networks.
    intrinsic_optimizer = torch.optim.Adam(
        list(embedding_network.parameters()) + list(rnd_predictor_network.parameters()),
        lr=FLAGS.int_learning_rate,
    )

    # Test network output.
    x = NguDqnNetworkInputs(
        s_t=torch.from_numpy(obs[None, None, ...]).float(),
        a_tm1=torch.zeros(1, 1).long(),
        ext_r_t=torch.zeros(1, 1).float(),
        int_r_t=torch.zeros(1, 1).float(),
        policy_index=torch.zeros(1, 1).long(),
        hidden_s=network.get_initial_hidden_state(1),
    )
    network_output = network(x)
    q_values = network_output.q_values
    q_hidden_s = network_output.hidden_s

    assert q_values.shape == (1, 1, num_actions)
    assert len(q_hidden_s) == 2

    # Create prioritized transition replay, no importance_sampling_exponent decay
    importance_sampling_exponent = FLAGS.importance_sampling_exponent

    def importance_sampling_exponent_schedule(x):
        return importance_sampling_exponent

    replay = replay_lib.PrioritizedReplay(
        capacity=FLAGS.replay_capacity,
        structure=agent.TransitionStructure,
        priority_exponent=FLAGS.priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        uniform_sample_probability=FLAGS.uniform_sample_probability,
        normalize_weights=FLAGS.normalize_weights,
        random_state=random_state,
        time_major=True,
    )

    # Create queue shared between actors and learner
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors)

    # Create NGU learner instance
    learner_agent = agent.Learner(
        network=network,
        optimizer=optimizer,
        embedding_network=embedding_network,
        rnd_target_network=rnd_target_network,
        rnd_predictor_network=rnd_predictor_network,
        intrinsic_optimizer=intrinsic_optimizer,
        replay=replay,
        min_replay_size=FLAGS.min_replay_size,
        target_network_update_frequency=FLAGS.target_network_update_frequency,
        unroll_length=FLAGS.unroll_length,
        burn_in=FLAGS.burn_in,
        retrace_lambda=FLAGS.retrace_lambda,
        transformed_retrace=FLAGS.transformed_retrace,
        priority_eta=FLAGS.priority_eta,
        batch_size=FLAGS.batch_size,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
    )

    # Create actor environments, actor instances.
    actor_envs = [environment_builder(i) for i in range(FLAGS.num_actors)]
    # TODO map to dedicated device if have multiple GPUs
    actor_devices = [runtime_device] * FLAGS.num_actors

    # Each actor has it's own embedding and RND predictor networks,
    # because we don't want to update these network weights in the middle of an episode,
    # it will only update these networks at the begining of an episode.
    actors = [
        agent.Actor(
            rank=i,
            data_queue=data_queue,
            network=NguDqnMlpNet(input_shape=input_shape, num_actions=num_actions, num_policies=FLAGS.num_policies),
            learner_network=network,
            rnd_target_network=rnd_target_network,
            rnd_predictor_network=RndMlpNet(input_shape=input_shape),
            embedding_network=NguEmbeddingMlpNet(input_shape=input_shape, num_actions=num_actions),
            learner_rnd_predictor_network=rnd_predictor_network,
            learner_embedding_network=embedding_network,
            random_state=np.random.RandomState(FLAGS.seed + int(i)),  # pylint: disable=no-member
            ext_discount=FLAGS.ext_discount,
            int_discount=FLAGS.int_discount,
            num_policies=FLAGS.num_policies,
            policy_beta=FLAGS.policy_beta,
            episodic_memory_capacity=FLAGS.episodic_memory_capacity,
            num_neighbors=FLAGS.num_neighbors,
            kernel_epsilon=FLAGS.kernel_epsilon,
            cluster_distance=FLAGS.cluster_distance,
            max_similarity=FLAGS.max_similarity,
            num_actors=FLAGS.num_actors,
            num_actions=num_actions,
            unroll_length=FLAGS.unroll_length,
            burn_in=FLAGS.burn_in,
            actor_update_frequency=FLAGS.actor_update_frequency,
            device=actor_devices[i],
        )
        for i in range(FLAGS.num_actors)
    ]

    # Create evaluation agent instance
    eval_agent = greedy_actors.NguEpsilonGreedyActor(
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
    checkpoint = PyTorchCheckpoint(FLAGS.checkpoint_path)
    state = checkpoint.state
    state.environment_name = FLAGS.environment_name
    state.iteration = 0
    state.network = network
    state.rnd_target_network = rnd_target_network
    state.rnd_predictor_network = rnd_predictor_network
    state.embedding_network = embedding_network

    # Run parallel traning N iterations.
    main_loop.run_parallel_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        network=network,
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
