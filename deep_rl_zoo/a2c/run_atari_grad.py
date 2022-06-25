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
"""A A2C agent training on Atari.

Specifically:
    * Actors sample batch of transitions to calculate loss, but not optimization step.
    * Actors collects local gradients, and send to master through multiprocessing.Queue.
    * Learner will aggregates batch of gradients then do the optimization step.
    * Learner update policy network weights for workers (shared_memory).

Note only supports training on single machine.

From the paper "Asynchronous Methods for Deep Reinforcement Learning"
https://arxiv.org/abs/1602.01783

Synchronous, Deterministic variant of A3C
https://openai.com/blog/baselines-acktr-a2c/.

"""
from absl import app
from absl import flags
from absl import logging
import multiprocessing
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.policy import ActorCriticConvNet
from deep_rl_zoo.a2c import agent_grad as agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
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
flags.DEFINE_integer('num_actors', 8, 'Number of worker processes to use.')
flags.DEFINE_bool('compress_gradient', True, 'Actor process to compress the local gradients before put on queue, default on.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 40.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('entropy_coef', 0.001, 'Coefficient for the entropy loss.')
flags.DEFINE_float('baseline_coef', 0.5, 'Coefficient for the state-value loss.')
flags.DEFINE_integer('n_step', 3, 'TD n-step bootstrap.')
flags.DEFINE_integer('batch_size', 32, 'Accumulate batch size transitions before do learning.')
flags.DEFINE_integer('num_iterations', 20, 'Number of iterations to run.')
flags.DEFINE_integer('num_train_steps', int(1e6), 'Number of training steps per iteration.')
flags.DEFINE_integer('num_eval_steps', int(2e5), 'Number of evaluation steps per iteration.')
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps per episode. 0 means no limit.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', 'logs/a2c_grad_atari_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_path', 'checkpoints/a2c_grad', 'Path for checkpoint directory.')


def main(argv):
    """Trains A2C agent on Atari."""
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
            terminal_on_life_loss=True,
        )

    eval_env = environment_builder()

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', eval_env.action_space.n)
    logging.info('Observation spec: %s', eval_env.observation_space.shape)

    input_shape = (
        FLAGS.environment_frame_stack,
        FLAGS.environment_height,
        FLAGS.environment_width,
    )
    num_actions = eval_env.action_space.n

    # Test environment and state shape.
    obs = eval_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == input_shape

    # Create policy network and optimizer
    policy_network = ActorCriticConvNet(input_shape=input_shape, num_actions=num_actions)
    policy_network.share_memory()
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=FLAGS.learning_rate)

    # Test network output.
    s = torch.from_numpy(obs[None, ...]).float()
    network_output = policy_network(s)
    pi_logits = network_output.pi_logits
    baseline = network_output.baseline
    assert pi_logits.shape == (1, num_actions)
    assert baseline.shape == (1, 1)

    # Create global gradient queue
    gradient_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors)
    lock = multiprocessing.Lock()

    # Create gradient replay, which is just a simple list with some optimization to speed up gradient aggregation
    gradient_replay = replay_lib.GradientReplay(FLAGS.num_actors, policy_network, FLAGS.compress_gradient)

    # Create A2C gradient learner agent instance
    learner_agent = agent.Learner(
        lock=lock,
        gradient_queue=gradient_queue,
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        gradient_replay=gradient_replay,
        num_actors=FLAGS.num_actors,
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
            lock=lock,
            gradient_queue=gradient_queue,
            policy_network=policy_network,
            transition_accumulator=replay_lib.NStepTransitionAccumulator(n=FLAGS.n_step, discount=FLAGS.discount),
            discount=FLAGS.discount,
            n_step=FLAGS.n_step,
            batch_size=FLAGS.batch_size,
            compress_gradient=FLAGS.compress_gradient,
            entropy_coef=FLAGS.entropy_coef,
            baseline_coef=FLAGS.baseline_coef,
            clip_grad=FLAGS.clip_grad,
            max_grad_norm=FLAGS.max_grad_norm,
            device=actor_devices[i],
        )
        for i in range(FLAGS.num_actors)
    ]

    # Create evaluation agent instance
    eval_agent = greedy_actors.PolicyGreedyActor(
        network=policy_network,
        device=runtime_device,
        name='A2C-GRAD',
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
        data_queue=gradient_queue,
        checkpoint=checkpoint,
        csv_file=FLAGS.results_csv_path,
        tensorboard=FLAGS.tensorboard,
        tag=FLAGS.tag,
    )


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)
