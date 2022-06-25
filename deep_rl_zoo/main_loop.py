# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
#
# The functions 'run_loop' has been modified
# by The Deep RL Zoo Authors to support gym environment
# without DeepMin's dm.env wrapper.
#
# ==============================================================================
"""Training loops for Deep RL Zoo."""
from typing import Iterable, List, Tuple, Text, Mapping, Union
import itertools
import collections
import sys
import time
import signal
import queue
import multiprocessing
import threading
from absl import logging
import gym
import torch

# pylint: disable=import-error
import deep_rl_zoo.trackers as trackers_lib
import deep_rl_zoo.types as types_lib
from deep_rl_zoo.log import CsvWriter
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo import gym_env


def run_loop(
    agent: types_lib.Agent, env: gym.Env
) -> Iterable[Tuple[gym.Env, types_lib.TimeStep, types_lib.Agent, types_lib.Action]]:
    """Repeatedly alternates step calls on environment and agent.

    At time `t`, `t + 1` environment timesteps and `t + 1` agent steps have been
    seen in the current episode. `t` resets to `0` for the next episode.

    Args:
      agent: Agent to be run, has methods `step(timestep)` and `reset()`.
      env: Environment to run, has methods `step(action)` and `reset()`.

    Yields:
      Tuple `(env, timestep_t, agent, a_t)` where
        `a_t = agent.step(timestep_t)`.

    Raises:
        RuntimeError if the `agent` do not have a callable reset() and step() method.
    """

    if not (hasattr(agent, 'reset') and callable(getattr(agent, 'reset'))) or not (
        hasattr(agent, 'step') and callable(getattr(agent, 'step'))
    ):
        raise RuntimeError('Expect agent to have a callable reset(), and step() method.')

    while True:  # For each episode.
        t = 0
        agent.reset()
        # Think reset is a special 'aciton' the agent take, thus given us some 'zero' reward, and new s_t
        observation = env.reset()
        reward = 0.0
        done = False
        first_step = True

        while True:  # For each step in the current episode.
            timestep_t = types_lib.TimeStep(observation=observation, reward=reward, done=done, first=first_step)
            a_t = agent.step(timestep_t)
            yield env, timestep_t, agent, a_t

            t += 1
            a_tm1 = a_t

            observation, reward, done, _ = env.step(a_tm1)
            first_step = False

            if done:
                # if we don't add additonal step to agent, with our way of construct the run loop,
                # the done state and reward will never be added to replay
                timestep_t = types_lib.TimeStep(observation=observation, reward=reward, done=done, first=first_step)
                unused_a = agent.step(timestep_t)  # noqa: F841
                yield env, timestep_t, agent, None
                break


def run_some_steps(
    num_steps: int,
    agent: types_lib.Agent,
    env: gym.Env,
    tb_log_dir: str = None,
) -> Mapping[Text, float]:
    """Run some steps and return the statistics, this could be either training, evaluation, or testing steps.

    Args:
        max_episode_steps: maximum steps per episode.
        agent: agent to run, expect the agent to have step(), reset(), and a agent_name property.
        train_env: training environment.
        tb_log_dir: tensorboard log dir.

    Returns:
        A Dict contains statistics about the result.

    """
    seq = run_loop(agent, env)
    seq_truncated = itertools.islice(seq, num_steps)
    trackers = trackers_lib.make_default_trackers(run_log_dir=tb_log_dir)
    stats = trackers_lib.generate_statistics(trackers, seq_truncated)
    return stats


def run_single_thread_training_iterations(
    num_iterations: int,
    num_train_steps: int,
    num_eval_steps: int,
    network: Union[torch.nn.Module, List[torch.nn.Module]],
    train_agent: types_lib.Agent,
    train_env: gym.Env,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    checkpoint: PyTorchCheckpoint,
    csv_file: str,
    tensorboard: bool,
    tag: str = None,
) -> None:
    """Runs single-thread training and evaluation for N iterations.
    The same code structure is shared by most single-threaded DQN agents,
    and some policy gradients agents like reinforce, actor-critic.

    Args:
        num_iterations: number of iterations to run.
        num_train_steps: number of train steps to run, per iteration.
        num_eval_steps: number of evaluation steps to run, per iteration.
        network: the main network (or list of networks), we switch between train and eval mode.
        train_agent: training agent, expect the agent to have step(), reset(), and a agent_name property.
        train_env: training environment.
        eval_agent: evaluation agent.
        eval_env: evaluation environment.
        checkpoint: checkpoint object.
        csv_file: csv log file path and name.
        tensorboard: if True, use tensorboard to log the runs.
        tag: tensorboard run log tag.

    """

    networks = []
    if isinstance(network, List):
        networks = network
    elif isinstance(network, torch.nn.Module):
        networks.append(network)

    # Create log file writer.
    writer = CsvWriter(csv_file)

    # Tensorboard log dir prefix.
    train_tb_log_prefix = get_tb_log_prefix(train_env.spec.id, train_agent.agent_name, tag, 'train')

    state = checkpoint.state
    # Start training
    while state.iteration < num_iterations:
        # Set netwrok in train mode.
        for net in networks:
            net.train()

        logging.info(f'Training iteration {state.iteration}')
        train_tb_log_dir = f'{train_tb_log_prefix}-{state.iteration}' if tensorboard else None

        # Run training steps.
        train_stats = run_some_steps(num_train_steps, train_agent, train_env, train_tb_log_dir)

        checkpoint.save()

        # Logging training statistics.
        log_output = [
            ('iteration', state.iteration, '%3d'),
            ('train_step', num_eval_steps, '%5d'),
            ('train_episode_return', train_stats['episode_return'], '%2.2f'),
            ('train_num_episodes', train_stats['num_episodes'], '%3d'),
            ('train_step_rate', train_stats['step_rate'], '%4.0f'),
            ('train_duration', train_stats['duration'], '%.2f'),
        ]

        if num_eval_steps > 0 and eval_agent is not None and eval_env is not None:
            eval_tb_log_prefix = get_tb_log_prefix(eval_env.spec.id, eval_agent.agent_name, tag, 'eval')

            # Set netwrok in eval mode.
            for net in networks:
                net.eval()

            logging.info(f'Evaluation iteration {state.iteration}')
            eval_tb_log_dir = f'{eval_tb_log_prefix}-{state.iteration}' if tensorboard else None

            # Run some evaluation steps.
            eval_stats = run_some_steps(num_eval_steps, eval_agent, eval_env, eval_tb_log_dir)

            # Logging evaluation statistics.
            eval_output = [
                ('eval_episode_return', eval_stats['episode_return'], '% 2.2f'),
                ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),
                ('eval_step_rate', eval_stats['step_rate'], '%4.0f'),
                ('eval_duration', eval_stats['duration'], '%.2f'),
            ]
            log_output.extend(eval_output)

        log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
        logging.info(log_output_str)
        writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
        state.iteration += 1
    writer.close()


def run_parallel_training_iterations(
    num_iterations: int,
    num_train_steps: int,
    num_eval_steps: int,
    network: torch.nn.Module,
    learner_agent: types_lib.Agent,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    actors: List[types_lib.Agent],
    actor_envs: List[gym.Env],
    data_queue: multiprocessing.Queue,
    checkpoint: PyTorchCheckpoint,
    csv_file: str,
    tensorboard: bool,
    tag: str = None,
) -> None:
    """Run parallel traning with multiple actors processes, single learner process.

    Args:
        num_iterations: number of iterations to run.
        num_train_steps: number of train steps to run, per iteration.
        num_eval_steps: number of evaluation steps to run, per iteration.
        network: the main network, we switch between train and eval mode.
        learner_agent: learner agent, expect the agent to have run_train_loop() method.
        eval_agent: evaluation agent.
        eval_env: evaluation environment.
        actors: list of actor instances we wish to run.
        actor_envs: list of gym.Env for each actor to run.
        data_queue: a multiprocessing.Queue used to send transitions from actors to learner.
        checkpoint: checkpoint object.
        csv_file: csv log file path and name.
        tensorboard: if True, use tensorboard to log the runs.
        tag: tensorboard run log tag.

    Raises:
        RuntimeError if the `learner_agent` do not have a callable run_train_loop().
    """
    if not (hasattr(learner_agent, 'run_train_loop') and callable(getattr(learner_agent, 'run_train_loop'))):
        raise RuntimeError('Expect learner_agent to have a callable run_train_loop() method.')

    # Create shared iteration count and start, end training event.
    # start_iteration_event is used to signaling actors to run one training iteration,
    # stop_event is used to signaling actors the end of training session.
    # The start_iteration_event and stop_event are only set by the main process.
    iteration_count = multiprocessing.Value('i', 0)
    start_iteration_event = multiprocessing.Event()
    stop_event = multiprocessing.Event()

    # To get training statistics from each actor and the learner. We use a single writer to write to csv file.
    log_queue = multiprocessing.SimpleQueue()

    # Run learner train loop on a new thread.
    learner = threading.Thread(
        target=run_learner,
        args=(
            num_iterations,
            num_eval_steps,
            network,
            learner_agent,
            eval_agent,
            eval_env,
            log_queue,
            iteration_count,
            start_iteration_event,
            stop_event,
            checkpoint,
            tensorboard,
            tag,
        ),
    )
    learner.start()

    # Start logging on a new thread, since it's very light-weight task.
    logger = threading.Thread(
        target=run_logger,
        args=(log_queue, csv_file),
    )
    logger.start()

    # Create and start actor processes once, this will preserve actor's internal state like steps etc.
    # Tensorboard log dir prefix. Only log to tensorboard for first and last actors.
    actor_tb_log_prefixs = [None for _ in range(len(actors))]
    if tensorboard:
        actor_tb_log_prefixs[0] = get_tb_log_prefix(actor_envs[0].spec.id, actors[0].agent_name, tag, 'train')
        actor_tb_log_prefixs[-1] = get_tb_log_prefix(actor_envs[-1].spec.id, actors[-1].agent_name, tag, 'train')

    processes = []
    for actor, actor_env, tb_log_prefix in zip(actors, actor_envs, actor_tb_log_prefixs):
        p = multiprocessing.Process(
            target=run_actor,
            args=(
                actor,
                actor_env,
                data_queue,
                log_queue,
                num_train_steps,
                iteration_count,
                start_iteration_event,
                stop_event,
                tb_log_prefix,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all actor to be finished.
    for p in processes:
        p.join()
        p.close()

    learner.join()
    logger.join()

    # Close queue.
    data_queue.close()


def run_actor(
    actor: types_lib.Agent,
    actor_env: gym.Env,
    data_queue: multiprocessing.Queue,
    log_queue: multiprocessing.SimpleQueue,
    num_train_steps: int,
    iteration_count: multiprocessing.Value,
    start_iteration_event: multiprocessing.Event,
    stop_event: multiprocessing.Event,
    tb_log_prefix: str = None,
) -> None:

    """
    Run actor process for as long as required, only terminate if the `stop_event` is set to True.
    Which is set by the main process.
    Actors will wait for the `start_iteration_event` signal to start run actual training session.
    The actor whoever finished the iteration first will reset `start_iteration_event` to False,
    so it does not run into a loop that is out of control.

    Args:
        actor: the actor to run.
        actor_env: environment for the actor instance.
        data_queue: multiprocessing.Queue used for transfering data from actor to learner.
        log_queue: multiprocessing.SimpleQueue used for transfering training statistics from actor,
            this is only for write to csv file, not for tensorboard.
        num_train_steps: number of training steps to run for one iteration.
        iteration: a counter which is updated by the main process.
        start_iteration_event: start training signal, set by the main process, clear by actor.
        stop_event: end traning signal.
        tb_log_prefix: tensorboard run log dir prefix.

    """
    # Initialize logging.
    init_absl_logging()

    # Listen to signals to exit process.
    handle_exit_signal()

    while not stop_event.is_set():
        # Wait for start training event signal, which is set by the main process.
        if not start_iteration_event.is_set():
            continue

        logging.info(f'Starting {actor.agent_name} ...')
        iteration = iteration_count.value
        train_tb_log_dir = f'{tb_log_prefix}-{iteration}' if tb_log_prefix is not None else None

        # Run training steps.
        train_stats = run_some_steps(num_train_steps, actor, actor_env, train_tb_log_dir)

        # Mark work done to avoid infinite loop in learner `run_train_loop`,
        # also possible multiprocessing.Queue deadlock.
        data_queue.put('PROCESS_DONE')

        # Whoever finished one iteration first will clear the start traning event.
        if start_iteration_event.is_set():
            start_iteration_event.clear()

        # Logging statistics after training finished
        log_output = [
            ('iteration', iteration, '%3d'),
            ('type', actor.agent_name, '%2s'),
            ('step', num_train_steps, '%5d'),
            ('episode_return', train_stats['episode_return'], '% 2.2f'),
            ('num_episodes', train_stats['num_episodes'], '%3d'),
            ('step_rate', train_stats['step_rate'], '%4.0f'),
            ('duration', train_stats['duration'], '%.2f'),
        ]

        # Add traning statistics to log queue, so the logger process can write to csv file.
        log_queue.put(log_output)


def run_learner(
    num_iterations: int,
    num_eval_steps: int,
    network: Union[torch.nn.Module, List[torch.nn.Module]],
    learner_agent: types_lib.Agent,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    log_queue: multiprocessing.SimpleQueue,
    iteration_count: multiprocessing.Value,
    start_iteration_event: multiprocessing.Event,
    stop_event: multiprocessing.Event,
    checkpoint: PyTorchCheckpoint,
    tensorboard: bool,
    tag: str = None,
) -> None:
    """Run learner for N iterations.

    At the begining of every iteration, learner will set the `start_iteration_event` to True, to signal actors to start training.
    The actor whoever finished the iteration first will reset `start_iteration_event` to False.
    Then on the next iteration, the learner will set the `start_iteration_event` to True.

    Args:
        num_iterations: number of iterations to run.
        num_eval_steps: number of evaluation steps to run, per iteration.
        network: the main network, we switch between train and eval mode.
        learner_agent: learner agent, expect the agent to have run_train_loop() method.
        eval_agent: evaluation agent.
        eval_env: evaluation environment.
        log_queue: a multiprocessing.SimpleQueue used send evaluation statistics to logger.
        start_iteration_event: a multiprocessing.Event signal to actors for start training.
        checkpoint: checkpoint object.
        tensorboard: if True, use tensorboard to log the runs.
        tag: tensorboard run log tag.

    """
    networks = []
    if isinstance(network, List):
        networks = network
    elif isinstance(network, torch.nn.Module):
        networks.append(network)

    state = checkpoint.state

    while state.iteration < num_iterations:
        # Set netwrok in train mode.
        for net in networks:
            net.train()

        logging.info(f'Training iteration {state.iteration}')
        logging.info(f'Starting {learner_agent.agent_name} ...')

        # Set start training event.
        start_iteration_event.set()

        # Start to run learner training loop for one iteration.
        learner_agent.run_train_loop()

        start_iteration_event.clear()
        checkpoint.save()

        # Run evaluation steps.
        if num_eval_steps > 0 and eval_agent is not None and eval_env is not None:
            eval_tb_log_prefix = None
            if tensorboard:
                eval_tb_log_prefix = get_tb_log_prefix(eval_env.spec.id, eval_agent.agent_name, tag, 'eval')

            # Set netwrok in eval mode.
            for net in networks:
                net.eval()

            logging.info(f'Evaluation iteration {state.iteration}')
            eval_tb_log_dir = f'{eval_tb_log_prefix}-{state.iteration}' if eval_tb_log_prefix is not None else None

            # Run some evaluation steps.
            eval_stats = run_some_steps(num_eval_steps, eval_agent, eval_env, eval_tb_log_dir)

            # Logging evaluation statistics
            log_output = [
                ('iteration', state.iteration, '%3d'),
                ('type', 'evaluation', '%3s'),
                ('step', num_eval_steps, '%5d'),
                ('episode_return', eval_stats['episode_return'], '%2.2f'),
                ('num_episodes', eval_stats['num_episodes'], '%3d'),
                ('step_rate', eval_stats['step_rate'], '%4.0f'),
                ('duration', eval_stats['duration'], '%.2f'),
            ]
            log_queue.put(log_output)

        state.iteration += 1

        # Update shared iteration count.
        iteration_count.value += 1

        time.sleep(5)

    # Signal actors training session ended.
    stop_event.set()
    # Signal logger training session ended, using stop_event seems not working.
    log_queue.put('PROCESS_DONE')


def run_logger(log_queue: multiprocessing.SimpleQueue, csv_file: str):
    """Run logger and csv file writer on a seperate thread,
    this is only for training/evaluation statistics."""

    # Create log file writer.
    writer = CsvWriter(csv_file)

    while True:
        try:
            log_output = log_queue.get()
            if log_output == 'PROCESS_DONE':
                break
            log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
            logging.info(log_output_str)
            writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
        except queue.Empty:
            pass
        except EOFError:
            pass


def run_test_iterations(
    num_iterations: int,
    num_eval_steps: int,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    tensorboard: bool,
    recording_video_dir: str = None,
):
    """Testing an agent restored from checkpoint.

    Args:
        num_iterations: number of iterations to run.
        num_eval_steps: number of evaluation steps, per iteration.
        eval_agent: evaluation agent, expect the agent has step(), reset(), and agent_name property.
        eval_env: evaluation environment.
        tensorboard: if True, use tensorboard to log the runs.
        recording_video_dir: folder to store agent self-play video for one episode.
    """

    # Tensorboard log dir prefix.
    test_tb_log_prefix = get_tb_log_prefix(eval_env.spec.id, eval_agent.agent_name, None, 'test')

    iteration = 0
    while iteration < num_iterations:
        logging.info(f'Testing iteration {iteration}')
        eval_tb_log_dir = f'{test_tb_log_prefix}-{iteration}' if tensorboard else None

        # Run some testing steps.
        eval_stats = run_some_steps(num_eval_steps, eval_agent, eval_env, eval_tb_log_dir)

        # Logging testing statistics.
        log_output = [
            ('iteration', iteration, '%3d'),
            ('step', iteration * num_eval_steps, '%5d'),
            ('episode_return', eval_stats['episode_return'], '% 2.2f'),
            ('num_episodes', eval_stats['num_episodes'], '%3d'),
            ('step_rate', eval_stats['step_rate'], '%4.0f'),
            ('duration', eval_stats['duration'], '%.2f'),
        ]

        log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
        logging.info(log_output_str)
        iteration += 1

    if recording_video_dir is not None and recording_video_dir != '':
        gym_env.play_and_record_video(eval_agent, eval_env, recording_video_dir)


def get_tb_log_prefix(env_id: str, agent_name: str, tag: str, suffix: str) -> str:
    """Returns the composed tensorboard log prefix,
    which is in the format {env_id}-{agent_name}-{tag}-{suffix}."""
    tb_log_prefix = f'{env_id}-{agent_name}'
    if tag is not None and tag != '':
        tb_log_prefix += f'-{tag}'
    tb_log_prefix += f'-{suffix}'
    return tb_log_prefix


def init_absl_logging():
    """Initialize absl.logging when run the process without app.run()"""
    logging._warn_preinit_stderr = 0  # pylint: disable=protected-access
    logging.set_verbosity(logging.INFO)
    logging.use_absl_handler()


def handle_exit_signal():
    """Listen to exit signal like ctrl-c or kill from os and try to exit the process forcefully."""

    def shutdown(signal_code, frame):
        del frame
        logging.info(
            f'Received signal {signal_code}: terminating process...',
        )
        sys.exit(128 + signal_code)

    # Listen to signals to exit process.
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
