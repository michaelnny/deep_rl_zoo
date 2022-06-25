# Deep RL Zoo
A collection of Deep RL algorithms implemented with PyTorch to solve classic control problems like CartPole, LunarLander, and Atari games.
This repos is based on DeepMind's [DQN Zoo](https://github.com/deepmind/dqn_zoo), we adapted the code and migrated to PyTorch, also implemented some new algorithms.


## Environment and requirements
* Python        3.9.12
* pip           22.0.3
* PyTorch       1.11.0
* openAI Gym    0.24.1
* tensorboard   2.8.0
* numpy         1.22.2


## Note
* Only support episodic environment with discrete action space.
* Focus on study and implementation for each algorithms, rather than create a standard library.
* Some code might not be optimal, especially the parts involving Python Multiprocessing, as speed of code execution is not our main focus.
* Try our best to replicate the implementation for the original paper, but may change some hyper-parameters to support low-end machine and speed up training.
* The hyper-parameters and network architectures are not fine-tuned.
* Most agents have been fully tested on classic control problems like CartPole, LunarLander on M1 Mac (CPU only), we also run some light tests on Unbuntu 18.04 with a single Nvidia RTX 2080Ti GPU.
* For Atari games, we only selectively run tests on Pong for some agents, there are agents (marked below) not tested on Atari due to lack of access to powerful machine and GPUs.
* We can't guarantee its bug free.


## Implemented algorithms (for discrete action space only)
### Policy gradient algorithms
<!-- mdformat off(for readability) -->
| Directory            | Reference Paper                                                                                                               | Note |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---- |
| `reinforce`          | [Policy Gradient Methods for RL](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)   | *    |
| `reinforce_baseline` | [Policy Gradient Methods for RL](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)   | *    |
| `actor_critic`       | [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)          | *    |
| `a2c`                | [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)                                      | P    |
|                      | [synchronous, deterministic variant of A3C](https://openai.com/blog/baselines-acktr-a2c/)                                     |      |
| `ppo`                | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)                                                   | P *  |
| `sac`                | [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning](https://arxiv.org/abs/1801.01290)                 | P *  |
|                      | [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207)                                            |      |
| `ppo_icm`            | [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)                                | P *  |
| `ppo_rnd`            | [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)                                                | P *  |
| `impala`             | [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561) | P    |
<!-- mdformat on -->

### Deep Q learning algorithms
<!-- mdformat off(for readability) -->
| Directory            | Reference Paper                                                                                               | Note |
| -------------------- | ------------------------------------------------------------------------------------------------------------- | ---- |
| `dqn`                | [Human Level Control Through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236)        |      |
| `double_dqn`         | [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)                        |      |
| `prioritized_dqn`    | [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)                                             |      |
| `rainbow`            | [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)            |      |
| `drqn`               | [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)                   | *    |
| `r2d2`               | [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/pdf?id=r1lyTjAqYX) | P *  |
| `ngu`                | [Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/abs/2002.06038)                   | P *  |
| `agent57`            | [Agent57: Outperforming the Atari Human Benchmark](https://arxiv.org/pdf/2003.13350)                          | P *  |

<!-- mdformat on -->

### Distributional Q learning algorithms
<!-- mdformat off(for readability) -->
| Directory            | Reference Paper                                                                                               | Note |
| -------------------- | ------------------------------------------------------------------------------------------------------------- | ---- |
| `c51_dqn`            | [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)                    |      |
| `qr_dqn`             | [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)            | *    |
| `iqn`                | [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)      | *    |

<!-- mdformat on -->
Notes:
* `P` means support parallel training with multiple actors and a single learner on a single machine.
* `*` means not tested on Atari games due to lack of access to powerful machine and GPUs.


## Code structure
*   `deep_rl_zoo` directory contains all the source code for different algorithms, specificlly:
    *   Each directory contains a algorithm, more specifically:
        - `agent.py` module contains an agent class that includes `reset()`, `step()` methods,
        for agent that supports parallel training, we have `Actor` and `Learner` classes for the specific agent.
        - `run_classic.py` module use simple MLP network to solve classic problems like CartPole, MountainCar, and LunarLander.
        - `run_atari.py` module use Conv2d networks to solve Atari games, the default environment_name is set to Pong.
        - `eval_agent.py` module evaluate trained agents by using a greedy actor and loading model state from checkpoint file,
        you can run testing on both classic problems like CartPole, MountainCar, LunarLander, and Atari games.
    *   `main_loop.py` module contains functions to run single thread and parallel traning loops.
    *   `networks` directory contains both policy networks and q networks used by the agents.
    *   `trackers.py` module is used to accumulating statistics during training and testing/evaluation,
        it also writes log to Tensorboard if desired.
    *   `replay.py` module contains functions and classes relating to experience replay.
    *   `value_learning.py` module contains functions to calculate q learning loss.
    *   `policy_gradient.py` module contains functions to calculate policy gradient loss.
    *   `gym_env.py` module contains components for standard Atari environment preprocessing.
    *   `greedy_actors.py` module contains all the greedy actors for testing/evaluation.
        for example `EpsilonGreedyActor` for DQN agents, `PolicyGreedyActor` for general policy gradient agents.
*   `tests` directory contains all the code for unit testing and end-to-end testing.
*   `screenshots` directory contains images of Tensorboard statistics for some of the agent.

## Quick start

### Install required packages on Mac
```
# install homebrew, skip this step if already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# upgrade pip
python3 -m pip install --upgrade pip setuptools

# install swig which is required for box-2d
brew install swig

# install ffmpeg for recording agent self-play
brew install ffmpeg

# install snappy for compress numpy.array on M1 mac
brew install snappy
CPPFLAGS="-I/opt/homebrew/include -L/opt/homebrew/lib" pip3 install python-snappy

pip3 install -r requirements.txt
```

### Install required packages on Ubuntu Linux
```
# install swig which is required for box-2d
sudo apt install swig

# install ffmpeg for recording agent self-play
sudo apt-get install ffmpeg

# upgrade pip
python3 -m pip install --upgrade pip setuptools

pip3 install -r requirements.txt
```


## Train agents

### CartPole, LunarLander, and MountainCar
By default, we have the following settings for all `run_classic.py` file:

* `num_iterations: 2`
* `num_train_steps: 5e5`
* `num_eval_steps: 2e5`

For some problem like LunarLander and MountainCar, you may need to increase the `num_train_steps`.

#### Note
* For some agents (like advanced DQN agents, most of the policy gradient agents), it's impossible to solve MountainCar due to the nature of the problem (sparse reward).
* For some of the advanced agents (like NGU, Agent57) will only converge on LunarLander, to make it work on CartPole, you may need to fine-tune the hyper-parameters.

```
python3 -m deep_rl_zoo.dqn.run_classic

python3 -m deep_rl_zoo.dqn.run_classic --environment_name=MountainCar-v0

python3 -m deep_rl_zoo.dqn.run_classic --environment_name=LunarLander-v2
```

### Atari environment
By default, we have the following settings for all `run_atari.py` file:

* `num_iterations: 20`
* `num_train_steps: 1e6`
* `num_eval_steps: 2e5`

For Atari, we omit the need to include 'NoFrameskip' and version in the `environment_name` args, as it will be handled by `create_atari_environment` in the `gym_env.py` module.
By default, it uses `NoFrameskip-v4` for the specified game.

We don't scale the observation with the atari wrappers BEFORE store the states into experience replay, as that will require 4-5x more RAM.
We only scale the states inside the model.forward() method.
As a reference, consider tuple (Stm1, Atm1, Rt, St, Done) as one sample and we stack 4 frames, and we store 100000 samples in experience replay.
If we do scale before store into replay, it will allocate ~14GB of RAM. But when don't scale the states, it only requires ~3GB of RAM.

#### Note
* Due to hardware limitation, for DQN (and the enhancements like double Q, rainbow, IQN, etc.), we set the maximum experience replay size to 200000 instead of 1000000.
* To speed up training on Atari games with DQN (and the enhancements like double Q, rainbow, IQN, etc.), we use 2500 instead of 10000 as interval to update target Q network.
* We experience that, for DQN, using the following configuration will speed up training on Pong (1 million env steps), but may not work on other games.
    * `--exploration_epsilon_decay_step=500000`
    * `--min_replay_size=10000`
    * `--replay_capacity=100000`
    * `--target_network_update_frequency=2500`
* As a reference, when using double Q learning, it's best to increase the interval to 2-3x naive DQN when update target Q network.

```
python3 -m deep_rl_zoo.dqn.run_atari --environment_name=Pong --num_iterations=1 --num_train_steps=1000000 --target_network_update_frequency=2500 --min_replay_size=10000 --target_network_update_frequency=2500--replay_capacity=100000 --exploration_epsilon_decay_step=50000

# Train DQN on Breakout may take 20-50 million env steps, and the hyper-parameters are not tuned.
python3 -m deep_rl_zoo.dqn.run_atari --environment_name=Breakout
```

### Multiple actors (on single machine)
When running multiple actors on GPU, watching out for possible CUDA OUT OF MEMORY error.

```
python3 -m deep_rl_zoo.a2c.run_classic --num_actors=8

python3 -m deep_rl_zoo.impala.run_atari --num_actors=8
```

## Test agents
Before you test the agent, make sure you have a valid checkpoint file for the specific agent and environment.
By valid checkpoint file, we mean saved by either the `run_classic.py` or `run_atari.py` module.
By default, it will record a single episode of agent's self-play at the `recordings` directory.

```
python3 -m deep_rl_zoo.dqn.eval_agent

python3 -m deep_rl_zoo.dqn.eval_agent --environment_name=MountainCar-v0

# load checkpoint file from a specific checkpoint file
python3 -m deep_rl_zoo.dqn.eval_agent --checkpoint_path=checkpoints/dqn/CartPole-v1_iteration_0.ckpt
```

## Monitoring performance and statistics with Tensorboard
By default, both training, evaluation, and testing will log to Tensorboard at the `runs` directory.
To disable this, use the option `--notensorboard`.

```
tensorboard --logdir=runs
```

The classes `TensorboardEpisodTracker`, `TensorboardStepRateTracker`, and `TensorboardAgentStatisticsTracker` for write to tensorboard is implemented in `trackers.py` module.

* we only write logs after episode terminates
* we separate training and evaluation logs
* the statistics are measured over env steps, or frames, if use frame_skip, it does't count the skipped frames
* for agents that support parallel training, only log the first and last actors, this is controlled by `run_parallel_training_iterations` in `main_loop.py` module

Here are the performance(env_steps) measurements available on Tensorboard:
* `episode_return` the last episode return
* `episode_steps` the last episode steps
* `num_episodes` how many episodes have been conducted
* `run_duration(minutes)` the duration (in minutes) since the start of the session
* `step_rate(second)` step pre seconds, pre actors

In addition, it'll log whatever is exposed in the `agent.statistics` such as train loss, learning rate, discount, updates etc.

### Add tags to Tensorboard
This could be handy if we want to compare different hyper parameter's performances
```
python3 -m deep_rl_zoo.impala.run_classic --use_lstm --learning_rate=0.001 --tag=LSTM-LR0.001
```

### Example of DQN agent (training on M1 Mac CPU)

#### CartPole
![DQN on CartPole](../main/screenshots/DQN_on_CartPole.png)

#### MountainCar
![DQN on MountainCar](../main/screenshots/DQN_on_MountainCar.png)

#### LunarLander
![DQN on LunarLander](../main/screenshots/DQN_on_LunarLander.png)

#### Pong
![DQN on Pong](../main/screenshots/DQN_on_Pong.png)

## Acknowledgments

### This project is based on the work of DeepMind's projects.
* [DeepMind DQN Zoo](http://github.com/deepmind/dqn_zoo) (for code strcture, replay, run_loops, trackers, and more.)
* [DeepMind RLax](https://github.com/deepmind/rlax) (for modules to calculate losses for all different algorithms)
* [DeepMind TRFL](https://github.com/deepmind/trfl) (for modules to calculate losses for all different algorithms)

### Other reference projects which have been very helpful when we build our project
* [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) (for basic policy gradient agents)
* [OpenAI Spinning Up](https://github.com/openai/spinningup) (for basic policy gradient agents)
* [SEED RL](https://github.com/google-research/seed_rl) (for IMPALA, R2D2 and more)
* [TorchBeast](https://github.com/facebookresearch/torchbeast) (for IMPALA)

## License

This project is licensed under the Apache License, Version 2.0 (the "License")
see the LICENSE file for details


## Citing our work

If you reference or use our project in your research, please cite:

```
@software{deep_rl_zoo2022github,
  title = {{Deep RL Zoo}: A collections of Deep RL algorithms implemented with PyTorch},
  author = {Michael Hu},
  url = {https://github.com/michaelnny/deep_rl_zoo},
  version = {1.0.0},
  year = {2022},
}
```
