Deep RL Zoo
=============================
A collection of Deep RL algorithms implemented with PyTorch to solve Atari games and classic control tasks like CartPole, LunarLander, and MountainCar.
This repo is based on DeepMind's [DQN Zoo](https://github.com/deepmind/dqn_zoo). We adapted the code to support PyTorch, and implemented some SOTA algorithms like PPO, IMPALA, R2D2, and Agent57.


# Content
- [Environment and Requirements](#environment-and-requirements)
- [Implemented Algorithms](#implemented-algorithms)
- [Code Structure](#code-structure)
- [Author's Notes](#authors-notes)
- [Quick Start](#quick-start)
- [Train Agents](#train-agents)
- [Evaluate Agents](#evaluate-agents)
- [Monitoring with Tensorboard](#monitoring-with-tensorboard)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Citing our work](#citing-our-work)


# Environment and Requirements
* Python        3.9.12
* pip           22.0.3
* PyTorch       1.11.0
* openAI Gym    0.25.2
* tensorboard   2.8.0
* numpy         1.22.2



# Implemented Algorithms
## Policy Gradient Algorithms
<!-- mdformat off(for readability) -->
| Directory            | Reference Paper                                                                                                               | Note |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---- |
| `reinforce`          | [Policy Gradient Methods for RL](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)   | *    |
| `reinforce_baseline` | [Policy Gradient Methods for RL](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)   | *    |
| `actor_critic`       | [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)          | *    |
| `a2c`                | [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) \| [synchronous, deterministic variant of A3C](https://openai.com/blog/baselines-acktr-a2c/)  | P    |
| `sac`                | [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning](https://arxiv.org/abs/1801.01290) \| [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207) | P *  |
| `ppo`                | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)                                                   | P    |
| `ppo_icm`            | [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)                                | P    |
| `ppo_rnd`            | [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)                                                | P    |
| `impala`             | [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561) | P    |
<!-- mdformat on -->


## Deep Q Learning Algorithms
<!-- mdformat off(for readability) -->
| Directory            | Reference Paper                                                                                               | Note |
| -------------------- | ------------------------------------------------------------------------------------------------------------- | ---- |
| `dqn`                | [Human Level Control Through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236)        |      |
| `double_dqn`         | [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)                        |      |
| `prioritized_dqn`    | [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)                                             |      |
| `drqn`               | [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)                   | *    |
| `r2d2`               | [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/pdf?id=r1lyTjAqYX) | P    |
| `ngu`                | [Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/abs/2002.06038)                   | P *  |
| `agent57`            | [Agent57: Outperforming the Atari Human Benchmark](https://arxiv.org/pdf/2003.13350)                          | P *   |

<!-- mdformat on -->


## Distributional Q Learning Algorithms
<!-- mdformat off(for readability) -->
| Directory            | Reference Paper                                                                                               | Note |
| -------------------- | ------------------------------------------------------------------------------------------------------------- | ---- |
| `c51_dqn`            | [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)                    |      |
| `rainbow`            | [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)            |      |
| `qr_dqn`             | [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)            |      |
| `iqn`                | [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)      |      |

<!-- mdformat on -->
**Notes**:
* `P` means support parallel training with multiple actors and a single learner, all running on a single machine.
* `*` means not fully tested on Atari games.

# Code Structure
*   `deep_rl_zoo` directory contains all the source code for different algorithms:
    *   each directory contains a algorithm, more specifically:
        - `agent.py` module contains an agent class that includes `reset()`, `step()` methods,
        for agent that supports parallel training, we have `Actor` and `Learner` classes for the specific agent.
        - `run_classic.py` module use simple MLP network to solve classic problems like CartPole, MountainCar, and LunarLander.
        - `run_atari.py` module use Conv2d networks to solve Atari games, the default environment_name is set to Pong.
        - `eval_agent.py` module evaluate trained agents by using a greedy actor and loading model state from checkpoint file,
        you can run testing on both classic problems like CartPole, MountainCar, LunarLander, and Atari games.
    *   `main_loop.py` module contains functions run single thread and parallel training loops,
        it also contains the `run_env_loop` function where the agent interaction with the environment.
    *   `networks` directory contains both policy networks and q networks used by the agents.
    *   `trackers.py` module is used to accumulating statistics during training and testing/evaluation,
        it also writes log to Tensorboard if desired.
    *   `replay.py` module contains functions and classes relating to experience replay.
    *   `value_learning.py` module contains functions to calculate q learning loss.
    *   `policy_gradient.py` module contains functions to calculate policy gradient loss.
    *   `gym_env.py` module contains components for standard Atari environment preprocessing.
    *   `greedy_actors.py` module contains all the greedy actors for testing/evaluation.
        for example `EpsilonGreedyActor` for DQN agents, `PolicyGreedyActor` for general policy gradient agents.
*   `tests` directory contains the code for unit and end-to-end testing.
*   `screenshots` directory contains images of Tensorboard statistics for some of the test runs.


# Author's Notes
* Only support episodic environment with discrete action space (except PPO which also supports continuous action space).
* Focus on study and implementation for each algorithms, rather than create a standard library.
* Some code might not be optimal, especially the parts involving Python Multiprocessing, as speed of code execution is not our main focus.
* Try our best to replicate the implementation for the original paper, but may change some hyper-parameters to support low budget setup.
* The hyper-parameters and network architectures are not fine-tuned.
* All agents have been fully tested on classic control tasks like CartPole, LunarLander on M1 Mac (CPU only), we also run some light tests on Ubuntu 18.04 with a single Nvidia RTX 2080Ti GPU.
* For Atari games, we only use Pong or Breakout for most of the agents, and we stop training once the agent have made some progress.
* We can't guarantee it's bug free.


# Quick Start

## Install required packages on openSUSE 15 Tumbleweed Linux
```
# install required dev packages
sudo zypper install gcc gcc-c++ python3-devel

# install swig which is required for box-2d
sudo zypper install swig

# install ffmpeg for recording agent self-play
sudo zypper install ffmpeg

# upgrade pip
python3 -m pip install --upgrade pip setuptools

pip3 install -r requirements.txt

# optional, install pre-commit and hooks
pip3 install pre-commit

pre-commit install
```

## Install required packages on Mac
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

# optional, install pre-commit and hooks
pip3 install pre-commit

pre-commit install
```

## Install required packages on Ubuntu Linux
```
# install swig which is required for box-2d
sudo apt install swig

# install ffmpeg for recording agent self-play
sudo apt-get install ffmpeg

# upgrade pip
python3 -m pip install --upgrade pip setuptools

pip3 install -r requirements.txt

# optional, install pre-commit and hooks
pip3 install pre-commit

pre-commit install
```


# Train Agents

## Classic Control Tasks
* We maintain a list of environment names at `gym_env.py` module, by default it contains ```['CartPole-v1', 'LunarLander-v2', 'MountainCar-v0', 'Acrobot-v1']```.
* For some agents (like advanced DQN agents, most of the policy gradient agents except agents using curiosity-driven exploration), it's impossible to solve MountainCar due to the nature of the problem (sparse reward).

To run a agent on classic control problem, use the following command, replace the <agent_name> with the sub-directory name.
```
python3 -m deep_rl_zoo.<agent_name>.run_classic

# example of running DQN agents
python3 -m deep_rl_zoo.dqn.run_classic --environment_name=MountainCar-v0

python3 -m deep_rl_zoo.dqn.run_classic --environment_name=LunarLander-v2
```

## Atari games
* By default, we uses gym `NoFrameskip-v4` for Atari game, and we omit the need to include 'NoFrameskip' and version in the `environment_name` args, as it will be handled by `create_atari_environment` in the `gym_env.py` module.
* We don't scale the images before store into experience replay, as that will require 4-5x more RAM, we only scale them inside the model.forward() method.

To run a agent on Atari game, use the following command, replace the <agent_name> with the sub-directory name.
```
python3 -m deep_rl_zoo.<agent_name>.run_atari

# example of running DQN on Atari Pong and Breakout
python3 -m deep_rl_zoo.dqn.run_atari --environment_name=Pong

python3 -m deep_rl_zoo.dqn.run_atari --environment_name=Breakout
```

## Training with multiple actors and single learner (on the same machine)
For agents that support parallel training, we can adjust the parameter `num_actors` to specify how many actors to run.
When running multiple actors on GPU, watching out for possible CUDA OUT OF MEMORY error.

```
python3 -m deep_rl_zoo.a2c.run_classic --num_actors=8
```

Notice the code DOES NOT SUPPORT running on multiple GPUs out of the box, you can try to adapt the code in either `run_classic.py` or `run_atari.py` modules to support your needs, but there's no guarantee it will work.
```
# Change here to map to dedicated device if have multiple GPUs
actor_devices = [runtime_device] * FLAGS.num_actors
```

# Evaluate Agents
Before you run the eval_agent module, make sure you have a valid checkpoint file for the specific agent and environment.
By default, it will record a video of agent self-play at the `recordings` directory.

To run a agent on Atari game, use the following command, replace the <agent_name> with the sub-directory name.
```
python3 -m deep_rl_zoo.<agent_name>.eval_agent

# Using pre-trained rainbow model on Pong
python3 -m deep_rl_zoo.rainbow.eval_agent --environment_name=Pong --load_checkpoint_file=saved_checkpoints/Rainbow_Pong_0.ckpt
```


# Monitoring with Tensorboard
By default, both training, evaluation, and testing will log to Tensorboard at the `runs` directory.
To disable this, use the option `--notensorboard`.

```
tensorboard --logdir=runs
```

The classes for write logs to Tensorboard is implemented in `trackers.py` module.

* to improve performance, we only write logs at end of episode
* we separate training and evaluation logs
* if algorithm support parallel training, we separate actor, learner logs
* for agents that support parallel training, only log the first and last actors, this is controlled by `run_parallel_training_iterations` in `main_loop.py` module

## Measurements available on Tensorboard
`performance(env_steps)`:
* the statistics are measured over env steps, or frames, if use frame_skip, it does't count the skipped frames
* `episode_return` the non-discounted sum of raw rewards of last episode
* `episode_steps` the last episode steps
* `num_episodes` how many episodes have been conducted
* `step_rate(second)` step per seconds, per actors
![Tensorboard performance](/screenshots/Rainbow_Pong_performance.png)

`agent_statistics(env_steps)`:
* the statistics are measured over env steps, or frames, if use frame_skip, it does't count the skipped frames
* it'll log whatever is exposed in the agent's `statistics` property such as train loss, learning rate, discount, updates etc.
* for algorithm support parallel training (multiple actors), this is only the statistics for the actors.
![Tensorboard agent_statistics](/screenshots/Rainbow_Pong_agent_statistics.png)

`learner_statistics(learner_steps)`:
* only available if the agent supports parallel training (multiple actors one learner)
* it'll log whatever is exposed in the learner's `statistics` property such as train loss, learning rate, discount, updates etc.
* to improve performance, it only logs every 100 learner steps
![Tensorboard learner_statistics](/screenshots/R2D2_CartPole_learner_statistics.png)

## Add tags to Tensorboard
This could be handy if we want to compare different hyper parameter's performances
```
python3 -m deep_rl_zoo.impala.run_classic --use_lstm --learning_rate=0.001 --tag=LSTM-LR0.001
```

## Debug with environment screenshots
This could be handy if we want to see what's happening during the training, we can set the `debug_screenshots_frequency` (measured over episode) to some value, and it'll add screenshots of the terminal state to Tensorboard. This should be used for debug only as it may use more resources and slows down the training process.

```
python3 -m deep_rl_zoo.r2d2.run_classic --environment_name=MountainCar-v0 --debug_screenshots_frequency=100
```
![Tensorboard debug screenshots](/screenshots/Tensorboard_debug_screenshots.png)


# Acknowledgments

## This project is based on the work of DeepMind's projects.
* [DeepMind DQN Zoo](http://github.com/deepmind/dqn_zoo) (for code strcture, replay, DQN agents, trackers, and more)
* [DeepMind RLax](https://github.com/deepmind/rlax) (for modules to calculate losses for all different algorithms)
* [DeepMind TRFL](https://github.com/deepmind/trfl) (for modules to calculate losses for all different algorithms)

## Other reference projects which have been very helpful when we build our project
* [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) (for basic policy gradient agents)
* [OpenAI Spinning Up](https://github.com/openai/spinningup) (for basic policy gradient agents)
* [SEED RL](https://github.com/google-research/seed_rl) (for IMPALA, R2D2 and more)
* [TorchBeast](https://github.com/facebookresearch/torchbeast) (for IMPALA)


# License

This project is licensed under the Apache License, Version 2.0 (the "License")
see the LICENSE file for details


# Citing our work

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
