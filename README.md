# Deep RL Zoo
## A collection of Deep RL algorithms implemented with PyTorch

This repository contains a collection of Deep RL algorithms to solve openAI Gym problems like CartPole, LunarLander, and Atari games.


## Note
* Only support episodic problems with discrete action space.
* We focus on study the individual algorithm and implementation, rather than creating a library.
* We try to follow the original paper as close as possible for each implementation, but may change some hyper-parameters to speed up training.
* The hyper-parameters and network architectures are not fine-tuned.
* Most agents have been fully tested on classic control problems like CartPole, LunarLander on M1 Mac (CPU only), we also run some light tests on Unbuntu 18.04 with a single Nvidia RTX 2080Ti GPU.
* For Atari games, we only run partially tests on Pong, there are some agents are not tested on Atari due to lack of access to powerful machine and GPUs.
* We can not guarantee its bug free.


## Environment and requirements
* Python        3.9.12
* pip           22.0.3
* PyTorch       1.11.0
* openAI Gym    0.24.1
* tensorboard   2.8.0
* numpy         1.22.2


## Implemented algorithms (for discrete action space only)
### Policy gradient algorithms
<!-- mdformat off(for readability) -->

| Directory            | Reference Paper                                                                                                               | Note |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---- |
| `reinforce`          | [Policy Gradient Methods for RL](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)   | *    |
| `reinforce_baseline` | [Policy Gradient Methods for RL](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)   | *    |
| `actor_critic`       | [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)          |      |
| `a2c`                | [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)                                      | P    |
|                      | [synchronous, deterministic variant of A3C](https://openai.com/blog/baselines-acktr-a2c/)                                     |      |
| `ppo`                | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)                                                   | P    |
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
| `c51_dqn`            | [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)                    |      |
| `rainbow_dqn`        | [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)            |      |
| `qr_dqn`             | [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)            |      |
| `iqn`                | [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)      | *    |
| `drqn`               | [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)                   | *    |
| `r2d2`               | [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/pdf?id=r1lyTjAqYX) | P    |
| `ngu`                | [Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/abs/2002.06038)                   | P *  |
| `agent57`            | [Agent57: Outperforming the Atari Human Benchmark](https://arxiv.org/pdf/2003.13350)                          | P *  |

<!-- mdformat on -->
Notes:
* `P` means support parallel training with multiple actors on a single machine.
* `*` means not tested on Atari games due to lack of access to powerful machine and GPUs.


## Code structure

*   Each directory contains a algorithm, more specifically:
    - `agent.py` in each agent directory contains an agent class that includes `reset()`, `step()` methods,
    for agent that supports parallel training, we have `Actor` and `Learner` classes for the specific agent.
    - `run_classic.py` use simple MLP network to solve classic problems like CartPole, MountainCar, and LunarLander.
    - `run_atari.py`  use Conv2d networks to solve Atari games, the default environment_name is set to Pong.
    - `eval_agent.py`  testing agents by using a greedy actor and loading model state from checkpoint file,
    you can run testing on both classic problems like CartPole, MountainCar, LunarLander, and Atari games
*   `main_loop.py` contains functions to run single thread and parallel traning loops.
*   `networks` contains both policy networks and q networks used by the agents.
*   `trackers.py` is used to accumulating statistics during training and testing/evaluation,
    it also writes log to Tensorboard if desired.
*   `replay.py` contains functions and classes relating to experience replay.
*   `value_learning.py` contains functions to calculate q learning loss.
*   `policy_gradient.py` contains functions to calculate policy gradient loss.
*   `gym_env.py` contains components for standard Atari environment preprocessing.
*   `greedy_actors.py` contains all the greedy actors for testing/evaluation.
    for example `EpsilonGreedyActor` for DQN agents, `PolicyGreedyActor` for general policy gradient agents.


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
* `num_train_steps: 2e5`
* `num_eval_steps: 1e5`

For some problem like LunarLander and MountainCar, you may need to increase the `num_train_steps`.

#### Note
* For some agents (like advanced DQN agents, most of the policy gradient agents), it's impossible to solve MountainCar due to the nature of the problem (sparse reward).
* For some of the advanced agents (like NGU, Agent57) will only converge on LunarLander, to make it work on CartPole, you may need to fine-tune the hyper-parameters. 

```
python3 -m deep_rl_zoo.dqn.run_classic

python3 -m deep_rl_zoo.dqn.run_classic --environment_name=MountainCar-v0 --num_train_steps=500000
```

### Atari environment
By default, we have the following settings for all `run_atari.py` file:

* `num_iterations: 10`
* `num_train_steps: 1e6`
* `num_eval_steps: 1e5`

For Atari, we omit the need to include 'NoFrameskip' and version in the `environment_name` args, as it will be handled by `create_atari_environment` in the `gym_env.py` module.
By default, it uses `NoFrameskip-v4` for the specified game.

#### Note
* For Atari games, we scale the observation with the atari wrappers BEFORE store the states into experience replay, as a reference 100000 samples some times can use ~14GB of RAM, pay attention if you want to increase the size of experience replay.
* Due to hardware limitation, for DQN (and the enhancements like double Q, rainbow, IQN, etc.), we set the maximum experience replay size to 50000 instead of 1000000.
* To speed up training on Atari games with DQN (and the enhancements like double Q, rainbow, IQN, etc.), we use 1000 instead of 10000 as interval to update target Q network.

```
python3 -m deep_rl_zoo.dqn.run_atari

# change environment to Breakout
python3 -m deep_rl_zoo.dqn.run_atari --environment_name=Breakout
```

### Multiple actors (on single machine)
When running multiple actors on GPU, watching out for possible CUDA OUT OF MEMORY error.

```
python3 -m deep_rl_zoo.a2c.run_classic --num_actors=16

python3 -m deep_rl_zoo.impala.run_atari --num_actors=16
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

The main logic to write to tensorboard is implemented in `trackers.py`,
mainly the classes `TensorboardEpisodTracker`, `TensorboardStepRateTracker`, and `TensorboardAgentStatisticsTracker`.

* we only write logs after episode terminates
* we separate training and evaluation logs
* we don't write loss to Tensorboard, as in RL the loss is not used to assess agent performance
* for agents that support parallel training, only log the first and last actors, this is controlled by `run_parallel_training_iterations` in `main_loop.py` module

Here are the performance measurements (measured over env steps) available on Tensorboard:
* `episode_return` the last episode return
* `episode_steps` the last episode steps
* `num_episodes` how many episodes have been conducted
* `run_duration(minutes)` the duration (in minutes) since the start of the session
* `step_rate(second)` step pre seconds, pre actors

In addition, it'll log whatever is exposed in the `agent.statistics` such as learning rate, discount, updates etc.

### Add tags to Tensorboard
This could be handy if we want to compare different hyper parameter's performances
```
python3 -m deep_rl_zoo.impala.run_classic --use_lstm --learning_rate=0.001 --tag=LSTM-LR0.001
```

### Example of DQN agent

#### CartPole (training on CPU)
![Tensorboard performance](../main/screenshots/DQN_on_CartPole_01.png)
![Tensorboard agent statistics](../main/screenshots/DQN_on_CartPole_02.png)

#### Pong (training on CPU)
![Tensorboard performance](../main/screenshots/DQN_on_Pong_01.png)
![Tensorboard agent statistics](../main/screenshots/DQN_on_Pong_02.png)

#### Breakout (training on CPU)
![Tensorboard performance](../main/screenshots/DQN_on_Breakout_01.png)
![Tensorboard agent statistics](../main/screenshots/DQN_on_Breakout_02.png)


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
