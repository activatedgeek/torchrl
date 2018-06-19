# TorchRL

[![Build Status](https://travis-ci.org/activatedgeek/torchrl.svg?branch=master)](https://travis-ci.org/activatedgeek/torchrl)
[![PyPI version](https://badge.fury.io/py/torchrl.svg)](https://pypi.org/project/torchrl/)
![Project Status](https://img.shields.io/badge/status-stable-green.svg)


## Objectives

* Modularity in the RL pipeline

* Clean implementations of basic ideas

* Scalability

# Install

```
$ pip install torchrl
```

# Usage

```
$ torchrl -h
usage: RL Experiment Runner [-h] --problem PROBLEM --hparam-set HPARAM_SET
                            [--seed] [--usr-dirs] [--cuda] [--no-cuda]
                            [--log-dir] [--save-dir] [--load-dir]
                            [--log-interval] [--eval-interval] [--num-eval]

optional arguments:
  -h, --help            show this help message and exit
  --problem PROBLEM     Problem name (default: None)
  --hparam-set HPARAM_SET
                        Hyperparameter set name (default: None)
  --seed                Random seed (default: None)
  --usr-dirs            Comma-separated list of user module directories
                        (default: )
  --cuda                Enable CUDA (default: True)
  --no-cuda             Disable CUDA (default: True)
  --log-dir             Directory to store logs (default: log)
  --save-dir            Directory to store agent (default: None)
  --load-dir            Directory to load agent (default: None)
  --log-interval        Log interval w.r.t epochs (default: 100)
  --eval-interval       Eval interval w.r.t epochs (default: 500)
  --num-eval            Number of evaluations (default: 10)
```

# Experiments

## DQN on CartPole-v1

```
$ torchrl --problem=dqn-cartpole-v1 --hparam-set=dqn-cartpole --seed=1 --usr-dirs=experiments
```


## A2C on CartPole-v0

```
$ torchrl --problem=a2c-cartpole-v0 --hparam-set=a2c-cartpole --seed=1 --usr-dirs=experiments
```

## DDPG on Pendulum-v0


```
$ torchrl --problem=ddpg-pendulum-v0 --hparam-set=ddpg-pendulum --seed=1 --usr-dirs=experiments
```

## PPO on Pendulum-v0

```
$ torchrl --problem=ppo-pendulum-v0 --hparam-set=ppo-pendulum --seed=1 --usr-dirs=experiments
```

