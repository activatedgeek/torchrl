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
usage: RL Experiment Runner [-h] [--problem] [--hparam-set] [--extra-hparams]
                            [--seed] [--progress] [--no-cuda] [--device]
                            [--usr-dirs] [--log-dir] [--load-dir]
                            [--log-interval] [--eval-interval] [--num-eval]

optional arguments:
  -h, --help        show this help message and exit
  --problem         Problem name (default: )
  --hparam-set      Hyperparameter set name (default: )
  --extra-hparams   Comma-separated list of extra key-value pairs,
                    automatically handles types int/float/str (default: )
  --seed            Random seed (default: None)
  --progress        Show epoch progress (default: False)
  --no-cuda         Disable CUDA (default: False)
  --device          Device selection for GPU (default: cuda)
  --usr-dirs        Comma-separated list of user module directories (default:
                    )
  --log-dir         Directory to store logs (default: log)
  --load-dir        Directory to load agent (default: None)
  --log-interval    Log interval w.r.t epochs (default: 100)
  --eval-interval   Eval interval w.r.t epochs (default: 1000)
  --num-eval        Number of evaluations (default: 10)
```

# Experiments

## DQN on CartPole-v1

```
$ torchrl --problem=dqn-cartpole-v1 --hparam-set=dqn-cartpole --seed=1 \
    --usr-dirs=experiments --log-dir=log/dqn --progress
```


## A2C on CartPole-v0

```
$ torchrl --problem=a2c-cartpole-v0 --hparam-set=a2c-cartpole --seed=1 \
    --usr-dirs=experiments --log-dir=log/a2c --progress
```

## DDPG on Pendulum-v0


```
$ torchrl --problem=ddpg-pendulum-v0 --hparam-set=ddpg-pendulum --seed=1 \
    --usr-dirs=experiments --log-dir=log/ddpg --progress
```

## PPO on Pendulum-v0

```
$ torchrl --problem=ppo-pendulum-v0 --hparam-set=ppo-pendulum --seed=1 \
    --usr-dirs=experiments --log-dir=log/ppo --progress
```

# Resume Experiments

To reload an experiment from previous run, say for instance the DQN run
above,

```
$ torchrl --usr-dirs=experiments --load-dir=log/dqn --log-dir=log/dqn-resumed \
    --progress
```

This will read all the other parameters from the directory and load the latest
checkpoint.
