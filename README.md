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
                            [--usr-dirs] [--cuda] [--no-cuda] [--log-dir]
                            [--save-dir] [--load-dir]

optional arguments:
  -h, --help            show this help message and exit
  --problem PROBLEM     Problem name
  --hparam-set HPARAM_SET
                        Hyperparameter set name
  --usr-dirs            Comma-separated list of user module directories
  --cuda                Enable CUDA
  --no-cuda             Disable CUDA
  --log-dir             Directory to store logs
  --save-dir            Directory to store agent
  --load-dir            Directory to load agent
```

# Experiments

See [experiments](./experiments.md) for ongoing implementations.
