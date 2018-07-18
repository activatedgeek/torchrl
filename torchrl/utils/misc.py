import os
import sys
import re
import importlib
import torch
import numpy as np
import random


# @NOTE: https://stackoverflow.com/a/1176023/2425365
first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z])([A-Z])')


def to_camel_case(name: str):
  cap_sub = first_cap_re.sub(r'\1_\2', name)
  return all_cap_re.sub(r'\1_\2', cap_sub).lower()


def import_usr_dir(usr_dir):
  dir_path = os.path.abspath(os.path.expanduser(usr_dir).rstrip("/"))
  containing_dir, module_name = os.path.split(dir_path)
  sys.path.insert(0, containing_dir)
  importlib.import_module(module_name)
  sys.path.pop(0)


def set_seeds(seed):
  """
  Set the seed value for PyTorch, NumPy and Python.
  Important for reproducible runs!
  :param seed: seed value
  :return:
  """
  if seed is None:
    return
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)


def minibatch_generator(*args, minibatch_size=5):
  total_len = len(args[0])
  minibatch_idx = np.random.choice(total_len, minibatch_size)
  for _ in range(total_len // minibatch_size):
    yield tuple(map(lambda item: item[minibatch_idx, :], args))
    minibatch_idx = np.random.choice(total_len, minibatch_size)
