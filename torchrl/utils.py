import torch
import numpy as np
import random


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

