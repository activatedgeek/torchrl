import numpy as np


def epsilon_greedy(action_size: int, choices: np.array, eps: float = 0.1):
  """
  Batched epsilon-greedy
  :param action_size: Total number of actions
  :param choices: A list of choices
  :param eps: Value of epsilon
  :return:
  """
  distribution = np.ones((len(choices), action_size), dtype=np.float32) * eps / action_size
  distribution[np.arange(len(choices)), choices] += 1.0 - eps
  actions = np.array([
    np.random.choice(np.arange(action_size), p=dist)
    for dist in distribution
  ])
  return np.expand_dims(actions, axis=1)
