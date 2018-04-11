import numpy as np


def epsilon_greedy(action_size, choice, eps=0.1):
    """
    Return the action chosen epsilon-greedily from a set of n actions
    numbered 0 to n - 1

    :param action_size: Total number of actions, int
    :param choice: Choice of the action chosen, range [0, n - 1]
    :param eps: Float value of epsilon, range [0, 1)
    :return: Action chosen and its probability, range [0, n - 1], [0, 1)
    """
    distribution = np.ones(action_size, dtype=np.float32) * eps / action_size
    distribution[choice] += 1.0 - eps
    action = np.random.choice(np.arange(action_size), p=distribution)
    return action, distribution[action]
