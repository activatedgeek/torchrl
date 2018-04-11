import numpy as np


def epsilon_greedy(n, choice, eps=0.1):
    """
    Return the action chosen epsilon-greedily from a set of n actions
    numbered 0 to n - 1

    :param n: Total number of actions, int
    :param choice: Choice of the action chosen, range [0, n - 1]
    :param eps: Float value of epsilon, range [0, 1)
    :return: Action chosen and its probability, range [0, n - 1], [0, 1)
    """
    p = np.ones(n, dtype=np.float32) * eps / n
    p[choice] += 1.0 - eps
    action = np.random.choice(np.arange(n), p=p)
    return action, p[action]
