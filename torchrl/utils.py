import torch
import numpy as np
import random


def set_seeds(seed):
    """
    Set the seed value for PyTorch, NumPy and Python. Important for reproducible runs!
    :param seed: seed value
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
