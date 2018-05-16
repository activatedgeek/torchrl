import torch
import torch.nn as nn
import numpy as np
import random
import gym


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


def get_gym_spaces(env_id: str):
    env = gym.make(env_id)
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    return observation_space, action_space


def polyak_average(source, target, tau=1e-3):
    """
    Polyak Average from the source to the target
    :param tau: Polyak Averaging Parameter
    :param source: Source Module
    :param target: Target Module
    :return:
    """
    assert isinstance(source, nn.Module), '"source" should be of type nn.Module, found "{}"'.format(type(source))
    assert isinstance(target, nn.Module), '"target" should be of type nn.Module, found "{}"'.format(type(target))

    for src_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * src_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """
    This class implements the Ornstein-Uhlenbeck process for noise, generously taken from OpenAI
    Baselines
    """
    def __init__(self, mean, sigma, theta=0.15, delta_t=1e-2, x_init=None):
        self.theta = theta
        self.mean = mean
        self.sigma = sigma
        self.delta_t = delta_t
        self.x_init = x_init
        self.x_prev = None
        self.reset()

    def __call__(self):
        noise = self.x_prev + \
                self.theta * (self.mean - self.x_prev) * self.delta_t + \
                self.sigma * np.sqrt(self.delta_t) * np.random.normal(size=self.mean.shape)
        self.x_prev = noise
        return noise

    def reset(self):
        self.x_prev = self.x_init if self.x_init is not None else np.zeros_like(self.mean)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mean, self.sigma)
