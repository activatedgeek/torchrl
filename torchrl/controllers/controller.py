import abc
import gym


class Controller(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def act(self, obs):
    raise NotImplementedError


class RandomController(Controller):
  def __init__(self, action_space: gym.Space):
    self.action_space = action_space

  def act(self, *_):
    return self.action_space.sample()
