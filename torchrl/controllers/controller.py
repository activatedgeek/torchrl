import abc
import gym


class Controller(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def act(self, obs):
    raise NotImplementedError

  def learn(self) -> dict:
    '''Placeholder method for the learning algorithm
    '''
    return {}


class RandomController(Controller):
  def __init__(self, action_space: gym.Space):
    self.action_space = action_space

  def act(self, obs):
    return [self.action_space.sample()
            for _ in range(len(obs))]
