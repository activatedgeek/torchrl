import abc


class BaseAgent(metaclass=abc.ABCMeta):
  """
  This is base agent specification which can encapsulate everything
  how a Reinforcement Learning Algorithm would function.
  """
  def __init__(self, observation_space, action_space):
    self.is_cuda = False
    self.observation_space = observation_space
    self.action_space = action_space

  @property
  def models(self) -> list:
    """
    This routine must return the list of trainable
    networks which external routines might want to
    generally operate on
    :return:
    """
    raise NotImplementedError

  @property
  def checkpoint(self) -> object:
    """
    This method must return an arbitrary object
    which defines the complete state of the agent
    to restore at any point in time
    """
    raise NotImplementedError

  @checkpoint.setter
  def checkpoint(self, cp):
    """
    This method must be the complement of
    `self.state` and restore the state
    """
    raise NotImplementedError

  @abc.abstractmethod
  def act(self, *args, **kwargs):
    """
    This is the method that should be called at every step of the episode.
    IMPORTANT: This method should be compatible with batches

    :param state: Representation of the state
    :return: identity of the action to be taken, as desired by the environment
    """
    raise NotImplementedError

  @abc.abstractmethod
  def learn(self, *args, **kwargs) -> dict:
    """
    This method represents the learning step
    """
    raise NotImplementedError

  def reset(self):
    """
    Optional function to reset learner's internals
    :return:
    """
    pass
