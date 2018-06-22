import abc


class BaseLearner(metaclass=abc.ABCMeta):
  """
  This is base runner specification which can encapsulate everything
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
  def learn(self, *args, **kwargs):
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

  @abc.abstractmethod
  def save(self, save_dir):
    """
    Store the agent for future usage
    :param save_dir: The directory to save arbitrary files to
    :return:
    """
    raise NotImplementedError

  @abc.abstractmethod
  def load(self, load_dir):
    """
    Load a pre-trained agent
    :param load_dir: The directory from a pre-trained agent
    (must be inverse of save)
    :return:
    """
    raise NotImplementedError
