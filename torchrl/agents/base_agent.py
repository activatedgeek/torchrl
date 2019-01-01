import abc
import torch
import numpy as np


class BaseAgent(metaclass=abc.ABCMeta):
  """
  This is base agent specification which can encapsulate everything
  how a Reinforcement Learning Algorithm would function.
  """
  def __init__(self, observation_space, action_space):
    self.observation_space = observation_space
    self.action_space = action_space
    self.device = None

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

  def to(self, device: torch.device):
    """
    This routine is takes the agent's :code:`models` attribute
    and sends them to a device.

    See https://pytorch.org/docs/stable/nn.html#torch.nn.Module.to.

    Args:
        device (:class:`torch.device`):

    Returns:
        Updated class reference.
    """
    self.device = device
    for model in self.models:
      model.to(device)

    return self

  def train(self, flag: bool = True):
    """
    This routine is takes the agent's :code:`models` attribute
    and applies the training flag.

    See https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train.

    Args:
        flag (bool): :code:`True` or :code:`False`
    """
    for model in self.models:
      model.train(flag)

  def obs_to_tensor(self, obs):
    with torch.no_grad():
      batch_obs_tensor = torch.from_numpy(
          np.array(obs)
      ).float()

    if self.device:
      batch_obs_tensor = batch_obs_tensor.to(self.device)

    return batch_obs_tensor
