from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
  """Take random actions on any environment.

  @NOTE: Work in Progress. Not supported yet.
  """
  @property
  def models(self) -> list:
    return []

  @property
  def state(self) -> object:
    return None

  def act(self, *args, **kwargs):
    return self.action_space.sample()

  def learn(self, *args, **kwargs):
    pass

