from .base_agent import BaseAgent


class GymRandomAgent(BaseAgent):
  """Take random actions on a Gym environment.

  This is only tested on Classic Control
  environments from OpenAI Gym. It is only
  meant to get started working with new environments.
  """
  @property
  def models(self) -> list:
    return []

  @property
  def checkpoint(self) -> object:
    return None

  def act(self, obs):
    return [[self.action_space.sample()] for _ in range(len(obs))]

  def learn(self, *args, **kwargs):
    return {}
