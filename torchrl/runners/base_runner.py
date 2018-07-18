import abc
from ..agents import BaseAgent


class BaseRunner(metaclass=abc.ABCMeta):
  """An abstract class for runner.

  This class provides the spec for what
  a runner should be.
  """
  MAX_STEPS = int(1e6)

  @abc.abstractmethod
  def make_env(self, seed: int = None):
    """Create the environment, optionally with seed."""
    raise NotImplementedError

  @abc.abstractmethod
  def compute_action(self, agent: BaseAgent, obs_list: list):
    """Use the agent to get actions.

    This method should be overridden to
    make use of the agent.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def process_transition(self, history,
                         transition: tuple) -> list:
    """Process the transition tuple and update history.

    This routine takes the transition tuple of
    (state, action, reward, next_state, done, info),
    applies transformation and appends to the history
    list. This method should be overridden for any
    non-trivial transformations needed, for instance
    conversion of boolean done to stack of ints.

    The first call to this method will have history as None.
    This allows for flexibility in terms of the format to
    storing history. Make sure to handle the None case and
    mutate as desired.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def rollout(self, agent, steps: int = None,
              render: bool = False, fps: int = 30):
    """Execute a rollout of the given environment.

    This is a simple utility and the main entrypoint to
    the runner. It allows flags for rendering and the
    maximum number of steps to execute in the current
    rollout.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def close(self):
    """Method to cleanup the runner instance."""
    raise NotImplementedError
