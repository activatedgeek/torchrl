import abc
from ..agents import BaseAgent


class BaseRunner(metaclass=abc.ABCMeta):
  """
  This class defines how any environment must be
  executed to generate trajectories. The
  :code:`MAX_STEPS` property must be respected by
  any derived runner so as to prevent infinite
  horizon trajectories during rollouts.
  """

  MAX_STEPS = int(1e6)

  @abc.abstractmethod
  def make_env(self, seed: int = None):
    """
    This method **must** be overriden by a derived class
    and create the environment. For uniformity, any
    subsequent usage of the environment must be via the
    runner so that they are reproducible (for instance in
    terms of the arguments like seed and so on).

    Args:
        seed (int): Optional seed for the environment creation.

    Returns:
        An object representing the environment. For instance it could
        be of type :class:`gym.Env`.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def compute_action(self, agent: BaseAgent, obs_list: list):
    """
    This helper method **must** be overriden by any derived
    class. It allows for flexible runners where any pre/post
    processing might be needed before/after the :code:`agent`'s
    :meth:`~torchrl.agents.base_agent.BaseAgent.act` is called.

    Args:
        agent (:class:`~torchrl.agents.base_agent.BaseAgent`): Any derived \
          agent.
        obs_list (list): A list of observations corresponding to each parallel \
          environment.

    Returns:
        A (potentially post-processed) action returned by any
        :class:`~torchrl.agents.base_agent.BaseAgent`.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def process_transition(self, history,
                         transition: tuple) -> list:
    """
    This helper method **must** be overriden by any derived
    class. Effectively, this method should take in all
    previous history and append the current transition tuple.

    .. warning::

        The first call to this method will have history as :code:`None`.
        This allows for flexibility in terms of the storage format of
        history. Make sure to handle this case and mutate as desired.
        See :class:`~torchrl.runners.gym_runner.GymRunner` for example.

    Args:
        history: A set of history items. The derived class is free to \
          choose any type.
        transition (tuple): A transition tuple which represents current \
          observation, action, reward, next observation and termination flag.
          Typically, this is a 5-tuple however the derived class is free to
          add more information here as long as handled appropriately.

    Returns:
        The update history object.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def rollout(self, agent, steps: int = None,
              render: bool = False, fps: int = 30):
    """
    This is the main entrypoint for a runner object. Given
    an agent, it rolls out a trajectory of specified length.
    Optionally, it also allows a render flag.

    .. warning::

        Care must be taken when an environment reaches its terminal state.
        This could either be transparently resetting the environment or by
        other means. See :class:`~torchrl.runners.gym_runner.GymRunner` for
        example which resets the environment as and when needed.

    Args:
        agent (:class:`~torchrl.agents.base_agent.BaseAgent`): Any derived \
          agent object.
        steps (int): An optional maximum number of steps to rollout the \
          environments for. If :code:`None`, the :code:`MAX_STEPS` is used.
        render (bool): A flag to render the environment.
        fps (int): Amount of sleep before the code can start executing after \
          each render.

    Returns:
        A list of all objects needed for each parallel environment. Typically,
        this would involve the full trajectory for each environment which
        is defined by a list of transition tuples.
        See :class:`~torchrl.runners.gym_runner.GymRunner` for a concrete
        example.

    Todo:
        * :code:`render` flag does not work across multiple threads while
          debugging. Tracked by
          `#53 <https://github.com/activatedgeek/torchrl/issues/53>`_.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def close(self):
    """
    Cleanup any artifacts created by the runner. Typically,
    this will involve shutting down the environments and
    cleaning up the parallel trajectory threads.
    """
    raise NotImplementedError
