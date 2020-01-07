import gym
from typing import Optional, Callable, Tuple

from .wrappers import TransitionMonitor


def make_gym_env(spec_id: str, seed: Optional[int] = None) -> gym.Env:
  env = gym.make(spec_id)
  env.seed(seed)
  return TransitionMonitor(env)


def get_gym_spaces(make_env_fn: Callable[..., gym.Env]) -> Tuple[gym.Space, gym.Space]:  # pylint: disable=line-too-long
  """
  A utility function to get observation and actions spaces of a
  Gym environment
  """
  env = make_env_fn()
  observation_space = env.observation_space
  action_space = env.action_space
  env.close()
  return observation_space, action_space

