import gym
from typing import Optional


def make_gym_env(spec_id: str, seed: Optional[int] = None) -> gym.Env:
  env = gym.make(spec_id)
  env.seed(seed)
  return env
