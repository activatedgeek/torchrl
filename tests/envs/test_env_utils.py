import pytest

import gym
from torchrl.envs import make_gym_env


@pytest.mark.parametrize('spec_id', [
    'Acrobot-v1',
    'CartPole-v1',
    'MountainCar-v0',
    'MountainCarContinuous-v0',
    'Pendulum-v0',
])
def test_make_gym_env(spec_id: str):
  env = make_gym_env(spec_id)
  assert isinstance(env, gym.Env)
  assert env.spec.id == spec_id
  env.close()
