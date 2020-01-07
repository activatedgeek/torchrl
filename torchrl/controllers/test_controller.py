import gym
import pytest

from torchrl.controllers import RandomController


@pytest.mark.parametrize('spec_id', [
    'Acrobot-v1', 'CartPole-v1', 'MountainCar-v0',
    'MountainCarContinuous-v0', 'Pendulum-v0'])
def test_random_controller(spec_id: str):
  env = gym.make(spec_id)
  ctrl = RandomController(env.action_space)

  done = False
  obs = env.reset()
  while not done:
    action = ctrl.act([obs])[0]

    assert env.action_space.contains(action)

    _, _, done, _ = env.step(action)

  env.close()
  