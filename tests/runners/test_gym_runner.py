import pytest

from torchrl.runners.gym_runner import GymRunner
from torchrl.agents.gym_random_agent import GymRandomAgent
from torchrl.utils import get_gym_spaces


@pytest.mark.parametrize('env_id', [
    'Acrobot-v1',
    'CartPole-v1',
    'MountainCar-v0',
    'MountainCarContinuous-v0',
    'Pendulum-v0',
])
def test_gym_runner(env_id: str):
  runner = GymRunner(env_id)
  agent = GymRandomAgent(*get_gym_spaces(runner.make_env))
  runner.rollout(agent)
  runner.close()
