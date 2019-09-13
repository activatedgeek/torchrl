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
  observation_space, action_space = get_gym_spaces(runner.make_env)
  agent = GymRandomAgent(observation_space, action_space)
  trajectory_list = runner.rollout(agent)
  runner.close()

  assert len(trajectory_list) == runner.n_envs
  assert trajectory_list[0].obs.ndim == 2
  assert trajectory_list[0].action.ndim == 2
  assert trajectory_list[0].reward.ndim == 2
  assert trajectory_list[0].next_obs.ndim == 2
  assert trajectory_list[0].done.ndim == 2
