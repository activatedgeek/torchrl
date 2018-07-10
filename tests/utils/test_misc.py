import gym
from torchrl.utils.misc import get_gym_spaces


def test_gym_spaces():
  def make_env():
    return gym.make('CartPole-v0')

  observation_space, action_space = get_gym_spaces(make_env)

  assert isinstance(observation_space, gym.Space)
  assert isinstance(action_space, gym.Space)
