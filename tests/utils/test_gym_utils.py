import gym
from torchrl.utils.gym_utils import get_gym_spaces, init_run_history


def make_env():
  return gym.make('CartPole-v0')


def test_gym_spaces():
  observation_space, action_space = get_gym_spaces(make_env)

  assert isinstance(observation_space, gym.Space)
  assert isinstance(action_space, gym.Space)


def test_init_run_history():
  env = make_env()
  history = init_run_history(env.observation_space, env.action_space)

  assert len(history) == 5
