import os
import sys
import gym
from torchrl.utils.misc import get_gym_spaces, import_usr_dir, to_camel_case


def test_import_usr_dir():
  usr_dir = os.path.dirname(__file__)
  import_usr_dir(usr_dir)

  assert os.path.dirname(usr_dir) not in sys.path


def test_gym_spaces():
  def make_env():
    return gym.make('CartPole-v0')

  observation_space, action_space = get_gym_spaces(make_env)

  assert isinstance(observation_space, gym.Space)
  assert isinstance(action_space, gym.Space)


def test_camel_case():
  assert to_camel_case('A2C4') == 'a2c4'
  assert to_camel_case('ABC') == 'abc'
  assert to_camel_case('DoesThisWork') == 'does_this_work'
  assert to_camel_case('doesThisAlsoWork') == 'does_this_also_work'
