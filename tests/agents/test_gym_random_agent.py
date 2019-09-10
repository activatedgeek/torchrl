# pylint: disable=redefined-outer-name

import pytest
import argparse
from torchrl import utils
from torchrl.problems import base_hparams
from torchrl.problems.gym_problem import GymProblem
from torchrl.agents.gym_random_agent import GymRandomAgent


class RandomGymProblem(GymProblem):
  def init_agent(self):
    observation_space, action_space = utils.get_gym_spaces(
        self.runner.make_env)

    return GymRandomAgent(observation_space, action_space)

  def train(self, history_list: list) -> dict:
    return {}

  @staticmethod
  def hparams_random_gym_problem():
    params = base_hparams.base()
    return params


@pytest.fixture(scope='function')
def problem_argv(request):
  env_id = request.param
  args_dict = {
      'seed': None,
      'extra_hparams': {
          'env_id': env_id,
          'num_total_steps': 100,
      },
      'log_interval': 50,
      'eval_interval': 50,
      'num_eval': 1,
  }

  yield args_dict


@pytest.mark.parametrize('problem_argv', [
    'Acrobot-v1',
    'CartPole-v1',
    'MountainCar-v0',
    'MountainCarContinuous-v0',
    'Pendulum-v0',
], indirect=['problem_argv'])
def test_gym_agent(problem_argv):

  def wrap(extra_hparams, **kwargs):
    hparams = RandomGymProblem.hparams_random_gym_problem()
    hparams.update(extra_hparams)

    problem = RandomGymProblem(hparams, argparse.Namespace(**kwargs),
                               None, device='cpu')

    problem.run()

  wrap(**problem_argv)
