# pylint: disable=redefined-outer-name

import pytest
from torchrl import registry
from torchrl import utils
from torchrl.cli.commands.run import do_run
from torchrl.problems import base_hparams
from torchrl.problems.gym_problem import GymProblem
from torchrl.agents.gym_random_agent import GymRandomAgent


@pytest.fixture(scope='function')
def problem_argv(request):
  env_id = request.param
  args_dict = {
      'problem': 'random_gym_problem',
      'hparam_set': 'random_gym_problem',
      'seed': None,
      'extra_hparams': {
          'num_total_steps': 100,
      },
      'log_interval': 50,
      'eval_interval': 50,
      'num_eval': 1,
  }

  @registry.register_problem  # pylint: disable=unused-variable
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
      params.env_id = env_id
      return params

  yield args_dict

  registry.remove_problem('random_gym_problem')
  registry.remove_hparam('random_gym_problem')


@pytest.mark.parametrize('problem_argv', [
    'Acrobot-v1',
    'CartPole-v1',
    'MountainCar-v0',
    'MountainCarContinuous-v0',
    'Pendulum-v0',
], indirect=['problem_argv'])
def test_gym_agent(problem_argv):
  problem = problem_argv.pop('problem')
  do_run(problem, **problem_argv)
