# pylint: disable=redefined-outer-name

"""Test Experiments.

This test runs all problems and hyperparameter
pairs for 100 time steps. It only guarantees
correct API compatiblity and not the problem
performance metrics.
"""

import pytest
from torchrl import registry
from torchrl.cli.commands.run import do_run


problem_hparams_tuples = []
for problem_id, hparams_list in registry.list_problem_hparams().items():
  for hparam_set_id in hparams_list:
    problem_hparams_tuples.append((problem_id, hparam_set_id))


@pytest.fixture(scope='function')
def problem_argv(request):
  problem_id, hparam_set_id = request.param
  args_dict = {
      'problem': problem_id,
      'hparam_set': hparam_set_id,
      'seed': None,
      'extra_hparams': {
          'num_total_steps': 100,
      },
      'log_interval': 50,
      'eval_interval': 50,
      'num_eval': 1,
  }

  yield args_dict


@pytest.mark.parametrize('problem_argv', problem_hparams_tuples,
                         indirect=['problem_argv'])
def test_problem(problem_argv):
  problem = problem_argv.pop('problem')
  do_run(problem, **problem_argv)
