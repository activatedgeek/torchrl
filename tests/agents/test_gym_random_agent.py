# pylint: disable=redefined-outer-name

import gym
import argparse
import pytest
import torchrl.registry as registry
import torchrl.utils as utils
import torchrl.problems.base_hparams as base_hparams
from torchrl.agents.random_gym_agent import RandomGymClassicControlAgent


@pytest.mark.parametrize('env_id', [
    'Acrobot-v1',
    'CartPole-v1',
    'MountainCar-v0',
    'MountainCarContinuous-v0',
    'Pendulum-v0',
])
def test_classic_control_problem(env_id):
  class RandomGymProblem(registry.Problem):
    def make_env(self):
      return gym.make(env_id)

    def init_agent(self):
      observation_space, action_space = utils.get_gym_spaces(self.make_env)

      return RandomGymClassicControlAgent(observation_space, action_space)

    def train(self, history_list: list) -> dict:
      return {}

  args = argparse.Namespace(**{
      'seed': None, 'num_eval': 1,
      'log_interval': 1000, 'eval_interval': 1000})
  hparams = base_hparams.base()

  problem = RandomGymProblem(hparams, args, None,
                             device='cpu', show_progress=False)
  problem.run()
