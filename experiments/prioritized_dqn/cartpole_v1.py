from torchrl import utils
from torchrl.contrib.problems import PrioritizedDQNProblem
from torchrl.contrib.agents import BaseDQNAgent

from ..dqn.cartpole_v1 import DQNCartpole


class PERCartpole(PrioritizedDQNProblem):
  def init_agent(self):
    observation_space, action_space = utils.get_gym_spaces(self.runner.make_env)

    agent = BaseDQNAgent(
        observation_space,
        action_space,
        double_dqn=self.hparams.double_dqn,
        lr=self.hparams.actor_lr,
        gamma=self.hparams.gamma,
        num_eps_steps=self.hparams.num_eps_steps,
        target_update_interval=self.hparams.target_update_interval)

    return agent

  @staticmethod
  def hparams_per_cartpole():
    params = DQNCartpole.hparams_dqn_cartpole()

    params.alpha = 0.6
    params.beta = 0.4
    params.beta_anneal_steps = 1000

    return params
