from torchrl import utils
from torchrl.problems import base_hparams
from torchrl.contrib.problems import A2CProblem
from torchrl.contrib.agents import BaseA2CAgent


class A2CCartpole(A2CProblem):
  def init_agent(self):
    observation_space, action_space = utils.get_gym_spaces(self.runner.make_env)

    agent = BaseA2CAgent(
        observation_space,
        action_space,
        lr=self.hparams.actor_lr,
        gamma=self.hparams.gamma,
        lmbda=self.hparams.lmbda,
        alpha=self.hparams.alpha,
        beta=self.hparams.beta)

    return agent

  @staticmethod
  def hparams_a2c_cartpole():
    params = base_hparams.base_pg()

    params.env_id = 'CartPole-v0'

    params.num_processes = 16

    params.rollout_steps = 5
    params.max_episode_steps = 500
    params.num_total_steps = int(1e6)

    params.alpha = 0.5
    params.gamma = 0.99
    params.beta = 1e-3
    params.lmbda = 1.0

    params.batch_size = 128
    params.tau = 1e-2
    params.actor_lr = 3e-4

    return params
