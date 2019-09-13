from torchrl import utils
from torchrl.problems import base_hparams
from torchrl.contrib.problems import PPOProblem
from torchrl.contrib.agents import BasePPOAgent


class PPOPendulum(PPOProblem):
  def init_agent(self):
    observation_space, action_space = utils.get_gym_spaces(self.runner.make_env)

    agent = BasePPOAgent(
        observation_space,
        action_space,
        lr=self.hparams.actor_lr,
        gamma=self.hparams.gamma,
        lmbda=self.hparams.lmbda,
        alpha=self.hparams.alpha,
        beta=self.hparams.beta,
        max_grad_norm=self.hparams.max_grad_norm)

    return agent

  @staticmethod
  def hparams_ppo_pendulum():
    params = base_hparams.base_ppo()

    params.env_id = 'Pendulum-v0'

    params.rollout_steps = 20
    params.num_processes = 16
    params.num_total_steps = int(5e6)

    params.batch_size = 64

    params.actor_lr = 3e-4

    params.alpha = 0.5
    params.gamma = 0.99
    params.beta = 1e-3
    params.lmbda = 0.95

    params.clip_ratio = 0.2
    params.max_grad_norm = 1.0
    params.ppo_epochs = 4

    return params
