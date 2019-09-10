from torchrl import utils
from torchrl.problems import base_hparams, DQNProblem
from torchrl.agents import BaseDQNAgent


class DQNCartpole(DQNProblem):
  def init_agent(self):
    observation_space, action_space = utils.get_gym_spaces(self.runner.make_env)

    agent = BaseDQNAgent(
        observation_space,
        action_space,
        double_dqn=self.hparams.double_dqn,
        lr=self.hparams.actor_lr,
        gamma=self.hparams.gamma,
        target_update_interval=self.hparams.target_update_interval)

    return agent

  @staticmethod
  def hparams_dqn_cartpole():
    params = base_hparams.base_dqn()

    params.env_id = 'CartPole-v1'

    params.rollout_steps = 1
    params.num_processes = 1
    params.actor_lr = 1e-3
    params.gamma = 0.99
    params.target_update_interval = 10
    params.eps_min = 1e-2
    params.buffer_size = 1000
    params.batch_size = 32
    params.num_total_steps = 10000
    params.num_eps_steps = 500

    return params

  @staticmethod
  def hparams_double_dqn_cartpole():
    params = DQNCartpole.hparams_dqn_cartpole()

    params.double_dqn = True
    params.target_update_interval = 5

    return params
