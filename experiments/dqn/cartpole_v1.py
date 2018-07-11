import gym
import torchrl.registry as registry
import torchrl.utils as utils
from torchrl.problems import base_hparams, DQNProblem
from torchrl.agents import BaseDQNAgent


@registry.register_problem('dqn-cartpole-v1')
class CartPoleDQNProblem(DQNProblem):
  def make_env(self):
    return gym.make('CartPole-v1')

  def init_agent(self):
    observation_space, action_space = utils.get_gym_spaces(self.make_env)

    agent = BaseDQNAgent(
        observation_space,
        action_space,
        double_dqn=self.hparams.double_dqn,
        lr=self.hparams.actor_lr,
        gamma=self.hparams.gamma,
        target_update_interval=self.hparams.target_update_interval)

    return agent


@registry.register_hparam('dqn-cartpole')
def hparam_dqn_cartpole():
  params = base_hparams.base_dqn()

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

@registry.register_hparam('double-dqn-cartpole')
def hparam_double_dqn_cartpole():
  params = hparam_dqn_cartpole()

  params.double_dqn = True
  params.target_update_interval = 5

  return params
