import gym
import torchrl.registry as registry
from torchrl.problems import base_hparams, DQNProblem, PrioritizedDQNProblem
from torchrl.learners import BaseDQNLearner


class CartPoleDQNLearner(BaseDQNLearner):
  def compute_q_values(self, obs, action, reward, next_obs, done):
    for i, _ in enumerate(reward):
      if done[i] == 1:
        reward[i] = -1.0

    return super(CartPoleDQNLearner, self).compute_q_values(
        obs, action, reward, next_obs, done)


@registry.register_problem('dqn-cartpole-v1')
class CartPoleDQNProblem(DQNProblem):
  def make_env(self):
    return gym.make('CartPole-v1')

  def init_agent(self):
    observation_space, action_space = self.get_gym_spaces()

    agent = CartPoleDQNLearner(
        observation_space,
        action_space,
        double_dqn=self.hparams.double_dqn,
        lr=self.hparams.actor_lr,
        gamma=self.hparams.gamma,
        target_update_interval=self.hparams.target_update_interval)

    return agent


@registry.register_problem('prioritized-dqn-cartpole-v1')
class PrioritizedCartPoleDQNProblem(PrioritizedDQNProblem):
  def make_env(self):
    return gym.make('CartPole-v1')

  def init_agent(self):
    observation_space, action_space = self.get_gym_spaces()

    agent = CartPoleDQNLearner(
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
  params.gamma = 0.8
  params.target_update_interval = 5
  params.eps_min = 0.15
  params.buffer_size = 5000
  params.batch_size = 64
  params.num_total_steps = 10000

  return params

@registry.register_hparam('ddqn-cartpole')
def hparam_ddqn_cartpole():
  params = hparam_dqn_cartpole()

  params.double_dqn = True

  return params
