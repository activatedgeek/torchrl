import gym
import numpy as np

from torchrl import CPUReplayBuffer, EpisodeRunner
import torchrl.registry as registry
from torchrl.registry.problems import Problem
from torchrl.learners import BaseDQNLearner


class CartPoleDQNLearner(BaseDQNLearner):
  def learn(self, obs, action, reward, next_obs, done, **kwargs):
    for i in range(len(reward)):
      if done[i] == 1:
        reward[i] = -1.0

    return super(CartPoleDQNLearner, self).learn(obs, action, reward, next_obs, done)


@registry.register_problem('dqn-cartpole-v1')
class CartPoleDQN(Problem):
  def __init__(self, args):
    args.env = 'CartPole-v1'
    super(CartPoleDQN, self).__init__(args)

    self.buffer = CPUReplayBuffer(args.buffer_size)

  def init_agent(self):
    args = self.args
    env = self.make_env()

    agent = CartPoleDQNLearner(
      env.observation_space,
      env.action_space,
      lr=args.actor_lr,
      gamma=args.gamma,
      target_update_interval=args.target_update_interval)

    if args.cuda:
        agent.cuda()

    env.close()
    return agent

  def make_env(self):
    return gym.make('CartPole-v1')

  def train(self, history_list: list):
    # Populate the buffer
    batch_history = EpisodeRunner.merge_histories(
      self.agent.observation_space, self.agent.action_space, *history_list)
    transitions = list(zip(*batch_history))
    self.buffer.extend(transitions)

    if len(self.buffer) >= self.args.batch_size:
      transition_batch = self.buffer.sample(self.args.batch_size)
      transition_batch = list(zip(*transition_batch))
      transition_batch = [np.array(item) for item in transition_batch]
      value_loss = self.agent.learn(*transition_batch)
      return value_loss


@registry.register_hparam('dqn-cartpole')
def hparam():
  import torchrl.registry.hparams as hparams
  params = hparams.base_dqn()

  params.seed = 1
  params.rollout_steps = 1
  params.num_processes = 1
  params.actor_lr = 1e-3
  params.gamma = 0.8
  params.target_update_interval = 5
  params.eps_min = 0.1
  params.buffer_size = 5000
  params.batch_size = 64
  params.num_total_steps = 12000

  params.eval_interval = 250

  return params
