import abc
import numpy as np
import gym
import torch
from tensorboardX import SummaryWriter

from .hparams import HParams
from .. import MultiEpisodeRunner, EpisodeRunner, CPUReplayBuffer
from ..learners import BaseLearner
from ..utils import set_seeds, minibatch_generator


class Problem(metaclass=abc.ABCMeta):
  """
  An abstract class which defines functions to define
  any RL problem
  """

  def __init__(self, args: HParams):
    """
    The constructor takes in all the parameters needed by
    the problem.
    :param args:
    """
    self.args = args
    self.agent = self.init_agent()
    self.runner = self.make_runner(n_runners=args.num_processes,
                                   base_seed=args.seed)
    self.logger = SummaryWriter(log_dir=args.log_dir)

  @abc.abstractmethod
  def init_agent(self) -> BaseLearner:
    """
    Use this method to initialize the learner using `self.args`
    object
    :return: BaseLearner
    """
    raise NotImplementedError

  @abc.abstractmethod
  def make_env(self):
    """
    This method should return a Gym-like environment
    :return: gym.Env
    """
    raise NotImplementedError

  # TODO: this should use `self.make_env`
  def make_runner(self, n_runners=1, base_seed=None) -> MultiEpisodeRunner:
    return MultiEpisodeRunner(self.args.env,
                              max_steps=self.args.max_episode_steps,
                              n_runners=n_runners,
                              base_seed=base_seed)

  def get_gym_spaces(self):
    """
    A utility function to get observation and actions spaces of a
    Gym environment
    """
    env = self.make_env()
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    return observation_space, action_space

  @abc.abstractmethod
  def train(self, history_list: list):
    """
    This method is called after a rollout and must
    contain the logic for updating the agent's weights
    :return:
    """
    raise NotImplementedError

  def eval(self):
    """
    This method is called after a rollout and must
    contain the logic for updating the agent's weights
    :return:
    """
    runner = EpisodeRunner(self.make_env(),
                           max_steps=self.args.max_episode_steps)
    rewards = []
    for _ in range(self.args.num_eval):
      runner.reset()
      _, _, reward_history, _, _ = runner.collect(self.agent, store=True)
      rewards.append(np.sum(reward_history, axis=0))
    runner.stop()

    return np.average(rewards)

  # TODO: add logging back
  def run(self):
    """
    This is the entrypoint to a problem class and can be overridden
    if the train and eval need to be done at a different point in the
    epoch
    :return:
    """
    args = self.args
    set_seeds(args.seed)

    n_epochs = args.num_total_steps // args.rollout_steps // args.num_processes

    for epoch in range(1, n_epochs + 1):
      self.agent.train()
      history_list = self.runner.collect(self.agent,
                                         steps=args.rollout_steps,
                                         store=True)

      self.train(history_list)

      for i, history in enumerate(history_list):
        if history[-1][-1] == 1:
          self.runner.reset(i)
          self.agent.reset()

      if epoch % args.eval_interval == 0:
        self.agent.eval()
        print('Avg. Reward at Epoch {}: {}'.format(epoch, self.eval()))

      if args.save_dir and epoch % args.save_interval == 0:
        self.agent.save(args.save_dir)

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.runner.stop()
    self.logger.close()


class DQNProblem(Problem):
  def __init__(self, args):
    super(DQNProblem, self).__init__(args)

    self.buffer = CPUReplayBuffer(args.buffer_size)

  def make_env(self):
    return gym.make(self.args.env)

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
      loss = self.agent.learn(*transition_batch)
      return loss


class DDPGProblem(DQNProblem):
  def train(self, history_list: list):
    # only overriding to make the return decomposition clear
    loss = super(DDPGProblem, self).train(history_list)
    if loss is not None:
      actor_loss, critic_loss = loss
      return actor_loss, critic_loss


class A2CProblem(Problem):
  def make_env(self):
    return gym.make(self.args.env)

  def train(self, history_list: list):
    # Merge histories across multiple trajectories
    batch_history = EpisodeRunner.merge_histories(
        self.agent.observation_space, self.agent.action_space, *history_list)
    returns = np.concatenate([self.agent.compute_returns(*history)
                              for history in history_list], axis=0)

    actor_loss, critic_loss, entropy_loss = self.agent.learn(*batch_history,
                                                             returns)
    return actor_loss, critic_loss, entropy_loss


class PPOProblem(Problem):
  def make_env(self):
    return gym.make(self.args.env)

  def train(self, history_list: list):
    # Merge histories across multiple trajectories
    batch_history = EpisodeRunner.merge_histories(
        self.agent.observation_space, self.agent.action_space, *history_list)

    # Compute returns, log probabilities and values
    returns, log_probs, values = np.empty((0, 1), dtype=float), \
                                 torch.empty(0, 1), \
                                 np.empty((0, 1), dtype=float)
    for history in history_list:
      returns_, log_probs_, values_ = self.agent.compute_returns(*history)
      returns = np.concatenate((returns, returns_), axis=0)
      log_probs = torch.cat([log_probs, log_probs_], dim=0)
      values = np.concatenate((values, values_), axis=0)
    advantages = returns - values

    # Train the agent
    actor_loss, critic_loss, entropy_loss = None, None, None
    for _ in range(self.args.ppo_epochs):
      for data in minibatch_generator(*batch_history,
                                      returns, log_probs, advantages,
                                      minibatch_size=self.args.batch_size):
        actor_loss, critic_loss, entropy_loss = self.agent.learn(*data)

    return actor_loss, critic_loss, entropy_loss
